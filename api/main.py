import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, generate_latest
from sqlmodel import Session, select

from classifier import classify
from database import Entry, ReceiptLog, check_sqlite, get_session, init_db
from embedder import embed, init_embedder
from mattermost import (
    detect_intent,
    extract_project,
    parse_webhook,
    post_message,
    validate_webhook_token,
)
from models import (
    CaptureRequest,
    CaptureResponse,
    ContextResponse,
    EntryDetail,
    FixRequest,
    HealthResponse,
    SearchResult,
)
from scheduler import build_digest, init_scheduler, send_digest, shutdown_scheduler
from vectorstore import check_qdrant, delete, init_collection, search, upsert

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Prometheus metrics
entries_total = Counter("synaptic_entries_total", "Total entries stored", ["type", "source"])
bounced_total = Counter("synaptic_bounced_total", "Total bounced entries")
captures_today = Counter("synaptic_captures_today", "Captures received today")

# Stream state: key = (user_name, channel_id), value = {"project": str, "updated_at": datetime}
_active_streams: dict[tuple[str, str], dict] = {}
STREAM_TIMEOUT_MINUTES = 30


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    init_embedder()
    init_collection()
    init_scheduler()
    logger.info("Synaptic API started")
    yield
    shutdown_scheduler()
    logger.info("Synaptic API stopped")


app = FastAPI(title="Synaptic", version="1.0.0", lifespan=lifespan)


def _entry_to_detail(e: Entry) -> EntryDetail:
    return EntryDetail(
        id=e.id,
        type=e.type,
        title=e.title,
        tags=json.loads(e.tags) if e.tags else [],
        summary=e.summary,
        raw_text=e.raw_text,
        source=e.source,
        confidence=e.confidence,
        status=e.status,
        project=e.project,
        created_at=e.created_at.isoformat(),
        updated_at=e.updated_at.isoformat(),
    )


def _get_active_stream(user_name: str, channel_id: str) -> dict | None:
    """Get active stream for user/channel, checking timeout."""
    key = (user_name, channel_id)
    stream = _active_streams.get(key)
    if not stream:
        return None

    # Check timeout
    age = datetime.utcnow() - stream["updated_at"]
    if age > timedelta(minutes=STREAM_TIMEOUT_MINUTES):
        del _active_streams[key]
        return None

    return stream


def _open_stream(user_name: str, channel_id: str, project: str) -> None:
    """Open a new stream for user/channel."""
    key = (user_name, channel_id)
    _active_streams[key] = {"project": project, "updated_at": datetime.utcnow()}


def _close_stream(user_name: str, channel_id: str) -> None:
    """Close stream for user/channel."""
    key = (user_name, channel_id)
    if key in _active_streams:
        del _active_streams[key]


# --- /webhook ---


@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.json()

    if not validate_webhook_token(payload):
        return {"text": "Unauthorized"}

    parsed = parse_webhook(payload)
    user_name = parsed["user_name"]
    channel_id = parsed["channel_id"]
    text = parsed["text"]

    # Extract project/mode from text BEFORE intent detection
    clean_text, project, mode = extract_project(text)

    # Handle stream modes
    if mode == "stream_close":
        _close_stream(user_name, channel_id)
        return {"text": f"Stream closed."}

    if mode == "stream_open":
        _open_stream(user_name, channel_id, project)
        if clean_text:
            # Stream open with text: capture and add to stream
            return await _handle_capture_webhook(
                clean_text, channel_id, user_name, project=project, mode=mode
            )
        else:
            # Stream open without text: just open and confirm
            return {"text": f"{project} → Opened stream"}

    # Check for active stream if no explicit project prefix
    if mode is None and project is None:
        active_stream = _get_active_stream(user_name, channel_id)
        if active_stream:
            project = active_stream["project"]
            _active_streams[(user_name, channel_id)]["updated_at"] = datetime.utcnow()

    # Now detect intent on potentially cleaned text
    intent, argument = detect_intent(text)
    digest_channel = os.getenv("MATTERMOST_DIGEST_CHANNEL_ID", channel_id)

    if intent == "fix":
        return await _handle_fix_webhook(argument, channel_id)
    elif intent == "search":
        return await _handle_search_webhook(argument, channel_id)
    elif intent == "report":
        return await _handle_report_webhook(argument, digest_channel)
    elif intent == "recent":
        return await _handle_recent_webhook(digest_channel)
    elif intent == "toc":
        return await _handle_toc_webhook(digest_channel)
    elif intent == "projects":
        return await _handle_projects_webhook(digest_channel)
    else:
        # Capture with potentially resolved project from stream
        return await _handle_capture_webhook(
            clean_text, channel_id, user_name, project=project, mode=mode
        )


async def _handle_fix_webhook(type_hint: str, channel_id: str) -> dict:
    """Handle !fix <type> — reclassify last bounced entry for this channel."""
    from database import get_engine

    engine = get_engine()
    with Session(engine) as session:
        last_bounced = session.exec(
            select(ReceiptLog)
            .where(ReceiptLog.disposition == "bounced")
            .order_by(ReceiptLog.created_at.desc())
            .limit(1)
        ).first()

        if not last_bounced:
            return {"text": "No bounced entries to fix."}

        result = await classify(last_bounced.raw_text, hint=type_hint)
        vector = await embed(last_bounced.raw_text)

        entry = Entry(
            type=result["type"],
            title=result["title"],
            tags=json.dumps(result["tags"]),
            summary=result["summary"],
            raw_text=last_bounced.raw_text,
            source=last_bounced.source,
            confidence=result["confidence"],
            status="stored",
        )
        session.add(entry)

        receipt = ReceiptLog(
            raw_text=last_bounced.raw_text,
            source=last_bounced.source,
            classified_as=result["type"],
            confidence=result["confidence"],
            disposition="fixed",
            entry_id=entry.id,
        )
        session.add(receipt)
        session.commit()

    await upsert(
        entry.id,
        vector,
        {"type": entry.type, "title": entry.title, "source": entry.source, "project": entry.project},
    )
    entries_total.labels(type=entry.type, source=entry.source).inc()

    return {"text": f"Filed as **{entry.type}**: {entry.title}"}


async def _handle_search_webhook(query: str, channel_id: str) -> dict:
    """Handle !search or ? queries via webhook."""
    results = await _do_search(query, limit=5)
    if not results:
        return {"text": f"No results for: {query}"}

    lines = [f"- **{r.title}** ({r.type}) — {r.summary}" for r in results[:5]]
    return {"text": f"**Search: {query}**\n" + "\n".join(lines)}


async def _handle_capture_webhook(
    text: str, channel_id: str, user_name: str = "", project: str | None = None, mode: str | None = None
) -> dict:
    """Capture via webhook — run classifier, apply bouncer.

    Supports #project (single) or [project] (stream) prefix modes.
    """
    req = CaptureRequest(text=text, source="@synaptic", channel_id=channel_id, project=project)
    result = await _do_capture(req, hint=project if mode == "single" else None)

    if result.status == "held_for_review":
        if channel_id:
            await post_message(
                channel_id,
                f"Not sure if this is a **{result.type}** — reply `!fix <type>` to file it.",
            )
        return {"text": ""}

    # Format response based on mode
    if mode == "single" and project:
        return {"text": f"**{result.type}** [{project}]: {result.title}"}
    elif mode == "stream_open" and project:
        return {"text": f"{project} → **{result.type}**: {result.title}"}
    else:
        # No project
        return {"text": f"Captured as **{result.type}**: {result.title}"}


async def _handle_report_webhook(subject: str, digest_channel: str) -> dict:
    """Handle !report <subject> — generate a formatted report for a subject/project."""
    from database import get_engine
    from collections import defaultdict

    if not subject:
        return {"text": "Usage: `!report <subject>` — e.g. `!report mohawk` or `!report kitchen`"}

    engine = get_engine()

    # Try project-filtered search first, fall back to text search
    results = await _do_search(subject, project_filter=subject.lower(), limit=30)
    if not results:
        results = await _do_search(subject, limit=30)

    if not results:
        return {"text": f"No entries found for **{subject}**"}

    # Group by type
    by_type = defaultdict(list)
    for r in results:
        by_type[r.type].append(r)

    # Build formatted report
    is_project = any(r.project and r.project.lower() == subject.lower() for r in results)
    if is_project:
        lines = [f"## Report: [{subject}]", ""]
    else:
        lines = [f"## Report: {subject}", ""]
    type_order = ["Project", "Task", "Idea", "Admin", "Contact"]
    for t in type_order:
        entries = by_type.get(t, [])
        if not entries:
            continue
        lines.append(f"### {t}s ({len(entries)})")
        for e in entries:
            tags_str = ", ".join(e.tags[:4]) if e.tags else ""
            lines.append(f"- **{e.title}** — {e.summary}")
            if tags_str:
                lines[-1] += f"  `{tags_str}`"
        lines.append("")

    lines.append(f"_Total: {len(results)} entries across {len(by_type)} categories_")
    report_text = "\n".join(lines)

    # Post to digest channel
    await post_message(digest_channel, report_text)
    return {"text": f"Report for **{subject}** posted to #synaptic-digest ({len(results)} entries)"}


async def _handle_recent_webhook(digest_channel: str) -> dict:
    """Handle !recent — show most recently updated subjects grouped by tag."""
    from database import get_engine
    from collections import defaultdict

    engine = get_engine()
    with Session(engine) as session:
        entries = session.exec(
            select(Entry).order_by(Entry.updated_at.desc()).limit(30)
        ).all()

    if not entries:
        return {"text": "No entries yet."}

    # Group by first tag (primary subject)
    subjects = defaultdict(lambda: {"count": 0, "latest": None, "types": set(), "project": None})
    for e in entries:
        tags = json.loads(e.tags) if e.tags else ["untagged"]
        primary = tags[0] if tags else "untagged"
        s = subjects[primary]
        s["count"] += 1
        s["types"].add(e.type)
        if e.project and not s["project"]:
            s["project"] = e.project
        if s["latest"] is None or e.updated_at > s["latest"]:
            s["latest"] = e.updated_at

    # Sort by most recent
    sorted_subjects = sorted(subjects.items(), key=lambda x: x[1]["latest"], reverse=True)

    lines = ["## Recently Updated Subjects", ""]
    lines.append("| Subject | Entries | Types | Last Updated |")
    lines.append("|---------|---------|-------|-------------|")
    for name, info in sorted_subjects[:20]:
        types_str = ", ".join(sorted(info["types"]))
        date_str = info["latest"].strftime("%Y-%m-%d %H:%M") if info["latest"] else "—"
        lines.append(f"| **{name}** | {info['count']} | {types_str} | {date_str} |")

    report_text = "\n".join(lines)
    await post_message(digest_channel, report_text)
    return {"text": f"Recent subjects posted to #synaptic-digest ({len(sorted_subjects)} subjects)"}


async def _handle_toc_webhook(digest_channel: str) -> dict:
    """Handle !toc — table of contents of all stored knowledge."""
    from database import get_engine
    from collections import defaultdict

    engine = get_engine()
    with Session(engine) as session:
        entries = session.exec(select(Entry)).all()

    if not entries:
        return {"text": "No entries yet."}

async def _handle_projects_webhook(digest_channel: str) -> dict:
    """Handle !projects — list all projects with entry counts and activity."""
    from database import get_engine
    from collections import defaultdict

    engine = get_engine()
    with Session(engine) as session:
        entries = session.exec(
            select(Entry).where(Entry.project != None)
        ).all()

    if not entries:
        return {"text": "No project-tagged entries yet. Use `#project` or `[project]` to tag captures."}

    # Group by project
    projects = defaultdict(lambda: {"count": 0, "types": defaultdict(int), "latest": None})
    for e in entries:
        p = projects[e.project]
        p["count"] += 1
        p["types"][e.type] += 1
        if p["latest"] is None or e.updated_at > p["latest"]:
            p["latest"] = e.updated_at

    # Sort by most recent activity
    sorted_projects = sorted(projects.items(), key=lambda x: x[1]["latest"], reverse=True)

    lines = ["## Active Projects", ""]
    lines.append("| Project | Entries | Breakdown | Last Activity |")
    lines.append("|---------|---------|-----------|--------------|")
    for name, info in sorted_projects:
        breakdown = ", ".join(f"{count} {t}" for t, count in sorted(info["types"].items(), key=lambda x: -x[1]))
        date_str = info["latest"].strftime("%Y-%m-%d %H:%M") if info["latest"] else "—"
        lines.append(f"| **{name}** | {info['count']} | {breakdown} | {date_str} |")

    report_text = "\n".join(lines)
    await post_message(digest_channel, report_text)
    return {"text": f"Projects posted to #synaptic-digest ({len(sorted_projects)} projects)"}


    # Build TOC grouped by type, then by primary tag
    by_type = defaultdict(lambda: defaultdict(list))
    for e in entries:
        tags = json.loads(e.tags) if e.tags else ["untagged"]
        primary = tags[0] if tags else "untagged"
        by_type[e.type][primary].append(e)

    lines = ["## Table of Contents", ""]
    lines.append(f"_Total: {len(entries)} entries_")
    lines.append("")

    type_order = ["Project", "Task", "Idea", "Admin", "Contact"]
    for t in type_order:
        subjects = by_type.get(t, {})
        if not subjects:
            continue
        total = sum(len(v) for v in subjects.values())
        lines.append(f"### {t}s ({total})")
        # Sort subjects by entry count descending
        for subj, ents in sorted(subjects.items(), key=lambda x: -len(x[1])):
            if len(ents) == 1:
                lines.append(f"- **{subj}**: {ents[0].title}")
            else:
                lines.append(f"- **{subj}** ({len(ents)} entries)")
                for e in sorted(ents, key=lambda x: x.updated_at, reverse=True)[:5]:
                    lines.append(f"  - {e.title}")
                if len(ents) > 5:
                    lines.append(f"  - _...and {len(ents) - 5} more_")
        lines.append("")

    report_text = "\n".join(lines)
    await post_message(digest_channel, report_text)
    return {"text": f"Table of contents posted to #synaptic-digest ({len(entries)} entries)"}


# --- /capture ---


@app.post("/capture", response_model=CaptureResponse)
async def capture(req: CaptureRequest, session: Session = Depends(get_session)):
    return await _do_capture(req, session, hint=None)


async def _do_capture(req: CaptureRequest, session: Session | None = None, hint: str | None = None) -> CaptureResponse:
    from database import get_engine

    result = await classify(req.text, hint=hint)
    # Inject hint tag if provided
    if hint:
        tags = result.get("tags", [])
        if hint not in tags:
            tags.insert(0, hint)
        result["tags"] = tags
    threshold = float(os.getenv("BOUNCER_THRESHOLD", "0.70"))
    captures_today.inc()

    own_session = session is None
    if own_session:
        engine = get_engine()
        session = Session(engine)

    try:
        if result["confidence"] < threshold:
            # Bouncer: hold for review
            receipt = ReceiptLog(
                raw_text=req.text,
                source=req.source,
                classified_as=result["type"],
                confidence=result["confidence"],
                disposition="bounced",
            )
            session.add(receipt)
            session.commit()
            bounced_total.inc()

            return CaptureResponse(
                id=receipt.id,
                status="held_for_review",
                type=result["type"],
                title=result["title"],
                tags=result["tags"],
                summary=result["summary"],
                confidence=result["confidence"],
                project=req.project,
            )

        # Store entry
        entry = Entry(
            type=result["type"],
            title=result["title"],
            tags=json.dumps(result["tags"]),
            summary=result["summary"],
            raw_text=req.text,
            source=req.source,
            confidence=result["confidence"],
            status="stored",
            project=req.project,
        )
        session.add(entry)

        receipt = ReceiptLog(
            raw_text=req.text,
            source=req.source,
            classified_as=result["type"],
            confidence=result["confidence"],
            disposition="stored",
            entry_id=entry.id,
        )
        session.add(receipt)
        session.commit()

        # Embed and upsert to Qdrant
        vector = await embed(req.text)
        await upsert(
            entry.id,
            vector,
            {"type": entry.type, "title": entry.title, "source": entry.source, "project": entry.project},
        )
        entries_total.labels(type=entry.type, source=entry.source).inc()

        return CaptureResponse(
            id=entry.id,
            status="stored",
            type=entry.type,
            title=entry.title,
            tags=result["tags"],
            summary=entry.summary,
            confidence=entry.confidence,
            project=entry.project,
        )
    finally:
        if own_session:
            session.close()


# --- /search ---


async def _do_search(
    q: str,
    type_filter: str | None = None,
    source_filter: str | None = None,
    project_filter: str | None = None,
    limit: int = 10,
) -> list[SearchResult]:
    from database import get_engine

    engine = get_engine()

    # Fan out to both stores
    # SQLite text search
    sql_results: dict[str, SearchResult] = {}
    with Session(engine) as session:
        query = select(Entry).where(
            Entry.title.contains(q)
            | Entry.summary.contains(q)
            | Entry.tags.contains(q)
        )
        if type_filter:
            query = query.where(Entry.type == type_filter)
        if source_filter:
            query = query.where(Entry.source == source_filter)
    if project_filter:
        query = query.where(Entry.project == project_filter)
        query = query.limit(limit)

        for e in session.exec(query).all():
            sql_results[e.id] = SearchResult(
                id=e.id,
                type=e.type,
                title=e.title,
                tags=json.loads(e.tags) if e.tags else [],
                summary=e.summary,
                source=e.source,
                confidence=e.confidence,
                score=0.0,
                project=e.project,
            )

    # Qdrant semantic search
    qdrant_results: list[SearchResult] = []
    try:
        qdrant_filters = {"project": project_filter} if project_filter else None
        qdrant_results = await search(q, limit=limit, filters=qdrant_filters)
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")

    # Merge results (Qdrant + SQL, deduplicated)
    merged = {r.id: r for r in qdrant_results}
    for r in sql_results.values():
        if r.id not in merged:
            merged[r.id] = r

    # Return sorted by score (Qdrant results have scores, SQL results have 0.0)
    results = sorted(merged.values(), key=lambda r: r.score, reverse=True)[:limit]
    return results


@app.get("/search", response_model=list[SearchResult])
async def search_endpoint(
    q: str = Query(..., min_length=1),
    type: str | None = Query(None),
    source: str | None = Query(None),
    project: str | None = Query(None),
    limit: int = Query(10, ge=1, le=100),
):
    return await _do_search(q, type_filter=type, source_filter=source, project_filter=project, limit=limit)


# --- /context ---


@app.get("/context", response_model=ContextResponse)
async def context(session: Session = Depends(get_session)):
    """Get recent entries and pending fixes."""
    recent = session.exec(
        select(Entry).order_by(Entry.created_at.desc()).limit(10)
    ).all()

    pending = session.exec(
        select(Entry).where(Entry.status == "pending_fix").limit(10)
    ).all()

    return ContextResponse(
        recent=[_entry_to_detail(e) for e in recent],
        pending_fix=[_entry_to_detail(e) for e in pending],
    )


# --- /entries ---


@app.get("/entries", response_model=list[EntryDetail])
async def list_entries(
    type: str | None = Query(None),
    source: str | None = Query(None),
    project: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    session: Session = Depends(get_session),
):
    """List entries with optional filters."""
    query = select(Entry)

    if type:
        query = query.where(Entry.type == type)
    if source:
        query = query.where(Entry.source == source)
    if project:
        query = query.where(Entry.project == project)

    query = query.order_by(Entry.created_at.desc()).limit(limit)
    entries = session.exec(query).all()

    return [_entry_to_detail(e) for e in entries]


@app.get("/entries/{entry_id}", response_model=EntryDetail)
async def get_entry(entry_id: str, session: Session = Depends(get_session)):
    """Get a single entry by ID."""
    entry = session.exec(select(Entry).where(Entry.id == entry_id)).first()
    if not entry:
        return {"error": "Entry not found"}, 404
    return _entry_to_detail(entry)


# --- /health ---


@app.get("/health", response_model=HealthResponse)
async def health():
    sqlite_status = await check_sqlite()
    qdrant_status = await check_qdrant()
    return HealthResponse(
        status="ok",
        sqlite=sqlite_status,
        qdrant=qdrant_status,
        version="1.0.0",
    )


# --- /metrics ---


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest())
