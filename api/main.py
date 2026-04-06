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
    RecallRequest,
    RecallResponse,
    SearchResult,
)
from recall import recall as do_recall
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
            return {"text": f"{project} -> Opened stream"}

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
    """Handle !fix type - reclassify last bounced entry for this channel."""
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

    lines = [f"- **{r.title}** ({r.type}) -- {r.summary}" for r in results[:5]]
    return {"text": f"**Search: {query}**\n" + "\n".join(lines)}


async def _handle_capture_webhook(
    text: str, channel_id: str, user_name: str = "", project: str | None = None, mode: str | None = None
) -> dict:
    """Capture via webhook - run classifier, apply bouncer.

    Supports #project (single) or [project] (stream) prefix modes.
    """
    req = CaptureRequest(text=text, source="@synaptic", channel_id=channel_id, project=project)
    result = await _do_capture(req, hint=project if mode == "single" else None)

    if result.status == "held_for_review":
        if channel_id:
            await post_message(
                channel_id,
                f"Not sure if this is a **{result.type}** -- reply `!fix <type>` to file it.",
            )
        return {"text": ""}

    # Format response based on mode
    if mode == "single" and project:
        return {"text": f"**{result.type}** [{project}]: {result.title}"}
    elif mode == "stream_open" and project:
        return {"text": f"{project} -> **{result.type}**: {result.title}"}
    else:
        # No project
        return {"text": f"Captured as **{result.type}**: {result.title}"}


async def _handle_report_webhook(subject: str, digest_channel: str) -> dict:
    """Handle !report subject - generate a formatted report for a subject/project."""
    from database import get_engine
    from collections import defaultdict

    if not subject:
        return {"text": "Usage: `!report <subject>` -- e.g. `!report mohawk` or `!report kitchen`"}

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
            summary = e.summary[:60] + "..." if len(e.summary) > 60 else e.summary
            lines.append(f"- **{e.title}** -- {summary}")
        lines.append("")

    await post_message(digest_channel, "\n".join(lines))
    return {"text": ""}


async def _handle_recent_webhook(digest_channel: str) -> dict:
    """Handle !recent - show 10 most recent entries."""
    from database import get_engine

    engine = get_engine()
    with Session(engine) as session:
        entries = session.exec(
            select(Entry).order_by(Entry.created_at.desc()).limit(10)
        ).all()

    if not entries:
        return {"text": "No recent entries."}

    lines = [f"- **{e.title}** ({e.type}) -- {e.created_at.strftime(%Y-%m-%d)}"]
    for e in entries:
        lines.append(f"- **{e.title}** ({e.type}) -- {e.created_at.strftime(%Y-%m-%d)}")

    await post_message(digest_channel, "\n".join(lines))
    return {"text": ""}


async def _handle_toc_webhook(digest_channel: str) -> dict:
    """Handle !toc - table of contents (group by type)."""
    from database import get_engine
    from collections import defaultdict

    engine = get_engine()
    with Session(engine) as session:
        entries = session.exec(select(Entry)).all()

    if not entries:
        return {"text": "No entries."}

    by_type = defaultdict(int)
    for e in entries:
        by_type[e.type] += 1

    lines = ["## Table of Contents", ""]
    for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
        lines.append(f"- **{t}**: {count}")

    await post_message(digest_channel, "\n".join(lines))
    return {"text": ""}


async def _handle_projects_webhook(digest_channel: str) -> dict:
    """Handle !projects - list all projects with entry counts."""
    from database import get_engine
    from collections import defaultdict

    engine = get_engine()
    with Session(engine) as session:
        entries = session.exec(select(Entry).where(Entry.project != None)).all()

    if not entries:
        return {"text": "No project entries."}

    by_project = defaultdict(int)
    for e in entries:
        if e.project:
            by_project[e.project] += 1

    lines = ["## Projects", ""]
    for p, count in sorted(by_project.items(), key=lambda x: -x[1]):
        lines.append(f"- **{p}**: {count}")

    await post_message(digest_channel, "\n".join(lines))
    return {"text": ""}


# --- REST API ---


@app.get("/health", response_class=PlainTextResponse)
async def health():
    checks = {
        "sqlite": await check_sqlite(),
        "qdrant": await check_qdrant(),
    }
    status = "ok" if all(checks.values()) else "degraded"
    return f"status={status}\n" + "\n".join(f"{k}={v}" for k, v in checks.items())


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return generate_latest()


@app.post("/capture", response_model=CaptureResponse)
async def capture(req: CaptureRequest, session: Session = Depends(get_session)):
    return await _do_capture(req)


@app.post("/search", response_model=list[SearchResult])
async def search_endpoint(req: RecallRequest):
    return await _do_search(req.query, limit=req.limit or 5)


@app.post("/recall", response_model=RecallResponse)
async def recall_endpoint(req: RecallRequest, session: Session = Depends(get_session)):
    return await do_recall(req, session)


@app.get("/context/{entry_id}", response_model=ContextResponse)
async def context(entry_id: int, session: Session = Depends(get_session)):
    entry = session.get(Entry, entry_id)
    if not entry:
        return {"error": f"Entry {entry_id} not found"}
    return {"entry": _entry_to_detail(entry)}


async def _do_capture(req: CaptureRequest, hint: str | None = None) -> CaptureResponse:
    """Classify text, check bouncer, store if pass, return response."""
    from database import get_engine

    result = await classify(req.text, hint=hint)

    # Bouncer: check confidence threshold
    if result["confidence"] < 0.7:
        engine = get_engine()
        with Session(engine) as session:
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
            id=None,
            type=result["type"],
            title=result["title"],
            tags=result["tags"],
            summary=result["summary"],
            status="held_for_review",
            project=None,
        )

    # Passed bouncer: embed, store, upsert to Qdrant
    vector = await embed(req.text)

    engine = get_engine()
    with Session(engine) as session:
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
        session.commit()
        entry_id = entry.id

    await upsert(
        entry_id,
        vector,
        {"type": result["type"], "title": result["title"], "source": req.source, "project": req.project},
    )
    entries_total.labels(type=result["type"], source=req.source).inc()
    captures_today.inc()

    return CaptureResponse(
        id=entry_id,
        type=result["type"],
        title=result["title"],
        tags=result["tags"],
        summary=result["summary"],
        status="stored",
        project=req.project,
    )


async def _do_search(query: str, limit: int = 5, project_filter: str | None = None) -> list[SearchResult]:
    """Search the vector store, optionally filtered by project."""
    from database import get_engine

    # Embed the query
    query_vector = await embed(query)

    # Search Qdrant
    results = await search(query_vector, limit=limit, project_filter=project_filter)

    if not results:
        return []

    # Look up full entries from database for display
    engine = get_engine()
    search_results = []
    with Session(engine) as session:
        for hit in results:
            entry = session.get(Entry, hit["id"])
            if entry:
                search_results.append(
                    SearchResult(
                        id=entry.id,
                        type=entry.type,
                        title=entry.title,
                        summary=entry.summary,
                        score=hit["score"],
                    )
                )

    return search_results
