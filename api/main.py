import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

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
        created_at=e.created_at.isoformat(),
        updated_at=e.updated_at.isoformat(),
    )


# --- /webhook ---


@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.json()

    if not validate_webhook_token(payload):
        return {"text": "Unauthorized"}

    parsed = parse_webhook(payload)
    intent, argument = detect_intent(parsed["text"])

    if intent == "fix":
        return await _handle_fix_webhook(argument, parsed["channel_id"])
    elif intent == "search":
        return await _handle_search_webhook(argument, parsed["channel_id"])
    else:
        return await _handle_capture_webhook(parsed["text"], parsed["channel_id"])


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
        {"type": entry.type, "title": entry.title, "source": entry.source},
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


async def _handle_capture_webhook(text: str, channel_id: str) -> dict:
    """Capture via webhook — run classifier, apply bouncer."""
    req = CaptureRequest(text=text, source="@synaptic", channel_id=channel_id)
    result = await _do_capture(req)

    if result.status == "held_for_review":
        if channel_id:
            await post_message(
                channel_id,
                f"Not sure if this is a **{result.type}** — reply `!fix <type>` to file it.",
            )
        return {"text": ""}

    return {"text": f"Captured as **{result.type}**: {result.title}"}


# --- /capture ---


@app.post("/capture", response_model=CaptureResponse)
async def capture(req: CaptureRequest, session: Session = Depends(get_session)):
    return await _do_capture(req, session)


async def _do_capture(req: CaptureRequest, session: Session | None = None) -> CaptureResponse:
    from database import get_engine

    result = await classify(req.text)
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
            {"type": entry.type, "title": entry.title, "source": entry.source},
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
        )
    finally:
        if own_session:
            session.close()


# --- /search ---


async def _do_search(
    q: str,
    type_filter: str | None = None,
    source_filter: str | None = None,
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
                score=0.5,
            )

    # Qdrant semantic search
    vector = await embed(q)
    filters = {}
    if type_filter:
        filters["type"] = type_filter
    if source_filter:
        filters["source"] = source_filter

    qdrant_hits = await search(vector, limit=limit, filters=filters if filters else None)

    for hit in qdrant_hits:
        eid = hit["id"]
        if eid in sql_results:
            # Boost score for items found in both
            sql_results[eid].score = max(sql_results[eid].score, hit["score"])
        else:
            # Fetch full entry from SQLite
            with Session(engine) as session:
                entry = session.get(Entry, eid)
                if entry:
                    sql_results[eid] = SearchResult(
                        id=entry.id,
                        type=entry.type,
                        title=entry.title,
                        tags=json.loads(entry.tags) if entry.tags else [],
                        summary=entry.summary,
                        source=entry.source,
                        confidence=entry.confidence,
                        score=hit["score"],
                    )

    results = sorted(sql_results.values(), key=lambda r: r.score, reverse=True)
    return results[:limit]


@app.get("/search", response_model=list[SearchResult])
async def search_endpoint(
    q: str,
    type: str | None = Query(None),
    source: str | None = Query(None),
    limit: int = Query(10, ge=1, le=100),
):
    return await _do_search(q, type_filter=type, source_filter=source, limit=limit)


# --- /recall/{id} ---


@app.get("/recall/{entry_id}", response_model=EntryDetail)
async def recall(entry_id: str, session: Session = Depends(get_session)):
    entry = session.get(Entry, entry_id)
    if not entry:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Entry not found")
    return _entry_to_detail(entry)


# --- /entries/{id} PATCH (Fix Button) ---


@app.patch("/entries/{entry_id}", response_model=CaptureResponse)
async def fix_entry(entry_id: str, fix: FixRequest, session: Session = Depends(get_session)):
    entry = session.get(Entry, entry_id)
    if not entry:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Entry not found")

    result = await classify(entry.raw_text, hint=fix.type)

    entry.type = result["type"]
    entry.title = result["title"]
    entry.tags = json.dumps(result["tags"])
    entry.summary = result["summary"]
    entry.confidence = result["confidence"]
    entry.status = "stored"
    entry.updated_at = datetime.utcnow()
    session.add(entry)

    receipt = ReceiptLog(
        raw_text=entry.raw_text,
        source=entry.source,
        classified_as=result["type"],
        confidence=result["confidence"],
        disposition="fixed",
        entry_id=entry.id,
    )
    session.add(receipt)
    session.commit()

    # Re-embed and upsert
    vector = await embed(entry.raw_text)
    await upsert(
        entry.id,
        vector,
        {"type": entry.type, "title": entry.title, "source": entry.source},
    )

    return CaptureResponse(
        id=entry.id,
        status="stored",
        type=entry.type,
        title=entry.title,
        tags=result["tags"],
        summary=entry.summary,
        confidence=entry.confidence,
    )


# --- /context ---


@app.get("/context", response_model=ContextResponse)
async def context(session: Session = Depends(get_session)):
    recent = session.exec(
        select(Entry).order_by(Entry.created_at.desc()).limit(10)
    ).all()

    pending = session.exec(
        select(Entry).where(Entry.status == "pending_fix")
    ).all()

    return ContextResponse(
        recent=[_entry_to_detail(e) for e in recent],
        pending_fix=[_entry_to_detail(e) for e in pending],
    )


# --- /digest ---


@app.post("/digest")
async def trigger_digest():
    await send_digest()
    return {"status": "digest_sent"}


# --- /health ---


@app.get("/health", response_model=HealthResponse)
async def health():
    sqlite_status = await check_sqlite()
    qdrant_status = await check_qdrant()
    overall = "ok" if sqlite_status == "ok" and qdrant_status == "ok" else "degraded"

    return HealthResponse(
        status=overall,
        sqlite=sqlite_status,
        qdrant=qdrant_status,
        version="1.0.0",
    )


# --- /metrics ---


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
