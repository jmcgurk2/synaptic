"""
Orex Agent Service — conversational AI assistant backed by Synaptic.

Receives Mattermost webhooks from #orex, classifies intent, routes to
the appropriate backend (Synaptic for memory, LiteLLM for conversation),
and posts replies.
"""

import logging
import os
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request

from intent import classify_intent, Intent
from synaptic_client import SynapticClient
from llm_client import LLMClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("orex")

BOT_MENTION = "@orex"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.synaptic = SynapticClient(
        base_url=os.getenv("SYNAPTIC_URL", "http://localhost:8000")
    )
    app.state.llm = LLMClient(
        base_url=os.getenv("LITELLM_BASE_URL", "http://litellm.mohawkops.ai:4000"),
        api_key=os.getenv("LITELLM_API_KEY", ""),
        model=os.getenv("LITELLM_MODEL", "mistral-small"),
        classify_model=os.getenv("LITELLM_CLASSIFY_MODEL", "mistral-small"),
    )
    logger.info("Orex agent service started")
    yield
    logger.info("Orex agent service stopped")


app = FastAPI(title="Orex", version="0.1.0", lifespan=lifespan)


def _strip_mention(text: str) -> str:
    """Remove @orex mention prefix from message text."""
    for prefix in [f"{BOT_MENTION} ", f"{BOT_MENTION}\n"]:
        if text.lower().startswith(prefix.lower()):
            return text[len(prefix) :].strip()
    return text


def _validate_token(payload: dict) -> bool:
    """Validate Mattermost outgoing webhook token."""
    expected = os.getenv("MATTERMOST_WEBHOOK_TOKEN", "")
    if not expected:
        logger.warning("MATTERMOST_WEBHOOK_TOKEN not set — accepting all webhooks")
        return True
    return payload.get("token") == expected


# ---------------------------------------------------------------------------
# Webhook endpoint — Mattermost posts here
# ---------------------------------------------------------------------------


@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.json()

    if not _validate_token(payload):
        return {"text": "Unauthorized"}

    text = _strip_mention(payload.get("text", "").strip())
    user_name = payload.get("user_name", "")
    channel_id = payload.get("channel_id", "")

    if not text:
        return {"text": ""}

    synaptic: SynapticClient = request.app.state.synaptic
    llm: LLMClient = request.app.state.llm

    # Classify intent
    intent = classify_intent(text)

    # If keyword rules didn't match, ask the LLM
    if intent.action == "unknown":
        intent = await llm.classify_intent(text)

    logger.info(
        "user=%s intent=%s text=%.60s", user_name, intent.action, text
    )

    # Route to the right handler
    try:
        if intent.action == "capture":
            return await _handle_capture(synaptic, text, intent, channel_id)
        elif intent.action == "search":
            return await _handle_search(synaptic, intent.argument or text)
        elif intent.action == "recall":
            return await _handle_recall(synaptic, intent.argument or text)
        elif intent.action == "briefing":
            return await _handle_briefing(synaptic)
        elif intent.action == "conversation":
            return await _handle_conversation(llm, synaptic, text, user_name)
        else:
            return await _handle_conversation(llm, synaptic, text, user_name)
    except Exception as e:
        logger.exception("Error handling message")
        return {"text": f"Something went wrong: {e}"}


# ---------------------------------------------------------------------------
# Intent handlers
# ---------------------------------------------------------------------------


async def _handle_capture(
    synaptic: SynapticClient, text: str, intent: Intent, channel_id: str
) -> dict:
    """Capture knowledge to Synaptic."""
    capture_text = intent.argument or text
    project = intent.project

    result = await synaptic.capture(
        text=capture_text, source="@orex", project=project
    )

    if result.get("status") == "held_for_review":
        return {
            "text": f"Hmm, not confident about that one — classified as **{result.get('type', '?')}**. "
            f"You can reclassify in #synaptic-brain with `!fix <type>`."
        }

    entry_type = result.get("type", "Note")
    title = result.get("title", capture_text[:50])
    proj = result.get("project")

    if proj:
        return {"text": f"Got it. **{entry_type}** [{proj}]: {title}"}
    else:
        return {"text": f"Got it. **{entry_type}**: {title}"}


async def _handle_search(synaptic: SynapticClient, query: str) -> dict:
    """Search Synaptic knowledge base."""
    results = await synaptic.search(query, limit=5)

    if not results:
        return {"text": f"Nothing found for: {query}"}

    lines = []
    for r in results[:5]:
        lines.append(f"- **{r.get('title', '?')}** ({r.get('type', '?')}) — {r.get('summary', '')[:80]}")

    return {"text": f"**Found {len(results)} result(s):**\n" + "\n".join(lines)}


async def _handle_recall(synaptic: SynapticClient, query: str) -> dict:
    """Use Synaptic's recall endpoint for LLM-powered answers over knowledge."""
    result = await synaptic.recall(query)

    answer = result.get("answer", "I couldn't find an answer.")
    sources = result.get("sources", [])

    if sources:
        source_lines = ", ".join(
            s.get("title", "?") for s in sources[:3]
        )
        return {"text": f"{answer}\n\n_Sources: {source_lines}_"}
    return {"text": answer}


async def _handle_briefing(synaptic: SynapticClient) -> dict:
    """Trigger a Synaptic digest / briefing."""
    result = await synaptic.digest()
    return {"text": result.get("digest", "No briefing content available.")}


async def _handle_conversation(
    llm: LLMClient, synaptic: SynapticClient, text: str, user_name: str
) -> dict:
    """General conversation — optionally pull Synaptic context, then respond via LLM."""
    # Pull recent context from Synaptic to give the LLM some memory
    context_entries = await synaptic.recent(limit=5)

    context_text = ""
    if context_entries:
        context_text = "\n".join(
            f"- [{e.get('type')}] {e.get('title')}: {e.get('summary', '')[:100]}"
            for e in context_entries[:5]
        )

    response = await llm.chat(
        user_message=text,
        user_name=user_name,
        context=context_text,
    )

    return {"text": response}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "service": "orex"}
