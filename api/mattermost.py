import os
import re
import logging

import httpx

logger = logging.getLogger(__name__)


def validate_webhook_token(payload: dict) -> bool:
    expected = os.getenv("MATTERMOST_WEBHOOK_TOKEN", "")
    if not expected:
        logger.warning("MATTERMOST_WEBHOOK_TOKEN not set — accepting all webhooks")
        return True
    return payload.get("token") == expected


def parse_webhook(payload: dict) -> dict:
    """Extract relevant fields from Mattermost outgoing webhook payload."""
    text = payload.get("text", "").strip()
    channel_id = payload.get("channel_id", "")
    user_name = payload.get("user_name", "")

    # Strip bot mention prefix (@synaptic)
    for prefix in ["@synaptic ", "@synaptic\n"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            break

    return {
        "text": text,
        "channel_id": channel_id,
        "user_name": user_name,
    }


def extract_project(text: str) -> tuple[str, str | None, str | None]:
    """Extract project mode from message text.

    Returns (cleaned_text, project_name_or_None, mode_or_None).

    Modes:
    - "single": #project prefix (single-shot capture)
    - "stream_open": [project] prefix (opens a stream)
    - "stream_close": [done] (closes stream)
    - None: no prefix

    Examples:
        "#orex the proxmox cluster needs a reboot plan"
        -> ("the proxmox cluster needs a reboot plan", "orex", "single")

        "[orex] ordered the backsplash tile from Home Depot"
        -> ("ordered the backsplash tile from Home Depot", "orex", "stream_open")

        "[orex]" (alone, no text after)
        -> ("", "orex", "stream_open")

        "[done]"
        -> ("", None, "stream_close")

        "just a plain message"
        -> ("just a plain message", None, None)
    """
    # Match [done] to close stream
    if text.lower().strip() == "[done]":
        return "", None, "stream_close"

    # Match #tag at start of message (single-shot)
    m = re.match(r"^#(\S+)\s+(.+)", text, re.DOTALL)
    if m:
        return m.group(2).strip(), m.group(1).lower(), "single"

    # Match [tag] at start of message (stream open)
    m = re.match(r"^\[([^\]]+)\]\s*(.+)", text, re.DOTALL)
    if m:
        return m.group(2).strip(), m.group(1).strip().lower(), "stream_open"

    # Match [tag] alone (stream open, no following text)
    m = re.match(r"^\[([^\]]+)\]\s*$", text, re.DOTALL)
    if m:
        return "", m.group(1).strip().lower(), "stream_open"

    # No project prefix
    return text, None, None


def extract_hint(text: str) -> tuple[str, str | None]:
    """Legacy alias for backward compatibility. Uses extract_project internally."""
    clean_text, project, mode = extract_project(text)
    # For backward compatibility, return (clean_text, project_hint)
    # Only single-shot mode counts as a "hint" for the classifier
    if mode == "single":
        return clean_text, project
    return text, None


def detect_intent(text: str) -> tuple[str, str]:
    """Detect command intent from message text."""
    lower = text.lower().strip()

    if lower.startswith("!fix "):
        return "fix", text[5:].strip()

    if lower.startswith("!search "):
        return "search", text[8:].strip()

    if lower.startswith("!report"):
        arg = text[7:].strip()
        return "report", arg if arg else ""

    if lower.startswith("!recent"):
        return "recent", ""

    if lower.startswith("!toc"):
        return "toc", ""

    if lower.startswith("?"):
        return "search", text[1:].strip()

    if lower.startswith("!projects"):
        return "projects", ""
    return "capture", text


async def post_message(channel_id: str, text: str) -> None:
    """Post a message to a Mattermost channel via API v4."""
    url = os.getenv("MATTERMOST_URL", "https://chat.mohawkops.ai")
    token = os.getenv("MATTERMOST_BOT_TOKEN", "")

    if not token:
        logger.error("MATTERMOST_BOT_TOKEN not set — cannot post message")
        return

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{url}/api/v4/posts",
            json={"channel_id": channel_id, "message": text},
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if resp.status_code != 201:
            logger.error(
                "Failed to post to Mattermost: %s %s", resp.status_code, resp.text
            )
