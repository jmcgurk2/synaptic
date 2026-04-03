import os
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


def detect_intent(text: str) -> tuple[str, str]:
    """Detect command intent from message text.

    Returns (intent, argument) where intent is one of:
    - "fix": !fix <type> command
    - "search": !search <query> or ?<query>
    - "capture": default — store as knowledge
    """
    lower = text.lower().strip()

    if lower.startswith("!fix "):
        return "fix", text[5:].strip()

    if lower.startswith("!search "):
        return "search", text[8:].strip()

    if lower.startswith("?"):
        return "search", text[1:].strip()

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
