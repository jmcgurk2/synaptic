import json
import os
import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a personal knowledge classifier. Classify the input into exactly one type:
- Project: something with multiple steps or ongoing work
- Idea: a thought, insight, or future possibility
- Task: a single discrete action to take
- Contact: information about a person or organisation
- Admin: reference info, credentials, logistics

Return ONLY valid JSON. No preamble. No markdown. JSON only:
{
  "type": "Project|Idea|Task|Contact|Admin",
  "title": "short descriptive title, max 8 words",
  "tags": ["tag1", "tag2"],
  "summary": "one sentence summary",
  "confidence": 0.0
}"""

VALID_TYPES = {"Project", "Idea", "Task", "Contact", "Admin"}


def _get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=os.getenv("LITELLM_BASE_URL", "http://litellm.mohawkops.ai:4000"),
        api_key=os.getenv("LITELLM_API_KEY", "sk-placeholder"),
    )


async def classify(text: str, hint: str | None = None) -> dict:
    """Classify text via LiteLLM. Returns dict with type, title, tags, summary, confidence."""
    client = _get_client()
    model = os.getenv("LITELLM_MODEL", "claude-haiku")

    user_message = text
    if hint:
        user_message = f"The user says this should be classified as {hint}. Reclassify with that in mind.\n\n{text}"

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
        )

        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)

        # Validate required fields
        if result.get("type") not in VALID_TYPES:
            raise ValueError(f"Invalid type: {result.get('type')}")
        if not isinstance(result.get("tags"), list):
            result["tags"] = []
        if not isinstance(result.get("confidence"), (int, float)):
            result["confidence"] = 0.0

        result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))

        return {
            "type": result["type"],
            "title": str(result.get("title", "Untitled"))[:80],
            "tags": [str(t) for t in result["tags"][:10]],
            "summary": str(result.get("summary", ""))[:500],
            "confidence": result["confidence"],
        }

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("Classification failed: %s", e)
        return {
            "type": "Admin",
            "title": "Unclassified",
            "tags": [],
            "summary": text[:200],
            "confidence": 0.0,
        }
    except Exception as e:
        logger.error("LiteLLM call failed: %s", e)
        return {
            "type": "Admin",
            "title": "Unclassified",
            "tags": [],
            "summary": text[:200],
            "confidence": 0.0,
        }
