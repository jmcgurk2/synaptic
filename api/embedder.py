import os
import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_local_model = None
_litellm_client: AsyncOpenAI | None = None
_use_litellm: bool = False


def init_embedder():
    """Initialise embedding backend. Prefer LiteLLM if LITELLM_EMBED_MODEL is set."""
    global _local_model, _litellm_client, _use_litellm

    embed_model = os.getenv("LITELLM_EMBED_MODEL", "")
    litellm_base = os.getenv("LITELLM_BASE_URL", "")

    if embed_model and litellm_base:
        _litellm_client = AsyncOpenAI(
            base_url=litellm_base,
            api_key=os.getenv("LITELLM_API_KEY", "sk-placeholder"),
        )
        _use_litellm = True
        logger.info("Embedder: using LiteLLM model %s", embed_model)
    else:
        from sentence_transformers import SentenceTransformer

        _local_model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )
        _use_litellm = False
        logger.info("Embedder: using local nomic-embed-text-v1.5")


def embed_sync(text: str) -> list[float]:
    """Synchronous embedding for use where async is not available."""
    if _local_model is None:
        raise RuntimeError("Local embedder not initialised — call init_embedder first")
    return _local_model.encode(text, normalize_embeddings=True).tolist()


async def embed(text: str) -> list[float]:
    """Embed text using LiteLLM or local model."""
    if _use_litellm and _litellm_client is not None:
        model = os.getenv("LITELLM_EMBED_MODEL", "nomic-embed-text")
        resp = await _litellm_client.embeddings.create(model=model, input=[text])
        return resp.data[0].embedding

    if _local_model is not None:
        return _local_model.encode(text, normalize_embeddings=True).tolist()

    raise RuntimeError("No embedding backend initialised — call init_embedder first")


def get_vector_size() -> int:
    return 768
