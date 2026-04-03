import os
import logging

from qdrant_client import QdrantClient, models

from embedder import get_vector_size

logger = logging.getLogger(__name__)

COLLECTION = "synaptic"

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "qdrant"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
        )
    return _client


def init_collection():
    """Create the Qdrant collection if it doesn't exist."""
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION not in collections:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=get_vector_size(),
                distance=models.Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection: %s", COLLECTION)


async def upsert(entry_id: str, vector: list[float], payload: dict):
    """Upsert a single point into Qdrant."""
    client = get_client()
    client.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=entry_id,
                vector=vector,
                payload=payload,
            )
        ],
    )


async def search(vector: list[float], limit: int = 10, filters: dict | None = None) -> list[dict]:
    """Semantic search. Returns list of {id, score, payload}."""
    client = get_client()

    qdrant_filter = None
    if filters:
        conditions = []
        for key, value in filters.items():
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            )
        qdrant_filter = models.Filter(must=conditions)

    results = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=limit,
        query_filter=qdrant_filter,
    )

    return [
        {"id": str(hit.id), "score": hit.score, "payload": hit.payload}
        for hit in results.points
    ]


async def delete(entry_id: str):
    """Delete a point by ID."""
    client = get_client()
    client.delete(
        collection_name=COLLECTION,
        points_selector=models.PointIdsList(points=[entry_id]),
    )


async def check_qdrant() -> str:
    try:
        client = get_client()
        client.get_collections()
        return "ok"
    except Exception:
        return "error"
