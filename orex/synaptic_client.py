"""
HTTP client for the Synaptic API (VM 120, :8000).

Orex is a consumer and producer of Synaptic knowledge. All persistent
memory operations go through this client.
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger("orex.synaptic")


class SynapticClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    async def capture(
        self,
        text: str,
        source: str = "@orex",
        project: str | None = None,
    ) -> dict[str, Any]:
        """POST /capture — store knowledge in Synaptic."""
        payload: dict[str, Any] = {"text": text, "source": source}
        if project:
            payload["project"] = project

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(f"{self.base_url}/capture", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def search(
        self, query: str, limit: int = 5, project: str | None = None
    ) -> list[dict[str, Any]]:
        """GET /search — semantic search across all sources."""
        params: dict[str, Any] = {"q": query, "limit": limit}
        if project:
            params["project"] = project

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{self.base_url}/search", params=params)
            resp.raise_for_status()
            return resp.json()

    async def recall(
        self, query: str, project: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        """POST /recall — LLM-powered answer from knowledge base."""
        payload: dict[str, Any] = {"query": query, "limit": limit, "mode": "recall"}
        if project:
            payload["project"] = project

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{self.base_url}/recall", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def digest(self) -> dict[str, Any]:
        """POST /digest — trigger digest generation."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{self.base_url}/digest")
            resp.raise_for_status()
            return resp.json()

    async def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        """GET /context — get recent entries for conversation context."""
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                resp = await client.get(f"{self.base_url}/context")
                resp.raise_for_status()
                data = resp.json()
                # context endpoint returns {"recent": [...], "pending_fix": [...]}
                return data.get("recent", [])[:limit]
            except Exception as e:
                logger.warning("Failed to get recent context: %s", e)
                return []

    async def health(self) -> bool:
        """GET /health — check if Synaptic is up."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/health")
                return resp.status_code == 200
        except Exception:
            return False
