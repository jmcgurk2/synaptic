"""Recall engine — synthesises knowledge from Synaptic entries via LiteLLM."""

import json
import os
import re
import logging
from datetime import datetime

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

RECALL_SYSTEM_PROMPT = """You are Synaptic, a personal knowledge assistant. The user is asking a question or requesting a briefing.

Below you will receive a set of knowledge entries retrieved from the user's second brain. Each entry has a type, title, summary, project tag, and raw text.

Your job:
1. **Synthesise** — don't list entries, write a coherent narrative answer grounded in the provided entries.
2. **Cite** — reference entry titles naturally (e.g. "Based on your note about X...").
3. **Connect** — if you notice relationships between entries, surface them.
4. **Flag gaps** — if the entries don't fully answer the question, say what's missing.
5. **Open items** — if there are tasks, pending decisions, or stale items, call them out.

Keep it concise but complete. Use markdown formatting for readability.

If there are no relevant entries, say so honestly — don't fabricate knowledge."""

BRIEF_SYSTEM_PROMPT = """You are Synaptic, a personal knowledge assistant generating a project briefing.

Below you will receive all knowledge entries tagged to a specific project. Each entry has a type, title, summary, project tag, timestamps, and raw text.

Generate a structured project briefing:
1. **Status summary** — one paragraph on where this project stands
2. **Key decisions** — any decisions captured or still pending
3. **Open tasks** — tasks that haven't been marked complete
4. **Recent activity** — what's happened lately (based on timestamps)
5. **Ideas & notes** — relevant ideas or admin items
6. **Connections** — anything that links to other projects or entries

Be concise and actionable. Use markdown formatting."""


def _get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=os.getenv("LITELLM_BASE_URL", "http://litellm.mohawkops.ai:4000"),
        api_key=os.getenv("LITELLM_API_KEY", "sk-placeholder"),
    )


def _entries_to_context(entries: list[dict]) -> str:
    """Format entries as context block for the LLM."""
    lines = []
    for i, e in enumerate(entries, 1):
        proj = f" [{e.get('project', '')}]" if e.get("project") else ""
        created = e.get("created_at", "")
        lines.append(
            f"--- Entry {i} ---\n"
            f"Type: {e['type']}{proj}\n"
            f"Title: {e['title']}\n"
            f"Summary: {e['summary']}\n"
            f"Tags: {', '.join(e.get('tags', []))}\n"
            f"Created: {created}\n"
            f"Raw: {e.get('raw_text', '')[:300]}"
        )
    return "\n\n".join(lines)


async def recall(query: str, entries: list[dict], mode: str = "recall") -> dict:
    """Synthesise an answer from retrieved entries.
    
    Args:
        query: The user's question or topic
        entries: List of entry dicts from search
        mode: "recall" for Q&A, "brief" for project briefing
    
    Returns:
        dict with: answer (str), sources (list of entry IDs/titles), entry_count (int)
    """
    client = _get_client()
    model = os.getenv("LITELLM_RECALL_MODEL", os.getenv("LITELLM_MODEL", "claude-haiku"))
    
    if not entries:
        return {
            "answer": f"I don't have any knowledge entries relevant to '{query}'. Try capturing some notes first.",
            "sources": [],
            "entry_count": 0,
        }
    
    context = _entries_to_context(entries)
    system_prompt = BRIEF_SYSTEM_PROMPT if mode == "brief" else RECALL_SYSTEM_PROMPT
    
    user_message = f"Question: {query}\n\n--- Retrieved Knowledge ({len(entries)} entries) ---\n\n{context}"
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=1500,
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Build sources list
        sources = [
            {"id": e.get("id", ""), "title": e["title"], "type": e["type"], "project": e.get("project")}
            for e in entries
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "entry_count": len(entries),
        }
    
    except Exception as e:
        logger.error("Recall LLM call failed: %s", e)
        return {
            "answer": f"Recall failed — LLM error: {str(e)[:200]}",
            "sources": [],
            "entry_count": len(entries),
        }
