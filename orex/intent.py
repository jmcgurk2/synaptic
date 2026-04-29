"""
Intent classification for Orex — hybrid keyword + LLM fallback (DEC-002).

Keyword rules handle the obvious patterns cheaply. Anything that doesn't
match falls through to the LLM classifier in llm_client.py.
"""

import re
from dataclasses import dataclass, field


@dataclass
class Intent:
    action: str  # capture, search, recall, briefing, conversation, unknown
    argument: str | None = None  # the query/text after the command
    project: str | None = None  # project name if #project or [project] syntax used


# Keyword patterns — ordered by specificity
_CAPTURE_KEYWORDS = [
    "remember",
    "save",
    "note",
    "note that",
    "log that",
    "store",
    "capture",
    "jot down",
    "keep track",
    "don't forget",
    "remind me",
]

_SEARCH_KEYWORDS = [
    "find",
    "search",
    "look up",
    "lookup",
    "look for",
    "what was",
    "what were",
    "where was",
    "where is",
    "when was",
    "when did",
    "who was",
    "do i have",
    "did i",
    "have i",
    "any notes",
    "anything about",
    "anything on",
]

_BRIEFING_KEYWORDS = [
    "morning briefing",
    "briefing",
    "daily briefing",
    "brief me",
    "what do i have going on",
    "what's going on",
    "what's on my plate",
    "what's happening",
    "daily digest",
    "digest",
]

_RECALL_KEYWORDS = [
    "tell me about",
    "explain",
    "summarize",
    "summary of",
    "what do i know about",
    "what do we know about",
    "give me context on",
    "context on",
    "background on",
]


def classify_intent(text: str) -> Intent:
    """
    Rule-based intent classification. Returns Intent with action="unknown"
    if no keyword rule matches — caller should fall through to LLM.
    """
    stripped = text.strip()
    lower = stripped.lower()

    # --- Explicit command prefixes (from Synaptic convention) ---
    if lower.startswith("!search "):
        return Intent(action="search", argument=stripped[8:].strip())

    if lower.startswith("!report"):
        return Intent(action="recall", argument=stripped[7:].strip() or None)

    if lower.startswith("!recent"):
        return Intent(action="briefing")

    if lower.startswith("!brief"):
        return Intent(action="briefing", argument=stripped[6:].strip() or None)

    if lower.startswith("!recall "):
        return Intent(action="recall", argument=stripped[8:].strip())

    if lower.startswith("?"):
        return Intent(action="search", argument=stripped[1:].strip())

    # --- Project prefix modes (#project / [project]) ---
    # #project text → capture with project
    m = re.match(r"^#(\S+)\s+(.+)", stripped, re.DOTALL)
    if m:
        return Intent(
            action="capture",
            argument=m.group(2).strip(),
            project=m.group(1).lower(),
        )

    # [project] text → capture with project (stream-style, but Orex doesn't
    # manage streams — it delegates to Synaptic's /capture with project param)
    m = re.match(r"^\[([^\]]+)\]\s+(.+)", stripped, re.DOTALL)
    if m:
        return Intent(
            action="capture",
            argument=m.group(2).strip(),
            project=m.group(1).strip().lower(),
        )

    # --- Keyword matching ---
    # Briefing (check before search since "what's going on" is ambiguous)
    for kw in _BRIEFING_KEYWORDS:
        if lower.startswith(kw) or lower == kw:
            arg = stripped[len(kw) :].strip() if len(stripped) > len(kw) else None
            return Intent(action="briefing", argument=arg)

    # Capture
    for kw in _CAPTURE_KEYWORDS:
        if lower.startswith(kw):
            remainder = stripped[len(kw) :].strip()
            # Strip common filler: "remember that ...", "remind me to ..."
            for filler in ["that ", "to ", "about "]:
                if remainder.lower().startswith(filler):
                    remainder = remainder[len(filler) :]
                    break
            return Intent(action="capture", argument=remainder)

    # Recall (LLM-powered answer from knowledge)
    for kw in _RECALL_KEYWORDS:
        if lower.startswith(kw):
            return Intent(action="recall", argument=stripped[len(kw) :].strip())

    # Search
    for kw in _SEARCH_KEYWORDS:
        if lower.startswith(kw):
            remainder = stripped[len(kw) :].strip()
            # Strip filler
            for filler in ["for ", "about ", "on "]:
                if remainder.lower().startswith(filler):
                    remainder = remainder[len(filler) :]
                    break
            return Intent(action="search", argument=remainder)

    # No keyword match — let the LLM decide
    return Intent(action="unknown", argument=stripped)
