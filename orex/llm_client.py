"""
LLM client for Orex — all model access goes through LiteLLM.

Handles two tasks:
1. Intent classification fallback (when keyword rules don't match)
2. General conversation (chat with context from Synaptic)
"""

import json
import logging

import httpx

from intent import Intent

logger = logging.getLogger("orex.llm")

CLASSIFY_SYSTEM_PROMPT = """You are an intent classifier for a personal AI assistant called Orex.

Given a user message, classify it into exactly one of these intents:
- "capture": The user wants to save/remember/note something (knowledge to store)
- "search": The user wants to find something they previously stored
- "recall": The user wants an LLM-powered answer drawing from their stored knowledge
- "briefing": The user wants a summary or briefing of recent activity
- "conversation": General chat, questions, or requests that don't involve stored knowledge

Respond with ONLY a JSON object: {"intent": "<one of the above>"}

Examples:
- "what's the name of that plumber we used?" → {"intent": "search"}
- "oh also the garage door sensor battery is low" → {"intent": "capture"}
- "tell me everything I know about the kitchen renovation" → {"intent": "recall"}
- "what's the weather like?" → {"intent": "conversation"}
- "good morning, what's on my plate?" → {"intent": "briefing"}
- "can you help me write an email?" → {"intent": "conversation"}
"""

CHAT_SYSTEM_PROMPT = """You are Orex, a personal AI assistant for John. You live inside his home network and help with home projects, personal tasks, work coordination, and general questions.

Your personality:
- Direct and helpful — no filler, no corporate tone
- You know John's context from his Synaptic knowledge base (provided below when available)
- You're conversational but efficient
- If you don't know something, say so honestly
- You can suggest capturing useful info to memory ("Want me to remember that?")

Recent context from John's knowledge base:
{context}

Keep responses concise unless asked for detail."""


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "mistral-small",
        classify_model: str = "mistral-small",
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.classify_model = classify_model

    async def classify_intent(self, text: str) -> Intent:
        """Use LLM to classify intent when keyword rules don't match."""
        try:
            response = await self._chat_completion(
                model=self.classify_model,
                messages=[
                    {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=50,
            )

            content = response.strip()
            # Parse JSON response
            if content.startswith("{"):
                data = json.loads(content)
                action = data.get("intent", "conversation")
                if action in ("capture", "search", "recall", "briefing", "conversation"):
                    return Intent(action=action, argument=text)

            # Fallback if parsing fails
            return Intent(action="conversation", argument=text)

        except Exception as e:
            logger.warning("LLM classify failed, defaulting to conversation: %s", e)
            return Intent(action="conversation", argument=text)

    async def chat(
        self,
        user_message: str,
        user_name: str = "",
        context: str = "",
    ) -> str:
        """General conversation with optional Synaptic context."""
        system = CHAT_SYSTEM_PROMPT.format(
            context=context if context else "(no recent context available)"
        )

        try:
            return await self._chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.7,
                max_tokens=500,
            )
        except Exception as e:
            logger.exception("LLM chat failed")
            return f"Sorry, I had trouble thinking about that: {e}"

    async def _chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """Call LiteLLM's OpenAI-compatible chat completions endpoint."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
