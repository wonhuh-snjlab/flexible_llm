"""OpenAI GPT provider."""

import os
from typing import List, Optional, Dict

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from llm.base import ChatMessage, LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.api_key = self.config.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
        self.model = self.config.get("model", "gpt-4o")

        if OpenAI is None:
            raise ImportError("Install openai package: pip install openai")

        self.client = OpenAI(api_key=self.api_key)

    @property
    def name(self) -> str:
        return "openai"

    def _check_config(self) -> bool:
        return bool(self.api_key) and self.api_key != ""

    def chat(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> str:
        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        for m in messages:
            chat_messages.append({"role": m.role, "content": m.content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def chat_stream(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        for m in messages:
            chat_messages.append({"role": m.role, "content": m.content})

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def list_models(self) -> List[str]:
        try:
            models = self.client.models.list()
            return [m.id for m in models.data if "gpt" in m.id]
        except Exception:
            return [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
            ]