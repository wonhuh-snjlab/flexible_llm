"""Anthropic Claude provider."""

import os
import json
from typing import List, Optional, Dict

try:
    import anthropic
except ImportError:
    anthropic = None

from llm.base import ChatMessage, LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.api_key = self.config.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
        self.model = self.config.get("model", "claude-3.5-sonnet-20241022")

        if anthropic is None:
            raise ImportError("Install anthropic package: pip install anthropic")

        self.client = anthropic.Anthropic(api_key=self.api_key)

    @property
    def name(self) -> str:
        return "anthropic"

    def _check_config(self) -> bool:
        return bool(self.api_key) and self.api_key != ""

    def chat(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> str:
        params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
        }
        if system_prompt:
            params["system"] = system_prompt

        response = self.client.messages.create(**params)
        return response.content[0].text

    def chat_stream(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
        }
        if system_prompt:
            params["system"] = system_prompt

        stream = self.client.messages.create(stream=True, **params)
        for chunk in stream:
            if chunk.type == "content_block_delta":
                yield chunk.delta.text

    def list_models(self) -> List[str]:
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
            "claude-3.5-sonnet-20241022",
            "claude-3.5-haiku-20241022",
        ]