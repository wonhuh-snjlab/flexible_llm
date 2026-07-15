"""Local LLM provider (Ollama, LM Studio, OpenWebUI - OpenAI-compatible endpoints)."""

import os
import json
from typing import List, Optional, Dict

try:
    import httpx
except ImportError:
    httpx = None

from llm.base import ChatMessage, LLMProvider


class LocalProvider(LLMProvider):
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.base_url = self.config.get("base_url", os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1"))
        self.model = self.config.get("model", os.environ.get("LOCAL_LLM_MODEL", "llama3.3"))

        if httpx is None:
            raise ImportError("Install httpx package: pip install httpx")

        self.client = httpx.Client(base_url=self.base_url.rstrip("/"))

    @property
    def name(self) -> str:
        return "local"

    def _check_config(self) -> bool:
        try:
            response = self.client.get("/models", timeout=3)
            response.raise_for_status()
            return True
        except Exception:
            return False

    def chat(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> str:
        chat_messages = list(messages)
        if system_prompt:
            chat_messages = [ChatMessage("system", system_prompt)] + chat_messages

        payload = {
            "model": self.model,
            "messages": [
                {"role": m.role, "content": m.content} for m in chat_messages
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = self.client.post("/chat/completions", json=payload, timeout=300.0)
        response.raise_for_status()
        data = response.json()

        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        return json.dumps(data)

    def chat_stream(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        chat_messages = list(messages)
        if system_prompt:
            chat_messages = [ChatMessage("system", system_prompt)] + chat_messages

        payload = {
            "model": self.model,
            "messages": [
                {"role": m.role, "content": m.content} for m in chat_messages
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        response = self.client.post("/chat/completions", json=payload, timeout=300.0)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                text = line.decode("utf-8")
                if text.startswith("data: "):
                    data_str = text[6:]
                    if data_str == "[DONE]":
                        return
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        if delta and delta.get("content"):
                            yield delta["content"]
                    except (json.JSONDecodeError, KeyError):
                        continue

    def list_models(self) -> List[str]:
        try:
            response = self.client.get("/models", timeout=3)
            response.raise_for_status()
            data = response.json()
            # Ollama format
            if "models" in data:
                return [m.get("name", m.get("id", "")) for m in data["models"]]
            # OpenAI-compatible format
            if "data" in data:
                return [m.get("id", "") for m in data["data"]]
            return []
        except Exception:
            return [self.model] if self.model else []