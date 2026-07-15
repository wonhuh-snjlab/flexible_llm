"""Resolve which LLM provider to use (global default → per-memory override)."""

import os
from pathlib import Path
from typing import Optional, Dict

from llm.base import ChatMessage, LLMProvider, build_provider
from memory.manager import (
    load_global_config,
    get_or_create_config_dir,
    get_memory,
    _load_json,
    _manifest_path,
)


class LLMRouter:
    """Resolves the active LLM provider from config + memory context."""

    def __init__(
        self,
        provider: Optional[str] = None,
        config_dir: Optional[Path] = None,
    ):
        if config_dir is None:
            config_dir = get_or_create_config_dir()
        self.config_dir = config_dir
        self.config = load_global_config(config_dir) or {}
        self.default_provider = self.config.get("default_llm", {}).get("provider", "anthropic")
        self.default_model = self.config.get("default_llm", {}).get("model", "")
        self.local_llm_config = self.config.get("local_llm", {
            "base_url": os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1"),
            "model": os.environ.get("LOCAL_LLM_MODEL", "llama3.3"),
        })

        # Override with explicit argument
        if provider:
            self.default_provider = provider

    def _get_api_keys(self) -> Dict[str, str]:
        keys = {}
        for k, v in (self.config.get("api_keys") or {}).items():
            if v:
                keys[k] = v
        return keys

    def get_active_provider(self, memory_name: Optional[str] = None) -> tuple:
        """
        Returns (provider_name, model_name, provider_config_dict).
        Priority: .llmrc > memory manifest > global config .
        """
        provider = self.default_provider
        model = self.default_model

        # Check per-memory LLM config
        if memory_name:
            llmrc_path = None
            if self.config_dir:
                from memory.manager import get_memory_root
                mem_root = get_memory_root(self.config_dir)
                llmrc_path = mem_root / memory_name / ".llmrc"
            if llmrc_path and llmrc_path.exists():
                llmrc = _load_json(llmrc_path)
                if llmrc:
                    if llmrc.get("provider"):
                        provider = llmrc["provider"]
                    if llmrc.get("model"):
                        model = llmrc["model"]
            else:
                mem_data = get_memory(memory_name, self.config_dir)
                if mem_data and mem_data.get("default_llm"):
                    if mem_data["default_llm"].get("provider"):
                        provider = mem_data["default_llm"]["provider"]
                    if mem_data["default_llm"].get("model"):
                        model = mem_data["default_llm"]["model"]

        # Build provider config
        if provider == "local":
            provider_config = {"base_url": self.local_llm_config.get("base_url", ""), "model": model or self.local_llm_config.get("model", "")}
        else:
            provider_config = {**self._get_api_keys(), "model": model}

        return provider, provider_config

    def create(self, memory_name: Optional[str] = None) -> LLMProvider:
        """Create and return an LLMProvider instance for the given memory."""
        provider_name, provider_config = self.get_active_provider(memory_name)
        return build_provider(provider_name, provider_config)

    def chat(
        self,
        messages,
        memory_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> str:
        """Convenience method: resolve provider and send a chat request."""
        if isinstance(messages, str):
            messages = [ChatMessage("user", messages)]
        provider = self.create(memory_name)
        return provider.chat(
            messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def chat_stream(
        self,
        messages,
        memory_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        """Convenience method: resolve provider and stream the response."""
        if isinstance(messages, str):
            messages = [ChatMessage("user", messages)]
        provider = self.create(memory_name)
        return provider.chat_stream(
            messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def list_providers(self) -> list:
        """List all available providers with their status."""
        providers = []
        for name in ["anthropic", "openai", "local"]:
            try:
                p = build_provider(name, self.config.get("api_keys") or {} if name != "local" else self.local_llm_config)
                models = p.list_models()
                available = p.available if isinstance(p, LLMProvider) else False
                providers.append({
                    "name": name,
                    "available": available,
                    "models": models[:5],
                })
            except Exception:
                providers.append({"name": name, "available": False, "models": []})
        return providers