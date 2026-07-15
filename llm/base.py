"""Abstract base class for all LLM providers."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict


class ChatMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class LLMProvider(ABC):
    """All LLM providers must implement the chat method."""

    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> str:
        """Send messages to the LLM and return the response text."""
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        """Generator that yields chunks of the response."""
        ...

    @abstractmethod
    def list_models(self) -> List[str]:
        """Return list of available model names for this provider."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name label (e.g. 'anthropic', 'openai', 'local')."""
        ...

    @property
    def available(self) -> bool:
        """Check if provider is properly configured."""
        try:
            return self._check_config()
        except Exception:
            return False

    @abstractmethod
    def _check_config(self) -> bool:
        """Internal check that provider is ready."""
        ...


def build_provider(provider: str, config: Optional[Dict] = None) -> LLMProvider:
    """Factory to create the right provider instance from a config dict."""
    from llm.anthropic import AnthropicProvider
    from llm.openai import OpenAIProvider
    from llm.local import LocalProvider

    if provider == "anthropic":
        return AnthropicProvider(config or {})
    elif provider == "openai":
        return OpenAIProvider(config or {})
    elif provider == "local":
        return LocalProvider(config or {})
    else:
        raise ValueError(f"Unknown provider: {provider}")