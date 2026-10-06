"""
Pydantic models for the shield's client-facing (OpenAI-compatible) surface.

`extra="allow"` on ChatCompletionRequest means fields the shield doesn't
explicitly know about (e.g. provider-specific sampling params) pass through
untouched to the upstream adapter rather than being silently dropped.
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[Any], None] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None


class ShieldBlockedError(BaseModel):
    error: dict = Field(
        default_factory=lambda: {
            "message": "blocked",
            "type": "prompt_shield_blocked",
            "code": "blocked",
        }
    )
