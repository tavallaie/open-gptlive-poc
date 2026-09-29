"""Conversational text generation port."""

from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class TranscriptTurn:
    """One recent input or output turn."""

    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True, slots=True)
class TalkRequest:
    """Context supplied to a conversational text adapter."""

    transcript: str
    instructions: Sequence[str] = field(default_factory=tuple)
    thinking: Sequence[str] = field(default_factory=tuple)
    history: Sequence[TranscriptTurn] = field(default_factory=tuple)


class Talker(Protocol):
    """Generate a plain conversational reply."""

    async def reply(self, request: TalkRequest) -> str: ...
