"""Automatic speech recognition port and result type."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Transcript:
    """Text recognized from one speech turn."""

    text: str
    start_ms: int
    end_ms: int


class ASR(Protocol):
    """Transcribe one 24 kHz PCM16LE speech turn."""

    def transcribe(self, pcm16le: bytes, start_ms: int) -> Transcript: ...
