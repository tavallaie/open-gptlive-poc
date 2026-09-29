"""Voice activity detection port and events."""

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True, slots=True)
class VADEvent:
    """A speech boundary relative to the start of the session audio stream."""

    type: Literal["speech_started", "speech_stopped"]
    offset_ms: int


class VAD(Protocol):
    """Detect speech boundaries from 24 kHz PCM16LE audio."""

    def append_audio(self, pcm16le: bytes) -> list[VADEvent]: ...

    def set_muted(self, muted: bool) -> None: ...
