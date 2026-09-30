"""Dependency-injection boundaries for the local model pipeline."""

from typing import Protocol

from .asr import ASR, Transcript
from .router import RouteDecision, Router
from .speaker import Speaker
from .talker import TalkRequest, Talker, TranscriptTurn
from .vad import VAD, VADEvent


class Tasks(Protocol):
    """Asynchronous task delegation port."""

    async def create(self, session_id: str, transcript: str, labels: dict[str, object]) -> str: ...


__all__ = ["ASR", "RouteDecision", "Router", "Speaker", "TalkRequest", "Tasks", "Talker", "Transcript", "TranscriptTurn", "VAD", "VADEvent"]
