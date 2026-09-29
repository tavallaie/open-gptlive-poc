"""Dependency-injection boundaries for the local model pipeline."""

from typing import Protocol

from .asr import ASR, Transcript
from .router import RouteDecision, Router
from .talker import TalkRequest, Talker, TranscriptTurn
from .vad import VAD, VADEvent


class Speaker(Protocol):
    """Text-to-speech port."""


class Tasks(Protocol):
    """Asynchronous task delegation port."""


__all__ = ["ASR", "RouteDecision", "Router", "Speaker", "TalkRequest", "Tasks", "Talker", "Transcript", "TranscriptTurn", "VAD", "VADEvent"]
