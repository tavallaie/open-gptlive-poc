"""Dependency-injection boundaries for the local model pipeline."""

from typing import Protocol

from .asr import ASR, Transcript
from .vad import VAD, VADEvent


class Router(Protocol):
    """Turn routing port."""


class Talker(Protocol):
    """Conversational text generation port."""


class Speaker(Protocol):
    """Text-to-speech port."""


class Tasks(Protocol):
    """Asynchronous task delegation port."""


__all__ = ["ASR", "Router", "Speaker", "Tasks", "Talker", "Transcript", "VAD", "VADEvent"]
