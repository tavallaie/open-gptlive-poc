"""Dependency-injection boundaries for the local model pipeline."""

from typing import Protocol


class VAD(Protocol):
    """Voice activity detection port."""


class ASR(Protocol):
    """Automatic speech recognition port."""


class Router(Protocol):
    """Turn routing port."""


class Talker(Protocol):
    """Conversational text generation port."""


class Speaker(Protocol):
    """Text-to-speech port."""


class Tasks(Protocol):
    """Asynchronous task delegation port."""
