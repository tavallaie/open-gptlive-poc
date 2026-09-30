"""Text-to-speech output port."""

from typing import Protocol


class Speaker(Protocol):
    """Synthesize one utterance as mono 24 kHz PCM16LE."""

    def synthesize(self, text: str, voice: str) -> bytes:
        """Return the complete utterance as mono 24 kHz PCM16LE."""
        ...
