"""Supertonic text-to-speech adapter."""

from __future__ import annotations

from collections.abc import Mapping
from threading import Lock
from typing import Any

import numpy as np


_VOICE_STYLES = {"F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"}


class SupertonicSpeaker:
    """Synthesize mono PCM16LE audio with Supertonic."""

    def __init__(
        self,
        voice_map: Mapping[str, str] | None = None,
        *,
        tts: Any | None = None,
    ) -> None:
        self.tts = tts
        self.voice_map = dict(voice_map or {"marin": "F1"})
        self._lock = Lock()

    def synthesize(self, text: str, voice: str) -> bytes:
        """Return one 24 kHz mono utterance encoded as signed little-endian PCM."""
        if not text.strip():
            raise ValueError("text must not be empty")
        style_name = self.voice_map.get(voice, self.voice_map.get("marin", "F1"))
        if style_name not in _VOICE_STYLES:
            raise ValueError(f"Unsupported Supertonic voice style: {style_name}")
        with self._lock:
            if self.tts is None:
                try:
                    from supertonic import TTS
                except ImportError as exc:
                    raise RuntimeError("Install supertonic to use SupertonicSpeaker") from exc
                self.tts = TTS(auto_download=True)
            style = self.tts.get_voice_style(voice_name=style_name)
            waveform, _ = self.tts.synthesize(text, voice_style=style, lang="na")
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return b""
        source_positions = np.arange(samples.size, dtype=np.float64)
        target_positions = np.arange(int(samples.size * 24_000 / 44_100), dtype=np.float64) * (44_100 / 24_000)
        resampled = np.interp(target_positions, source_positions, samples)
        pcm16 = np.rint(np.clip(resampled, -1.0, 1.0) * 32767).astype("<i2")
        return pcm16.tobytes()
