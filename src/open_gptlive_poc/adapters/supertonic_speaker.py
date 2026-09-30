"""Supertonic text-to-speech adapter."""

from __future__ import annotations

from collections.abc import Mapping
from threading import BoundedSemaphore, Event, Lock
from textwrap import wrap
from typing import Any

import numpy as np


_VOICE_STYLES = {"F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"}
_SOURCE_RATE = 44_100
_TARGET_RATE = 24_000
_MAX_TEXT_CHUNK = 180
_RESAMPLE_TAPS = 63


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
        self._load_lock = Lock()
        self._synthesis_slots = BoundedSemaphore(2)

    def synthesize(self, text: str, voice: str, cancel_event: Event | None = None) -> bytes:
        """Return one 24 kHz mono utterance encoded as signed little-endian PCM."""
        if not text.strip():
            raise ValueError("text must not be empty")
        style_name = self.voice_map.get(voice, self.voice_map.get("marin", "F1"))
        if style_name not in _VOICE_STYLES:
            raise ValueError(f"Unsupported Supertonic voice style: {style_name}")
        with self._load_lock:
            if self.tts is None:
                try:
                    from supertonic import TTS
                except ImportError as exc:
                    raise RuntimeError("Install supertonic to use SupertonicSpeaker") from exc
                self.tts = TTS(auto_download=True)
            tts = self.tts
        style = tts.get_voice_style(voice_name=style_name)
        audio_chunks: list[bytes] = []
        for text_chunk in wrap(text, width=_MAX_TEXT_CHUNK, break_long_words=True):
            if cancel_event is not None and cancel_event.is_set():
                break
            while not self._synthesis_slots.acquire(timeout=0.1):
                if cancel_event is not None and cancel_event.is_set():
                    return b"".join(audio_chunks)
            try:
                waveform, _ = tts.synthesize(
                    text_chunk,
                    voice_style=style,
                    lang="na",
                    max_chunk_length=_MAX_TEXT_CHUNK,
                    silence_duration=0,
                )
            finally:
                self._synthesis_slots.release()
            audio_chunks.append(_resample_pcm16le(waveform))
        return b"".join(audio_chunks)


def _resample_pcm16le(waveform: Any) -> bytes:
    """Low-pass and resample Supertonic float audio to signed 24 kHz PCM16LE."""
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return b""
    half = _RESAMPLE_TAPS // 2
    positions = np.arange(-half, half + 1, dtype=np.float64)
    cutoff = _TARGET_RATE / (2 * _SOURCE_RATE)
    kernel = 2 * cutoff * np.sinc(2 * cutoff * positions) * np.kaiser(_RESAMPLE_TAPS, 8.6)
    kernel /= kernel.sum()
    filtered = np.convolve(np.pad(samples, (half, half), mode="edge"), kernel, mode="valid")
    output_count = round(samples.size * _TARGET_RATE / _SOURCE_RATE)
    source_positions = np.arange(samples.size, dtype=np.float64)
    target_positions = np.arange(output_count, dtype=np.float64) * (_SOURCE_RATE / _TARGET_RATE)
    resampled = np.interp(target_positions, source_positions, filtered)
    pcm16 = np.rint(np.clip(resampled, -1.0, 1.0) * 32767).astype("<i2")
    return pcm16.tobytes()
