"""Local Faster-Whisper ASR adapter."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from ..ports.asr import Transcript


class FasterWhisperASR:
    """Transcribe local PCM16LE turns with a local CTranslate2 Whisper model."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        compute_type: str = "auto",
        language: str | None = None,
        model: Any | None = None,
    ) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(model_path)
        self.language = language
        self._requires_numpy = model is None
        if model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("FasterWhisperASR requires faster-whisper") from exc
            model = WhisperModel(model_path, device=device, compute_type=compute_type)
        self.model = model

    def transcribe(self, pcm16le: bytes, start_ms: int) -> Transcript:
        """Transcribe one PCM16LE turn and return session-relative timestamps."""
        _validate_pcm16le(pcm16le)
        if start_ms < 0:
            raise ValueError("start_ms must not be negative")
        samples = _pcm16le_to_floats(pcm16le)
        if self._requires_numpy:
            try:
                import numpy as np
            except ImportError as exc:
                raise RuntimeError("FasterWhisperASR requires numpy") from exc
            samples = np.asarray(samples, dtype=np.float32)
        segments, _ = self.model.transcribe(
            samples,
            language=self.language,
            vad_filter=False,
        )
        collected = list(segments)
        text = "".join(segment.text for segment in collected).strip()
        duration_ms = round(len(pcm16le) / 2 / 24_000 * 1000)
        segment_end_ms = round(collected[-1].end * 1000) if collected else duration_ms
        end_ms = start_ms + min(segment_end_ms, duration_ms)
        return Transcript(text, start_ms, end_ms)


def _validate_pcm16le(pcm16le: bytes) -> None:
    if not isinstance(pcm16le, bytes) or len(pcm16le) % 2:
        raise ValueError("audio must be PCM16LE bytes")


def _pcm16le_to_floats(pcm16le: bytes) -> list[float]:
    count = len(pcm16le) // 2
    return [sample / 32768 for sample in struct.unpack(f"<{count}h", pcm16le)]
