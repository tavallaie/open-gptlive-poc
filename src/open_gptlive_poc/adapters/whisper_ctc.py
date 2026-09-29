"""Local Whisper CTC ASR adapter."""

from __future__ import annotations

import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..ports.asr import Transcript


class WhisperCTC:
    """Transcribe local PCM16LE turns with a local Whisper CTC checkpoint."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(model_path)
        self.device = device
        self.processor = processor
        self.model = model
        self._torch = None
        if self.processor is None or self.model is None:
            try:
                import torch
                from transformers import AutoModelForCTC, AutoProcessor
            except ImportError as exc:
                raise RuntimeError("WhisperCTC requires torch and transformers") from exc
            self._torch = torch
            self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModelForCTC.from_pretrained(model_path, local_files_only=True).to(device).eval()

    def transcribe(self, pcm16le: bytes, start_ms: int) -> Transcript:
        """Transcribe one PCM16LE turn and return session-relative timestamps."""
        _validate_pcm16le(pcm16le)
        if start_ms < 0:
            raise ValueError("start_ms must not be negative")
        samples = _pcm16le_to_floats(pcm16le)
        inputs = self.processor(samples, sampling_rate=24_000, return_tensors="pt")
        if isinstance(inputs, Mapping):
            inputs = {key: _to_device(value, self.device) for key, value in inputs.items()}
        if self._torch is None:
            outputs = self.model(**inputs)
        else:
            with self._torch.no_grad():
                outputs = self.model(**inputs)
        token_ids = outputs.logits.argmax(dim=-1)
        text = self.processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
        duration_ms = round(len(pcm16le) / 2 / 24_000 * 1000)
        return Transcript(text, start_ms, start_ms + duration_ms)


def _to_device(value: Any, device: str) -> Any:
    return value.to(device) if hasattr(value, "to") else value


def _validate_pcm16le(pcm16le: bytes) -> None:
    if not isinstance(pcm16le, bytes) or len(pcm16le) % 2:
        raise ValueError("audio must be PCM16LE bytes")


def _pcm16le_to_floats(pcm16le: bytes) -> list[float]:
    count = len(pcm16le) // 2
    return [sample / 32768 for sample in struct.unpack(f"<{count}h", pcm16le)]
