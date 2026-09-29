"""Silero VAD adapter with a dependency-free PCM resampler."""

from __future__ import annotations

import struct
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..ports.vad import VADEvent


class SileroVAD:
    """Run Silero VAD on local 24 kHz PCM16LE input."""

    def __init__(
        self,
        model_path: str,
        *,
        pause_ms: int = 700,
        threshold: float = 0.5,
        infer: Callable[[Sequence[float]], float] | None = None,
    ) -> None:
        if pause_ms < 0:
            raise ValueError("pause_ms must not be negative")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if not Path(model_path).exists():
            raise FileNotFoundError(model_path)
        self.pause_ms = pause_ms
        self.threshold = threshold
        self._infer = infer or _load_onnx_model(model_path)
        self._buffer = bytearray()
        self._muted = False
        self._speaking = False
        self._silence_ms = 0
        self._processed_samples = 0

    def append_audio(self, pcm16le: bytes) -> list[VADEvent]:
        """Consume PCM16LE audio and return any speech boundary events."""
        _validate_pcm16le(pcm16le)
        if self._muted:
            self._processed_samples += len(pcm16le) // 2
            return []
        self._buffer.extend(pcm16le)
        events: list[VADEvent] = []
        source_bytes_per_chunk = 768 * 2
        while len(self._buffer) >= source_bytes_per_chunk:
            chunk = bytes(self._buffer[:source_bytes_per_chunk])
            del self._buffer[:source_bytes_per_chunk]
            samples = _pcm16le_to_floats(resample_pcm16le(chunk, 24_000, 16_000))
            offset_ms = round(self._processed_samples * 1000 / 24_000)
            chunk_ms = round(768 * 1000 / 24_000)
            self._processed_samples += 768
            try:
                probability = self._infer(samples)
            except Exception:
                self._buffer.clear()
                self._reset_detection()
                raise
            if probability >= self.threshold:
                self._silence_ms = 0
                if not self._speaking:
                    self._speaking = True
                    events.append(VADEvent("speech_started", offset_ms))
            elif self._speaking:
                self._silence_ms += chunk_ms
                if self._silence_ms >= self.pause_ms:
                    self._speaking = False
                    self._silence_ms = 0
                    events.append(VADEvent("speech_stopped", offset_ms + chunk_ms))
        return events

    def set_muted(self, muted: bool) -> None:
        """Enable or disable audio consumption without resetting the session."""
        self._muted = muted
        if muted:
            self._buffer.clear()
            self._reset_detection()

    def _reset_detection(self) -> None:
        """Reset speech state and model context after a discontinuity."""
        self._speaking = False
        self._silence_ms = 0
        reset = getattr(self._infer, "reset", None)
        if reset is not None:
            reset()


def resample_pcm16le(pcm16le: bytes, source_rate: int, target_rate: int) -> bytes:
    """Linearly resample mono little-endian PCM16 audio."""
    _validate_pcm16le(pcm16le)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if source_rate == target_rate or not pcm16le:
        return pcm16le
    count = len(pcm16le) // 2
    source = struct.unpack(f"<{count}h", pcm16le)
    output_count = round(count * target_rate / source_rate)
    output: list[int] = []
    for index in range(output_count):
        position = index * source_rate / target_rate
        left = min(int(position), count - 1)
        right = min(left + 1, count - 1)
        fraction = position - left
        output.append(round(source[left] + (source[right] - source[left]) * fraction))
    return struct.pack(f"<{len(output)}h", *output)


def _validate_pcm16le(pcm16le: bytes) -> None:
    """Reject non-byte-aligned PCM16LE input."""
    if not isinstance(pcm16le, bytes) or len(pcm16le) % 2:
        raise ValueError("audio must be PCM16LE bytes")


def _pcm16le_to_floats(pcm16le: bytes) -> list[float]:
    """Decode PCM16LE samples into normalized floats."""
    count = len(pcm16le) // 2
    return [sample / 32768 for sample in struct.unpack(f"<{count}h", pcm16le)]


def _load_onnx_model(model_path: str) -> Callable[[Sequence[float]], float]:
    """Load Silero lazily so protocol and fake-port tests need no ML runtime."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("SileroVAD requires numpy and onnxruntime") from exc

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros(64, dtype=np.float32)

    def infer(samples: Sequence[float]) -> float:
        """Run one stateful Silero inference window."""
        nonlocal context, state
        values = np.asarray(samples, dtype=np.float32).reshape(1, -1)
        values = np.concatenate((context, values.reshape(-1))).reshape(1, -1)
        inputs: dict[str, Any] = {}
        for item in session.get_inputs():
            name = item.name.lower()
            if "state" in name:
                inputs[item.name] = state
            elif name in {"sr", "sampling_rate"}:
                inputs[item.name] = np.array([16_000], dtype=np.int64)
            else:
                inputs[item.name] = values
        outputs = session.run(None, inputs)
        if len(outputs) > 1:
            state = outputs[1]
        context = values.reshape(-1)[-64:]
        return float(np.asarray(outputs[0]).reshape(-1)[0])

    def reset() -> None:
        """Reset recurrent state and the preceding-sample context."""
        nonlocal context, state
        context = np.zeros(64, dtype=np.float32)
        state = np.zeros((2, 1, 128), dtype=np.float32)

    infer.reset = reset  # type: ignore[attr-defined]

    return infer
