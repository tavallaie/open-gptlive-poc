"""Small client for exercising the GPT-Live WebSocket endpoint."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np


TARGET_SAMPLE_RATE = 24_000


@dataclass(frozen=True, slots=True)
class DemoResult:
    """User-visible result returned by one demo request."""

    status: str
    transcript: str
    audio: tuple[int, np.ndarray] | None = None


def convert_to_pcm16le(audio: tuple[int, np.ndarray] | np.ndarray) -> bytes:
    """Convert Gradio audio to mono 24 kHz PCM16LE bytes."""
    sample_rate, samples = audio if isinstance(audio, tuple) else (TARGET_SAMPLE_RATE, audio)
    if sample_rate <= 0:
        raise ValueError("sample rate must be positive")
    values = np.asarray(samples)
    integer_audio = np.issubdtype(values.dtype, np.integer)
    integer_max = np.iinfo(values.dtype).max if integer_audio else 1
    if values.ndim == 2:
        values = values.mean(axis=1)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("audio must contain one non-empty waveform")
    values = values.astype(np.float32, copy=False)
    if integer_audio:
        values = values / integer_max
    if sample_rate != TARGET_SAMPLE_RATE:
        source_positions = np.arange(values.size, dtype=np.float64)
        target_size = round(values.size * TARGET_SAMPLE_RATE / sample_rate)
        target_positions = np.linspace(0, values.size - 1, target_size)
        values = np.interp(target_positions, source_positions, values)
    return (np.clip(values, -1, 1) * 32767).astype("<i2").tobytes()


def endpoint_url(value: str) -> str:
    """Normalize a configured HTTP or WebSocket endpoint URL."""
    url = value.rstrip("/")
    if url.startswith("https://"):
        url = "wss://" + url.removeprefix("https://")
    elif url.startswith("http://"):
        url = "ws://" + url.removeprefix("http://")
    if not url.endswith("/v1/live/sessions"):
        url += "/v1/live/sessions"
    return url


class LiveDemoClient:
    """Send one recorded waveform to the configured Live endpoint."""

    def __init__(self, url: str | None = None, token: str | None = None) -> None:
        self.url = url or os.getenv("GPTLIVE_DEMO_WS_URL") or os.getenv("GPTLIVE_ENDPOINT_URL")
        self.token = token or os.getenv("GPTLIVE_DEMO_TOKEN") or os.getenv("GPTLIVE_BEARER_TOKEN")

    def run(self, pcm16le: bytes) -> DemoResult:
        """Run the async WebSocket exchange from a synchronous Gradio callback."""
        if not self.url:
            return DemoResult("Endpoint is not configured; showing local input preview.", "", (TARGET_SAMPLE_RATE, _pcm16le_array(pcm16le)))
        if not self.token:
            return DemoResult("GPTLIVE_DEMO_TOKEN is missing.", "")
        return asyncio.run(self._run(pcm16le))

    async def _run(self, pcm16le: bytes) -> DemoResult:
        """Perform the WebSocket exchange and collect server output."""
        try:
            import websockets
        except ImportError as exc:
            return DemoResult(f"WebSocket client unavailable: {exc}", "")

        transcripts: list[str] = []
        audio_chunks: list[bytes] = []
        try:
            async with websockets.connect(
                endpoint_url(self.url or ""),
                additional_headers={"Authorization": f"Bearer {self.token}"},
                max_size=16 * 1024 * 1024,
            ) as socket:
                await socket.send(json.dumps({
                    "type": "session.start",
                    "event_id": "demo_start",
                    "model": "gpt-live-1",
                    "audio": {"format": {"encoding": "pcm16le", "sample_rate": TARGET_SAMPLE_RATE, "channels": 1}},
                }))
                await socket.send(json.dumps({
                    "type": "session.input_audio.append",
                    "event_id": "demo_audio",
                    "audio": base64.b64encode(pcm16le).decode("ascii"),
                }))
                await socket.send(json.dumps({"type": "session.close", "event_id": "demo_close"}))
                async for message in socket:
                    event = json.loads(message)
                    event_type = event.get("type")
                    if event_type in {"session.input_transcript.delta", "session.output_transcript.delta"}:
                        transcripts.append(str(event.get("text", event.get("delta", ""))))
                    elif event_type == "session.output_audio.delta":
                        encoded = event.get("audio", event.get("delta"))
                        if encoded:
                            audio_chunks.append(base64.b64decode(encoded))
                    elif event_type == "error":
                        return DemoResult(event.get("error", {}).get("message", "Server error"), "")
                    elif event_type == "session.closed":
                        break
        except Exception as exc:
            return DemoResult(f"Endpoint request failed: {exc}", "")

        output = b"".join(audio_chunks)
        audio = (TARGET_SAMPLE_RATE, _pcm16le_array(output)) if output else None
        return DemoResult("Completed.", "".join(transcripts), audio)


def _pcm16le_array(pcm16le: bytes) -> np.ndarray:
    """Decode PCM16LE bytes into Gradio-compatible samples."""
    return np.frombuffer(pcm16le, dtype="<i2").copy()
