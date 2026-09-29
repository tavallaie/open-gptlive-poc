"""Runtime configuration for the GPT-Live server."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Mapping


class ConfigurationError(ValueError):
    """Raised when the server configuration cannot be used safely."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated settings loaded from GPTLIVE_* environment variables."""

    host: str = "127.0.0.1"
    port: int = 8000
    bearer_token: str | None = None
    silero_model_path: str | None = None
    whisper_model_path: str | None = None
    gliner_model_path: str | None = None
    gliner_device: str = "cpu"
    lm_studio_base_url: str = "http://127.0.0.1:1234/api/v1"
    lm_studio_model: str = "local-model"
    supertonic_voice_map: dict[str, str] | None = None
    tasks_url: str = "http://127.0.0.1:8080/tasks"
    vad_pause_ms: int = 700
    gliner_task_threshold: float = 0.5
    max_session_duration_s: int = 3600

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        """Load settings from environment variables and validate them."""
        values = dict(os.environ if environ is None else environ)
        prefix = "GPTLIVE_"
        raw = {key.removeprefix(prefix).lower(): value for key, value in values.items() if key.startswith(prefix)}
        defaults = cls()

        settings = cls(
            host=raw.get("host", defaults.host),
            port=_integer(raw.get("port"), defaults.port, "GPTLIVE_PORT"),
            bearer_token=raw.get("bearer_token") or None,
            silero_model_path=raw.get("silero_model_path") or None,
            whisper_model_path=raw.get("whisper_model_path") or None,
            gliner_model_path=raw.get("gliner_model_path") or None,
            gliner_device=raw.get("gliner_device", defaults.gliner_device),
            lm_studio_base_url=raw.get("lm_studio_base_url", defaults.lm_studio_base_url),
            lm_studio_model=raw.get("lm_studio_model", defaults.lm_studio_model),
            supertonic_voice_map=_voice_map(raw.get("supertonic_voice_map")),
            tasks_url=raw.get("tasks_url", defaults.tasks_url),
            vad_pause_ms=_integer(raw.get("vad_pause_ms"), defaults.vad_pause_ms, "GPTLIVE_VAD_PAUSE_MS"),
            gliner_task_threshold=_number(raw.get("gliner_task_threshold"), defaults.gliner_task_threshold, "GPTLIVE_GLINER_TASK_THRESHOLD"),
            max_session_duration_s=_integer(raw.get("max_session_duration_s"), defaults.max_session_duration_s, "GPTLIVE_MAX_SESSION_DURATION_S"),
        )
        return settings.validate()

    def validate(self) -> "Settings":
        """Return this settings object or raise a useful configuration error."""
        required = {
            "GPTLIVE_BEARER_TOKEN": self.bearer_token,
            "GPTLIVE_SILERO_MODEL_PATH": self.silero_model_path,
            "GPTLIVE_WHISPER_MODEL_PATH": self.whisper_model_path,
            "GPTLIVE_GLINER_MODEL_PATH": self.gliner_model_path,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ConfigurationError(f"Missing required configuration: {', '.join(missing)}")
        if not 1 <= self.port <= 65535:
            raise ConfigurationError("GPTLIVE_PORT must be between 1 and 65535")
        if self.vad_pause_ms < 0:
            raise ConfigurationError("GPTLIVE_VAD_PAUSE_MS must not be negative")
        if not 0 <= self.gliner_task_threshold <= 1:
            raise ConfigurationError("GPTLIVE_GLINER_TASK_THRESHOLD must be between 0 and 1")
        if self.max_session_duration_s <= 0:
            raise ConfigurationError("GPTLIVE_MAX_SESSION_DURATION_S must be positive")
        return self


def _integer(value: str | None, default: int, name: str) -> int:
    """Parse an integer setting or return its default."""
    try:
        return default if value is None else int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc


def _number(value: str | None, default: float, name: str) -> float:
    """Parse a numeric setting or return its default."""
    try:
        return default if value is None else float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc


def _voice_map(value: str | None) -> dict[str, str] | None:
    """Parse the optional JSON voice map or return the default mapping."""
    if value is None:
        return {"marin": "F1"}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ConfigurationError("GPTLIVE_SUPERTONIC_VOICE_MAP must be valid JSON") from exc
    if not isinstance(parsed, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()):
        raise ConfigurationError("GPTLIVE_SUPERTONIC_VOICE_MAP must be a JSON object of strings")
    return parsed
