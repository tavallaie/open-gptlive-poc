"""Parsing and serialization for the slice-1 GPT-Live event contract."""

from __future__ import annotations

import base64
import binascii
import hmac
import json
from dataclasses import dataclass, field
from typing import Any, Mapping


SUPPORTED_CLIENT_EVENTS = frozenset(
    {
        "session.start",
        "session.update",
        "session.input_audio.append",
        "session.input_audio.mute",
        "session.input_audio.unmute",
        "session.instructions.append",
        "session.thinking.append",
        "session.commentary.append",
        "session.close",
    }
)
APPEND_EVENTS = frozenset(
    {
        "session.instructions.append",
        "session.thinking.append",
        "session.commentary.append",
    }
)


class ProtocolError(ValueError):
    """A client event that cannot be accepted by the protocol."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        param: str | None = None,
        client_event_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.param = param
        self.client_event_id = client_event_id

    def as_error(self) -> "ServerEvent":
        error: dict[str, Any] = {
            "type": "invalid_request_error",
            "code": self.code,
            "message": self.message,
        }
        if self.param is not None:
            error["param"] = self.param
        if self.client_event_id is not None:
            error["client_event_id"] = self.client_event_id
        return ServerEvent("error", {"error": error})


@dataclass(frozen=True, slots=True)
class ClientEvent:
    """A validated client event with unsupported fields preserved but ignored."""

    type: str
    event_id: str | None = None
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ServerEvent:
    """A server event ready to send over the WebSocket."""

    type: str
    data: Mapping[str, Any] = field(default_factory=dict)
    event_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        event: dict[str, Any] = {"type": self.type}
        if self.event_id is not None:
            event["event_id"] = self.event_id
        event.update(self.data)
        return event

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), separators=(",", ":"))


def parse_client_event(raw: str | bytes) -> ClientEvent:
    """Parse and validate one client JSON event."""
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid_value", "Event must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError("invalid_value", "Event must be a JSON object")

    event_type = value.get("type")
    if not isinstance(event_type, str) or event_type not in SUPPORTED_CLIENT_EVENTS:
        raise ProtocolError("invalid_value", "Unknown or missing event type", param="type")
    event_id = value.get("event_id")
    if event_id is not None and not isinstance(event_id, str):
        raise ProtocolError("invalid_value", "event_id must be a string", param="event_id")

    data = {key: item for key, item in value.items() if key not in {"type", "event_id"}}
    if event_type == "session.start":
        _validate_start(data, event_id)
    elif event_type == "session.update":
        _validate_update(data, event_id)
    elif event_type == "session.input_audio.append":
        _validate_audio(data, event_id)
    elif event_type in APPEND_EVENTS:
        _validate_append(data, event_id)

    return ClientEvent(event_type, event_id, data)


def authenticate(authorization: str | None, expected_token: str) -> bool:
    """Return whether an Authorization header contains the configured bearer token."""
    scheme, separator, token = (authorization or "").partition(" ")
    return bool(
        separator
        and scheme.lower() == "bearer"
        and token
        and hmac.compare_digest(token, expected_token)
    )


def _validate_start(data: Mapping[str, Any], event_id: str | None) -> None:
    if data.get("model") != "gpt-live-1":
        _fail("invalid_value", "model must be gpt-live-1", "model", event_id)
    delegation = data.get("delegation", {})
    if not isinstance(delegation, dict):
        _fail("invalid_value", "delegation must be an object", "delegation", event_id)
    if delegation.get("type", "client") != "client":
        _fail("invalid_value", "delegation.type must be client", "delegation.type", event_id)
    _validate_audio_format(data.get("audio"), event_id)


def _validate_update(data: Mapping[str, Any], event_id: str | None) -> None:
    changed = data.get("session", data)
    if not isinstance(changed, dict):
        _fail("invalid_value", "session must be an object", "session", event_id)
    for field_name in ("model", "instructions", "audio", "delegation"):
        if field_name in changed:
            _fail(
                "immutable_field_update",
                f"{field_name} cannot be changed after session.start",
                f"session.{field_name}",
                event_id,
            )


def _validate_append(data: Mapping[str, Any], event_id: str | None) -> None:
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        _fail("invalid_value", "text must be a non-empty string", "text", event_id)
    if len(text.split()) > 500:
        _fail("invalid_value", "text must contain at most 500 tokens", "text", event_id)
    delegation_id = data.get("delegation_id")
    if delegation_id is not None and not isinstance(delegation_id, str):
        _fail("invalid_value", "delegation_id must be a string or null", "delegation_id", event_id)


def _validate_audio(data: Mapping[str, Any], event_id: str | None) -> None:
    encoded = data.get("audio")
    if not isinstance(encoded, str):
        _fail("invalid_audio", "audio must be base64 PCM16 bytes", "audio", event_id)
    try:
        audio = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProtocolError("invalid_audio", "audio must be valid base64", param="audio", client_event_id=event_id) from exc
    if not audio or len(audio) % 2:
        _fail("invalid_audio", "audio must contain PCM16 bytes", "audio", event_id)


def _validate_audio_format(audio: Any, event_id: str | None) -> None:
    if audio is None:
        return
    if not isinstance(audio, dict):
        _fail("invalid_value", "audio must be an object", "audio", event_id)
    formats = [audio]
    formats.extend(value for value in audio.values() if isinstance(value, dict))
    for value in formats:
        audio_format = value.get("format")
        if audio_format is None:
            continue
        if not isinstance(audio_format, dict):
            _fail("invalid_value", "audio.format must be an object", "audio.format", event_id)
        expected = {"encoding": "pcm16le", "sample_rate": 24000, "channels": 1}
        for key, required in expected.items():
            if key in audio_format and audio_format[key] != required:
                _fail("invalid_value", f"audio.format.{key} must be {required}", f"audio.format.{key}", event_id)


def _fail(code: str, message: str, param: str, event_id: str | None) -> None:
    raise ProtocolError(code, message, param=param, client_event_id=event_id)
