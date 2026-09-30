"""Configuration helpers for the local live-demo WebSocket bridge."""

from __future__ import annotations

import os


def endpoint_url(value: str) -> str:
    """Normalize a server URL to its GPT-Live WebSocket route."""
    url = value.rstrip("/")
    if url.startswith("https://"):
        url = "wss://" + url.removeprefix("https://")
    elif url.startswith("http://"):
        url = "ws://" + url.removeprefix("http://")
    if not url.endswith("/v1/live/sessions"):
        url += "/v1/live/sessions"
    return url


def upstream_settings() -> tuple[str, str]:
    """Read the endpoint and credential without exposing either to the browser."""
    url = os.getenv("GPTLIVE_DEMO_WS_URL") or os.getenv("GPTLIVE_ENDPOINT_URL", "")
    token = os.getenv("GPTLIVE_DEMO_TOKEN") or os.getenv("GPTLIVE_BEARER_TOKEN", "")
    return (endpoint_url(url) if url else "", token)
