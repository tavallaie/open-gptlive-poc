"""HTTP task delegation adapter."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from urllib.request import Request, urlopen
from uuid import uuid4


Transport = Callable[[str, bytes, Mapping[str, str]], bytes]


class HttpTasks:
    """POST transcript and GLiNER labels to the configured task system."""

    def __init__(self, url: str, *, transport: Transport | None = None) -> None:
        self.url = url
        self.transport = transport or _post_json

    async def create(self, session_id: str, transcript: str, labels: dict[str, object]) -> str:
        """Create a delegation and return its client-visible identifier."""
        delegation_id = f"item_{uuid4().hex}"
        payload = {
            "session_id": session_id,
            "delegation_id": delegation_id,
            "transcript": transcript,
            "kind": labels.get("kind", "task"),
            "gliner": labels,
        }
        await asyncio.to_thread(
            self.transport,
            self.url,
            json.dumps(payload).encode("utf-8"),
            {"Content-Type": "application/json"},
        )
        return delegation_id


def _post_json(url: str, body: bytes, headers: Mapping[str, str]) -> bytes:
    request = Request(url, data=body, headers=dict(headers), method="POST")
    with urlopen(request, timeout=30) as response:
        return response.read()
