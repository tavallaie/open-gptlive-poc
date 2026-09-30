"""HTTP task delegation adapter."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from urllib.request import Request, urlopen

from .sqlite_delegations import SQLiteDelegations


Transport = Callable[[str, bytes, Mapping[str, str]], bytes]


class HttpTasks:
    """POST transcript and GLiNER labels to the configured task system."""

    def __init__(
        self,
        url: str,
        *,
        delegations: SQLiteDelegations,
        transport: Transport | None = None,
    ) -> None:
        self.url = url
        self.delegations = delegations
        self.transport = transport or _post_json

    async def create(
        self, delegation_id: str, session_id: str, transcript: str, labels: dict[str, object]
    ) -> None:
        """Register, then POST a delegation without waiting for task completion."""
        payload = {
            "session_id": session_id,
            "delegation_id": delegation_id,
            "transcript": transcript,
            "kind": labels.get("kind", "task"),
            "gliner": labels,
        }
        try:
            await asyncio.to_thread(self.delegations.register, delegation_id, session_id)
            await asyncio.to_thread(
                self.transport,
                self.url,
                json.dumps(payload).encode("utf-8"),
                {"Content-Type": "application/json"},
            )
        except Exception:
            await asyncio.to_thread(self.delegations.unregister_delegation, delegation_id)
            raise


def _post_json(url: str, body: bytes, headers: Mapping[str, str]) -> bytes:
    request = Request(url, data=body, headers=dict(headers), method="POST")
    with urlopen(request, timeout=30) as response:
        return response.read()
