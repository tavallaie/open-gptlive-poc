"""Outbound asynchronous task delegation port."""

from typing import Protocol


class Tasks(Protocol):
    """Create work in an external task system without waiting for completion."""

    async def create(
        self,
        delegation_id: str,
        session_id: str,
        transcript: str,
        labels: dict[str, object],
    ) -> None: ...
