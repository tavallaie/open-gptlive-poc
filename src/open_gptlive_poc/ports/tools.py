"""Allow-listed tools that a Talker may invoke."""

from collections.abc import Mapping, Sequence
from typing import Protocol


class ToolExecutor(Protocol):
    """Expose JSON schemas and execute only registered functions."""

    @property
    def schemas(self) -> Sequence[Mapping[str, object]]: ...

    async def execute(self, name: str, arguments: Mapping[str, object]) -> str: ...
