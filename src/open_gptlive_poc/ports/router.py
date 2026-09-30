"""Turn-routing port and its typed decision."""

from dataclasses import dataclass
from typing import Literal, Mapping, Protocol


RouteKind = Literal["chat", "task", "function_call"]


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Classification result used by the session orchestration layer."""

    transcript: str
    needs_task: float
    kind: RouteKind
    talk: bool
    labels: Mapping[str, object]
    task_threshold: float = 0.5
    interrupt_current: bool = False
    wait_for_user: bool = False

    @property
    def task(self) -> bool:
        """Whether the task system should receive this turn."""
        return self.needs_task >= self.task_threshold and self.kind != "function_call" and not self.wait_for_user


class Router(Protocol):
    """Classify one transcript into talk/task behavior."""

    def classify(self, transcript: str) -> RouteDecision:
        """Return a typed decision for a transcript."""
