"""Turn-routing port and its typed decision."""

from dataclasses import dataclass
from typing import Literal, Mapping, Protocol


RouteKind = Literal["chat", "task", "function_call"]
BackgroundDelivery = Literal[
    "silent", "after_playback", "interrupt_after_sentence", "wait_for_user"
]
ToolExecution = Literal["inline", "background"]
ToolUrgency = Literal["normal", "urgent"]


@dataclass(frozen=True, slots=True)
class ToolProfile:
    """Runtime behavior of a tool, separate from model-generated arguments."""

    execution: ToolExecution
    urgency: ToolUrgency = "normal"
    can_request_input: bool = True


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

    def classify_background_delivery(
        self, request: str, result: str, profile: ToolProfile | None = None
    ) -> BackgroundDelivery:
        """Choose how a background result should be delivered."""
