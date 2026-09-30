"""ONNX Laya System 1 transcript router."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from numbers import Real
from pathlib import Path
from typing import Any, cast

from ..ports.router import RouteDecision, RouteKind, Router


_QUESTIONS = {
    "needs_task": {
        "type": "choice",
        "instructions": "Does this transcript require an external task or action?",
        "criteria": {
            "task": "The transcript requires an external task or action",
            "no_task": "The transcript only needs a conversational answer",
        },
    },
    "kind": {
        "type": "choice",
        "instructions": "What kind of request is this?",
        "criteria": {
            "chat": "A conversational request that needs no external action",
            "task": "A request that requires an external task or action",
            "function_call": "A request that requires invoking a function",
        },
    },
    "response": {
        "type": "choice",
        "instructions": "What should the assistant do next?",
        "criteria": {
            "talk": "Answer the user conversationally",
            "task": "Delegate the request to the task system",
            "both": "Delegate the task and tell the user what happened",
        },
    },
}


class LayaRouter(Router):
    """Route transcripts with an ONNX-backed Laya agent."""

    def __init__(
        self,
        model_path: str,
        *,
        onnx_path: str,
        subfolder: str | None = "multilingual",
        task_threshold: float = 0.5,
        model: Any | None = None,
        model_factory: Callable[[str, str], Any] | None = None,
    ) -> None:
        if not 0 <= task_threshold <= 1:
            raise ValueError("task_threshold must be between 0 and 1")
        self.task_threshold = task_threshold
        self.model = model or self._load_model(model_path, onnx_path, subfolder, model_factory)

    def classify(self, transcript: str) -> RouteDecision:
        """Return a typed route from one transcript."""
        if not transcript.strip():
            raise ValueError("transcript must not be empty")
        result = self.model.predict(transcript, _QUESTIONS)
        answers = _answers(result)
        needs_task = _task_probability(answers["needs_task"])
        kind = _choice(answers["kind"], ("chat", "task", "function_call"), "chat")
        response = _choice(answers["response"], ("talk", "task", "both"), "talk" if kind == "chat" else "task")
        task = needs_task >= self.task_threshold
        talk = not task or kind == "chat" or response in {"talk", "both"}
        return RouteDecision(
            transcript=transcript,
            needs_task=needs_task,
            kind=cast(RouteKind, kind),
            talk=talk,
            labels=dict(answers),
            task_threshold=self.task_threshold,
        )

    @staticmethod
    def _load_model(
        model_path: str,
        onnx_path: str,
        subfolder: str | None,
        model_factory: Callable[[str, str], Any] | None,
    ) -> Any:
        if model_factory is not None:
            return model_factory(model_path, onnx_path)
        try:
            from laya.onnx_agent import ONNXAgent
        except ImportError as exc:
            raise RuntimeError("Install laya[onnx] to use LayaRouter") from exc
        if subfolder and Path(model_path).is_dir() and not (Path(model_path) / subfolder).is_dir():
            subfolder = None
        return ONNXAgent(model_path, onnx_path=onnx_path, subfolder=subfolder)


def _answers(result: object) -> Mapping[str, object]:
    if not isinstance(result, Mapping) or not isinstance(result.get("answers"), Mapping):
        raise TypeError("Laya response must contain an answers mapping")
    answers = result["answers"]
    assert isinstance(answers, Mapping)
    return answers


def _task_probability(value: object) -> float:
    if isinstance(value, Mapping):
        probabilities = value.get("probabilities")
        if isinstance(probabilities, Mapping):
            task = probabilities.get("task")
            if isinstance(task, Real):
                return float(task)
        choice = value.get("choice")
        if choice == "task":
            return float(value.get("confidence", 1.0))
        if choice == "no_task":
            return 0.0
    raise TypeError("Laya task answer must contain task probabilities")


def _choice(value: object, choices: tuple[str, ...], default: str) -> str:
    if isinstance(value, str):
        return value if value in choices else default
    if isinstance(value, Mapping):
        candidate = value.get("choice")
        if isinstance(candidate, str) and candidate in choices:
            return candidate
        scores = {str(key): item for key, item in value.items() if isinstance(item, Real)}
        if scores:
            return max((choice for choice in choices if choice in scores), key=scores.get, default=default)
    return default
