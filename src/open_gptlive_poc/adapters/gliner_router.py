"""GLiNER2.5-Decide transcript router."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from numbers import Real
from typing import Any, cast

from ..ports.router import RouteDecision, RouteKind, Router


_QUESTION = {
    "needs_task": ["task", "no_task"],
    "kind": ["chat", "task", "function_call"],
    "response": ["talk", "task", "both"],
}


class GLiNERRouter(Router):
    """Classify transcripts with the local GLiNER2.5-Decide model."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        task_threshold: float = 0.5,
        model: Any | None = None,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not 0 <= task_threshold <= 1:
            raise ValueError("task_threshold must be between 0 and 1")
        self.task_threshold = task_threshold
        self.model = model or self._load_model(model_path, device, model_factory)

    def classify(self, transcript: str) -> RouteDecision:
        """Return a typed route from one transcript."""
        if not transcript.strip():
            raise ValueError("transcript must not be empty")
        result = self.model.classify_text(transcript, _QUESTION)
        if not isinstance(result, Mapping):
            raise TypeError("GLiNER response must be a mapping")
        needs_task = _score(result.get("needs_task"), positive=("task", "yes", "true"))
        kind = _choice(result.get("kind"), ("chat", "task", "function_call"), "chat")
        response = _choice(result.get("response"), ("talk", "task", "both"), "talk" if kind == "chat" else "task")
        return RouteDecision(
            transcript=transcript,
            needs_task=needs_task,
            kind=cast(RouteKind, kind),
            talk=response in {"talk", "both"},
            labels=dict(result),
            task_threshold=self.task_threshold,
        )

    @staticmethod
    def _load_model(model_path: str, device: str, model_factory: Callable[[str], Any] | None) -> Any:
        if model_factory is not None:
            return model_factory(model_path)
        try:
            from gliner2 import Extractor
        except ImportError as exc:
            raise RuntimeError("Install gliner2 to use GLiNERRouter") from exc
        return Extractor.from_pretrained(model_path, map_location=device)


def _score(value: object, *, positive: tuple[str, ...]) -> float:
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, (tuple, list)) and len(value) >= 2 and isinstance(value[1], Real):
        return float(value[1]) if str(value[0]).lower() in positive else 1.0 - float(value[1])
    if isinstance(value, Mapping):
        values = {str(key).lower(): float(item) for key, item in value.items() if isinstance(item, Real)}
        for key in positive:
            if key in values:
                return values[key]
        if values:
            return max(values.values())
    raise TypeError("GLiNER needs_task output must contain a numeric score")


def _choice(value: object, choices: tuple[str, ...], default: str) -> str:
    if isinstance(value, str):
        return value if value in choices else default
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], str):
        return value[0] if value[0] in choices else default
    if isinstance(value, Mapping):
        scores = {str(key): float(item) for key, item in value.items() if isinstance(item, Real)}
        if scores:
            return max((choice for choice in choices if choice in scores), key=scores.get, default=default)
    return default
