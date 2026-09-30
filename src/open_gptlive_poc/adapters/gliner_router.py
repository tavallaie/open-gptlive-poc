"""ONNX GLiNER2.5-Decide transcript router."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import importlib.util
from numbers import Real
from pathlib import Path
import re
import sys
from typing import Any, cast

from ..ports.router import RouteDecision, RouteKind, Router


_QUESTION = {
    "needs_task": ["task", "no_task"],
    "kind": ["chat", "task", "function_call"],
    "response": ["talk", "task", "both"],
    "interrupt": ["interrupt_current_response", "continue_current_response"],
    "turn_state": ["wait_for_user", "ready_to_process"],
}
_WAIT_FOR_USER_THRESHOLD = 0.7


class GLiNERRouter(Router):
    """Classify transcripts with the local GLiNER2.5-Decide model."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        task_threshold: float = 0.7,
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
        result = self.model.classify_text(transcript, _QUESTION, include_confidence=True)
        if not isinstance(result, Mapping):
            raise TypeError("GLiNER response must be a mapping")
        needs_task = _score(_label_confidence(result.get("needs_task")), positive=("task", "yes", "true"))
        kind = _choice(_label_confidence(result.get("kind")), ("chat", "task", "function_call"), "chat")
        response = _choice(_label_confidence(result.get("response")), ("talk", "task", "both"), "talk" if kind == "chat" else "task")
        interrupt_result = result.get("interrupt")
        interrupt = interrupt_result is not None and _score(
            _label_confidence(interrupt_result),
            positive=("interrupt_current_response", "interrupt", "stop", "cancel", "yes", "true"),
        ) >= max(self.task_threshold, _WAIT_FOR_USER_THRESHOLD)
        interrupt = interrupt or _has_explicit_interruption(transcript)
        wait_for_user = _is_wait_for_user(result.get("turn_state"))
        task = needs_task >= self.task_threshold
        talk = not task or kind in {"chat", "function_call"} or response in {"talk", "both"}
        return RouteDecision(
            transcript=transcript,
            needs_task=needs_task,
            kind=cast(RouteKind, kind),
            talk=talk,
            labels=dict(result),
            task_threshold=self.task_threshold,
            interrupt_current=interrupt,
            wait_for_user=wait_for_user,
        )

    @staticmethod
    def _load_model(model_path: str, device: str, model_factory: Callable[[str], Any] | None) -> Any:
        if model_factory is not None:
            return model_factory(model_path)
        local_path = Path(model_path)
        if local_path.is_dir() and (local_path / "gliner_onnx.py").is_file():
            return _load_local_onnx_model(local_path)
        try:
            from gliner2 import GLiNER2
        except ImportError as exc:
            raise RuntimeError("Install gliner2 to use GLiNERRouter") from exc
        # GLiNER2 selects the ONNX Runtime backend when the model repository
        # contains an ONNX graph. The device remains a configuration concern
        # for repositories that provide a CPU/GPU-specific runtime.
        return GLiNER2.from_pretrained(model_path)


def _label_confidence(value: object) -> object:
    """Normalize GLiNER's confidence result to the router's label-score shape."""
    if isinstance(value, Mapping) and isinstance(value.get("label"), str) and isinstance(value.get("confidence"), Real):
        return value["label"], value["confidence"]
    return value


def _load_local_onnx_model(model_path: Path) -> Any:
    """Load the torch-free runtime shipped with the ONNX model bundle."""
    spec = importlib.util.spec_from_file_location("gptlive_gliner_onnx", model_path / "gliner_onnx.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load GLiNER ONNX runtime from {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model_file = next(model_path.glob("model*.onnx"), None)
    if model_file is None:
        raise FileNotFoundError(f"No ONNX graph found in {model_path}")
    return _LocalGlinerClassifier(module, model_file, model_path / "tokenizer.json")


class _LocalGlinerClassifier:
    """Adapt the bundle's probability API to the existing router port."""

    def __init__(self, module: Any, model_path: Path, tokenizer_path: Path) -> None:
        self._module = module
        self._model = module.GlinerOnnx(str(model_path), str(tokenizer_path))

    def classify_text(
        self,
        transcript: str,
        questions: Mapping[str, list[str]],
        *,
        include_confidence: bool = False,
    ) -> Mapping[str, Mapping[str, float]]:
        tasks = [
            self._module.Task(name, {label: None for label in labels}, exclusive=True)
            for name, labels in questions.items()
        ]
        return self._model.probabilities(transcript, tasks)


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


def _is_wait_for_user(value: object) -> bool:
    """Wait only when GLiNER is confident the utterance is incomplete."""
    value = _label_confidence(value)
    if isinstance(value, Mapping):
        score = value.get("wait_for_user")
        return isinstance(score, Real) and float(score) >= _WAIT_FOR_USER_THRESHOLD
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], str):
        return (
            value[0] == "wait_for_user"
            and len(value) > 1
            and isinstance(value[1], Real)
            and float(value[1]) >= _WAIT_FOR_USER_THRESHOLD
        )
    return value == "wait_for_user"


def _has_explicit_interruption(transcript: str) -> bool:
    return re.search(r"\b(no|stop|wait|hold on|I mean)\b", transcript, re.IGNORECASE) is not None
