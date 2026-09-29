"""LM Studio OpenAI-compatible conversational adapter."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from urllib.request import Request, urlopen

from ..ports.talker import TalkRequest, Talker


Transport = Callable[[str, bytes, Mapping[str, str]], bytes]


class LMStudioTalker(Talker):
    """Generate plain reply text through LM Studio's native or OpenAI API."""

    def __init__(self, base_url: str, model: str, *, transport: Transport | None = None) -> None:
        self.native = base_url.rstrip("/").endswith("/api/v1")
        self.url = base_url.rstrip("/") + ("/chat" if self.native else "/chat/completions")
        self.model = model
        self.transport = transport or _post_json

    async def reply(self, request: TalkRequest) -> str:
        """Generate one reply without blocking the event loop."""
        payload = _native_payload(request, self.model) if self.native else {
            "model": self.model,
            "messages": _messages(request),
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        raw = await asyncio.to_thread(self.transport, self.url, body, {"Content-Type": "application/json"})
        try:
            response = json.loads(raw)
            content = _native_content(response) if self.native else response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("LM Studio returned an invalid chat response") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LM Studio returned empty reply text")
        return content


def _messages(request: TalkRequest) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if request.instructions:
        messages.append({"role": "system", "content": "\n\n".join(request.instructions)})
    if request.thinking:
        messages.append({"role": "system", "content": "Private context:\n" + "\n".join(request.thinking)})
    messages.extend({"role": turn.role, "content": turn.text} for turn in request.history)
    messages.append({"role": "user", "content": request.transcript})
    return messages


def _native_payload(request: TalkRequest, model: str) -> dict[str, object]:
    """Build the LM Studio native v1 request shape."""
    payload: dict[str, object] = {"model": model, "input": request.transcript, "store": False}
    if request.instructions:
        payload["system_prompt"] = "\n\n".join(request.instructions)
    if request.thinking or request.history:
        context = [*request.thinking]
        context.extend(f"{turn.role}: {turn.text}" for turn in request.history)
        payload["input"] = "Context:\n" + "\n".join(context) + "\n\nUser: " + request.transcript
    return payload


def _native_content(response: object) -> str:
    if not isinstance(response, Mapping) or not isinstance(response.get("output"), list):
        raise TypeError("output must be a list")
    messages = [item for item in response["output"] if isinstance(item, Mapping) and item.get("type") == "message"]
    if not messages or not isinstance(messages[-1].get("content"), str):
        raise TypeError("output does not contain a message")
    return messages[-1]["content"]


def _post_json(url: str, body: bytes, headers: Mapping[str, str]) -> bytes:
    request = Request(url, data=body, headers=dict(headers), method="POST")
    with urlopen(request, timeout=60) as response:
        return response.read()
