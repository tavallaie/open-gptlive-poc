"""FastAPI composition root for the GPT-Live server."""

import json
from dataclasses import dataclass
from collections.abc import Callable

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect

from .config import Settings
from .ports import ASR, Router, Speaker, Tasks, Talker, VAD


_MAX_CALLBACK_BODY_BYTES = 64 * 1024
_MAX_CALLBACK_WORDS = 500


@dataclass(frozen=True, slots=True)
class Ports:
    """The six replaceable collaborators used by one server process."""

    vad: VAD
    asr: ASR
    router: Router
    talker: Talker
    speaker: Speaker
    tasks: Tasks


def create_app(
    settings: Settings,
    ports_factory: Callable[[], Ports | None] | None = None,
) -> FastAPI:
    """Create the FastAPI application with validated settings and injected ports."""
    settings.validate()
    app = FastAPI(title="GPT-Live server")
    app.state.settings = settings
    app.state.ports_factory = ports_factory or (lambda: None)
    app.state.sessions = {}

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return a minimal liveness response."""
        return {"status": "ok"}

    @app.post("/internal/delegations/{delegation_id}/result")
    async def delegation_result(delegation_id: str, request: Request) -> dict[str, str]:
        """Deliver task callback text to its live session for speech output."""
        from .live.protocol import authenticate

        if not authenticate(request.headers.get("authorization"), settings.bearer_token or ""):
            raise HTTPException(status_code=401, detail="Unauthorized")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > _MAX_CALLBACK_BODY_BYTES:
                raise HTTPException(status_code=413, detail="callback body is too large")
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail="request body must be valid JSON") from exc
        content = payload.get("content") if isinstance(payload, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise HTTPException(status_code=422, detail="content must be a non-empty string")
        if len(content.split()) > _MAX_CALLBACK_WORDS:
            raise HTTPException(status_code=422, detail="content must contain at most 500 words")
        for session in app.state.sessions.values():
            if session.handle_task_result(delegation_id, content):
                return {"status": "accepted"}
        raise HTTPException(status_code=404, detail="Unknown delegation")

    @app.websocket("/v1/live/sessions")
    async def live_sessions(websocket: WebSocket) -> None:
        """Authenticate and run one isolated GPT-Live session."""
        from .live.protocol import authenticate
        from .live.session import LiveSession

        if not authenticate(websocket.headers.get("authorization"), settings.bearer_token or ""):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        session = LiveSession(websocket, settings, app.state.ports_factory)
        app.state.sessions[session.state.session_id] = session
        try:
            await session.run()
        except WebSocketDisconnect:
            await session.close("connection_lost")
        finally:
            app.state.sessions.pop(session.state.session_id, None)

    return app
