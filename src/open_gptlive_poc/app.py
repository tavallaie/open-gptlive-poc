"""FastAPI composition root for the GPT-Live server."""

import asyncio
import json
import logging
import sqlite3
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import monotonic

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect

from .adapters.sqlite_delegations import SQLiteDelegations
from .config import Settings
from .ports import ASR, Router, Speaker, Tasks, Talker, VAD


_MAX_CALLBACK_BODY_BYTES = 64 * 1024
_MAX_CALLBACK_WORDS = 500
_CALLBACK_WAIT_SECONDS = 5
_logger = logging.getLogger(__name__)


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
    delegations: SQLiteDelegations | None = None,
) -> FastAPI:
    """Create the FastAPI application with validated settings and injected ports."""
    settings.validate()
    callback_store = delegations or SQLiteDelegations(
        settings.delegation_db_path,
        session_ttl_seconds=settings.max_session_duration_s,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        callback_store.initialize()
        worker = asyncio.create_task(_dispatch_callbacks(app, callback_store))
        try:
            yield
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            callback_store.close_worker()

    app = FastAPI(title="GPT-Live server", lifespan=lifespan)
    app.state.settings = settings
    app.state.ports_factory = ports_factory or (lambda: None)
    app.state.sessions = {}
    app.state.delegations = callback_store

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
        try:
            callback_id = callback_store.enqueue(delegation_id, content)
        except sqlite3.Error as exc:
            raise HTTPException(status_code=503, detail="Callback store unavailable") from exc
        if callback_id is None:
            raise HTTPException(status_code=404, detail="Unknown delegation")
        deadline = monotonic() + _CALLBACK_WAIT_SECONDS
        while monotonic() < deadline:
            try:
                accepted = callback_store.result(callback_id)
            except sqlite3.Error as exc:
                raise HTTPException(status_code=503, detail="Callback store unavailable") from exc
            if accepted is not None:
                if accepted:
                    return {"status": "accepted"}
                raise HTTPException(status_code=404, detail="Unknown delegation")
            await asyncio.sleep(0.02)
        raise HTTPException(status_code=503, detail="Session worker did not acknowledge callback")

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
            callback_store.unregister_session(session.state.session_id)

    return app


async def _dispatch_callbacks(app: FastAPI, callback_store: SQLiteDelegations) -> None:
    """Poll the local SQLite queue and deliver callbacks to sessions owned here."""
    last_heartbeat = 0.0
    while True:
        try:
            if monotonic() - last_heartbeat >= 5:
                callback_store.heartbeat()
                last_heartbeat = monotonic()
            for callback_id, delegation_id, content in callback_store.pending():
                accepted = any(
                    session.handle_task_result(delegation_id, content)
                    for session in app.state.sessions.values()
                )
                callback_store.complete(callback_id, accepted)
        except Exception:
            _logger.exception("Delegation callback polling failed")
        await asyncio.sleep(0.05)
