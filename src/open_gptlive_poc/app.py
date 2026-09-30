"""FastAPI composition root for the GPT-Live server."""

import asyncio
import json
import logging
import sqlite3
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from loguru import logger

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
        logger.info("Initializing delegation store", database_path=settings.delegation_db_path)
        await asyncio.to_thread(callback_store.initialize)
        worker = asyncio.create_task(_dispatch_callbacks(app, callback_store))
        try:
            yield
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await asyncio.to_thread(callback_store.close_worker)
            logger.info("Delegation callback worker stopped")

    app = FastAPI(title="GPT-Live server", lifespan=lifespan)
    app.state.settings = settings
    app.state.ports_factory = ports_factory or (lambda: None)
    app.state.sessions = {}
    app.state.delegations = callback_store

    def is_authorized(authorization: str | None) -> bool:
        from .live.protocol import authenticate

        return not settings.bearer_token or authenticate(authorization, settings.bearer_token)

    async def run_live_session(websocket: WebSocket) -> None:
        from .live.session import LiveSession

        session = LiveSession(websocket, settings, app.state.ports_factory)
        app.state.sessions[session.state.session_id] = session
        logger.info("Live WebSocket accepted", session_id=session.state.session_id, active_sessions=len(app.state.sessions))
        try:
            await session.run()
        except WebSocketDisconnect:
            await session.close("connection_lost")
        finally:
            app.state.sessions.pop(session.state.session_id, None)
            await asyncio.to_thread(callback_store.unregister_session, session.state.session_id)
            logger.info("Live WebSocket removed", session_id=session.state.session_id, active_sessions=len(app.state.sessions))

    @app.get("/", include_in_schema=False)
    def voice_demo() -> FileResponse:
        return FileResponse(Path(__file__).resolve().parents[2] / "demo" / "index.html")

    @app.get("/status", include_in_schema=False, response_class=HTMLResponse)
    def demo_status() -> str:
        return '<p id="configuration">GPT-Live server connected</p>'

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return a minimal liveness response."""
        return {"status": "ok"}

    @app.post("/internal/delegations/{delegation_id}/result")
    async def delegation_result(delegation_id: str, request: Request) -> dict[str, str]:
        """Deliver task callback text to its live session for speech output."""
        if not is_authorized(request.headers.get("authorization")):
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
            callback_id = await asyncio.to_thread(callback_store.enqueue, delegation_id, content)
        except sqlite3.Error as exc:
            raise HTTPException(status_code=503, detail="Callback store unavailable") from exc
        if callback_id is None:
            logger.warning("Callback rejected for unknown delegation", delegation_id=delegation_id)
            raise HTTPException(status_code=404, detail="Unknown delegation")
        deadline = monotonic() + _CALLBACK_WAIT_SECONDS
        while monotonic() < deadline:
            try:
                accepted = await asyncio.to_thread(callback_store.result, callback_id)
            except sqlite3.Error as exc:
                raise HTTPException(status_code=503, detail="Callback store unavailable") from exc
            if accepted is not None:
                if accepted:
                    logger.info("Delegation callback accepted", delegation_id=delegation_id)
                    return {"status": "accepted"}
                raise HTTPException(status_code=404, detail="Unknown delegation")
            await asyncio.sleep(0.05)
        raise HTTPException(status_code=503, detail="Session worker did not acknowledge callback")

    @app.websocket("/v1/live/sessions")
    async def live_sessions(websocket: WebSocket) -> None:
        """Authenticate and run one isolated GPT-Live session."""
        if not is_authorized(websocket.headers.get("authorization")):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        await run_live_session(websocket)

    @app.websocket("/ws/live")
    async def browser_live(websocket: WebSocket) -> None:
        """Authenticate the browser UI then run the same live session handler."""
        from .live.protocol import ServerEvent

        await websocket.accept()
        try:
            auth_frame = await websocket.receive_text()
            if len(auth_frame) > 4096:
                await websocket.close(code=1008, reason="Invalid authentication frame")
                return
            payload = json.loads(auth_frame)
            token = payload.get("token") if isinstance(payload, dict) and payload.get("type") == "auth" else None
            authorization = f"Bearer {token}" if isinstance(token, str) and token else None
            if not is_authorized(authorization):
                logger.warning("Browser WebSocket authentication rejected")
                await websocket.close(code=1008, reason="Unauthorized")
                return
            await websocket.send_text(ServerEvent("session.authenticated").to_json())
            await run_live_session(websocket)
        except (WebSocketDisconnect, json.JSONDecodeError):
            return

    return app


async def _dispatch_callbacks(app: FastAPI, callback_store: SQLiteDelegations) -> None:
    """Poll the local SQLite queue and deliver callbacks to sessions owned here."""
    last_heartbeat = 0.0
    accepted_callbacks: set[str] = set()
    while True:
        try:
            if monotonic() - last_heartbeat >= 5:
                await asyncio.to_thread(callback_store.heartbeat)
                last_heartbeat = monotonic()
            callbacks = await asyncio.to_thread(callback_store.pending)
            accepted_callbacks.intersection_update(callback[0] for callback in callbacks)
            for callback_id, delegation_id, content in callbacks:
                accepted = callback_id in accepted_callbacks
                if not accepted:
                    accepted = any(
                        session.handle_task_result(delegation_id, content)
                        for session in app.state.sessions.values()
                    )
                    if accepted:
                        accepted_callbacks.add(callback_id)
                await asyncio.to_thread(callback_store.complete, callback_id, accepted)
                accepted_callbacks.discard(callback_id)
        except Exception:
            _logger.exception("Delegation callback polling failed")
        await asyncio.sleep(0.05)
