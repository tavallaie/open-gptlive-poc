"""FastAPI composition root for the GPT-Live server."""

from dataclasses import dataclass
from collections.abc import Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .config import Settings
from .ports import ASR, Router, Speaker, Tasks, Talker, VAD


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

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return a minimal liveness response."""
        return {"status": "ok"}

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
        try:
            await session.run()
        except WebSocketDisconnect:
            await session.close("connection_lost")

    return app
