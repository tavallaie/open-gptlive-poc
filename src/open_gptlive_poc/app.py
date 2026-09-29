"""FastAPI composition root for the GPT-Live server."""

from dataclasses import dataclass
from fastapi import FastAPI

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


def create_app(settings: Settings, ports: Ports | None = None) -> FastAPI:
    """Create the FastAPI application with validated settings and injected ports."""
    settings.validate()
    app = FastAPI(title="GPT-Live server")
    app.state.settings = settings
    app.state.ports = ports

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
