"""ASGI entry point for local development."""

from .adapters.faster_whisper import FasterWhisperASR
from .adapters.gliner_router import GLiNERRouter
from .adapters.lmstudio_talker import LMStudioTalker
from .adapters.silero_vad import SileroVAD
from .app import Ports, create_app
from .config import Settings


settings = Settings.from_env()
router = GLiNERRouter(
    settings.gliner_model_path or "",
    device=settings.gliner_device,
    task_threshold=settings.gliner_task_threshold,
)
talker = LMStudioTalker(settings.lm_studio_base_url, settings.lm_studio_model)


def ports_factory() -> Ports:
    """Create stateful speech adapters for one WebSocket session."""
    return Ports(
        vad=SileroVAD(settings.silero_model_path or "", pause_ms=settings.vad_pause_ms),
        asr=FasterWhisperASR(settings.whisper_model_path or ""),
        router=router,
        talker=talker,
        speaker=object(),
        tasks=object(),
    )


app = create_app(settings, ports_factory)
