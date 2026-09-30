"""ASGI entry point for local development."""

from .adapters.faster_whisper import FasterWhisperASR
from .adapters.gliner_router import GLiNERRouter
from .adapters.http_tasks import HttpTasks
from .adapters.laya_router import LayaRouter
from .adapters.lmstudio_talker import LMStudioTalker
from .adapters.sqlite_delegations import SQLiteDelegations
from .adapters.silero_vad import SileroVAD
from .adapters.supertonic_speaker import SupertonicSpeaker
from .app import Ports, create_app
from .config import Settings


settings = Settings.from_env()
if settings.router_provider == "laya":
    router = LayaRouter(
        settings.laya_model_path or "",
        onnx_path=settings.laya_onnx_path or "",
        subfolder=settings.laya_subfolder,
        task_threshold=settings.gliner_task_threshold,
    )
else:
    router = GLiNERRouter(
        settings.gliner_model_path or "",
        device=settings.gliner_device,
        task_threshold=settings.gliner_task_threshold,
    )
talker = LMStudioTalker(
    settings.lm_studio_base_url,
    settings.lm_studio_model,
    reasoning_effort=settings.lm_studio_reasoning_effort,
)
speaker = SupertonicSpeaker(settings.supertonic_voice_map)
delegations = SQLiteDelegations(
    settings.delegation_db_path,
    session_ttl_seconds=settings.max_session_duration_s,
)


def ports_factory() -> Ports:
    """Create stateful speech adapters for one WebSocket session."""
    return Ports(
        vad=SileroVAD(settings.silero_model_path or "", pause_ms=settings.vad_pause_ms),
        asr=FasterWhisperASR(settings.whisper_model_path or ""),
        router=router,
        talker=talker,
        speaker=speaker,
        tasks=HttpTasks(settings.tasks_url, delegations=delegations),
    )


app = create_app(settings, ports_factory, delegations=delegations)
