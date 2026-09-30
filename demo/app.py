"""FastAPI UI and server-side WebSocket bridge for live microphone testing."""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from loguru import logger
import websockets

try:
    from client import upstream_settings
except ImportError:
    from .client import upstream_settings


_log = logger.bind(component="demo-bridge")
_DEMO_DIR = Path(__file__).resolve().parent
load_dotenv(_DEMO_DIR.parent / ".env")

app = FastAPI(title="GPT-Live local voice demo")


@app.get("/")
async def index() -> FileResponse:
    """Serve the microphone UI."""
    return FileResponse(_DEMO_DIR / "index.html", media_type="text/html")


@app.get("/status", response_class=HTMLResponse)
async def status() -> str:
    """Report whether the server-side WebSocket bridge has its configuration."""
    endpoint, token = upstream_settings()
    message = "Backend endpoint and token configured." if endpoint and token else (
        "Set GPTLIVE_DEMO_WS_URL and GPTLIVE_DEMO_TOKEN in the root .env."
    )
    return f'<p id="configuration">{message}</p>'


@app.websocket("/ws/live")
async def live_bridge(browser: WebSocket) -> None:
    """Proxy one browser session to the authenticated GPT-Live WebSocket."""
    endpoint, token = upstream_settings()
    if not endpoint or not token:
        _log.error("Demo WebSocket bridge is missing endpoint or token configuration")
        await browser.close(code=1011, reason="Set GPTLIVE_DEMO_WS_URL and GPTLIVE_DEMO_TOKEN in .env")
        return

    await browser.accept()
    _log.info("Browser WebSocket accepted")
    try:
        async with websockets.connect(
            endpoint,
            additional_headers={"Authorization": f"Bearer {token}"},
            max_size=16 * 1024 * 1024,
        ) as server:
            _log.info("Connected to GPT-Live backend")

            async def browser_to_server() -> None:
                while True:
                    await server.send(await browser.receive_text())

            async def server_to_browser() -> None:
                async for message in server:
                    await browser.send_text(message)

            tasks = {
                asyncio.create_task(browser_to_server()),
                asyncio.create_task(server_to_browser()),
            }
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            await asyncio.gather(*done, return_exceptions=True)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    except WebSocketDisconnect:
        _log.info("Browser WebSocket disconnected")
        pass
    except Exception:
        _log.exception("Live demo WebSocket bridge failed")
        try:
            await browser.close(code=1011, reason="Could not connect to the GPT-Live server")
        except RuntimeError:
            pass
    else:
        _log.info("GPT-Live backend disconnected")
        try:
            await browser.close()
        except RuntimeError:
            pass
