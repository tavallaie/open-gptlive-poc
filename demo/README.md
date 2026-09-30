# Local live voice demo

The browser streams microphone audio over WebSocket while you speak. The small
FastAPI bridge keeps the GPT-Live bearer token on the server and forwards the
browser session to `/v1/live/sessions`.

Install the demo's lightweight environment:

```bash
uv sync --project demo --python 3.14
```

Set these values in the repository-root `.env`:

```dotenv
GPTLIVE_DEMO_WS_URL=ws://127.0.0.1:8000/v1/live/sessions
GPTLIVE_DEMO_TOKEN=your-local-server-token
```

`GPTLIVE_DEMO_TOKEN` may instead use the server's existing
`GPTLIVE_BEARER_TOKEN` value.

Start the GPT-Live backend in one terminal (WebSocket support is required):

```bash
uv run --with uvicorn --with python-dotenv --with websockets --no-sync uvicorn --env-file .env --app-dir src open_gptlive_poc.server:app --host 127.0.0.1 --port 8000
```

Start the live voice UI in another:

```bash
uv run --project demo --python 3.14 uvicorn --app-dir demo app:app --host 127.0.0.1 --port 7860
```

Open <http://127.0.0.1:7860>, allow microphone access, and press **Connect
microphone**. Speak naturally; press **Disconnect** to end the session. This
is a local development demo, not a deployed or security-hardened service.
The bridge logs connection lifecycle and failures with Loguru; backend JSON
logs provide the per-session and per-turn processing trace.
