---
title: GPT-Live demo
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
---

# GPT-Live Hugging Face Space

Copy the contents of this directory into a Gradio Space, or configure the
Space repository to use this directory as its app root.

Add these Space secrets:

- `GPTLIVE_DEMO_WS_URL`: public server URL, for example
  `wss://example.com/v1/live/sessions`.
- `GPTLIVE_DEMO_TOKEN`: the server bearer token.

For local testing from this repository:

```bash
set -a
source .env
set +a
uv run --with uvicorn uvicorn --app-dir src open_gptlive_poc.server:app
```

The current server slice verifies session, audio, and close events. It will
return transcript and output-audio events after the ASR, GLiNER, LM Studio,
and TTS adapters are wired into the session pipeline. The token is never
rendered in the UI.
