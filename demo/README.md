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

If no endpoint is configured, the demo runs in local preview mode and returns
the recorded input as output audio. The token is never rendered in the UI.
