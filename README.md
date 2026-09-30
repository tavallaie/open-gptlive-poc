# open-gptlive-poc

Minimal GPT-Live 1 WebSocket server foundation.

The server exposes the Live event contract over WebSocket and keeps model
implementations behind replaceable ports. Python 3.14 is the local development
version; package compatibility starts at Python 3.12.

## High-level architecture

```mermaid
flowchart LR
    Client[Live client] <-->|WebSocket /v1/live/sessions| Session[Per-session event loop]
    Session --> VAD[VAD port]
    Session --> ASR[ASR port]
    Session --> Router[Router port]
    Session --> Talker[Talker port]
    Session --> Speaker[Speaker port]
    Session --> Tasks[Tasks port]
    VAD --> Silero[Silero adapter]
    ASR --> Whisper[Faster-Whisper adapter]
    Router --> Decision[Selectable System 1 provider]
    Decision --> GLiNER[GLiNER2.5-Decide ONNX]
    Decision --> Laya[Laya multilingual ONNX]
    Talker --> LM[LM Studio adapter]
    Speaker --> Supertonic[Supertonic adapter]
    Tasks --> HTTP[Async task HTTP adapter]
```

The session layer owns protocol and lifecycle state. Adapters own external
libraries and services; the session layer does not import model libraries.

## Session flow

```mermaid
sequenceDiagram
    participant C as Client
    participant S as FastAPI WebSocket
    participant P as Session ports
    participant T as Task system
    participant D as Shared local SQLite queue

    C->>S: session.start
    S-->>C: session.started
    C->>S: session.input_audio.append
    S->>P: feed audio
    P-->>S: transcript / decision / speech
    S-->>C: transcript and output audio events
    S->>T: async task POST
    T->>S: delegation result callback
    S->>D: enqueue for owning worker
    D-->>S: worker delivery acknowledgement
    S-->>C: callback speech
    C->>S: session.close
    S-->>C: session.closed + usage
```

Sessions report cumulative wall-clock time in `session.usage.updated` about
once per minute and repeat the final seconds in `session.closed`. The server
closes expired sessions with reason `expired`; configure the limit with
`GPTLIVE_MAX_SESSION_DURATION_S` (default `3600`). Explicit closes use
`close_requested`, and disconnected sockets use `connection_lost`. Closing a
session cancels current and queued speech but does not cancel tasks already
submitted to the external task system.

## Development

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Configuration is loaded from `GPTLIVE_*` environment variables. Required
credentials and local model paths must remain outside the repository.

Set the required model paths in an untracked local `.env` file or in the
deployment environment. Do not add machine-specific paths to the repository.

The router uses one selectable System 1 decision provider per process. GLiNER
and Laya are alternatives, not sequential stages. The default is the ONNX
export of [GLiNER2.5-Decide](https://huggingface.co/nishparadox/gliner2.5-decide-onnx):

```env
GPTLIVE_ROUTER_PROVIDER=gliner
GPTLIVE_GLINER_MODEL_PATH=/models/gliner2.5-decide-onnx
```

To evaluate or run the multilingual Laya provider instead, install the ONNX
extra and configure the original Laya tokenizer/configuration plus its ONNX
graph:

```env
GPTLIVE_ROUTER_PROVIDER=laya
GPTLIVE_LAYA_MODEL_PATH=/models/laya-multilingual-onnx-int4
GPTLIVE_LAYA_SUBFOLDER=multilingual
GPTLIVE_LAYA_ONNX_PATH=/models/laya-multilingual-onnx-int4/laya-multilingual-int4-blk32.onnx
```

Laya uses typed `choice`, `score`, and `noul` questions and returns the same
`RouteDecision` interface as GLiNER. Keep model downloads outside the
repository; model paths and Hub IDs are configuration only. See the
[Laya ONNX runtime](https://nandhakishorm.github.io/laya/reference/agent/)
and [multilingual ONNX export](https://huggingface.co/yehor-oleksiuk/laya-multilingual-onnx).

LM Studio uses the OpenAI-compatible `/v1/chat/completions` endpoint by
default. Set `GPTLIVE_LM_STUDIO_BASE_URL` to `/api/v1` only when using the
native LM Studio API.

Supertonic supplies speech output. Its Python package downloads model files to
the Hugging Face cache on first use. The Live voice `marin` maps to Supertonic
style `F1`; override voice/style mappings with
`GPTLIVE_SUPERTONIC_VOICE_MAP`, for example
`{"marin":"F1","cedar":"M1"}`. Audio is resampled to mono 24 kHz PCM16LE
and emitted as 100 ms deltas. The authenticated task callback endpoint accepts
`POST /internal/delegations/{delegation_id}/result` with `{"content":"..."}`.
Delegation IDs are registered before the task POST, so a fast callback cannot
arrive before the live session knows the ID. For multiple workers on one
machine, callback ownership and results are relayed through SQLite. All workers
must use the same `GPTLIVE_DELEGATION_DB_PATH` on a local filesystem; do not put
the database on a network share. The default is `.gptlive-delegations.sqlite3`
in the current working directory. Callback rows are consumed once, and unknown
or closed delegation IDs return 404. The result callback is authenticated with
the configured bearer token.

Test OpenAI-compatible streaming directly:

```bash
curl -N http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-2b-mtp-voodoo",
    "messages": [
      {"role": "system", "content": "You answer only in rhymes."},
      {"role": "user", "content": "What is your favorite color?"}
    ],
    "reasoning_effort": "none",
    "stream": true
  }'
```
