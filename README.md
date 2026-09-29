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
    Router --> Laya[Laya adapter]
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

    C->>S: session.start
    S-->>C: session.started
    C->>S: session.input_audio.append
    S->>P: feed audio
    P-->>S: transcript / decision / speech
    S-->>C: transcript and output audio events
    S->>T: async task POST
    T-->>S: delegation result callback
    S-->>C: callback speech
    C->>S: session.close
    S-->>C: session.closed + usage
```

## Development

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Configuration is loaded from `GPTLIVE_*` environment variables. Required
credentials and local model paths must remain outside the repository.

Example local model configuration:

```bash
export GPTLIVE_SILERO_MODEL_PATH=/home/ali/Models/silero_vad.onnx
export GPTLIVE_WHISPER_MODEL_PATH=/home/ali/Models/hub/models--deepdml--faster-whisper-large-v3-turbo-ct2/snapshots/4df90f75321148c3a29a9e2351b7ddf8f5b115a8
```
