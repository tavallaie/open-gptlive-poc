# Compatible GPT-Live 1 server (slice 1)

Date: 2026-09-30
Status: approved for slice 1; more detail later
Repo: `open-gptlive-poc`

This process is an open GPT-Live 1 WebSocket server. A client that speaks OpenAI Live events can connect. Behind the socket, a local pipeline listens, decides, talks, speaks, and fires tasks.

Slice 1 is WebSocket only. One Python process. Real local models. No mocks.

## Goal

Implement enough of the GPT-Live 1 event contract that a client can:

1. Start a session.
2. Stream 24 kHz PCM16LE input.
3. Receive input transcripts, output transcripts, and output audio.
4. See `session.delegation.created` when a turn is a task.
5. Hear spoken results when the other system posts a callback.
6. Close the session and get usage.

Sources for the public contract: OpenAI GPT-Live getting started, delegation, session management, and the Live event reference.

## Locked product choices

| Choice | Value |
|---|---|
| Public API | GPT-Live 1 events on `/v1/live/sessions` |
| Delegation | Client mode only (`target: "client"`) |
| Transport | WebSocket first |
| VAD | Silero |
| ASR | Faster-Whisper, weights already on the machine |
| Router | Laya typed decisions |
| Conversational text | LM Studio `/v1/chat/completions` |
| TTS | Supertonic, resampled to 24 kHz |
| Tasks | Async HTTP POST out, callback in |
| Process shape | One process, six ports, adapters |

## Public surface

Listen on `wss://<host>/v1/live/sessions`.

Authenticate with `Authorization: Bearer <token>`. The token is local config. It is not an OpenAI key.

The first client message is `session.start`. The server replies with `session.started` before it accepts audio or later commands.

Same process also serves:

```
POST /internal/delegations/{delegation_id}/result
```

The other system posts spoken results here. This path is not part of the OpenAI Live spec.

Out of slice 1: WebRTC (`POST /v1/live/sessions` with SDP), SIP, sideband attach, fork, store, recording download.

## Session object

One WebSocket is one session. One session has one VAD, ASR, Laya, Talker, Speaker chain.

Fields locked at `session.start`:

- `model` must be `gpt-live-1`
- `instructions` (max 16,384 tokens; later append with `session.instructions.append`)
- `audio.output.voice` (maps to a Supertonic style)
- `delegation.type` must be `client` or omitted (omitted means client)
- `audio.format` is PCM16LE mono 24 kHz for both directions

`session.update` that changes a locked field fails with `immutable_field_update`. Unsupported fields in this slice are ignored. The session stays up.

Default voice mapping until a full table exists: Live `marin` maps to Supertonic `F1`.

## Lifecycle

```
client                    this process                 LM Studio / Laya / other system
  |  session.start              |
  |---------------------------->|  bind adapters for this session
  |  session.started            |
  |<----------------------------|
  |  session.input_audio.append |
  |---------------------------->|  Silero VAD on rolling buffer
  |                             |  long pause → Faster-Whisper
  |  session.input_transcript   |
  |         .delta              |
  |<----------------------------|
  |                             |  Laya on that transcript
  |                             |-- talk --> LM Studio chat
  |                             |-- task --> HTTP POST (do not wait)
  |  session.delegation.created |            (if task)
  |<----------------------------|
  |  output transcript + audio  |  Supertonic of Talker text
  |<----------------------------|
  |                             |<-- POST /internal/delegations/{id}/result
  |  more output audio          |  Supertonic of callback text
  |<----------------------------|
  |  session.close              |
  |---------------------------->|
  |  session.closed + usage     |
  |<----------------------------|
```

Input audio and TTS run together. Silero keeps watching during playback. A new `speech_started` cancels current TTS (barge-in).

## Event map (slice 1)

### Client to server

| Event | Action |
|---|---|
| `session.start` | Create session, emit `session.started` |
| `session.input_audio.append` | Decode PCM, feed VAD |
| `session.input_audio.mute` | Stop consuming VAD frames |
| `session.input_audio.unmute` | Resume VAD |
| `session.instructions.append` | Append trusted instructions; ack |
| `session.thinking.append` | Quiet context for later Talker turns; ack |
| `session.commentary.append` | Queue text for Speaker; ack |
| `session.close` | Drain, emit `session.closed`, drop socket |

Each append is a plain string, max 500 tokens, with `delegation_id` (`null` means session-wide). Match acks by `client_event_id` when the client sent `event_id`.

### Server to client

| Event | When |
|---|---|
| `session.started` | After start |
| `session.input_transcript.delta` | Whisper fragment with `start_ms` / `end_ms` |
| `session.output_transcript.delta` | Text about to be spoken |
| `session.output_audio.delta` | Base64 24 kHz PCM16LE plus timing |
| `session.delegation.created` | Task POST fired (`target: "client"`) |
| `session.instructions.appended` | Instructions append accepted |
| `session.thinking.appended` | Thinking append accepted |
| `session.commentary.appended` | Commentary append accepted |
| `session.input_audio.muted` | Mute applied |
| `session.input_audio.unmuted` | Unmute applied |
| `session.usage.updated` | About once per minute |
| `session.closed` | Final usage and `reason` |
| `error` | Bad command or adapter failure |

### Not in slice 1

`response.event`, `response.create`, `response.item.create`, MCP, Responses `session.update` fields.

## Internal ports

The session loop does not import Silero, Whisper, Laya, LM Studio, or Supertonic by name. It calls six ports. Adapters wrap the real libraries.

```
PCM in → VAD → ASR → Router → Talker
                              → Tasks  → (async HTTP)
                 Talker/Tasks → Speaker → PCM out
```

| Port | Job | First adapter |
|---|---|---|
| VAD | Speech start/stop on PCM frames | Silero |
| ASR | Turn PCM → transcript + timestamps | Faster-Whisper |
| Router | Transcript → talk / task / both | Laya |
| Talker | History + instructions → reply text | LM Studio |
| Speaker | Text → 24 kHz PCM16LE chunks | Supertonic then resample |
| Tasks | Fire-and-forget POST; later callback | HTTP client + callback route |

Each port is replaceable. Live event names stay fixed.

### VAD (Silero)

Input is 16-bit PCM, resampled to Silero's rate (16 kHz).

Output events: `speech_started(offset_ms)`, `speech_stopped(offset_ms)`.

A turn starts on `speech_stopped` after a long pause. Default pause is 700 ms of silence. Config can change that.

Mute means VAD consumes no frames. The session stays up.

Barge-in: `speech_started` while Speaker is playing cancels the current utterance.

### ASR (Faster-Whisper)

Input is PCM from last `speech_started` through `speech_stopped`.

Output is text plus `start_ms` / `end_ms` relative to session start.

The server emits `session.input_transcript.delta` from that result.

Weights load from a local path in config. No cloud ASR.

### Router (Laya)

One decision per turn. Laya does not write speech.

First question set:

| Id | Type | Meaning |
|---|---|---|
| `needs_task` | noul | Probability the other system should run |
| `kind` | choice | `chat`, `task`, or `function_call` |

Rules:

- `needs_task` below threshold → Talker only
- `needs_task` at or above threshold → Tasks POST; Talker also runs if `kind` is `chat` or the decision says both
- Default threshold is `0.5`, overridable in config

The Laya question JSON lives in one module. Change labels there, not in the session loop.

### Talker (LM Studio)

Config supplies base URL (example `http://127.0.0.1:1234/v1`) and model name.

Input is session instructions, thinking appends, and recent input/output transcripts.

Output is plain reply text.

If LM Studio fails, emit `error` with `internal_error` and skip that turn's speech. Do not kill the session.

Talker does not send the task body. Tasks send transcript and Laya labels as-is.

### Speaker (Supertonic)

Input text comes from Talker, from `session.commentary.append`, or from a task callback.

Voice maps from the Live voice name to a Supertonic style.

Output is PCM chunks at 24 kHz 16-bit mono for `session.output_audio.delta`.

Emit `session.output_transcript.delta` before the matching audio.

One utterance at a time per session. Barge-in drops the rest of the current utterance.

Supertonic native rate is 44.1 kHz. The adapter resamples.

### Tasks (async HTTP)

Outbound POST (do not wait for the work to finish):

```http
POST {TASKS_URL}
Content-Type: application/json

{
  "session_id": "sess_...",
  "delegation_id": "item_...",
  "transcript": "...",
  "kind": "task" | "function_call",
  "laya": { "needs_task": 0.91, "kind": "task" }
}
```

Then emit `session.delegation.created` with that `delegation_id` and `target: "client"`.

Inbound callback:

```http
POST /internal/delegations/{delegation_id}/result
Content-Type: application/json

{
  "delegation_id": "item_...",
  "content": "The order shipped today."
}
```

Unknown or closed `delegation_id` returns HTTP 404. Valid `content` is commentary: run Speaker, keep the session.

Polling is out of slice 1. Callback only.

## Errors

Envelope:

```json
{
  "type": "error",
  "event_id": "event_...",
  "error": {
    "type": "invalid_request_error",
    "code": "immutable_field_update",
    "message": "...",
    "param": "session.delegation.type",
    "client_event_id": "event_update"
  }
}
```

| Case | Code | Session |
|---|---|---|
| Unknown event type, bad JSON, missing field | `invalid_value` | stays up |
| Change of locked start field | `immutable_field_update` | stays up |
| Audio not even PCM16 bytes | `invalid_audio` | stays up |
| Command after close started | `session_closed` | ignore work |
| LM Studio / Whisper / Laya / TTS adapter fails | `internal_error` | stays up; skip that turn's speech |
| Callback for unknown id | HTTP 404; no Live `error` | — |

Set `client_event_id` only when the failure maps to one client `event_id`. Adapter failures omit it.

One failed model call does not end the session.

## Close

1. Client sends `session.close`, or the socket drops.
2. Stop new turns and new callbacks for new work. A callback already in flight may still speak if close has not finished.
3. Cancel Speaker. Drop remaining TTS.
4. Outbound tasks already POSTed keep running in the other system. This process does not cancel them.
5. Emit `session.closed` with final usage and `reason`, then close the socket.

| Reason | When |
|---|---|
| `close_requested` | Client sent `session.close` |
| `connection_lost` | Socket died before `session.close` |
| `expired` | Hit max duration (config; default 60 minutes) |

`content` (safety kill) and `remote_hangup` (SIP) are out of slice 1.

If the socket dies first, try to emit `session.closed` while the write path is alive. If that write fails, the client does not get final usage. The process still frees adapter handles.

## Usage

`session.usage.updated` about once per minute:

```json
{
  "type": "session.usage.updated",
  "event_id": "event_...",
  "usage": { "seconds": 12 },
  "context_window": { "usage_ratio": 0.0 }
}
```

`usage.seconds` is wall time of the Live session, cumulative. `context_window.usage_ratio` is `0.0` in slice 1.

`session.closed` repeats the final `usage.seconds`.

LM Studio and Laya token counts stay in those adapters. They are not in the Live usage object.

## Concurrency

- One VAD + ASR + Laya + Talker + Speaker chain per session
- Many sessions per process
- Task callbacks may arrive on any worker; they join the session Speaker queue by `delegation_id`
- One Speaker utterance at a time per session

## Config

Load from environment or a local file. Do not commit secrets.

- Live listen host, port, bearer token
- Paths for Silero and Faster-Whisper weights
- Laya checkpoint and device
- LM Studio base URL and model
- Supertonic voice map
- `TASKS_URL`
- VAD pause ms
- Laya `needs_task` threshold
- Max session duration (default 3600 s)

## Package layout (when we build)

Keep the current Python 3.14 package. Add modules under `src/open_gptlive_poc/`:

| Module | Role |
|---|---|
| `app.py` | Composition root: load config, wire adapters, serve |
| `live/protocol.py` | Event parse/serialize |
| `live/session.py` | Session loop |
| `live/http.py` | Callback route |
| `ports/` | Port types only |
| `adapters/silero_vad.py` | VAD |
| `adapters/faster_whisper.py` | ASR |
| `adapters/laya_router.py` | Router |
| `adapters/lmstudio_talker.py` | Talker |
| `adapters/supertonic_speaker.py` | Speaker |
| `adapters/http_tasks.py` | Tasks |

Dependencies need user approval before add. Expected later: a WebSocket server, ONNX runtime, `faster-whisper`, `supertonic`, Laya, Silero, HTTP client.

## Tests for slice 1

Use fake ports in unit tests. Do not require GPU or LM Studio for CI.

1. `session.start` → `session.started` with resolved config.
2. Pause → fake ASR text → Laya talk → output transcript and audio on the socket.
3. Laya task → outbound POST fired, `delegation.created` emitted, callback → more audio.
4. Barge-in: `speech_started` during TTS stops further `output_audio.delta`.
5. Bad JSON and immutable update emit `error` and leave the session up.
6. `session.close` → `session.closed` with `reason: close_requested` and `usage.seconds`.

## Out of slice 1

WebRTC SDP, SIP, sideband, fork, store, recording, Responses envelopes, polling the other system, full Live voice table, 128k context summarization, `gpt-realtime` (the older Realtime API).

## Later detail (explicitly deferred)

- Exact Laya question JSON and threshold tuning
- Faster-Whisper package and weight path
- Full Live voice → Supertonic style table
- LM Studio system prompt template
- Auth story beyond a single bearer token
- Whether the repo name stays `open-gptlive-poc` while the protocol is Live

## Key decisions

1. Speak the Live protocol, not the Realtime (`/v1/realtime`) protocol.
2. One process with ports, WebSocket first.
3. Laya only classifies. LM Studio writes chitchat. The other system writes task results. Supertonic only synthesizes.
4. Tasks are async. Callback only in slice 1.
5. Adapter failure skips the turn. It does not kill the session.
