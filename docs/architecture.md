# Architecture decisions

Why this system looks the way it does. The README says *what* the pieces
are; this document says *why* each one won over its alternatives.

## Why threads (and not async, and not a single loop)

Two facts force the design. First, `cv2.VideoCapture(0)` blocks for
seconds when it opens the camera and every `read()` after that is
blocking I/O. Second, the web server must answer requests the whole time.
Putting camera, detection, and ANPR each in their own thread means a slow
frame or a slow network call never freezes the dashboard.

Threads beat asyncio here because the CV work is *blocking by nature*
(OpenCV and TensorRT don't speak async). Wrapping blocking calls in an
event loop just hides them badly. So: sync threads for the pipeline, sync
`def` endpoints for the API, and exactly one island of async — SSE — where
it is genuinely needed.

The capture thread also enforces the memory rule: it keeps only the
*latest* frame (no queue of frames). On a 4 GB board shared with the GPU,
buffering video is how you meet the OOM killer.

## Why the pipeline hands events to the browser through a broker

A worker thread cannot touch an `asyncio.Queue` — those are not
thread-safe. `EventBroker` is the one bridge: `publish()` may be called
from any thread because it only schedules work onto the event loop via
`loop.call_soon_threadsafe(...)`; each connected browser has its own queue
subscribed by the async SSE endpoint. One writer rule, enforced in one
place, instead of locks sprinkled everywhere.

## Why SQLite

Single file, zero setup, ships with Python, survives power cuts well
enough for this scope, and comfortably handles a few events per minute
with one reader page open. A server database (PostgreSQL, MySQL) would add
a service to install, monitor, and migrate on a 4 GB board — for no
feature this appliance needs. The one SQLite quirk that matters here:
connections refuse cross-thread use by default, so the engine sets
`check_same_thread=False`; safety comes from never sharing a *session*
across threads (each thread opens its own from `SessionLocal`).

## Why a repository layer

All SQL-touching code lives in `repositories.py`. Three reasons:

1. **Two very different callers.** The HTTP routes and the pipeline
   threads both write events. Without a repository, query logic would be
   duplicated in both — and drift.
2. **Testable without HTTP.** The unit tests exercise queries against an
   in-memory SQLite directly; the integration tests go through FastAPI.
3. **One place to enforce rules** like the hard `MAX_LIST_LIMIT` cap, so
   no caller can ask for the whole table.

## Why htmx over a JS framework

The dashboard is read-mostly, single-user, on a LAN. Server-side rendering
with Jinja2 keeps all logic in one language and one process; htmx adds the
interactivity actually needed (fragment swaps, a search box, delete
buttons) via HTML attributes. No node, no npm, no build step, nothing to
compile on an ARM board. A React/Vue SPA would double the codebase and the
toolchain to deliver the same six panels.

## Why SSE and not polling or WebSockets

The dashboard originally polled every 2 s: ~1800 requests per browser per
hour, almost all answering "nothing changed". Server-Sent Events invert
it: the server pushes a tiny change-signal the moment an event is created,
updated, or deleted, and htmx re-fetches the affected fragments — so
rendering stays server-side and the payload stays HTML.

SSE won over WebSockets because this stream is strictly one-way
(server→browser), and SSE is plain HTTP: no protocol upgrade, works
through the same port, and the browser's `EventSource` **auto-reconnects
for free** — verified live when the process was SIGKILLed and every open
dashboard re-attached itself after systemd restarted the app. A heartbeat
every 30 s keeps intermediaries from silently dropping idle connections.

The signals carry no data ("created"/"updated"/"deleted" only). Dashboards
re-fetch fragments instead of patching DOM from a payload — one rendering
path instead of two, at the cost of an extra HTTP round-trip on the LAN.

## Time: store UTC, display local

Timestamps are written with `datetime.utcnow()` and converted only at the
template edge (a `localtime` Jinja filter using the OS timezone). UTC in
storage is unambiguous and sorts correctly across DST changes — during the
autumn fall-back, 03:30 local time happens twice, but UTC never repeats.
The display zone is whatever `timedatectl` says, so the same code shows
correct local time wherever the box lives.

## Deployment: systemd as the ops layer

The app does not daemonize, write pidfiles, or rotate logs — systemd does
all of it. The unit runs `alembic upgrade head` as `ExecStartPre` (schema
migrations become part of process start), `Restart=on-failure` resurrects
crashes (SIGKILL-tested), and stdout goes to journald as one JSON line per
event (`PYTHONUNBUFFERED=1`, or Python would buffer the pipe). A daily
04:00 timer restarts the service as a documented mitigation for
long-running-process memory fragmentation on this board. Logs are queried
with `journalctl -u car-logger`, structured fields intact.

## Testing strategy

Unit tests cover the logic that can be wrong in subtle ways: IoU geometry,
tracker deduplication rules, the broker's thread-to-async handoff, ANPR
response parsing against a mocked httpx. Integration tests run the real
FastAPI app against an in-memory SQLite (`StaticPool`, one shared
connection) with the `get_db` dependency overridden — real routing, real
serialization, no disk, no network. What cannot be simulated —
camera, CUDA, sunlight — is verified live on the device at explicit
checkpoints instead of pretended in mocks.
