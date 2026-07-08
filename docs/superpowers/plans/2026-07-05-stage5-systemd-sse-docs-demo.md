# Stage 5 — Systemd + SSE + Docs + Demo (car_logger) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the working app into an appliance: structured logging, a systemd service that auto-starts on boot and restarts on failure, live dashboard updates via Server-Sent Events (replacing 2s polling), a couple of polish features the student picks, and complete documentation + a demo.

**Architecture:** SSE bridges the background pipeline thread to the browser through an `EventBroker`: worker threads call `broker.publish(...)` (thread-safe, via `loop.call_soon_threadsafe`); the async `/stream/events` endpoint (sse-starlette) subscribes an `asyncio.Queue` per client and streams change-signals. htmx's SSE extension listens and re-fetches the existing partial routes, so rendering stays server-side. systemd owns the process lifecycle; structlog emits JSON to stdout, which `journalctl` captures.

**Tech Stack:** structlog 21.5.0, sse-starlette 0.10.3, `asyncio`, htmx SSE extension (CDN), systemd unit + timer, pytest-asyncio 0.16.0.

## Global Constraints

- **Python 3.6.9 target.** No 3.7+ syntax.
- **`async def` allowed only for SSE** (the endpoint + generator). Everything else stays sync.
- **Thread → async bridge must be thread-safe:** publish from worker threads only via `loop.call_soon_threadsafe`. Never touch an `asyncio.Queue` directly from a non-loop thread.
- **No secrets in git.** The systemd unit references `.env` via the working directory; it contains no keys itself.
- **Graceful shutdown:** systemd `stop` must let FastAPI's shutdown handler stop the threads (SSE, pipeline, ANPR, camera).
- **Daily restart** for memory fragmentation (Known footgun) via a systemd timer — pragmatic, documented.
- **Split execution:** **[LAPTOP — Claude]** writes/commits/pushes; **[JETSON — student]** pulls, installs, runs. Paste output at each **CHECKPOINT**.
- **Commit trailer:** `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## File structure (what this stage creates)

- `car_logger/logging_config.py` — structlog JSON setup
- `car_logger/services/broker.py` — `EventBroker` (thread→async fan-out)
- `car_logger/api/routes_stream.py` — `GET /stream/events` (SSE)
- `car_logger/main.py` — build broker, publish on event create/update, call logging setup
- `car_logger/templates/dashboard.html` — swap polling for SSE
- `car_logger/api/routes_events.py` + `repositories.py` — `DELETE /api/events/{id}` + search (polish)
- `deployment/car-logger.service`, `deployment/car-logger-restart.service`, `deployment/car-logger-restart.timer`
- `scripts/install_service.sh`
- `README.md` (rewrite), `docs/architecture.md`
- `tests/unit/test_broker.py`

---

### Task 1: Structured logging (structlog)

**Files:**
- Create: `car_logger/logging_config.py`
- Modify: `car_logger/main.py` (call `configure_logging()` at import)

**Interfaces:**
- Produces: `configure_logging(level: str)` — sets up structlog to emit JSON to stdout; `get_logger(name)` re-exported for modules.

- [x] **Step 1: Write the logging config** **[LAPTOP — Claude]**

`car_logger/logging_config.py`:
```python
"""structlog setup: JSON logs to stdout so journalctl/systemd capture them."""

import logging
import sys

import structlog


def configure_logging(level="INFO"):
    logging.basicConfig(format="%(message)s", stream=sys.stdout,
                        level=getattr(logging, level.upper(), logging.INFO))
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name):
    return structlog.get_logger(name)
```

- [x] **Step 2: Call it from `main.py`** **[LAPTOP — Claude]**

At the top of `car_logger/main.py`, after imports and before `app = FastAPI(...)`:
```python
from car_logger.logging_config import configure_logging, get_logger

configure_logging(settings.log_level)
log = get_logger("car_logger")
```
Then add a log line in `_startup()` after the pipeline starts:
```python
    log.info("pipeline_started", target_fps=settings.max_pipeline_fps)
```
and in `_shutdown()`:
```python
    log.info("app_shutdown")
```

- [x] **Step 3: Commit and push** **[LAPTOP — Claude]** — `ff2de66`

```bash
git add car_logger/logging_config.py car_logger/main.py
git commit -m "feat(logging): structlog JSON output to stdout

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

- [x] **Step 4: Verify JSON logs appear** **[JETSON — student]**

```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -c "
from car_logger.logging_config import configure_logging, get_logger
configure_logging('INFO'); get_logger('t').info('hello', k=1)
"
```
Expected: a single JSON line containing `"event": "hello"`, `"k": 1`, `"level": "info"`, a timestamp.

**CHECKPOINT:** paste the JSON line before Task 2.

> ✅ 2026-07-07: JSON line verified on the Jetson (`"event": "hello", "k": 1, "level": "info"`).
> Deviation (deliberate): the resolved level feeds BOTH `basicConfig` and
> `make_filtering_bound_logger` — the plan hardcoded INFO in the wrapper, which
> would have made `LOG_LEVEL=DEBUG` a no-op for structlog.

---

### Task 2: SSE — event broker (test-first)

**Files:**
- Create: `car_logger/services/broker.py`
- Test: `tests/unit/test_broker.py`

**Interfaces:**
- Produces: `EventBroker` with:
  - `set_loop(loop)` — remember the serving event loop (called from the async endpoint).
  - `async subscribe() -> asyncio.Queue`
  - `unsubscribe(queue)`
  - `publish(data: str)` — thread-safe; schedules `data` onto every subscriber queue via `call_soon_threadsafe`. Consumed by `main.py` worker callbacks (Task 4) and `/stream/events` (Task 3).

- [x] **Step 1: Write the failing test** **[LAPTOP — Claude]**

`tests/unit/test_broker.py`:
```python
import asyncio

import pytest

from car_logger.services.broker import EventBroker


@pytest.mark.asyncio
async def test_publish_reaches_subscriber():
    broker = EventBroker()
    broker.set_loop(asyncio.get_event_loop())
    queue = await broker.subscribe()
    broker.publish("changed")
    data = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert data == "changed"


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery():
    broker = EventBroker()
    broker.set_loop(asyncio.get_event_loop())
    queue = await broker.subscribe()
    broker.unsubscribe(queue)
    broker.publish("changed")
    await asyncio.sleep(0.05)
    assert queue.empty()


def test_publish_without_loop_is_noop():
    # publish before any subscriber/loop must not raise
    EventBroker().publish("changed")
```

- [x] **Step 2: Commit, push, confirm RED** **[LAPTOP — Claude then JETSON — student]** — `6efaf8f`, RED = collection error (module missing)

```bash
git add tests/unit/test_broker.py
git commit -m "test(broker): failing tests for thread-safe SSE fan-out

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
Then **[JETSON — student]**:
```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -m pytest tests/unit/test_broker.py -v
```
Expected: FAIL — module missing. *(If pytest-asyncio complains it needs a marker mode, add `asyncio_mode = auto` under `[tool:pytest]` in a `pytest.ini`; the async tests use `@pytest.mark.asyncio` which 0.16 supports.)*

- [x] **Step 3: Implement the broker** **[LAPTOP — Claude]**

`car_logger/services/broker.py`:
```python
"""Thread-to-async fan-out for SSE.

Worker threads call publish(); the async SSE endpoint subscribes a queue per
client. publish() is safe from any thread because it hands work to the event
loop via call_soon_threadsafe — asyncio.Queue is NOT thread-safe otherwise."""

import asyncio


class EventBroker(object):
    def __init__(self):
        self._subscribers = set()
        self._loop = None

    def set_loop(self, loop):
        self._loop = loop

    async def subscribe(self):
        queue = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue):
        self._subscribers.discard(queue)

    def publish(self, data):
        """Schedule `data` onto every subscriber queue. No-op if no loop yet."""
        loop = self._loop
        if loop is None:
            return
        for queue in list(self._subscribers):
            loop.call_soon_threadsafe(queue.put_nowait, data)
```

- [x] **Step 4: Commit, push, confirm GREEN** **[LAPTOP — Claude then JETSON — student]** — `dfdeb4d`

```bash
git add car_logger/services/broker.py
git commit -m "feat(broker): thread-safe EventBroker for SSE

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
Then **[JETSON — student]**:
```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -m pytest tests/unit/test_broker.py -v
```
Expected: `3 passed`.

**CHECKPOINT:** paste output before Task 3.

> ✅ 2026-07-07: `3 passed, 1 warning` on the Jetson.

---

### Task 3: SSE endpoint

**Files:**
- Create: `car_logger/api/routes_stream.py`
- Modify: `car_logger/main.py` (create broker on startup, include stream router)

**Interfaces:**
- Consumes: `app.state.broker`.
- Produces: `GET /stream/events` → `text/event-stream`; emits a `new_event` SSE on each publish and a `heartbeat` every 30s of silence.

- [x] **Step 1: Write the SSE router** **[LAPTOP — Claude]**

`car_logger/api/routes_stream.py`:
```python
"""GET /stream/events — Server-Sent Events. One-way server->browser stream.

We send lightweight change-signals ("new_event"); htmx re-fetches the partials
on receipt, so HTML rendering stays server-side (Task 5 template change)."""

import asyncio

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

router = APIRouter(tags=["stream"])

HEARTBEAT_SECONDS = 30


@router.get("/stream/events")
async def stream_events(request: Request):
    broker = request.app.state.broker
    broker.set_loop(asyncio.get_event_loop())
    queue = await broker.subscribe()

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(
                        queue.get(), timeout=HEARTBEAT_SECONDS
                    )
                except asyncio.TimeoutError:
                    # keep the connection alive + let the client detect death
                    yield {"event": "heartbeat", "data": "ping"}
                    continue
                yield {"event": "new_event", "data": data}
        finally:
            broker.unsubscribe(queue)

    return EventSourceResponse(event_generator())
```

- [x] **Step 2: Create the broker on startup + include the router** **[LAPTOP — Claude]** — deviation: `app.state.broker` is set BEFORE the `enable_pipeline` early-return, so SSE works on camera-less runs/tests

In `car_logger/main.py`:
- add `from car_logger.api.routes_stream import router as stream_router` and `from car_logger.services.broker import EventBroker`
- after the other `include_router` calls: `app.include_router(stream_router)`
- at module level: `broker = EventBroker()` and stash it in startup: `app.state.broker = broker`
- bump `APP_VERSION = "0.5.0"`

- [x] **Step 3: Commit and push** **[LAPTOP — Claude]** — `9540f70`

```bash
git add car_logger/api/routes_stream.py car_logger/main.py
git commit -m "feat(sse): /stream/events endpoint backed by EventBroker

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

- [x] **Step 4: Live-verify the stream** **[JETSON — student]**

Run the server, then in another terminal:
```bash
curl -N http://192.168.0.232:8000/stream/events
```
Expected: the connection stays open; within 30s a `event: heartbeat` / `data: ping` line appears. Leave it while showing a car to the camera — expect `event: new_event` lines when events are created (after Task 4 wiring). Ctrl+C to stop curl.

**CHECKPOINT:** paste the first few SSE lines before Task 4.

> ✅ 2026-07-07: connection stays open; our `heartbeat` at 30s of silence PLUS
> sse-starlette's built-in `event: ping` every 15s (library default — redundant
> keep-alives, harmless; htmx only listens for `new_event`). Jetson has no curl
> by default — installed via apt (wget -qO- also works).

---

### Task 4: Publish on writes + swap dashboard polling for SSE

**Files:**
- Modify: `car_logger/main.py` (publish after event create + ANPR update)
- Modify: `car_logger/templates/dashboard.html` (htmx SSE)

**Interfaces:**
- The `on_confirmed` closure publishes `"created"` after inserting the pending event; `on_result` publishes `"updated"` after the ANPR update. The dashboard connects to `/stream/events` and refreshes the feed/stats on `sse:new_event`.

- [x] **Step 1: Publish from the worker callbacks** **[LAPTOP — Claude]** — single `broker.publish("updated")` after the if/else (covers both branches)

In `car_logger/main.py`, inside the `on_confirmed` closure after the event is created:
```python
        app.state.broker.publish("created")
```
and inside `on_result` (in `_make_on_result`) after the DB update in both branches:
```python
        broker.publish("updated")
```
> `_make_on_result` must capture `broker` — change its signature to `_make_on_result(broker)` and call it as `_make_on_result(app.state.broker)` in `_startup` (set `app.state.broker = broker` first).

- [x] **Step 2: Swap polling for SSE in the dashboard** **[LAPTOP — Claude]** — DEVIATION: kept the existing editorial template (event-detail drawer, theme); only swapped `every Ns` triggers for `sse:new_event` + added the search box. sse.js pinned **1.9.12** to match the existing htmx pin, not 1.9.10.

`car_logger/templates/dashboard.html`:
```html
{% extends "base.html" %}
{% block content %}
<div hx-ext="sse" sse-connect="/stream/events"
     class="grid grid-cols-1 lg:grid-cols-3 gap-6">
  <section class="lg:col-span-2">
    <div class="flex items-center justify-between mb-3">
      <h2 class="text-lg font-medium">Live events</h2>
      <input type="search" name="q" placeholder="search plate…"
             class="bg-neutral-900 border border-neutral-800 rounded px-2 py-1 text-sm"
             hx-get="/partials/events-feed" hx-target="#events-feed"
             hx-trigger="input changed delay:300ms" hx-include="this">
    </div>
    <div id="events-feed"
         hx-get="/partials/events-feed"
         hx-trigger="load, sse:new_event"
         hx-swap="innerHTML">Loading…</div>
  </section>
  <aside class="space-y-6">
    <div>
      <h2 class="text-lg font-medium mb-3">Stats</h2>
      <div id="stats" hx-get="/partials/stats"
           hx-trigger="load, sse:new_event" hx-swap="innerHTML">…</div>
    </div>
    <div>
      <h2 class="text-lg font-medium mb-3">Vehicles</h2>
      <div id="vehicles" hx-get="/partials/vehicles-list"
           hx-trigger="load, sse:new_event" hx-swap="innerHTML">…</div>
    </div>
  </aside>
</div>
{% endblock %}
```
And add the SSE extension to `base.html` `<head>` (after the htmx script):
```html
  <script src="https://unpkg.com/htmx.org@1.9.10/dist/ext/sse.js"></script>
```

- [x] **Step 3: Make the feed route honour the search query** **[LAPTOP — Claude]** — feed limit stays 15 (earlier student decision), not 25

In `car_logger/api/routes_dashboard.py`, update `events_feed` to accept `q`:
```python
@router.get("/partials/events-feed", response_class=HTMLResponse)
def events_feed(request: Request, q: str = "", db: Session = Depends(get_db)):
    events = repositories.list_events(db, limit=25, plate_text=(q or None))
    return templates.TemplateResponse(
        "partials/events_feed.html", {"request": request, "events": events}
    )
```

- [x] **Step 4: Commit and push** **[LAPTOP — Claude]** — `3a1e0e7`

```bash
git add car_logger/main.py car_logger/templates/dashboard.html car_logger/templates/base.html car_logger/api/routes_dashboard.py
git commit -m "feat(sse): publish on writes, dashboard live-updates via SSE + search

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

- [x] **Step 5: Verify live updates + no polling** **[JETSON — student]**

Run the server, open `/` on the laptop, open DevTools → Network. Expected: exactly **one** long-lived `event-stream` connection to `/stream/events`, and **no** repeating `/partials/*` requests until an SSE `new_event` arrives (then a single burst of partial fetches). Show a car → feed/stats update within a second. Type in the search box → feed filters.

**CHECKPOINT:** confirm the single EventStream + event-driven refresh before Task 5.

> ✅ 2026-07-08 ~09:53: checkpoint complete. (a) no polling — one `events-feed`
> in 45s where 2s-polling would have shown ~20; (b) `events-feed?q=mmm` fired
> once, feed filtered; (c) via `~/e2e_fake_cam.py` (own full app instance —
> must STOP the real server first, it binds :8000; student hit Errno 98 twice
> before stopping it): EventSource died (server stop), 5 retries at ~2s
> (ERR_CONNECTION_REFUSED), auto-reconnected to the fake instance, then TWO
> SSE-triggered feed fetches (`created` + `updated`) — event #12 `mmm8748`
> appeared without refresh, stats 11→12, plates 5→6. 1 credit used (~94 left).
> Noted quirk (possible Task 5 polish): SSE-triggered feed re-fetch drops the
> active `?q=` filter — box still shows the query, feed shows everything.

---

### Task 5: Polish — delete event (student picks the rest)

**Files:**
- Modify: `car_logger/repositories.py`, `car_logger/api/routes_events.py`, `car_logger/templates/partials/events_feed.html`
- Test: append to `tests/integration/test_api_events.py`

**Interfaces:**
- Produces: `repositories.delete_event(db, event_id) -> bool`; `DELETE /api/events/{id}` → 204 or 404; a delete button per feed row.

> **STUDENT DECISION:** PLAN lists several polish features (plate search ✓ done in Task 4, "mark vehicle as known" notes edit, 24h/7d stats toggle, delete). Pick which to ship by time. Delete is implemented here as one concrete example; the others follow the same route+repo+template pattern.
>
> ✅ 2026-07-08: student picked a second polish after spotting that the feed
> showed 06:53 for a 09:53 event — timestamps are stored UTC (`utcnow()`) but
> were rendered raw. Added a `localtime` Jinja filter (routes_dashboard.py):
> `replace(tzinfo=utc).astimezone()` — the OS timezone (set via `timedatectl`)
> decides the display zone, DST handled by the system, zero new deps. Applied
> in events_feed, event_detail (label "Moment (UTC)" → "Moment"), and
> vehicles_list; invariant-tested in tests/unit/test_template_filters.py.

- [x] **Step 1: Write the failing test** **[LAPTOP — Claude]** — `d4ad7dc`

Append to `tests/integration/test_api_events.py`:
```python
def test_delete_event(client):
    created = client.post("/api/events", json={"plate_text": "DEL123"}).json()
    resp = client.delete("/api/events/" + str(created["id"]))
    assert resp.status_code == 204
    assert client.get("/api/events/" + str(created["id"])).status_code == 404


def test_delete_missing_event_is_404(client):
    assert client.delete("/api/events/9999").status_code == 404
```

- [x] **Step 2: Commit, push, confirm RED** **[LAPTOP — Claude then JETSON — student]** — RED confirmed 2026-07-08: `2 failed, 6 passed`

```bash
git add tests/integration/test_api_events.py
git commit -m "test(api): failing tests for delete event

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
Then **[JETSON — student]**: `git pull && python3 -m pytest tests/integration/test_api_events.py -v` → new tests FAIL.

- [x] **Step 3: Implement delete** **[LAPTOP — Claude]**

> DEVIATIONS 2026-07-08 (all deliberate):
> 1. htmx 1.x never swaps a 204 response (`shouldSwap` excludes it), so the
>    planned `hx-target="closest li"` button would have left the row on screen
>    and the stats stale. Instead the DELETE route publishes `"deleted"` on the
>    broker — the SSE round-trip refreshes feed+stats+vehicles, consistent with
>    Task 4's "publish on writes". The button has no hx-target at all.
> 2. `app.state.broker` moved from `_startup()` to module level in `main.py`:
>    the TestClient fixture never fires startup events, and the delete route
>    needs the broker during tests (publish with no loop is a documented no-op).
> 3. The feed row is a full-width `<button>` (detail drawer) — nesting the
>    delete button inside it is invalid HTML. The `<li>` became a flex row with
>    the delete button as a sibling; `hover:text-rose-400` matches the "eșuat"
>    badge palette; `hx-confirm` text is Romanian like the rest of the UI.

Append to `car_logger/repositories.py`:
```python
def delete_event(db, event_id):
    """Delete the event. Returns True if it existed, False otherwise."""
    event = db.query(Event).filter(Event.id == event_id).first()
    if event is None:
        return False
    db.delete(event)
    db.commit()
    return True
```
Append to `car_logger/api/routes_events.py` (add `from fastapi import Response`):
```python
@router.delete("/{event_id}", status_code=204)
def delete_event(event_id: int, db: Session = Depends(get_db)):
    if not repositories.delete_event(db, event_id):
        raise HTTPException(status_code=404, detail="Event not found")
    return Response(status_code=204)
```
Add a delete button in `car_logger/templates/partials/events_feed.html` inside each `<li>`:
```html
    <button class="text-xs text-red-400 hover:text-red-300"
            hx-delete="/api/events/{{ e.id }}"
            hx-confirm="Delete this event?"
            hx-target="closest li" hx-swap="outerHTML">✕</button>
```

- [x] **Step 4: Commit, push, confirm GREEN** **[LAPTOP — Claude then JETSON — student]** — `7c43ee7` (delete), `a7f43cb` (localtime polish)

```bash
git add car_logger/repositories.py car_logger/api/routes_events.py car_logger/templates/partials/events_feed.html
git commit -m "feat(events): DELETE /api/events/{id} + feed delete button

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
Then **[JETSON — student]**: `git pull && python3 -m pytest tests/ -v` → full suite green.

**CHECKPOINT:** paste the full suite result before Task 6.

> ✅ 2026-07-08 ~10:12: **53 passed** (full suite, Jetson). Live verify:
> DELETE seen in DevTools as a 204 followed by the SSE-triggered partial
> refresh; rows removed without reload, stats 12→9; timestamps now display
> Romania time (09:53:42 where UTC said 06:53:42). Task 5 done.

---

### Task 6: systemd service (auto-start, restart, daily refresh)

**Files:**
- Create: `deployment/car-logger.service`, `deployment/car-logger-restart.service`, `deployment/car-logger-restart.timer`
- Create: `scripts/install_service.sh`

**Interfaces:**
- Produces: an installed, enabled `car-logger` systemd service + a daily-restart timer.

- [x] **Step 1: Write the unit files** **[LAPTOP — Claude]** — deviation: added `Environment=PYTHONUNBUFFERED=1` (stdout to journald is a pipe; Python would block-buffer the JSON logs)

`deployment/car-logger.service`:
```ini
[Unit]
Description=Car Logger appliance (FastAPI + CV pipeline)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=tudor
WorkingDirectory=/home/tudor/jetson-car-logger
ExecStartPre=/home/tudor/jetson-car-logger/venv/bin/alembic upgrade head
ExecStart=/home/tudor/jetson-car-logger/venv/bin/uvicorn car_logger.main:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5
Environment=PATH=/home/tudor/jetson-car-logger/venv/bin:/usr/bin:/bin

[Install]
WantedBy=multi-user.target
```

`deployment/car-logger-restart.service`:
```ini
[Unit]
Description=Daily restart of car-logger (memory-fragmentation mitigation)

[Service]
Type=oneshot
ExecStart=/bin/systemctl restart car-logger.service
```

`deployment/car-logger-restart.timer`:
```ini
[Unit]
Description=Restart car-logger every day at 04:00

[Timer]
OnCalendar=*-*-* 04:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

- [x] **Step 2: Write the install script** **[LAPTOP — Claude]**

`scripts/install_service.sh`:
```bash
#!/usr/bin/env bash
# Install and enable the car-logger systemd service + daily-restart timer.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/../deployment" && pwd)"
sudo cp "$SRC/car-logger.service" /etc/systemd/system/
sudo cp "$SRC/car-logger-restart.service" /etc/systemd/system/
sudo cp "$SRC/car-logger-restart.timer" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now car-logger.service
sudo systemctl enable --now car-logger-restart.timer
sudo systemctl status car-logger.service --no-pager
```

- [ ] **Step 3: Commit and push** **[LAPTOP — Claude]**

```bash
git add deployment/ scripts/install_service.sh
git commit -m "feat(deploy): systemd unit, daily-restart timer, install script

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

- [ ] **Step 4: Install and verify** **[JETSON — student]**

```bash
cd ~/jetson-car-logger && git pull
chmod +x scripts/install_service.sh
./scripts/install_service.sh
```
Expected: `Active: active (running)`. Then confirm the dashboard is reachable at `http://192.168.0.232:8000/` **without** a manually-run uvicorn.

- [ ] **Step 5: Reboot + restart-on-failure tests** **[JETSON — student]**

```bash
sudo reboot
# wait ~60s, then from the laptop:
curl http://192.168.0.232:8000/health
# back on the Jetson, prove restart-on-failure — NOTE the -9: plain pkill
# sends SIGTERM, uvicorn exits CLEANLY (code 0) and Restart=on-failure
# would rightly NOT restart it. A crash is SIGKILL:
sudo pkill -9 -f uvicorn ; sleep 10 ; systemctl status car-logger --no-pager | head
journalctl -u car-logger -n 20 --no-pager
```
Expected: after reboot the dashboard answers within 60s; after `pkill`, systemd restarts it within ~10s; `journalctl` shows the JSON logs.

**CHECKPOINT:** paste `systemctl status` + the post-reboot `curl /health` before Task 7.

---

### Task 7: README + architecture doc + final cleanup

**Files:**
- Rewrite: `README.md`
- Create: `docs/architecture.md`
- Modify: various (remove debug logging, ensure docstrings, update `.env.example`)

**Interfaces:** documentation only.

- [ ] **Step 1: Draft the README** **[LAPTOP — Claude]**

Rewrite `README.md` covering: what it does; the layered architecture (with the ASCII diagram from the roadmap); hardware requirements; install from scratch (clone → venv `--system-site-packages` → `pip install -r requirements.txt` → `.env` → `alembic upgrade head` → `scripts/install_service.sh`); configuration (`.env` fields); dev-run vs production (uvicorn `--reload` vs systemd); known limitations; future work (CV v2). Keep placeholders out — write the real content.

> **Student owns the voice.** Per PLAN.md 5.6, the student rewrites anything that doesn't read as their own understanding. Claude drafts; the student edits.

- [ ] **Step 2: Write the architecture doc** **[LAPTOP — Claude]**

`docs/architecture.md` (1–2 pages): why threading (camera blocks; pipeline must not stall the API); why SQLite (single-file, zero-setup, enough for LAN scope); why the repository layer (testable, reusable from the pipeline); why htmx over React (no build step, server-rendered, one language); why SSE over polling/WebSocket (one-way, simpler, live). This is primarily the student's understanding — Claude reviews for clarity only.

- [ ] **Step 3: Final cleanup pass** **[LAPTOP — Claude]**

- Remove any debug/learning logging added in earlier stages (e.g. session create/close prints from PLAN 2.2).
- Ensure every public function has a docstring.
- Update `.env.example` to include every settings field used (`DATABASE_URL`, `ANPR_API_KEY`, `ANPR_API_URL`, `LOG_LEVEL`, `MAX_PIPELINE_FPS`, `DETECTOR_THRESHOLD`, `CAMERA_INDEX`, `ENABLE_PIPELINE`).
- Grep for `TODO`/`FIXME`/commented-out code and remove.

- [ ] **Step 4: Commit and push** **[LAPTOP — Claude]**

```bash
git add README.md docs/architecture.md .env.example car_logger
git commit -m "docs: README + architecture doc; final cleanup

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

- [ ] **Step 5: Full suite + 24h soak + tag** **[JETSON — student]**

```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -m pytest tests/ -v          # all green
# leave the service running ~24h, then check:
systemctl status car-logger --no-pager | head
free -h                              # RAM not climbing toward OOM
git tag v1.0 && git push --tags
```
Expected: all tests green; after 24h no crash/OOM/frozen UI; tag pushed.

- [ ] **Step 6: Demo video** **[JETSON/laptop — student, manual]**

2–3 min screen recording per PLAN 5.7: overview → live demo (webcam + dashboard) → `systemctl status`/`journalctl`/`tegrastats` proving on-device → code tour of the layers.

**CHECKPOINT:** confirm suite green, soak survived, tag pushed, video recorded. Stage 5 — and the project — is done.

---

## Self-Review

**1. Spec coverage** (against `PLAN.md` Week 5):
- structlog JSON to stdout, key events logged: Task 1. ✓
- systemd unit + install script + `Restart=on-failure` + reboot test + restart-on-failure test: Task 6. ✓
- SSE endpoint via sse-starlette, asyncio queue, heartbeat every 30s: Tasks 2–3. ✓
- htmx SSE replacing polling, single EventStream verified: Task 4. ✓
- Polish (search ✓, delete ✓; notes/toggle noted as student's pick): Tasks 4–5. ✓
- README (what/arch/hardware/install/config/dev-vs-prod/limits/future): Task 7. ✓
- architecture.md (why threading/SQLite/htmx/SSE): Task 7. ✓
- Final cleanup (remove debug logging, docstrings, `.env.example`, no TODO): Task 7 Step 3. ✓
- 24h soak, `v1.0` tag, demo video: Task 7 Steps 5–6. ✓

**2. Placeholder scan:** every code/config step is complete. README/architecture prose is delegated to the drafting step with an explicit content list (documentation, not code) — no code placeholders. ✓

**3. Type consistency:** `EventBroker` methods (`set_loop`/`subscribe`/`unsubscribe`/`publish`) identical across impl, tests, and the SSE endpoint. `broker.publish("created"/"updated")` strings are opaque signals the dashboard reacts to via `sse:new_event` — no schema coupling. `_make_on_result(broker)` signature updated consistently with its call site. `delete_event` repo/route/test names match. ✓

## Notes for the executor

- **SSE threading is the subtle part:** `publish` must only ever reach the `asyncio.Queue` through `loop.call_soon_threadsafe`. If events never arrive in the browser but heartbeats do, the loop wasn't captured — check `set_loop` runs in the async endpoint. Per CLAUDE.md, chase this yourself; Claude explains.
- The daily-restart timer is a pragmatic mitigation for the documented memory-fragmentation footgun, not a fix — note it as such in the README.
- `network-online.target` in the unit avoids the service starting before the LAN is up (dashboard unreachable on boot otherwise).
- Keep the student's voice in README/architecture — those docs are where a reviewer checks genuine understanding.
