# PLAN.md — Jetson Car Logger

> **Scope:** Self-contained car detection & logging appliance on Jetson Nano.
> Full stack: camera capture → detection → ANPR via API → database → web dashboard.
> **Duration:** 4-5 weeks part-time (assisted coding accelerates ~25% vs manual).
> **Student level:** beginner (first real project).
> **Focus:** end-to-end application engineering. CV is a black box.

## How this plan uses assisted coding

**Default mode: assisted.** Claude Code writes the boilerplate. The student
directs, reviews, modifies, and tests. Hand-writing every line as a ritual
is artificial — speed and understanding are not in conflict if the ritual
in CLAUDE.md (read → flag unclear → modify → explain → test) is followed.

**Exception: debugging.** When code breaks, Claude Code explains and asks
clarifying questions but does NOT produce the fix. The student writes the
fix. This is where intuition is built and cannot be shortcut.

**Exception: core business logic.** Decisions that define what the
product does (deduplication rules, confidence thresholds, event policies)
are the student's to make. Claude Code can list options; the student
picks.

## Weekly roadmap

| Week | Focus                                           | Key outcome                      |
|------|-------------------------------------------------|----------------------------------|
| 1    | Environment + "hello FastAPI"                   | Web server reachable from laptop |
| 2    | Database + API endpoints (no CV yet)            | CRUD works end-to-end            |
| 3    | Camera capture + detection (CV black box)       | Detections logged to DB          |
| 4    | ANPR integration + dashboard                    | Plates read + displayed live     |
| 5    | Systemd + live updates + docs + demo            | Appliance-ready + presentable    |

5 weeks target; 6 weeks if setbacks. Week 6 is buffer/polish if needed.

---

## Week 1 — Environment & "hello FastAPI"

**Goal:** a FastAPI app running on the Jetson, reachable from a browser on
the laptop. No CV, no database. Just "hello world" over HTTP.

### Concepts to learn (ask Claude Code to explain these BEFORE generating code)

- What is a virtual environment? Why do we need one?
- What is SSH? Why do we use keys instead of passwords?
- What is a web framework? What does "request/response cycle" mean?
- What is an HTTP method (GET, POST)? What is a URL path?
- What is the difference between a web **server** and a web **application**?
- What is `uvicorn` and why do we need it separately from FastAPI?

### Tasks — with assistance level hints

**1.1 — Jetson baseline setup** *(mostly manual — these are shell commands)*
- `cat /etc/nv_tegra_release` shows `R32` with revision 7.x. If not,
  flash the correct SD card image first.
- Switch to headless mode:
  `sudo systemctl set-default multi-user.target && sudo reboot`.
- Find Jetson IP with a monitor briefly (`hostname -I`), or from router.
- **Ask Claude Code** for the swap file setup commands (4GB swapfile).
  Student runs them, verifies with `swapon --show`.
- **Ask Claude Code** for a systemd service that enables the fan on boot.
  Generate it, install it, verify.
- Power via barrel jack with jumper J48.

**1.2 — Laptop setup** *(manual — tooling installation)*
- VS Code + "Remote - SSH" extension.
- Claude Code installed.
- `ssh-keygen -t ed25519 -C "laptop@<yourname>"`.
- `ssh-copy-id <user>@<jetson-ip>`.
- Test passwordless SSH.
- Connect via VS Code Remote-SSH.

**1.3 — Create the repo** *(manual — git basics the student should type)*
- Empty repo on GitHub: `jetson-car-logger`.
- Clone on Jetson.

**1.4 — Python environment** *(Claude Code assists with commands)*
- Create venv **with system packages** (needed for `jetson.inference` later):
  ```
  python3 -m venv --system-site-packages venv
  source venv/bin/activate
  pip install --upgrade pip
  ```
- **Ask Claude Code** to produce the pinned `requirements.txt` from the
  stack in CLAUDE.md. Student reads it, commits it, runs `pip install -r`.
- If any package fails: student debugs with Claude Code **asking
  questions**, not getting the answer directly. E.g., "the package X
  failed to install with error Y. What's likely the issue?"

**1.5 — Hello FastAPI** *(Claude Code generates, student applies the ritual)*
- **Ask Claude Code:** "Generate a minimal `car_logger/main.py` with
  FastAPI: a root endpoint returning a JSON greeting and a `/health`
  endpoint. Use sync `def` handlers, not async."
- Apply the 5-step ritual:
  1. Read every line out loud.
  2. Ask Claude Code: "Why do we import `FastAPI` from `fastapi`? What's
     the difference between `app = FastAPI()` and calling a function?"
  3. Modify: change the greeting message, add a `version` field to the
     response.
  4. Explain to yourself: what happens when a request hits `/`?
  5. Write a test in `tests/test_main.py` that uses `TestClient` to
     verify both endpoints return 200 and the expected JSON. Generate
     test scaffolding with Claude Code; student writes the assertions.

- Run: `uvicorn car_logger.main:app --host 0.0.0.0 --port 8000`.
- From laptop: `http://<jetson-ip>:8000` → sees JSON.
- Try `http://<jetson-ip>:8000/docs` → Swagger UI. Click every endpoint.

**1.6 — First commit** *(manual — git hygiene)*
- `.gitignore`: `venv/`, `__pycache__/`, `*.pyc`, `*.db`, `.env`, `data/`.
- Minimal README.
- `git commit -m "week1: fastapi hello world"` — message written by
  student, not generated.

### Verification

- [ ] `ssh <user>@<jetson>` works from laptop without password prompt.
- [ ] VS Code Remote-SSH can open files on the Jetson.
- [ ] `free -h` on Jetson shows ≥ 4GB swap.
- [ ] `uvicorn car_logger.main:app --host 0.0.0.0` runs without errors.
- [ ] Laptop browser reaches `http://<jetson-ip>:8000`.
- [ ] `http://<jetson-ip>:8000/docs` shows the endpoints.
- [ ] `pytest tests/` passes with at least 2 tests.
- [ ] GitHub shows the initial commit.
- [ ] **Student can explain** what `uvicorn` does vs what FastAPI does.

### Mentor checkpoint

None formal this week. 10-minute async message to Radu with a screenshot
of Swagger UI.

---

## Week 2 — Database + API (no CV yet)

**Goal:** a full CRUD API over SQLite. Create events, list events, get a
specific event. No camera, no detection — the "backend" in isolation so
the student learns it without CV complexity.

### Concepts to learn (ask Claude Code to explain BEFORE generating)

- What is a **database**? What is a **schema**? What is a **primary key**?
- What is an **ORM**? Why use SQLAlchemy instead of raw SQL?
- What is **dependency injection** in FastAPI (`Depends()`)?
- What is a **migration**? Why use Alembic instead of `CREATE TABLE` in code?
- What is a **repository pattern**? Why separate DB code from API code?
- What are **Pydantic models** and why are they different from SQLAlchemy models?

**These concepts are the core of the week.** Do not skip to code before
understanding them. 30-60 minutes of reading/asking > 3 hours of debugging
confused code.

### Tasks — with assistance level hints

**2.1 — Configuration** *(Claude Code generates, student reviews)*
- **Ask Claude Code:** "Generate `car_logger/config.py` using
  `pydantic.BaseSettings` (v1 syntax) that loads from a `.env` file.
  Include fields for database_url, anpr_api_key, anpr_api_url, log_level."
- Apply ritual. Modify: add a field for `max_pipeline_fps` with a sane
  default. Write a test that loads settings from a temporary .env.

**2.2 — Database setup** *(Claude Code generates, student MUST understand
the threading detail)*
- **Ask Claude Code:** "Generate `car_logger/database.py` with SQLAlchemy
  engine, SessionLocal, declarative Base, and a get_db() generator for
  FastAPI dependency injection. Use SQLite."
- **CRITICAL:** before accepting the code, ask: "Why does
  `connect_args={'check_same_thread': False}` appear here? What would go
  wrong without it?" Student must be able to explain before moving on.
- Modify: add logging that prints when a session is created/closed (for
  learning — remove before Week 5).

**2.3 — ORM models** *(student directs, Claude Code writes)*
- **Student's decision:** look at the data model in CLAUDE.md. Decide
  which fields are nullable, which have defaults, what indexes to add.
- **Ask Claude Code:** "Generate SQLAlchemy ORM models for Vehicle and
  Event per this schema: [paste the schema]. Add indexes on
  `events.timestamp` and `vehicles.plate_text`."
- Apply ritual. Modify: add a `__repr__` to each model for debugging.

**2.4 — Pydantic schemas** *(Claude Code generates)*
- **Ask Claude Code:** "Generate Pydantic v1 schemas for EventCreate,
  EventRead, VehicleRead. Use `orm_mode = True` in Config."
- Apply ritual. **Key question to ask Claude Code:** "Why do we have
  separate schemas from the ORM models?" Student should be able to
  explain the API/persistence boundary.

**2.5 — Repositories** *(Claude Code generates scaffolding, student adds
edge cases)*
- **Ask Claude Code:** "Generate `car_logger/repositories.py` with
  functions: create_event, get_event, list_events (with pagination and
  optional plate_text filter)."
- Apply ritual. **Student adds** (hand-written, deliberate):
  - What should `get_event` return if the ID doesn't exist? None? Raise?
    Decide, then implement.
  - What's the max `limit` for `list_events`? Cap it. The student picks
    the number.

**2.6 — Alembic migrations** *(manual — tooling, worth learning by typing)*
- `alembic init alembic` (student types).
- **Ask Claude Code:** "How do I configure alembic/env.py to use my
  SQLAlchemy models?" Apply the change manually.
- `alembic revision --autogenerate -m "initial schema"`.
- Review the generated migration file — does it look right? Ask Claude
  Code to explain any line that's unclear.
- `alembic upgrade head`. Verify with `sqlite3 car_logger.db ".schema"`.

**2.7 — API routes** *(Claude Code generates)*
- **Ask Claude Code:** "Generate `car_logger/api/routes_events.py` with
  endpoints POST /api/events, GET /api/events (with pagination and plate
  filter query params), GET /api/events/{id}. Use the repositories from
  step 2.5 via Depends."
- Apply ritual. Modify: add a `limit` query param max of 100.
- Wire into `main.py` with `app.include_router`.

**2.8 — Tests** *(student writes assertions, Claude Code scaffolds)*
- **Ask Claude Code:** "Generate a conftest.py with a fixture that gives
  me an in-memory SQLite session for tests."
- **Student writes assertions** for: create event, read event by ID,
  list events returns empty when DB is empty, list events respects limit,
  plate filter matches partial text.
- Test tally target: ≥ 8 tests for this week's code.

### Verification

- [ ] `pytest tests/ -v` passes — ≥ 8 tests across repository + API.
- [ ] Swagger UI shows the 3 event endpoints.
- [ ] From Swagger UI: POST an event → 200 with ID → GET it back.
- [ ] `sqlite3 car_logger.db "SELECT * FROM events;"` shows the row.
- [ ] `alembic downgrade base && alembic upgrade head` runs clean.
- [ ] **Student can explain** to Radu in 5 minutes: what is dependency
  injection, why have separate Pydantic schemas from ORM models, what
  does the repository pattern buy us.
- [ ] Commit: `git commit -m "week2: database + crud api"`.

### Mentor checkpoint (formal)

15-30 minute call with Radu. Student demos:
1. Swagger UI, creates and retrieves an event live.
2. Shows the tests running.
3. **Explains the architecture in their own words**, walking through
   the layers (API → schema → repository → ORM → DB).

If the explanation is hand-wavy, means Claude Code did the thinking.
Radu directs the student back to the concepts before Week 3.

---

## Week 3 — Camera + detection (CV as a black box)

**Goal:** the Jetson captures video from the USB webcam, runs detection,
creates events in the database for each car detected. No ANPR yet.

### Concepts to learn (explain before generating)

- What is a **thread**? Why do we need one for the camera?
- What is a **producer/consumer pattern**?
- Why can't we process video inside an API request handler?
- What is a **detection** (bbox + confidence + class)? What is **tracking**?
- Why do we need tracking? (Answer: to avoid creating 30 events per car.)
- What is a **race condition**? What is a **lock**?

### Tasks — assistance level hints

**3.1 — Camera capture worker** *(Claude Code generates, student debugs)*
- **Ask Claude Code:** "Generate `car_logger/services/capture.py` with a
  CameraWorker class that runs `cv2.VideoCapture(0)` in a background
  thread and exposes a thread-safe `get_latest_frame()` method."
- Apply ritual. **Key question to ask:** "How is thread safety achieved
  here? What happens if get_latest_frame is called at the exact moment
  the worker is writing a new frame?"
- Student modifies: add start/stop methods, make the thread a daemon.

**3.2 — Detector wrapper** *(Claude Code generates)*
- **Ask Claude Code:** "Generate `car_logger/services/detector.py` that
  wraps `jetson.inference.detectNet('ssd-mobilenet-v2', threshold=0.5)`.
  Method `detect(frame_bgr)` returns a list of Detection named tuples
  with x1, y1, x2, y2, confidence, class_id. Filter to COCO classes
  car=3, motorcycle=4, bus=6, truck=8."
- Apply ritual. Modify: make the threshold configurable via settings.

**3.3 — IoU tracker** *(STUDENT-LED — this is core business logic)*
- **Concepts first.** Ask Claude Code to explain IoU, bounding boxes,
  track birth/death, greedy matching. Student reads carefully.
- **Student decides the rules:**
  - IoU threshold for match: start with 0.3, document why.
  - Frames missed before track death: start with 5, document why.
  - Minimum frames before emitting an event: start with 5, document why.
- **Ask Claude Code:** "I want a simple IoU tracker with these rules:
  [list rules]. Generate `car_logger/services/tracker.py`."
- Apply ritual. **Student writes the tests** — Claude Code scaffolds,
  student defines the scenarios:
  - Two overlapping bboxes in consecutive frames → same track_id.
  - Track with no matches for 5 frames → removed.
  - Two non-overlapping boxes → two tracks.
  - What about when bboxes cross (occlusion)? Student thinks about it,
    writes a test documenting current behavior.

**3.4 — Pipeline orchestration** *(Claude Code generates skeleton,
student owns the event emission rule)*
- **Ask Claude Code:** "Generate `car_logger/services/pipeline.py` —
  a PipelineWorker that runs in a thread, pulls frames from
  CameraWorker, runs Detector, runs Tracker, and calls a callback for
  events to persist."
- Apply ritual.
- **Student writes** the event emission logic:
  - When to emit an event? (Confirmed tracks after N frames.)
  - How to avoid duplicate events per track? (track_id → last_emitted_at.)
  - This is the product logic. Don't let Claude Code decide it alone.

**3.5 — Wire it up** *(Claude Code generates startup/shutdown code)*
- **Ask Claude Code:** "Generate FastAPI startup/shutdown event handlers
  that start CameraWorker and PipelineWorker on app start, stop them on
  shutdown. The pipeline's event callback should persist to DB via the
  repository."
- Apply ritual. Key debugging skill moment: the student will likely hit
  threading + SQLite issues here. **Claude Code explains, student fixes.**
- Add `GET /api/status` returning pipeline FPS, camera health, last
  event timestamp. Student writes this endpoint with minimal Claude Code
  help (by now it's a repeatable pattern).

### Verification

- [ ] With webcam pointed at any scene, `GET /api/events` returns events
  with bboxes populated (plate_text null).
- [ ] Moving a phone-displayed car image past the camera creates a small
  number of events (not 30/sec — dedup works).
- [ ] `GET /api/status` returns pipeline FPS.
- [ ] Pipeline FPS ≥ 10 on the Jetson.
- [ ] RAM stays < 2.5 GB (`tegrastats`).
- [ ] Ctrl+C on uvicorn stops the camera thread cleanly (no zombie
  processes, no "device busy" on restart).
- [ ] All Week 2 tests still pass.
- [ ] Tracker tests: ≥ 4 scenarios.
- [ ] Commit: `git commit -m "week3: camera + detection pipeline"`.

### Mentor checkpoint (formal)

30-minute call. Student demos live + **draws the architecture on paper**
(threads, queues, DB writes). If the drawing is wrong, the student
doesn't understand the threading model — go back and fix.

---

## Week 4 — ANPR + dashboard

**Goal:** plates read via Plate Recognizer API; web dashboard shows events
and vehicles.

### Concepts to learn (explain before generating)

- What is a **REST API**? What is a **POST request with a file**?
- What are **HTTP status codes**? What does 429 mean?
- What is **rate limiting**? What is **exponential backoff**?
- What is a **template engine**? What does Jinja2 do?
- What is **htmx**? How does it differ from React?
- What do `hx-get`, `hx-post`, `hx-swap`, `hx-trigger` do?
- What is a **template partial** and why do we use them with htmx?

### Tasks — assistance level hints

**4.1 — Plate Recognizer account** *(manual)*
- Sign up at platerecognizer.com (free: 2500 calls/month).
- API key to `.env`: `ANPR_API_KEY=...`.
- Manual verification with `curl` and a sample image.

**4.2 — ANPR client** *(Claude Code generates, student owns retry policy)*
- **Concepts first:** ask Claude Code what retry strategies exist
  (immediate, linear backoff, exponential backoff, circuit breaker).
  Discuss which makes sense for ANPR (hint: exponential for 5xx, no
  retry for 429 or 4xx).
- **Student decides:**
  - Timeout per request: 3 seconds? 5?
  - Max retries for 5xx: 2? 3?
  - What to do on 429: log and skip, don't retry.
- **Ask Claude Code:** "Generate `car_logger/services/anpr_client.py`
  using httpx with these rules: [list rules]. Return a dataclass
  PlateResult."
- Apply ritual.
- **Tests:** student writes scenarios — Claude Code helps with httpx
  mocking. Scenarios:
  - 200 → PlateResult with plate_text populated.
  - Timeout → retry once, then fail.
  - 429 → fail immediately, status="throttled".
  - 500 → retry per policy.
  - Network error → fail gracefully.

**4.3 — Integrate ANPR into pipeline** *(Claude Code generates, student
verifies non-blocking)*
- **Critical concept:** the ANPR call takes 300-800ms. If the pipeline
  blocks on it, FPS drops to ~2. It MUST be async or run in a separate
  worker thread.
- **Ask Claude Code:** "Integrate ANPR calls into the pipeline such that
  detection FPS is not affected. Use a separate worker with a queue."
- Apply ritual. Student **verifies the non-blocking claim** by:
  - Adding a fake `time.sleep(1)` in the ANPR client.
  - Confirming pipeline FPS stays the same.
  - Removing the sleep.

**4.4 — Crop storage** *(Claude Code generates)*
- **Ask Claude Code:** "After the ANPR worker gets a result, save the
  cropped plate image to `data/plates/<event_id>.jpg`. Update the event
  in DB with plate_text, anpr_status, image_path."
- Apply ritual. Modify: add a cleanup function that deletes crops older
  than 30 days (student decides the retention; 30 days is a reasonable
  default for a personal project).

**4.5 — Dashboard template** *(Claude Code generates, UI UX Pro Max skill
if available)*
- **Invoke Radu's UI UX Pro Max skill** if configured. Otherwise, ask
  Claude Code directly:
  "Generate `car_logger/templates/base.html` and `dashboard.html` using
  Tailwind CDN + htmx CDN. Dark theme, editorial style, Playfair Display
  for headlines, DM Mono for monospaced content. Three panels: live
  events feed, recent vehicles, stats summary."
- Apply ritual on the HTML — yes, read it line by line. htmx attributes
  especially. If `hx-get` or `hx-trigger` or `hx-swap` is unclear, ask.

**4.6 — Dashboard routes** *(Claude Code generates)*
- **Ask Claude Code:** "Generate `car_logger/api/routes_dashboard.py`:
  GET / renders dashboard.html. GET /partials/events-feed renders the
  events feed fragment. Same for vehicles-list and stats."
- Apply ritual. Student modifies: add a `GET /partials/event/{id}` for
  the event detail drawer.

### Verification

- [ ] With webcam seeing a printed plate (or car photo on a screen), an
  event appears in DB with plate_text populated.
- [ ] `http://<jetson-ip>:8000/` shows the dashboard.
- [ ] Events feed auto-refreshes every 2 seconds (DevTools shows
  periodic htmx requests).
- [ ] **Network-down test:** disconnect Jetson from internet.
  Events still created, plates empty, anpr_status="failed".
  Reconnect: new events get plates.
- [ ] Mobile browser shows the dashboard correctly.
- [ ] Full stack RAM < 2.5 GB.
- [ ] Pipeline FPS ≥ 8 with ANPR integrated.
- [ ] All tests green.
- [ ] Commit: `git commit -m "week4: anpr + dashboard"`.

### Mentor checkpoint (CRITICAL)

**Live demo with Radu over screen share.** This is the "product works"
milestone. If the demo fails, **do not move to Week 5**.

Student shows:
1. Dashboard loaded.
2. Car photo held to webcam or car driving past.
3. Event appears. Plate gets read. Vehicle appears.
4. **Student explains**: why is the ANPR call async? what happens if
   the internet dies? how does htmx know to refresh?

---

## Week 5 — Systemd + SSE + docs + demo (combined final week)

**Goal:** appliance-ready deployment + live updates via SSE + complete
documentation + demo video. This is intentionally packed — assisted
coding makes it feasible.

### Concepts to learn

- What is **systemd**? What is a unit file?
- What are **Server-Sent Events (SSE)**? How do they differ from polling
  and WebSockets?
- What is **graceful shutdown**?
- What is **structured logging**?

### Tasks

**5.1 — Structured logging** *(Claude Code generates)*
- **Ask Claude Code:** "Configure structlog to output JSON logs to stdout.
  Log: pipeline start/stop, per-minute detection count, ANPR outcomes,
  errors with traceback."
- Apply ritual.

**5.2 — Systemd service** *(Claude Code generates, student installs)*
- **Ask Claude Code:** "Generate `deployment/car-logger.service` systemd
  unit file, plus a `scripts/install_service.sh` to install and enable it."
- Student reads the unit file line by line (it's short — ~15 lines). Key
  question: "what does `Restart=on-failure` do?"
- Install, enable, start, verify with `systemctl status`.
- **Reboot test:** `sudo reboot`. Verify service auto-starts and
  dashboard is reachable within 60 seconds.

**5.3 — SSE endpoint** *(Claude Code generates)*
- **Concepts first:** SSE vs WebSocket vs polling. Why SSE is enough
  here (one-way server-to-client, simpler).
- **Ask Claude Code:** "Generate `car_logger/api/routes_stream.py` with
  a GET /stream/events that returns text/event-stream. In the pipeline,
  push new events to an asyncio queue. The SSE endpoint reads from the
  queue and streams to connected clients. Use sse-starlette."
- Apply ritual. Modify: send a heartbeat every 30s to detect dead
  connections.

**5.4 — htmx SSE integration** *(Claude Code generates)*
- **Ask Claude Code:** "Modify dashboard.html to replace the 2-second
  polling with htmx SSE. Include the htmx-sse extension via CDN."
- Apply ritual. Verify in browser DevTools: one long-lived EventStream
  connection, no more periodic polling.

**5.5 — Polish features** *(Claude Code generates, student directs)*
- Search bar for plates (`?q=B12` filters events).
- "Mark vehicle as known" — edit notes field.
- 24h vs 7d stats toggle.
- Delete event button (with confirmation).
- **Student picks** which of these to include based on time. Not all are
  required.

**5.6 — README + architecture doc** *(Claude Code drafts, student edits)*
- **Ask Claude Code:** "Draft a README for this project covering: what
  it does, architecture (with an ASCII diagram), hardware requirements,
  installation from scratch, configuration, running in dev vs production,
  known limitations, future improvements."
- Student reads every section, **rewrites anything that doesn't feel
  accurate or authentic in their own voice**. The README is where the
  student's understanding shows — Claude Code can draft, but the final
  voice is the student's.
- `docs/architecture.md`: 1-2 pages. Why threading? Why SQLite? Why
  htmx? Why SSE? Student writes this primarily hand-written — these
  are their understanding. Claude Code only reviews for clarity.

**5.7 — Demo video** *(manual — student records)*
- 2-3 minutes. Screen recording + voice.
- 30s: project overview.
- 60s: live demo (webcam + dashboard).
- 30s: `systemctl status`, `journalctl`, `tegrastats` — prove it's on
  the Jetson.
- 30s: code tour in VS Code, highlight the layers.

**5.8 — Final cleanup**
- `pytest tests/` all green.
- No TODO, FIXME, commented-out code.
- All public functions have docstrings.
- `.env.example` up to date.
- Remove the debug logging added in Week 2.

### Verification

- [ ] `sudo systemctl status car-logger` shows active (running).
- [ ] After `sudo reboot`, dashboard reachable within 60 seconds.
- [ ] Killing uvicorn (`pkill uvicorn`) → systemd restarts within 10s.
- [ ] Dashboard updates in real-time via SSE (DevTools: one EventStream).
- [ ] Search by partial plate works.
- [ ] 24-hour continuous run: no crashes, no OOM, no frozen UI.
- [ ] README is complete; classmate can follow it to install on another
  Jetson (if available) or at least understand it in 5 minutes.
- [ ] Architecture doc explains design choices in the student's voice.
- [ ] Demo video recorded and uploaded.
- [ ] `git tag v1.0 && git push --tags`.
- [ ] **Student can explain the full system** to Radu in a 15-minute
  review without Claude Code open.

### Final mentor checkpoint

60-minute review with Radu:
1. Live demo from the Jetson.
2. Student walks Radu through the codebase, explaining each file.
3. Radu picks 3 random non-trivial functions and asks the student to
   explain what they do and why. If the student can't, Claude Code wrote
   something that shouldn't have shipped — identify and either rewrite
   or remove before calling done.
4. Discuss v2 ideas (custom CV, multi-camera, parking domain).

---

## Week 6 — Buffer / stretch goals (optional)

If Week 5 ran long, this week absorbs the slip. If Week 5 finished on
schedule, pick ONE of these:

- **Multi-camera stub:** add a second USB camera, show both feeds. Just
  to prove the architecture supports it; don't polish.
- **Export CSV:** `GET /api/events/export?from=X&to=Y` → CSV download.
  Good REST skill extension.
- **Docker image for deployment:** NOT for development, but as a v2
  prep. Package the app in a Dockerfile. Note the memory overhead.
- **Start the CV v2 research:** student reads YOLOv5 + TensorRT
  export docs to prepare for the follow-up project.

Do NOT add complexity that touches the core pipeline. This is polish,
not redesign.

---

## What is explicitly OUT of scope (repeating for clarity)

- Training a custom car/plate detector (v2 project).
- Multi-camera beyond a stub.
- Cloud sync beyond the Plate Recognizer API.
- User authentication (single-user LAN appliance).
- Replacing Plate Recognizer with self-hosted OCR (v2 project).
- Redis, Celery, RabbitMQ, Kafka, microservices, Docker for production,
  Kubernetes. Resist. The Jetson has 4GB RAM.

## Definition of "done" (repeating)

At the end:
1. Plug in Jetson, power on → 60s → dashboard live on LAN.
2. Webcam sees a car → event → plate → vehicle record.
3. Student can explain every architectural choice to a peer.
4. Student can explain every non-trivial code section to Radu.
5. Tests pass, logs are clean, no secrets in git.
6. Repo is public with a complete README, demo video linked.
