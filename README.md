# Jetson Car Logger

A self-contained car detection & logging appliance running entirely on an
NVIDIA Jetson Nano Developer Kit (original, 2019). A USB webcam watches the
street; the device detects cars on-device in real time, reads their license
plates through the Plate Recognizer API, stores every sighting in SQLite,
and serves a live web dashboard to any browser on the LAN.

**The device is the product.** Power it on and it works: systemd starts the
app at boot, restarts it on crashes, and the dashboard updates itself over
Server-Sent Events — no page refresh, no laptop, no cloud (except the OCR
call).

## How it works

```
┌─────────────────────────────────────────────────────────────────┐
│                        JETSON NANO (all on-device)               │
│                                                                  │
│   ┌──────────┐   frames    ┌──────────────┐                     │
│   │  Camera  │────────────>│  Detector    │                     │
│   │  worker  │             │  (SSD-MNet)  │                     │
│   └──────────┘             └──────┬───────┘                     │
│                                   │ detections                   │
│                                   v                              │
│                            ┌──────────────┐                     │
│                            │   Tracker    │  (simple IoU)       │
│                            └──────┬───────┘                     │
│                                   │ tracks                       │
│                                   v                              │
│                            ┌──────────────┐   crop + POST       │
│                            │  ANPR stage  │─────────────────────┼──>  Plate Recognizer
│                            │  (worker)    │<────────────────────┼──   API (external)
│                            └──────┬───────┘   plate text         │
│                                   │                              │
│                                   v                              │
│                            ┌──────────────┐                     │
│                            │  Event store │  (SQLAlchemy)       │
│                            │   (SQLite)   │                     │
│                            └──────┬───────┘                     │
│                                   │                              │
│                                   v                              │
│                            ┌──────────────┐    HTTP + SSE        │
│                            │   FastAPI    │<─────────────────────┼──── Browser on LAN
│                            │  + Jinja +   │    (laptop, phone)  │
│                            │    htmx      │                      │
│                            └──────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

The layers, in one paragraph each:

- **Capture** (`services/capture.py`) — a background thread owns the USB
  camera (opening it blocks for seconds, so the API server never touches
  it) and always keeps just the latest frame. No buffering: 4 GB of RAM
  are shared with the GPU.
- **Detect** (`services/detector.py`) — pre-trained SSD-Mobilenet-v2 via
  `jetson-inference`, GPU-accelerated. Deliberately a black box: this
  project is about everything *around* the model.
- **Track** (`services/tracker.py`) — a small IoU tracker turns per-frame
  detections into stable tracks, so one passing car becomes one event, not
  thirty.
- **Identify** (`services/anpr_client.py` + `anpr_worker.py`) — the plate
  crop goes to the Plate Recognizer API from a separate worker thread with
  a bounded queue; the pipeline never waits for the network.
- **Persist** (`models.py`, `repositories.py`) — SQLite + SQLAlchemy with
  Alembic migrations. Two tables: `events` (every sighting) and `vehicles`
  (every unique plate). All queries live in the repository layer.
- **Serve** (`api/`, `templates/`) — FastAPI + Jinja2 + htmx, Tailwind via
  CDN. The dashboard is server-rendered; live updates arrive as SSE
  change-signals (`/stream/events`) that make htmx re-fetch fragments.

See [docs/architecture.md](docs/architecture.md) for why each of these
choices was made.

## Hardware requirements

- NVIDIA Jetson Nano Developer Kit (original, 2019, 4 GB), JetPack 4.6.x
  (Ubuntu 18.04, Python 3.6.9, CUDA 10.2)
- USB webcam
- **Barrel-jack 5V/4A power supply with the J48 jumper set.** Micro-USB
  power browns out under GPU load — this is a hard requirement, not a
  recommendation.
- A Plate Recognizer account (free tier is enough)

## Install from scratch (on the Jetson)

```bash
git clone https://github.com/tudorraducristian/jetson-car-logger.git
cd jetson-car-logger

# --system-site-packages: jetson.inference and cv2 are built by JetPack
# system-wide; a clean venv would not see them.
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install -r requirements.txt    # pinned, Python 3.6-compatible versions

cp .env.example .env               # then edit: set ANPR_API_KEY
alembic upgrade head               # create the database schema

./scripts/install_service.sh       # install + enable systemd service & timer
```

From then on the appliance is autonomous: it starts at boot, restarts on
failure, and restarts daily at 04:00. Open `http://<jetson-ip>:8000/` from
any device on the LAN.

## Configuration

Everything configurable lives in `.env` (gitignored — secrets never reach
the repo). Fields and defaults:

| Variable             | Default                          | Meaning                                    |
|----------------------|----------------------------------|--------------------------------------------|
| `DATABASE_URL`       | `sqlite:///./car_logger.db`      | SQLAlchemy URL; SQLite file in the repo dir |
| `ANPR_API_KEY`       | *(empty — required)*             | Plate Recognizer API token                 |
| `ANPR_API_URL`       | Plate Recognizer v1 endpoint     | Swap for a mock in testing                 |
| `LOG_LEVEL`          | `INFO`                           | `DEBUG` for troubleshooting                |
| `MAX_PIPELINE_FPS`   | `15`                             | Detection loop ceiling (thermal headroom)  |
| `DETECTOR_THRESHOLD` | `0.5`                            | Min detection confidence                   |
| `CAMERA_INDEX`       | `0`                              | `/dev/video*` index of the webcam          |
| `ENABLE_PIPELINE`    | `true`                           | `false` = API/dashboard only, no camera    |

## Development vs production

**Production (the appliance):** systemd owns the process.

```bash
systemctl status car-logger            # health
journalctl -u car-logger -f            # live JSON logs
sudo systemctl restart car-logger      # manual restart
```

**Development on the Jetson:** stop the service first — it holds port 8000.

```bash
sudo systemctl stop car-logger
source venv/bin/activate
uvicorn car_logger.main:app --host 0.0.0.0 --port 8000
```

**Development off-device (any machine):** set `ENABLE_PIPELINE=false` in
`.env` — the API, dashboard, and tests run without camera or CUDA.

Deploying a change: `git push` from the dev machine, then on the Jetson
`git pull && sudo systemctl restart car-logger` (migrations run
automatically via `ExecStartPre`).

## HTTP surface

| Route                       | What it is                                        |
|-----------------------------|---------------------------------------------------|
| `GET /`                     | The dashboard (server-rendered, htmx-driven)      |
| `GET /partials/*`           | HTML fragments the dashboard panels re-fetch      |
| `GET /stream/events`        | SSE stream of change-signals (+ 30s heartbeats)   |
| `GET /api/events`           | JSON list, `?plate=` filter, `skip`/`limit` pages |
| `GET /api/events/{id}`      | One event                                         |
| `POST /api/events`          | Create (tests/tools; the pipeline writes directly)|
| `DELETE /api/events/{id}`   | Delete; dashboards refresh via SSE                |
| `GET /api/status`           | Pipeline health (fps, frames, camera)             |
| `GET /health`               | Liveness probe                                    |

Interactive OpenAPI docs at `/docs`.

## Testing

```bash
python3 -m pytest tests/ -v
```

- `tests/unit/` — tracker geometry and deduplication, the SSE broker's
  thread-safety, the ANPR client against a mocked httpx, template filters.
- `tests/integration/` — the HTTP API against an in-memory SQLite via
  FastAPI's TestClient; no camera or network needed.
- The full pipeline is verified live on the device (it needs the camera,
  the GPU, and daylight).

## Performance on this hardware

| Component                            | Measured / expected |
|--------------------------------------|---------------------|
| Detection (SSD-Mobilenet-v2)         | 20-25 FPS           |
| Full pipeline (detect + track + DB)  | 12-22 FPS           |
| ANPR API round-trip                  | 300-800 ms          |
| Dashboard page load                  | < 100 ms            |
| SSE update after a detection         | ~1 s end-to-end     |
| RAM, full stack                      | 1.5-2.2 GB / 4 GB   |

## Known limitations

- **Daily 04:00 restart** is a pragmatic mitigation for slow memory
  fragmentation in a long-running CPython process on this device — a
  documented workaround, not a fix.
- **Plate reads need cooperation from physics:** daylight, a reasonable
  camera angle, and a real car (photos of screens don't detect well; a
  steep top-down angle defeats both detector and OCR).
- **No authentication** — a single-user LAN appliance by design. Do not
  port-forward it to the internet.
- **Python 3.6 pins everything.** The stack (FastAPI 0.67, Pydantic v1,
  SQLAlchemy 1.3) is the last generation that runs on JetPack 4.6's
  Python. Upgrading any of it requires testing on the device first.
- The Plate Recognizer free tier is ~2500 lookups/month; the tracker's
  deduplication is what makes that budget survivable.

## Future work (v2)

- Replace the black-box detector with a custom-trained model (the CV
  project this one deliberately postponed).
- Self-hosted OCR to cut the external dependency.
- Editable notes per vehicle, stats time-window toggles.
- If it ever leaves the LAN: HTTPS, auth, and a real threat model.
