# Jetson Car Logger

Self-contained car detection & logging appliance running entirely on an
NVIDIA Jetson Nano Developer Kit (original, 2019). USB webcam captures
video, on-device CV detects cars in real-time, an external OCR API reads
license plates, events are stored in SQLite, and a web dashboard accessible
from any browser on the LAN shows the live feed and history.

## What this project is

An **end-to-end application engineering** project. The student learns to
build a complete system: capture → process → persist → serve. Computer
vision is treated as a black box (pre-trained SSD-Mobilenet-v2) so the
focus stays on clean architecture, database design, API design, and
frontend delivery — skills transferable to any backend engineering role.

## What this project is NOT

- Not a computer vision research project (CV is a black box — v2 idea).
- Not a cloud app (everything runs on one device on the LAN).
- Not a commercial product (no auth, no multi-tenancy, GDPR-minimal).
- Not a microservices architecture (one process, one device, on purpose).

## Stack

- **Device:** NVIDIA Jetson Nano 2019, JetPack 4.6, Python 3.6
- **Backend:** FastAPI 0.67 + Pydantic v1 + SQLAlchemy 1.3
- **Database:** SQLite with Alembic migrations
- **Frontend:** Jinja2 templates + htmx + Tailwind CSS (via CDN, no build)
- **Live updates:** Server-Sent Events (SSE)
- **CV:** `jetson-inference` SSD-Mobilenet-v2 (pre-installed)
- **ANPR:** Plate Recognizer API (cloud, free tier)
- **Process management:** systemd
- **Observability:** structured JSON logs via structlog

## Status

| Week | Theme                               | Status      |
|------|-------------------------------------|-------------|
| 1    | Environment + Hello FastAPI         | Not started |
| 2    | Database + CRUD API                 | Not started |
| 3    | Camera + Detection pipeline         | Not started |
| 4    | ANPR + Dashboard                    | Not started |
| 5    | Systemd + SSE live updates          | Not started |
| 6    | Demo + Documentation                | Not started |

See `PLAN.md` for weekly tasks and verification criteria.
See `CLAUDE.md` for hardware constraints and stack decisions that Claude
Code must respect.

## Running (after Week 5)

```bash
# On the Jetson, after setup is complete:
sudo systemctl start car-logger
sudo journalctl -u car-logger -f     # see logs

# From any device on the LAN:
open http://jetson.local:8000
```

## License

TBD
