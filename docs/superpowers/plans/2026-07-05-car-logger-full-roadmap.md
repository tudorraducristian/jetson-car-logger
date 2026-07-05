# Jetson Car Logger — Full Project Roadmap (Stages 1–5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement each stage plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the full self-contained car-detection & logging appliance described in `CLAUDE.md` — camera → detection → tracking → ANPR → SQLite → web dashboard — as five self-contained, independently testable stages, each with its own detailed plan.

**Architecture:** One `car_logger` FastAPI package grows layer by layer. Every stage attaches to the same `app` object created in Stage 1. Code is written on the laptop, flows to the Jetson via git (push → pull), and **all runtime verification happens on the Jetson** (Python 3.6.9). Stages are ordered so each one leaves a working, demoable system.

**Tech Stack:** FastAPI 0.67.0, uvicorn 0.15.0, pydantic 1.8.2, SQLAlchemy 1.3.24, alembic 1.7.7, httpx 0.22.0, jinja2 3.0.3, sse-starlette 0.10.3, structlog 21.5.0, pytest 7.0.1, `jetson.inference` (SSD-Mobilenet-v2), Tailwind + htmx via CDN, SQLite.

---

## Why this is split into five plans

This project spans several independent subsystems (persistence, real-time CV pipeline, external ANPR integration, web UI, deployment). Per the writing-plans scope rule, each is its own plan that produces working, testable software on its own. Execute them in order; do not start a stage until the previous stage's verification passes.

| Stage | Plan file | Builds | Key outcome |
|-------|-----------|--------|-------------|
| 1 | [`2026-07-04-stage1-hello-fastapi-car-logger.md`](2026-07-04-stage1-hello-fastapi-car-logger.md) | `car_logger/main.py`, `/` + `/health`, pinned deps | Web server reachable from the laptop **(DONE)** |
| 2 | [`2026-07-05-stage2-database-crud-api.md`](2026-07-05-stage2-database-crud-api.md) | config, DB, ORM, schemas, repositories, `/api/events`, Alembic | CRUD over SQLite works end-to-end |
| 3 | [`2026-07-05-stage3-camera-detection-pipeline.md`](2026-07-05-stage3-camera-detection-pipeline.md) | camera worker, detector, IoU tracker, pipeline, `/api/status` | Detections logged to DB, deduplicated |
| 4 | [`2026-07-05-stage4-anpr-dashboard.md`](2026-07-05-stage4-anpr-dashboard.md) | ANPR client + worker, crop storage, dashboard + partials | Plates read + displayed live |
| 5 | [`2026-07-05-stage5-systemd-sse-docs-demo.md`](2026-07-05-stage5-systemd-sse-docs-demo.md) | structlog, systemd unit, SSE, polish, README/docs | Appliance-ready + presentable |

> **On the "weeks":** these five stages map 1:1 onto the Week 1–5 sections of `PLAN.md`, but they are sequenced by **dependency and demoable outcome**, not by calendar. Pace them however you like — there are no calendar deadlines or mentor-checkpoint gates baked into the plans. The technical content of `PLAN.md`/`CLAUDE.md` (architecture, pinned stack, constraints, learning philosophy) still governs.

---

## Global Constraints (apply to EVERY stage)

Copied verbatim from `CLAUDE.md`. Every task in every stage plan implicitly includes these.

- **Target runtime is the Jetson: Python 3.6.9.** No Python 3.7+ syntax anywhere: no walrus (`:=`), no f-string `=` (`f"{x=}"`), no dict union (`|`), no `typing.Literal` (use `typing_extensions` if ever needed). Plain f-strings are fine (3.6 supports them).
- **No Pydantic v2 syntax.** `validator` not `field_validator`. `BaseSettings` from `pydantic` not `pydantic_settings`. Nested `class Config:` not `model_config`.
- **No SQLAlchemy 2.0 syntax.** Classic `Query` API: `db.query(Model).filter(...)`, not `db.execute(select(Model))`.
- **Sync `def` endpoints by default.** Only use `async def` where we specifically need it (SSE in Stage 5, httpx if ever async).
- **No state in code.** Everything persistable goes to SQLite via the repository layer. No module-level dicts or globals for data.
- **No secrets in git.** API keys live in `.env` (gitignored), loaded via `pydantic.BaseSettings`. `.env.example` holds placeholders only.
- **Memory-conscious.** 4GB shared CPU/GPU. Process frame-by-frame; never buffer video. If RAM > 3GB, investigate.
- **No new heavy infra.** No Redis/Celery/Docker/npm/webpack/modern-CV. Tailwind + htmx via CDN. `jetson.inference` is the entire CV layer.
- **Pinned versions from `requirements.txt`, verbatim.** Do not upgrade any package.

## Execution model (applies to every stage)

- **Split execution.** Steps marked **[LAPTOP — Claude]** are done by Claude in-session (write files, commit, push). Steps marked **[JETSON — student]** are commands the student runs over SSH: `ssh tudor@192.168.0.232`, repo at `~/jetson-car-logger`, venv activated with `source venv/bin/activate`. **Claude cannot run anything on the Jetson** — after each Jetson step the student pastes output back before the plan continues.
- **The laptop cannot run this code** (its Python can't install the 3.6 pins). Never pytest/run it on the laptop. All tests run on the Jetson.
- **Working method** (student's chosen mode): Claude writes the code and the tests; the student reads it, runs it on the Jetson, and confirms via the test output. The pure-logic modules (repositories, tracker, geometry, ANPR client) are written test-first so the student sees red → green; hardware/glue modules are verified green + live.
- **Business-logic decisions are the student's.** Tracker thresholds (Stage 3), ANPR retry policy (Stage 4), event-emission and retention rules — the plans ship sensible **documented defaults** and mark each as a decision to confirm or tune, not to accept blindly.
- **Commit style:** English messages, `type(scope): summary`, ending with the trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Do not touch** `experiments/lpr_batch/` or the docs unless a task says so.

## Layered architecture (target end state)

```
Camera worker ─frames→ Detector ─dets→ Tracker ─tracks→ [emit rule] ─event→ Event store (SQLite)
   (Stage 3)          (Stage 3)      (Stage 3)                              (Stage 2)
                                                    └─crop→ ANPR worker ──→ Plate Recognizer API
                                                              (Stage 4)          (external)
                                                                  │ plate text
                                                                  ▼
                                              FastAPI + Jinja + htmx  ←HTTP─ Browser on LAN
                                                 (Stage 1 app, Stage 4 UI, Stage 5 SSE)
```

## Definition of done (whole project)

1. Power on the Jetson → within 60s the dashboard is live on the LAN (Stage 5 systemd).
2. Webcam sees a car → event → plate → vehicle record (Stages 3–4).
3. Tests pass, logs are clean, no secrets in git.
4. The student can explain every architectural choice and every non-trivial code section.

---

## Per-stage verification summary (the gates between plans)

- **Stage 2 done:** `pytest tests/ -v` green (≥ 8 tests); Swagger shows the 3 event endpoints; POST→GET round-trips; `alembic downgrade base && alembic upgrade head` runs clean.
- **Stage 3 done:** `/api/events` shows deduplicated events with bboxes (plate null); `/api/status` reports FPS ≥ 10; RAM < 2.5GB; Ctrl+C stops threads cleanly; Stage 2 tests still green.
- **Stage 4 done:** printed plate → event with `plate_text`; dashboard at `/` auto-refreshes; internet-down test → events still created with `anpr_status="failed"`; reconnect → new events get plates.
- **Stage 5 done:** `systemctl status car-logger` active; survives `reboot` within 60s; dashboard updates via one SSE stream (no polling); 24h run without OOM/crash; README complete.
