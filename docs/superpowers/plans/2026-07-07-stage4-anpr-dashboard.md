# Stage 4 — ANPR + Dashboard (execution record)

> Executed 2026-07-07 directly from PLAN.md Week 4, on `main`
> (commits `d6ab491`, `32c7c4d`, `759d3c7`). This doc records the
> decisions, the pieces built, and the verification status.

## Student decisions (recorded)

| Decision | Value | Why |
|---|---|---|
| ANPR request timeout | 5 s | tolerates home Wi-Fi jitter; caller is a background worker, nothing blocks |
| Retries for 5xx/timeout/network | 2, exponential backoff 0.5s → 1s | transient server trouble; worst case ~12s/event |
| 429 handling | no retry, mark `throttled` | quota error is about us; retrying burns more quota |
| Crop retention | 30 days | plan default, sane for a personal project |
| Queue full | mark event `skipped` | shed load instead of growing memory (4GB budget) |
| 200 with no plate found | mark `failed`, keep crop | the crop on disk is the debugging evidence |

## Built

- [x] `services/anpr_client.py` — Plate Recognizer client, injectable
      transport, never raises (9 unit tests, httpx.MockTransport)
- [x] `services/anpr_worker.py` — bounded queue (32) + daemon thread;
      saves crops to `data/plates/<event_id>.jpg`; `cleanup_old_crops()`
      at startup (6 unit tests)
- [x] `services/pipeline.py` — emits the car crop with each event;
      degenerate bbox → event born `skipped`
- [x] `repositories.py` — `update_event_anpr` (upserts Vehicle, links
      event, bumps sightings), `list_vehicles`, `get_stats` (5 new tests)
- [x] Dashboard at `/` — dark editorial (Playfair Display + DM Mono,
      Tailwind CDN, htmx CDN): events feed (2s polling), vehicles,
      stats, detail drawer (8 integration tests)
- [x] `/api/status` reports `anpr_queue` depth
- [x] Crops served at `/data/plates/<event_id>.jpg` (StaticFiles mount)

## Verification — measured on the Jetson (2026-07-07)

- [x] `pytest tests/` — **51 passed** (one test-thread race found and
      fixed in `759d3c7`; the worker itself was correct)
- [x] Dashboard reachable over LAN from the laptop — HTTP 200, partials render
- [x] Pipeline FPS with ANPR integrated: **18 fps** (target ≥ 8)
- [x] RAM: uvicorn RSS **1.45 GB** (target < 2.5 GB)
- [ ] **ANPR key** (task 4.1, student): platerecognizer.com account,
      `ANPR_API_KEY` into `.env` on the Jetson — until then plates = `skipped`
- [ ] Live plate read: webcam sees a **printed** plate/car photo
      (screens don't detect — known constraint), event gets `plate_text`
- [ ] Non-blocking proof (student): `time.sleep(1)` in the ANPR client,
      confirm FPS unchanged, remove it
- [ ] Network-down test: unplug internet → events keep coming,
      status `failed`/`skipped`; reconnect → new events get plates
- [ ] Events feed refresh visible in DevTools (htmx request every 2s)
- [ ] Dashboard on a phone browser
- [ ] Mentor checkpoint: live demo with Radu (the "product works" milestone)

## Notes for the review with Radu

Questions to be ready for: why is the ANPR call on a separate thread with
a queue (not inline, not asyncio)? what happens when the internet dies?
how does htmx know to refresh? why is the queue bounded and what happens
when it fills? Timestamps are stored UTC (`datetime.utcnow`) — displayed
with a UTC label in the detail drawer; local-time display is a possible
Stage 5 polish item.
