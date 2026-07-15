# Camera self-healing + honest health — Design

**Date:** 2026-07-15 · **Status:** approved by student · **Sequencing:** standalone bugfix. The USB-autosuspend change is a deployment prerequisite (ops, not code — see Prerequisite).

## Problem

On 2026-07-15 the USB webcam (Trust Full HD Webcam, `145f:02aa`) dropped
off the bus and re-enumerated, after which the Tegra port autosuspended it.
`dmesg` evidence: `USB disconnect, device number N` → `new high-speed USB
device` → `usb_suspend_both` → `entering ELPG`. This repeated roughly every
few minutes (observed at 15:03 and again at 15:40).

The `CameraWorker` never noticed. On a failed `read()` it retries silently
and keeps the last good frame forever, so `get_latest_frame()` returns a
frozen pre-drop frame and `camera_ok` (defined as "frame is not None")
reports `True`. The pipeline keeps reprocessing the frozen frame, so
`frames_processed` climbs — every health signal looks green while no live
car is ever seen and no plate is read. Recovery required a manual
`systemctl restart car-logger`.

The same dishonest health check silently blindsided the 24h soak: a dead
camera and a carless scene produce identical output (no events), so "the
soak passed" never proved the camera was capturing.

Two distinct defects:

1. **No recovery** — the worker never reopens a lost device.
2. **Dishonest health** — `camera_ok` means "has a frame", not "has a
   *fresh* frame", so it cannot tell live capture from a frozen frame.

## Goals

1. **Self-heal.** When the camera stops delivering frames, the worker
   reopens it on its own and resumes — no manual restart. Loss is detected
   by staleness (no fresh frame for `stale_after_s`); reopen happens *only*
   on loss, never while the camera is healthy.
2. **Honest health.** `camera_ok` is true only when a fresh frame arrived
   within `stale_after_s`. `get_latest_frame()` returns `None` when stale,
   so the pipeline stops reprocessing a frozen frame and `frames_processed`
   becomes an honest liveness signal.
3. **Testable without hardware.** Reconnection and staleness logic are
   unit-tested with an injected fake camera and a fake clock.

## Non-goals

- **Disabling USB autosuspend / physical cable hygiene** — deployment
  prerequisite, not application code (see Prerequisite). Without it the
  camera still drops; this work makes the appliance *recover* from a drop.
- **Surfacing camera state in the dashboard UI** — deferred to a later
  "full observability" iteration.
- **Handling a `read()` that blocks forever** — evidence shows `read()`
  returns a *failure* (repeated GStreamer "Could not read from resource"),
  not a hang. A blocking read would need a separate watchdog; noted, out of
  scope.

## Student decisions (2026-07-15)

- Loss is detected on **time**: no fresh frame for `stale_after_s`
  (default `2.0`), not on a failed-read count. One tunable knob, easy to
  test with a fake clock. (`~2s` ≈ 30 missed frames at 15 fps — well above
  a normal inter-frame gap.)
- Reopen **only on loss**; retry **forever** with `reopen_backoff_s`
  between attempts, so the appliance recovers whenever the camera returns.
- **Approach A**: reconnection lives inside `CameraWorker`. Cheapest
  recovery — reopen the camera only (~3-5s), the TensorRT model stays
  loaded; cohesive and unit-testable. A separate watchdog thread and a
  systemd self-restart were both rejected (more moving parts / restarts
  everything and loses tracker state).
- Recovery window **~5-7s** per cycle is accepted. It is dominated by the
  `cv2.VideoCapture` open cost (~3-5s, a known Jetson footgun); shrinking
  the staleness threshold cannot reduce it.

## Design

### 1. `CameraWorker` becomes self-healing (`services/capture.py` — the core)

Constructor gains injectable dependencies and two tunables:

```
CameraWorker(device_index=0,
             stale_after_s=2.0,
             reopen_backoff_s=2.0,
             open_capture=<factory>,   # default: lazily returns cv2.VideoCapture(idx)
             now=time.monotonic)       # monotonic clock, immune to NTP steps
```

- `cv2` moves out of the module-top import into the default factory
  (mirrors `detector.py`'s lazy `jetson.inference` import), so the module
  imports without `cv2` present.
- `_last_frame_at` (monotonic) is the single source of truth for "fresh".
- `_loop`:
  - Ensure an open capture; if there is none, open one via the factory. If
    the open fails / the capture is not opened, sleep `reopen_backoff_s`
    and retry.
  - `ok, frame = cap.read()`. On `ok`: store the frame under the lock and
    set `_last_frame_at = now()`.
  - On not-`ok`: while `now() - _last_frame_at <= stale_after_s`, keep
    retrying (a single dropped frame is normal). Once it exceeds
    `stale_after_s`, treat it as a **loss**: log `camera_lost` once,
    `release()` the handle and drop it so the next iteration reopens.
  - When frames resume after a loss, log `camera_reconnected` once. Logs
    fire on **state transitions only** — no per-iteration spam.
- `get_latest_frame()`: returns a private copy of the frame, or `None`
  when there is no frame **or** `now() - _last_frame_at > stale_after_s`
  (stale reads as absent).
- `is_healthy()`: `True` iff a frame exists and
  `now() - _last_frame_at <= stale_after_s`.
- `start()` / `stop()` keep their contract; `stop()` still releases the
  capture.

### 2. Honest status (`api/routes_status.py`)

`camera_ok` becomes `camera.is_healthy()` instead of
`camera.get_latest_frame() is not None`. (`get_latest_frame()` is honest
now too, but `is_healthy()` avoids copying a frame just to answer a health
probe.) Other fields are unchanged; `frames_processed` is now trustworthy
because the pipeline stops advancing it while the camera is stale.

### 3. Config (`config.py`, `.env.example`, README)

Two new settings, tunable without touching code:

- `camera_stale_after_s: float = 2.0`
- `camera_reopen_backoff_s: float = 2.0`

Documented in `.env.example` and the README configuration table.

### 4. Wiring (`main.py`)

In `_startup`, construct
`CameraWorker(device_index=settings.camera_index,
stale_after_s=settings.camera_stale_after_s,
reopen_backoff_s=settings.camera_reopen_backoff_s)`. No other wiring
changes.

### 5. Pipeline (`services/pipeline.py`) — no change

`_tick` already returns early when `get_latest_frame()` is `None`
(sleeps 0.02). While the camera is stale/reopening it now receives `None`
and idles cleanly; `frames_processed` stops climbing. This is exactly why
the staleness rule belongs inside `get_latest_frame()`.

## Testing

TDD. Runs on the Jetson. Uses a `FakeCapture` + `FakeClock` — no real
camera. Student writes the assertions; Claude scaffolds the fixtures.

- **healthy:** reads succeed → `get_latest_frame()` returns the frame,
  `is_healthy()` True, `open_capture` called once.
- **single dropped frame under threshold:** one not-`ok` read, clock
  advanced `< stale_after_s` → still healthy, **no** reopen.
- **loss:** reads fail and the clock advances past `stale_after_s` →
  `is_healthy()` False, `get_latest_frame()` None, `open_capture` called
  again (reopen attempted), `camera_lost` logged once.
- **recovery:** the fake camera revives after reopen → frames resume,
  `is_healthy()` True again, `camera_reconnected` logged once.
- **reopen keeps failing:** `open_capture` returns a not-opened capture
  repeatedly → the worker retries with backoff, stays unhealthy, and never
  raises.

## Prerequisite (ops, not code)

Disable USB autosuspend on the Jetson so the port stops sleeping the
webcam: `usbcore.autosuspend=-1` on the `APPEND` line of
`/boot/extlinux/extlinux.conf` + reboot (back up the file first), or a
udev rule scoped to `145f:02aa`. Without this the camera still drops; this
code makes the appliance recover in ~5-7s instead of needing a manual
restart. Physical hygiene (good cable, direct Jetson port, no hub) further
reduces drops.

## Success criteria

After a USB drop, the appliance resumes live capture on its own within
~5-7s with no manual restart, and `camera_ok` reflects the true state
throughout (False during the outage, True once fresh frames resume). A
future 24h soak can trust `camera_ok` to reveal a dead camera.

## Risks

- **Blocking `read()`.** If `read()` ever blocks forever instead of
  returning a failure, the in-loop staleness check cannot fire (the loop
  is stuck). Current evidence shows failures, not hangs; a blocking read
  would need a separate watchdog. Noted, out of scope.
- **Reopen-forever.** A permanently-removed camera means one reopen attempt
  every `reopen_backoff_s` indefinitely. Bounded cost, logged on
  transitions only. Accepted for an appliance that should recover whenever
  the camera returns.
- **Threshold too low.** Too small a `stale_after_s` risks reopening during
  a brief legitimate stall. `2.0s` is comfortably above a normal
  inter-frame gap and is tunable via config.
