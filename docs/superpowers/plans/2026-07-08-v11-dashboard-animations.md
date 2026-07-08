# v1.1 "Living Dashboard" Animations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Four discrete CSS-only animations that make the dashboard feel alive — fresh event rows slide in with a fading gold edge, the pipeline status dot pulses, the stat counters pop when they change, the detail drawer fades up on open — plus one display polish: plate chips render in uppercase everywhere.

**Architecture:** All motion is CSS keyframes in `base.html` — zero new JavaScript, zero dependencies. The only server-side logic is "which row is new": the feed route passes a `fresh_cutoff` timestamp (now − 10s, naive UTC like the DB values) and the template marks fresher rows with class `row-new`. Server-side rendering stays the single source of truth, and the marking is integration-testable (RED→GREEN). Everything else re-renders via the existing SSE-triggered swaps, so entrance animations replay naturally when fragments are re-inserted.

**Tech Stack:** Tailwind Play CDN (built-in `animate-pulse`), hand-written CSS keyframes, Jinja2, FastAPI TestClient tests.

## Global Constraints

- **DO NOT start before the `v1.0` tag exists.** The 24h soak validates commit `87c0b33`; this plan is the first v1.1 work. (Student confirms soak + Claude tags v1.0 first.)
- **Python 3.6.9 target.** No 3.7+ syntax.
- **No JavaScript, no build step.** CSS-only motion; Tailwind utility classes where they exist (`animate-pulse`).
- **Discrete intensity (student decision 2026-07-08):** durations ≤ 400ms, small offsets (≤ 6px); the single exception is the gold edge fade-out at 2s.
- **Accessibility:** the existing `@media (prefers-reduced-motion: reduce)` rule in `base.html` already zeroes ALL animations/transitions — do not duplicate it, just keep new rules above it.
- **Timestamps are naive UTC on both sides** of the freshness comparison (`datetime.utcnow()` in the route, `Event.timestamp` from the DB). Never mix in aware datetimes here.
- **Split execution:** **[LAPTOP — Claude]** writes/commits/pushes; **[JETSON — student]** pulls, runs, verifies. Paste output at each **CHECKPOINT**.
- **Commit trailer:** `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## File structure (what this plan touches)

- `car_logger/templates/base.html` — keyframes + `.row-new` / `.pop-in` / `.fade-up` classes (Task 1)
- `car_logger/api/routes_dashboard.py` — `fresh_cutoff` in the feed context (Task 2)
- `car_logger/templates/partials/events_feed.html` — `row-new` marking + hover slide (Tasks 2, 4)
- `tests/integration/test_dashboard.py` — freshness-marking tests (Task 2)
- `car_logger/templates/partials/stats.html` — pulsing dot + counter pop (Task 3)
- `car_logger/templates/partials/event_detail.html` — drawer fade-up (Task 4)
- `car_logger/templates/partials/macros.html` — uppercase plate chips (Task 5)

---

### Task 1: Animation primitives (CSS keyframes)

**Files:**
- Modify: `car_logger/templates/base.html` (the existing `<style>` block, lines ~41-51)

**Interfaces:**
- Produces: CSS classes `row-new`, `pop-in`, `fade-up` consumed by Tasks 2-4. Keyframe names `row-in`, `gold-flash`, `pop-in`, `fade-up`.

- [ ] **Step 1: Extend the style block** **[LAPTOP — Claude]**

In `base.html`, replace the existing `<style>` element with (new rules go ABOVE the reduced-motion kill-switch, which stays last):

```html
  <style>
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-thumb { background: #2a2a32; border-radius: 4px; }
    ::selection { background: #d4a843; color: #0a0a0c; }

    /* v1.1 "living dashboard" motion. Entrance animations replay whenever
       htmx re-inserts a fragment; the reduced-motion rule below disables
       everything for users who ask for no motion. */
    @keyframes row-in {
      from { opacity: 0; transform: translateY(-4px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes gold-flash {
      from { box-shadow: inset 3px 0 0 #d4a843; }
      to   { box-shadow: inset 3px 0 0 rgba(212, 168, 67, 0); }
    }
    @keyframes pop-in {
      from { opacity: 0; transform: scale(0.96); }
      to   { opacity: 1; transform: scale(1); }
    }
    @keyframes fade-up {
      from { opacity: 0; transform: translateY(6px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    .row-new { animation: row-in 300ms ease-out, gold-flash 2s ease-out forwards; }
    .pop-in  { animation: pop-in 250ms ease-out; }
    .fade-up { animation: fade-up 250ms ease-out; }

    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        animation: none !important;
        transition: none !important;
      }
    }
  </style>
```

Why `box-shadow: inset` for the gold edge instead of `border-left`: a border changes the element's box and shifts content 3px; an inset shadow paints over it — no layout shift when the flash fades.

- [ ] **Step 2: Commit** **[LAPTOP — Claude]**

```bash
git add car_logger/templates/base.html
git commit -m "feat(ui): animation keyframes for the living dashboard

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Fresh rows marked server-side (TDD)

**Files:**
- Modify: `car_logger/api/routes_dashboard.py` (imports + `events_feed`)
- Modify: `car_logger/templates/partials/events_feed.html` (the `<li>`)
- Test: `tests/integration/test_dashboard.py` (append)

**Interfaces:**
- Consumes: `.row-new` class from Task 1; existing `client` / `db_session` fixtures from `tests/conftest.py`.
- Produces: template context key `fresh_cutoff` (naive-UTC `datetime`); module constant `FRESH_ROW_SECONDS = 10` in `routes_dashboard.py`.

- [ ] **Step 1: Write the failing tests** **[LAPTOP — Claude]**

Append to `tests/integration/test_dashboard.py`:

```python
def test_fresh_event_row_is_marked_new(client):
    # created "now" -> inside the freshness window -> animated entrance
    client.post("/api/events", json={"plate_text": "NEW111"})
    resp = client.get("/partials/events-feed")
    assert resp.status_code == 200
    assert "row-new" in resp.text


def test_old_event_row_is_not_marked_new(client, db_session):
    from datetime import datetime, timedelta

    from car_logger.models import Event

    db_session.add(Event(timestamp=datetime.utcnow() - timedelta(minutes=5),
                         anpr_status="pending"))
    db_session.commit()
    resp = client.get("/partials/events-feed")
    assert resp.status_code == 200
    assert "row-new" not in resp.text
```

- [ ] **Step 2: Commit, push, confirm RED** **[LAPTOP — Claude then JETSON — student]**

```bash
git add tests/integration/test_dashboard.py
git commit -m "test(ui): failing tests for fresh-row marking

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

Then **[JETSON — student]**:
```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -m pytest tests/integration/test_dashboard.py -v
```
Expected: the two new tests FAIL (`'row-new' in resp.text` is False — the template never emits the class yet); older tests stay green.

- [ ] **Step 3: Implement the marking** **[LAPTOP — Claude]**

In `car_logger/api/routes_dashboard.py`, change the datetime import and add the constant:

```python
from datetime import datetime, timedelta, timezone
```

Below `templates.env.filters["localtime"] = localtime` add:

```python
# Rows younger than this get the animated "new" entrance in the feed.
FRESH_ROW_SECONDS = 10
```

Replace `events_feed` with:

```python
@router.get("/partials/events-feed")
def events_feed(request: Request, q: str = "", db: Session = Depends(get_db)):
    """Feed fragment, newest first; `q` filters by plate substring."""
    events = repositories.list_events(db, limit=15, plate_text=(q or None))
    fresh_cutoff = datetime.utcnow() - timedelta(seconds=FRESH_ROW_SECONDS)
    return templates.TemplateResponse(
        "partials/events_feed.html",
        {"request": request, "events": events, "fresh_cutoff": fresh_cutoff})
```

In `car_logger/templates/partials/events_feed.html`, change the `<li>` line:

```html
  <li class="flex items-stretch{{ ' row-new' if event.timestamp > fresh_cutoff else '' }}">
```

- [ ] **Step 4: Commit, push, confirm GREEN** **[LAPTOP — Claude then JETSON — student]**

```bash
git add car_logger/api/routes_dashboard.py car_logger/templates/partials/events_feed.html
git commit -m "feat(ui): fresh feed rows slide in with a fading gold edge

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

Then **[JETSON — student]**: `git pull && python3 -m pytest tests/ -v` → full suite green (55 passed).

**CHECKPOINT:** paste the suite total before Task 3.

---

### Task 3: Pulsing status dot + counter pop

**Files:**
- Modify: `car_logger/templates/partials/stats.html`

**Interfaces:**
- Consumes: `.pop-in` from Task 1; Tailwind built-in `animate-pulse`.

- [ ] **Step 1: Edit the template** **[LAPTOP — Claude]**

In `stats.html`, the three counter `<p>` elements gain `pop-in` (the fragment only re-renders on real SSE events, so a pop always means a real change):

```html
    <p class="pop-in mt-1 font-mono text-3xl text-paper">{{ stats.total_events }}</p>
```
```html
    <p class="pop-in mt-1 font-mono text-3xl text-paper">{{ stats.unique_vehicles }}</p>
```
```html
    <p class="pop-in mt-1 font-mono text-3xl text-gold">{{ stats.plates_read }}</p>
```

And the status dot pulses only while the pipeline runs (a dead pipeline should look dead):

```html
    <span class="h-1.5 w-1.5 rounded-full {{ 'bg-emerald-400 animate-pulse' if pipeline_running else 'bg-rose-400' }}"></span>
```

- [ ] **Step 2: Commit and push** **[LAPTOP — Claude]**

```bash
git add car_logger/templates/partials/stats.html
git commit -m "feat(ui): pulsing pipeline dot + counter pop on change

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

---

### Task 4: Drawer fade-up + row hover slide

**Files:**
- Modify: `car_logger/templates/partials/event_detail.html` (the `<article>`)
- Modify: `car_logger/templates/partials/events_feed.html` (the row `<button>` classes)

**Interfaces:**
- Consumes: `.fade-up` from Task 1.

- [ ] **Step 1: Drawer entrance** **[LAPTOP — Claude]**

In `event_detail.html`, the root element gains `fade-up`:

```html
<article class="fade-up overflow-hidden rounded-lg border border-ink-700 bg-ink-900">
```

- [ ] **Step 2: Hover slide on feed rows** **[LAPTOP — Claude]**

In `events_feed.html`, on the row `<button>` (the one with `hx-get="/partials/event/..."`), change `transition-colors duration-150` to `transition duration-150` (so transform transitions too) and add `hover:translate-x-0.5`:

```html
            class="flex min-w-0 flex-1 cursor-pointer items-center gap-4 px-4 py-3 text-left transition duration-150 hover:translate-x-0.5 hover:bg-ink-850 focus:outline-none focus-visible:ring-2 focus-visible:ring-gold">
```

- [ ] **Step 3: Commit and push** **[LAPTOP — Claude]**

```bash
git add car_logger/templates/partials/event_detail.html car_logger/templates/partials/events_feed.html
git commit -m "feat(ui): drawer fade-up + subtle row hover slide

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

---

### Task 5: Uppercase plate chips (student request 2026-07-08)

**Files:**
- Modify: `car_logger/templates/partials/macros.html` (the `plate_chip` macro)

**Interfaces:**
- Display-only: the DB keeps whatever casing the ANPR API returns (lowercase); CSS `text-transform` changes rendering everywhere the shared macro is used (feed, drawer, vehicles). Search is unaffected — SQLite `LIKE` is already case-insensitive for ASCII.

- [ ] **Step 1: Add `uppercase` to the chip** **[LAPTOP — Claude]**

In `macros.html`, the populated-chip `<span>` gains the Tailwind `uppercase` class:

```html
    <span class="inline-block rounded border border-paper-faint/60 bg-ink-800 px-2 py-0.5 font-mono text-sm font-medium uppercase tracking-widest text-paper">
```

- [ ] **Step 2: Commit and push** **[LAPTOP — Claude]**

```bash
git add car_logger/templates/partials/macros.html
git commit -m "style(ui): plate chips render uppercase everywhere

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

---

### Task 6: Live checkpoint on the device

- [ ] **Step 1: Deploy** **[JETSON — student]**

```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -m pytest tests/ -v            # expected: 55 passed
sudo systemctl restart car-logger
```

- [ ] **Step 2: Visual verification** **[JETSON/laptop — student]**

Open the dashboard, then trigger one detection (printed photo or `~/e2e_fake_cam.py` — remember: `sudo systemctl stop car-logger` first if using the fake cam, then restart the service after). Expected, all without touching the page:

1. The new row slides in from above with a gold left edge that fades over ~2s.
2. The counters pop when the numbers change.
3. The green "pipeline activ" dot pulses continuously.
4. Clicking a row: the detail drawer content fades up.
5. Hovering a row: it slides 2px right, smoothly.
6. Every plate chip (feed, drawer, vehicles) reads uppercase: `MMM8748`.

**CHECKPOINT:** describe what you saw (or screen-record it — it doubles as demo-video footage). Any animation that did NOT play gets debugged by the student per CLAUDE.md (Claude explains, student fixes).

---

## Self-Review

**1. Spec coverage:** fresh-row entrance + gold flash (Tasks 1+2, tested), pulsing dot (Task 3), counter pop (Tasks 1+3), drawer fade + hover slide (Tasks 1+4), uppercase plate chips (Task 5, student request), discrete intensity (constraint + durations), reduced-motion (existing global rule, kept last), after-v1.0 gate (Global Constraints). ✓

**2. Placeholder scan:** every step carries exact code/commands; no TBDs. ✓

**3. Type consistency:** class names `row-new`/`pop-in`/`fade-up` identical in CSS (Task 1), template (Tasks 2-4), and tests (Task 2). `fresh_cutoff` name matches route context and template condition. `FRESH_ROW_SECONDS` used only in the route. Naive-UTC on both sides of the comparison. ✓
