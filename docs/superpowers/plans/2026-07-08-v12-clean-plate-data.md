# v1.2 "Clean Plate Data" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop phantom vehicles (confidence gate + region-aware format check + normalization + data-repair migration per the approved spec `docs/superpowers/specs/2026-07-08-v12-clean-plate-data-design.md`) and fix the real defects found in the 2026-07-08 codex architecture review.

**Architecture:** A new pure-function module `plate_rules` is the single identity gate; the ANPR client normalizes text and extracts the region at the system boundary; `on_result` verifies the event still exists before creating anything; deletion repairs vehicle aggregates and removes the crop file; the broker moves its whole fan-out onto the event loop with coalescing queues; the pipeline and ANPR worker become crash-proof at their loop boundaries. Two Alembic revisions: schema (`events.region`), then data repair (normalize, merge, drop orphans).

**Tech Stack:** Python 3.6 stdlib (`re`, `collections.namedtuple`), SQLAlchemy 1.3 Query API, Alembic 1.7, httpx 0.22 MockTransport tests, pytest + pytest-asyncio 0.16.

## Global Constraints

- **Execute only after the v1.1 animations plan is done** (which itself is gated on the `v1.0` tag).
- **Python 3.6.9.** In particular: `namedtuple` has NO `defaults=` parameter (3.7+) — when `PlateResult` grows a field, every construction site must pass it explicitly.
- **Confidence threshold 0.85**, configurable as `MIN_VEHICLE_CONFIDENCE` (spec, student decision).
- **RO format regex applies ONLY when the API region is `"ro"`** (spec, student decision — the CZ lesson).
- **Events always keep the reading** (text/confidence/region); the gate guards only vehicle creation.
- **Backup before data migration:** on the Jetson, copy `car_logger.db` before `alembic upgrade head` (spec risk note).
- **Deviation from spec (documented):** two Alembic revisions instead of one — the schema change keeps a real `downgrade()` (drop column) while the data repair's downgrade is an honest no-op; separable review and rollback.
- **Split execution:** **[LAPTOP — Claude]** writes/commits/pushes; **[JETSON — student]** pulls, runs, pastes output at each **CHECKPOINT**.
- **Commit trailer:** `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## Codex review findings — disposition (2026-07-08, gpt-5.5 read-only)

| # | Finding | Verdict | Where |
|---|---------|---------|-------|
| 1 | `on_result` creates a vehicle + crop for an event deleted while ANPR was in flight | **REAL** — race introduced by the v1.0 delete feature | Task 5 |
| 2 | Deleting an event leaves `vehicles` aggregates stale (phantom stays listed) | **REAL** | Task 6 |
| 3 | `upsert_vehicle_for_plate` read-modify-write not race-safe | **REJECTED (YAGNI)** — exactly one ANPR worker thread exists; there is no concurrent caller by design. Revisit only if workers scale (v2). Noted here so the decision is on record. | — |
| 4 | SQLite foreign keys defined but never enforced (`PRAGMA foreign_keys` off) | **REAL** | Task 7 |
| 5 | One unhandled exception permanently kills the pipeline thread; status still says running | **REAL — highest severity for an appliance** | Task 8 |
| 6 | Broker: cross-thread mutation of `_subscribers` during `publish()` iteration; unbounded queues | **REAL** (the set race can raise `RuntimeError`); backpressure resolved by coalescing (maxsize=1), not bigger buffers | Task 9 |
| 7 | Shutdown strands queued ANPR jobs as `pending` forever; httpx client never closed | **REAL** — daily 04:00 restart makes this routine, not rare | Task 10 |
| 8 | 200/201 with a non-JSON body raises out of `read_plate`, breaking its documented contract | **REAL** | Task 2 |
| 9 | Queue-full "skipped" path updates the DB but never publishes SSE | **REAL** | Task 5 |
| 10 | Deleting an event leaves its crop `/data/plates/<id>.jpg` fetchable for up to 30 days | **REAL** (privacy on LAN) | Task 6 |

## File structure

- Create: `car_logger/services/plate_rules.py`, `tests/unit/test_plate_rules.py`, `tests/unit/test_on_result.py`, `tests/unit/test_pipeline_resilience.py`, `tests/unit/test_database.py`, `alembic/versions/b21c47d1a9e0_add_region_to_events.py`, `alembic/versions/c9d3e58f2b41_normalize_and_merge_plates.py`
- Modify: `car_logger/services/anpr_client.py`, `car_logger/services/anpr_worker.py`, `car_logger/services/broker.py`, `car_logger/services/pipeline.py`, `car_logger/models.py`, `car_logger/schemas.py`, `car_logger/repositories.py`, `car_logger/database.py`, `car_logger/config.py`, `car_logger/main.py`, `car_logger/api/routes_events.py`, `car_logger/templates/partials/event_detail.html`, `.env.example`, `README.md`
- Test (modify): `tests/unit/test_anpr_client.py`, `tests/unit/test_anpr_worker.py`, `tests/unit/test_broker.py`, `tests/integration/test_api_events.py`

---

### Task 1: `plate_rules` — the identity gate (TDD)

**Files:**
- Create: `car_logger/services/plate_rules.py`
- Test: `tests/unit/test_plate_rules.py`

**Interfaces:**
- Produces: `normalize_plate(text: Optional[str]) -> Optional[str]`; `is_valid_ro_plate(text) -> bool`; `should_create_vehicle(plate_text, confidence, region, min_confidence) -> bool`. Consumed by Tasks 2 and 5.

- [ ] **Step 1: Write the failing tests** **[LAPTOP — Claude]**

`tests/unit/test_plate_rules.py`:
```python
"""The identity gate: when is an OCR reading trustworthy enough for a Vehicle?"""

from car_logger.services.plate_rules import (
    is_valid_ro_plate, normalize_plate, should_create_vehicle)


def test_normalize_uppercases_and_strips_separators():
    assert normalize_plate("b 123-abc") == "B123ABC"


def test_normalize_none_passthrough():
    assert normalize_plate(None) is None


def test_ro_county_format_valid():
    assert is_valid_ro_plate("CJ45XYZ") is True


def test_ro_bucharest_format_valid():
    assert is_valid_ro_plate("B123ABC") is True


def test_ro_rejects_four_trailing_digits():
    assert is_valid_ro_plate("ELT4740") is False  # one of today's phantoms


def test_gate_rejects_below_threshold():
    assert should_create_vehicle("MMM8748", 0.60, None, 0.85) is False


def test_gate_accepts_confident_foreign_read():
    # region "cz": no RO regex — confidence alone decides (the CZ lesson)
    assert should_create_vehicle("EL147AD", 0.97, "cz", 0.85) is True


def test_gate_rejects_ro_region_with_bad_format():
    assert should_create_vehicle("ELT4740", 0.95, "ro", 0.85) is False


def test_gate_accepts_ro_region_with_good_format():
    assert should_create_vehicle("B123ABC", 0.95, "ro", 0.85) is True


def test_gate_rejects_missing_text_or_confidence():
    assert should_create_vehicle(None, 0.99, "ro", 0.85) is False
    assert should_create_vehicle("B123ABC", None, "ro", 0.85) is False
```

- [ ] **Step 2: Commit, push, confirm RED** **[LAPTOP — Claude then JETSON — student]**

```bash
git add tests/unit/test_plate_rules.py
git commit -m "test(plates): failing tests for the vehicle identity gate

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
**[JETSON]** `git pull && python3 -m pytest tests/unit/test_plate_rules.py -v` → collection error (module missing) = RED.

- [ ] **Step 3: Implement** **[LAPTOP — Claude]**

`car_logger/services/plate_rules.py`:
```python
"""Pure decision rules for plate data quality.

OCR output is a hypothesis with a confidence score, not a fact. These
functions decide — in one place — when a reading is trustworthy enough to
mint a Vehicle identity. Events always keep the raw reading regardless."""

import re

# Bucharest: B + 2-3 digits + 3 letters. Counties: 2 letters + 2 digits +
# 3 letters. Applied to normalized text (uppercase, no separators).
_RO_PLATE_RE = re.compile(r"^(B\d{2,3}|[A-Z]{2}\d{2})[A-Z]{3}$")


def normalize_plate(text):
    """Uppercase and strip spaces/dashes; None passes through."""
    if text is None:
        return None
    return text.replace(" ", "").replace("-", "").upper()


def is_valid_ro_plate(text):
    """True if the normalized text looks like a Romanian plate."""
    if not text:
        return False
    return _RO_PLATE_RE.match(normalize_plate(text)) is not None


def should_create_vehicle(plate_text, confidence, region, min_confidence):
    """The identity gate: trustworthy enough to create/update a Vehicle?

    STUDENT DECISIONS (2026-07-08): threshold configurable (default 0.85);
    the RO format check applies ONLY when the API says region == "ro" — a
    Romanian regex applied blindly rejects correct foreign reads (the
    CZ-plate lesson)."""
    if not plate_text:
        return False
    if confidence is None or confidence < min_confidence:
        return False
    if region == "ro" and not is_valid_ro_plate(plate_text):
        return False
    return True
```

- [ ] **Step 4: Commit, push, confirm GREEN** **[LAPTOP — Claude then JETSON — student]**

```bash
git add car_logger/services/plate_rules.py
git commit -m "feat(plates): identity gate - normalization, RO format, confidence

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
**[JETSON]** `git pull && python3 -m pytest tests/unit/test_plate_rules.py -v` → `10 passed`.

**CHECKPOINT:** paste before Task 2.

---

### Task 2: ANPR client — region, normalization at the boundary, JSON guard (finding 8)

**Files:**
- Modify: `car_logger/services/anpr_client.py`, `car_logger/services/anpr_worker.py:60`
- Test: `tests/unit/test_anpr_client.py`

**Interfaces:**
- Consumes: `normalize_plate` from Task 1.
- Produces: `PlateResult(plate_text, confidence, status, region)` — 4 fields now; every construction passes `region` explicitly (3.6 namedtuples have no defaults). Consumed by Tasks 5 and 10.

- [ ] **Step 1: Update + add tests (failing)** **[LAPTOP — Claude]**

In `tests/unit/test_anpr_client.py`, normalization changes three existing assertions — update them:
- `test_200_returns_plate`: `assert result.plate_text == "B123XYZ"` (was `"b123xyz"`)
- `test_201_created_returns_plate`: `assert result.plate_text == "MMM8748"` (was `"mmm8748"`)
- `test_500_then_200_succeeds`: `assert result.plate_text == "CJ01AAA"` (was `"cj01aaa"`)

Append:
```python
def test_parse_extracts_region_and_normalizes(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    ac, _ = _client_returning([
        (201, {"results": [{"plate": "el1 47ad", "score": 0.97,
                            "region": {"code": "cz", "score": 0.8}}]}),
    ])
    result = ac.read_plate(b"jpegbytes")
    assert result.plate_text == "EL147AD"
    assert result.region == "cz"


def test_parse_without_region_is_none(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    ac, _ = _client_returning([
        (200, {"results": [{"plate": "b123xyz", "score": 0.9}]}),
    ])
    assert ac.read_plate(b"x").region is None


def test_success_status_with_invalid_json_is_failed(monkeypatch):
    # codex finding 8: a 200 with an HTML body (proxy error page) must not
    # raise out of read_plate — its contract says expected failures don't.
    monkeypatch.setattr("time.sleep", lambda *_: None)

    def handler(request):
        return httpx.Response(200, text="<html>gateway error</html>")

    http = httpx.Client(transport=httpx.MockTransport(handler))
    ac = AnprClient("http://anpr.test", "tok", client=http)
    assert ac.read_plate(b"x").status == "failed"
```

- [ ] **Step 2: Commit, push, confirm RED** **[LAPTOP — Claude then JETSON — student]**

```bash
git add tests/unit/test_anpr_client.py
git commit -m "test(anpr): region extraction, normalization, invalid-JSON guard

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
**[JETSON]** `git pull && python3 -m pytest tests/unit/test_anpr_client.py -v` → new tests FAIL (`region` attribute missing / raw lowercase text / JSON decode error raised).

- [ ] **Step 3: Implement** **[LAPTOP — Claude]**

In `car_logger/services/anpr_client.py`:
- add import: `from car_logger.services.plate_rules import normalize_plate`
- change the namedtuple:
```python
PlateResult = namedtuple(
    "PlateResult", ["plate_text", "confidence", "status", "region"]
)  # status: success | failed | throttled | skipped
```
- every existing `PlateResult(None, None, "...")` in this file (lines ~56, 61, 66, 71) becomes `PlateResult(None, None, "...", None)`.
- the success branch of `read_plate` gains the JSON guard:
```python
            if resp.status_code in (200, 201):
                try:
                    payload = resp.json()
                except ValueError:
                    # 200 with a non-JSON body (proxy error page…): the
                    # contract says expected failures never raise.
                    return PlateResult(None, None, "failed", None)
                return self._parse(payload)
```
- `_parse` becomes:
```python
    def _parse(self, payload):
        results = payload.get("results", [])
        if not results:
            return PlateResult(None, None, "failed", None)
        best = results[0]
        region = (best.get("region") or {}).get("code")
        return PlateResult(normalize_plate(best.get("plate")),
                           best.get("score"), "success", region)
```

In `car_logger/services/anpr_worker.py` line ~60: `result = PlateResult(None, None, "failed", None)`.

If `tests/unit/test_anpr_worker.py` constructs `PlateResult(...)` stubs anywhere, add the fourth `None` argument there too (grep the file for `PlateResult(`).

- [ ] **Step 4: Commit, push, confirm GREEN** **[LAPTOP — Claude then JETSON — student]**

```bash
git add car_logger/services/anpr_client.py car_logger/services/anpr_worker.py tests/unit/test_anpr_worker.py
git commit -m "feat(anpr): PlateResult carries region; normalize at boundary; JSON guard

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
**[JETSON]** `git pull && python3 -m pytest tests/unit/test_anpr_client.py tests/unit/test_anpr_worker.py -v` → all pass.

**CHECKPOINT:** paste before Task 3.

---

### Task 3: `events.region` — model, schemas, repo, drawer, schema migration

**Files:**
- Modify: `car_logger/models.py`, `car_logger/schemas.py`, `car_logger/repositories.py` (`update_event_anpr`), `car_logger/templates/partials/event_detail.html`
- Create: `alembic/versions/b21c47d1a9e0_add_region_to_events.py`
- Test: `tests/integration/test_api_events.py` (append)

**Interfaces:**
- Produces: `Event.region` (nullable String); `EventCreate.region` / `EventRead.region` (`Optional[str]`); `update_event_anpr(db, event_id, plate_text, confidence, status, image_path, vehicle_id=None, region=None)`. Consumed by Task 5.

- [ ] **Step 1: Failing test** **[LAPTOP — Claude]**

Append to `tests/integration/test_api_events.py`:
```python
def test_event_region_roundtrip(client):
    created = client.post(
        "/api/events", json={"plate_text": "B123ABC", "region": "ro"}).json()
    assert created["region"] == "ro"
    resp = client.get("/api/events/" + str(created["id"]))
    assert resp.json()["region"] == "ro"
```
Commit/push (`test(api): failing test for event region roundtrip` + trailer); **[JETSON]** confirm RED (`region` missing from response).

- [ ] **Step 2: Implement** **[LAPTOP — Claude]**

`car_logger/models.py`, in `Event` after `plate_confidence`:
```python
    # region code the ANPR API detected for the plate ("ro", "cz"…) — used
    # by the identity gate and shown in the detail drawer.
    region = Column(String, nullable=True)
```

`car_logger/schemas.py`: add `region: Optional[str] = None` to `EventCreate` (after `plate_confidence`) and `region: Optional[str]` to `EventRead` (same position).

`car_logger/repositories.py`:
```python
def update_event_anpr(db, event_id, plate_text, confidence, status,
                      image_path, vehicle_id=None, region=None):
    """Fill in ANPR results on an existing event. Returns the event or None."""
    event = db.query(Event).filter(Event.id == event_id).first()
    if event is None:
        return None
    event.plate_text = plate_text
    event.plate_confidence = confidence
    event.anpr_status = status
    event.image_path = image_path
    event.region = region
    if vehicle_id is not None:
        event.vehicle_id = vehicle_id
    db.commit()
    db.refresh(event)
    return event
```

`car_logger/templates/partials/event_detail.html` — add a `<div>` in the `<dl>` after the "Track" entry:
```html
    <div>
      <dt class="font-mono text-[11px] uppercase tracking-widest text-paper-faint">Regiune</dt>
      <dd class="mt-1 font-mono text-paper">{{ event.region.upper() if event.region else '—' }}</dd>
    </div>
```

`alembic/versions/b21c47d1a9e0_add_region_to_events.py`:
```python
"""add region column to events

Revision ID: b21c47d1a9e0
Revises: a714d2651be8
Create Date: 2026-07-08
"""
from alembic import op
import sqlalchemy as sa

revision = "b21c47d1a9e0"
down_revision = "a714d2651be8"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("events", sa.Column("region", sa.String(), nullable=True))


def downgrade():
    # batch mode: this SQLite (3.22) has no native DROP COLUMN
    with op.batch_alter_table("events") as batch_op:
        batch_op.drop_column("region")
```

- [ ] **Step 3: Commit, push, confirm GREEN** **[LAPTOP — Claude then JETSON — student]**

```bash
git add car_logger/models.py car_logger/schemas.py car_logger/repositories.py car_logger/templates/partials/event_detail.html alembic/versions/b21c47d1a9e0_add_region_to_events.py tests/integration/test_api_events.py
git commit -m "feat(events): region column + schema migration + drawer display

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
**[JETSON]** `git pull && python3 -m pytest tests/ -v` → green (tests use `create_all`, so no migration needed for tests; the live DB migrates in Task 11).

**CHECKPOINT:** paste before Task 4.

---

### Task 4: Config — `MIN_VEHICLE_CONFIDENCE`

**Files:**
- Modify: `car_logger/config.py`, `.env.example`, `README.md` (configuration table)

**Interfaces:**
- Produces: `settings.min_vehicle_confidence: float = 0.85`. Consumed by Task 5.

- [ ] **Step 1: Implement (no test — pure config, covered via Task 5's tests)** **[LAPTOP — Claude]**

`car_logger/config.py`, after `detector_threshold`:
```python
    # identity gate: a plate reading below this confidence never creates a
    # Vehicle (the event still keeps the reading). Student decision 2026-07-08.
    min_vehicle_confidence: float = 0.85
```
`.env.example`, after `DETECTOR_THRESHOLD`:
```
MIN_VEHICLE_CONFIDENCE=0.85
```
`README.md` configuration table, after the `DETECTOR_THRESHOLD` row:
```
| `MIN_VEHICLE_CONFIDENCE` | `0.85`                       | Min OCR confidence to create a vehicle     |
```

- [ ] **Step 2: Commit and push** **[LAPTOP — Claude]**

```bash
git add car_logger/config.py .env.example README.md
git commit -m "feat(config): MIN_VEHICLE_CONFIDENCE for the identity gate

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

---

### Task 5: `on_result` rework — gate + deleted-event check + skipped publish (findings 1, 9; TDD)

**Files:**
- Modify: `car_logger/main.py`
- Test: create `tests/unit/test_on_result.py`

**Interfaces:**
- Consumes: `should_create_vehicle` (Task 1), `PlateResult` 4-field (Task 2), `update_event_anpr(..., region=)` (Task 3), `settings.min_vehicle_confidence` (Task 4).
- Produces: same `on_result(event_id, plate_result, crop_bytes)` callback signature — internals change only.

- [ ] **Step 1: Failing tests** **[LAPTOP — Claude]**

`tests/unit/test_on_result.py`:
```python
"""on_result: ANPR completion — deleted-event check + the identity gate.

crop_bytes is b"" in every test (falsy) so no crop files get written."""

from car_logger import main as app_main
from car_logger import repositories, schemas
from car_logger.models import Vehicle
from car_logger.services.anpr_client import PlateResult


class FakeBroker(object):
    def __init__(self):
        self.published = []

    def publish(self, data):
        self.published.append(data)


def _on_result(monkeypatch, db_session):
    monkeypatch.setattr(app_main, "SessionLocal", lambda: db_session)
    broker = FakeBroker()
    return app_main._make_on_result(broker), broker


def test_deleted_event_creates_nothing(monkeypatch, db_session):
    # codex finding 1: event deleted while the ANPR call was in flight
    on_result, broker = _on_result(monkeypatch, db_session)
    on_result(9999, PlateResult("B123ABC", 0.99, "success", "ro"), b"")
    assert db_session.query(Vehicle).count() == 0
    assert broker.published == []


def test_low_confidence_keeps_text_but_no_vehicle(monkeypatch, db_session):
    on_result, broker = _on_result(monkeypatch, db_session)
    event = repositories.create_event(db_session, schemas.EventCreate())
    on_result(event.id, PlateResult("EL4740", 0.60, "success", "cz"), b"")
    refreshed = repositories.get_event(db_session, event.id)
    assert refreshed.plate_text == "EL4740"
    assert refreshed.region == "cz"
    assert refreshed.vehicle_id is None
    assert db_session.query(Vehicle).count() == 0
    assert broker.published == ["updated"]


def test_confident_read_creates_vehicle(monkeypatch, db_session):
    on_result, broker = _on_result(monkeypatch, db_session)
    event = repositories.create_event(db_session, schemas.EventCreate())
    on_result(event.id, PlateResult("B123ABC", 0.95, "success", "ro"), b"")
    refreshed = repositories.get_event(db_session, event.id)
    assert refreshed.vehicle_id is not None
    assert db_session.query(Vehicle).count() == 1
    assert broker.published == ["updated"]
```

Commit/push (`test(main): failing tests for on_result gate + deleted-event race` + trailer); **[JETSON]** confirm RED (vehicle created for the deleted event, low-confidence read creates a vehicle).

- [ ] **Step 2: Implement** **[LAPTOP — Claude]**

`car_logger/main.py` — add import `from car_logger.services.plate_rules import should_create_vehicle`, then replace `_make_on_result`:
```python
def _make_on_result(broker):
    """Build the ANPR result callback: verify the event still exists, save
    the crop, apply the identity gate, update the event.

    Student amendment (2026-07-07): the crop is saved for EVERY outcome —
    a failed read's image is the debugging evidence.
    Review fix (codex, 2026-07-08): the event may have been DELETED from
    the dashboard while the ANPR call was in flight; in that case do
    nothing (especially: no vehicle, no crop file). A delete in the tiny
    window between this check and the commit is accepted residual risk on
    a single-user LAN appliance."""
    def on_result(event_id, plate_result, crop_bytes):
        db = SessionLocal()
        try:
            if repositories.get_event(db, event_id) is None:
                log.info("anpr_result_for_deleted_event", event_id=event_id)
                return
            image_path = None
            if crop_bytes:
                os.makedirs(PLATES_DIR, exist_ok=True)
                image_path = os.path.join(PLATES_DIR, str(event_id) + ".jpg")
                with open(image_path, "wb") as fh:
                    fh.write(crop_bytes)
            if plate_result.status == "success":
                vehicle_id = None
                if should_create_vehicle(plate_result.plate_text,
                                         plate_result.confidence,
                                         plate_result.region,
                                         settings.min_vehicle_confidence):
                    vehicle = repositories.upsert_vehicle_for_plate(
                        db, plate_result.plate_text)
                    vehicle_id = vehicle.id
                repositories.update_event_anpr(
                    db, event_id, plate_result.plate_text,
                    plate_result.confidence, "success", image_path,
                    vehicle_id, region=plate_result.region)
            else:
                repositories.update_event_anpr(
                    db, event_id, None, None, plate_result.status,
                    image_path, region=plate_result.region)
            broker.publish("updated")
        finally:
            db.close()
    return on_result
```

Finding 9 — in `on_confirmed` (inside `_startup`), the queue-full branch publishes too:
```python
        if not submitted:
            db2 = SessionLocal()
            try:
                repositories.update_event_anpr(
                    db2, event_id, None, None, "skipped", None,
                )
            finally:
                db2.close()
            app.state.broker.publish("updated")
```

- [ ] **Step 3: Commit, push, confirm GREEN** **[LAPTOP — Claude then JETSON — student]**

```bash
git add car_logger/main.py tests/unit/test_on_result.py
git commit -m "fix(main): identity gate in on_result; no vehicle for deleted events; publish on skipped

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
**[JETSON]** `git pull && python3 -m pytest tests/ -v` → green.

**CHECKPOINT:** paste before Task 6.

---

### Task 6: Delete repairs vehicle aggregates + removes the crop (findings 2, 10; TDD)

**Files:**
- Modify: `car_logger/repositories.py` (`delete_event`), `car_logger/api/routes_events.py`
- Test: `tests/integration/test_api_events.py` (append)

**Interfaces:**
- Produces: `repositories.delete_event(db, event_id) -> (bool, Optional[str])` — `(existed, image_path)`. The route deletes the crop file after DB success.

- [ ] **Step 1: Failing tests** **[LAPTOP — Claude]**

Append to `tests/integration/test_api_events.py`:
```python
def test_delete_last_event_removes_vehicle(client, db_session):
    from datetime import datetime

    from car_logger.models import Vehicle

    vehicle = Vehicle(plate_text="GONE123", first_seen_at=datetime.utcnow(),
                      last_seen_at=datetime.utcnow(), total_sightings=1)
    db_session.add(vehicle)
    db_session.commit()
    created = client.post("/api/events", json={
        "plate_text": "GONE123", "vehicle_id": vehicle.id}).json()
    assert client.delete("/api/events/" + str(created["id"])).status_code == 204
    assert db_session.query(Vehicle).count() == 0


def test_delete_one_of_two_recomputes_sightings(client, db_session):
    from datetime import datetime

    from car_logger.models import Vehicle

    vehicle = Vehicle(plate_text="STAY123", first_seen_at=datetime.utcnow(),
                      last_seen_at=datetime.utcnow(), total_sightings=2)
    db_session.add(vehicle)
    db_session.commit()
    first = client.post("/api/events", json={
        "plate_text": "STAY123", "vehicle_id": vehicle.id}).json()
    client.post("/api/events", json={
        "plate_text": "STAY123", "vehicle_id": vehicle.id})
    assert client.delete("/api/events/" + str(first["id"])).status_code == 204
    db_session.refresh(vehicle)
    assert vehicle.total_sightings == 1
```

Commit/push (`test(events): failing tests for delete repairing vehicle aggregates` + trailer); **[JETSON]** confirm RED (vehicle survives with stale counts).

- [ ] **Step 2: Implement** **[LAPTOP — Claude]**

`car_logger/repositories.py` — replace `delete_event`:
```python
def delete_event(db, event_id):
    """Delete the event and repair its vehicle's aggregates.

    Returns (existed, image_path) — image_path so the caller can remove
    the crop file AFTER the DB commit succeeds. If this was the vehicle's
    last event, the vehicle goes too: an identity with zero evidence is
    noise (the phantom-vehicle lesson, 2026-07-08)."""
    event = db.query(Event).filter(Event.id == event_id).first()
    if event is None:
        return (False, None)
    image_path = event.image_path
    vehicle_id = event.vehicle_id
    db.delete(event)
    if vehicle_id is not None:
        remaining = (db.query(Event)
                       .filter(Event.vehicle_id == vehicle_id,
                               Event.id != event_id)
                       .all())
        vehicle = db.query(Vehicle).filter(Vehicle.id == vehicle_id).first()
        if vehicle is not None:
            if not remaining:
                db.delete(vehicle)
            else:
                vehicle.total_sightings = len(remaining)
                vehicle.last_seen_at = max(e.timestamp for e in remaining)
    db.commit()
    return (True, image_path)
```

`car_logger/api/routes_events.py` — add `import os` at the top, replace the route body:
```python
@router.delete("/{event_id}", status_code=204)
def delete_event(event_id: int, request: Request,
                 db: Session = Depends(get_db)):
    """Delete an event: 204 on success, 404 if unknown."""
    existed, image_path = repositories.delete_event(db, event_id)
    if not existed:
        raise HTTPException(status_code=404, detail="Event not found")
    if image_path:
        try:
            os.remove(image_path)
        except OSError:
            pass  # crop already gone; the DB row was the source of truth
    # A delete is a write like any other write: publish so every open
    # dashboard refreshes its feed AND stats via SSE. htmx ignores the 204
    # body, so the SSE round-trip is what removes the row from the page.
    request.app.state.broker.publish("deleted")
    return Response(status_code=204)
```

- [ ] **Step 3: Commit, push, confirm GREEN** **[LAPTOP — Claude then JETSON — student]**

```bash
git add car_logger/repositories.py car_logger/api/routes_events.py tests/integration/test_api_events.py
git commit -m "fix(events): delete repairs vehicle aggregates and removes the crop file

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
**[JETSON]** `git pull && python3 -m pytest tests/ -v` → green.

**CHECKPOINT:** paste before Task 7.

---

### Task 7: Enforce SQLite foreign keys (finding 4; TDD)

**Files:**
- Modify: `car_logger/database.py`
- Test: create `tests/unit/test_database.py`

- [ ] **Step 1: Failing test** **[LAPTOP — Claude]**

`tests/unit/test_database.py`:
```python
"""SQLite ships with foreign-key enforcement OFF; we turn it on per connection."""

import pytest
from sqlalchemy.exc import IntegrityError

from car_logger.models import Event


def test_sqlite_rejects_bogus_vehicle_fk(db_session):
    db_session.add(Event(vehicle_id=9999, anpr_status="pending"))
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()
```

Commit/push (`test(db): failing test for FK enforcement` + trailer); **[JETSON]** confirm RED (commit succeeds silently — no IntegrityError).

- [ ] **Step 2: Implement** **[LAPTOP — Claude]**

`car_logger/database.py` — add after the imports (listener registered on the Engine *class* so the test engines in conftest get it too):
```python
from sqlalchemy import event
from sqlalchemy.engine import Engine


@event.listens_for(Engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_conn, connection_record):
    """SQLite leaves foreign keys OFF per connection unless asked; without
    this an event can reference a vehicle id that doesn't exist."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
```

- [ ] **Step 3: Commit, push, confirm GREEN** **[LAPTOP — Claude then JETSON — student]**

```bash
git add car_logger/database.py tests/unit/test_database.py
git commit -m "fix(db): enforce SQLite foreign keys on every connection

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
**[JETSON]** `git pull && python3 -m pytest tests/ -v` → green (watch for any test that relied on bogus FKs — none known).

**CHECKPOINT:** paste before Task 8.

---

### Task 8: Pipeline survives a bad tick (finding 5; TDD)

**Files:**
- Modify: `car_logger/services/pipeline.py`
- Test: create `tests/unit/test_pipeline_resilience.py`

- [ ] **Step 1: Failing test** **[LAPTOP — Claude]**

`tests/unit/test_pipeline_resilience.py`:
```python
"""codex finding 5: one exception must not kill the appliance's CV thread."""

import time

from car_logger.services.pipeline import PipelineWorker


class OneFrameCamera(object):
    def get_latest_frame(self):
        return "frame"


class FlakyDetector(object):
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient CUDA hiccup")
        return []


class NullTracker(object):
    def update(self, boxes):
        return []

    def new_confirmed_tracks(self):
        return []


def test_pipeline_survives_detector_exception():
    detector = FlakyDetector()
    worker = PipelineWorker(camera=OneFrameCamera(), detector=detector,
                            tracker=NullTracker(),
                            on_confirmed=lambda t, f: None, target_fps=200)
    worker.start()
    deadline = time.time() + 3.0
    while worker.frames_processed < 2 and time.time() < deadline:
        time.sleep(0.05)
    worker.stop()
    assert detector.calls >= 2           # kept calling after the raise
    assert worker.frames_processed >= 1  # processed frames post-exception
```

Commit/push (`test(pipeline): failing test - thread must survive a detector exception` + trailer); **[JETSON]** confirm RED (thread dies on first raise; `frames_processed` stays 0 until the deadline, assertion fails).

- [ ] **Step 2: Implement** **[LAPTOP — Claude]**

`car_logger/services/pipeline.py` — add `import logging` and `log = logging.getLogger(__name__)` at module level, then split the loop:
```python
    def _loop(self):
        while self._running:
            try:
                self._tick()
            except Exception:
                # A transient failure (SQLite lock, detector hiccup, bad
                # frame) must not kill the appliance's only CV thread.
                # Short sleep so a persistent failure can't spin the CPU.
                log.exception("pipeline tick failed; continuing")
                time.sleep(0.5)

    def _tick(self):
        t0 = time.time()
        frame = self.camera.get_latest_frame()
        if frame is None:
            time.sleep(0.02)
            return
        detections = self.detector.detect(frame)
        boxes = [(d.x1, d.y1, d.x2, d.y2) for d in detections]
        self.tracker.update(boxes)
        for track in self.tracker.new_confirmed_tracks():
            self.last_event_at = time.time()
            self.on_confirmed(track, frame)
        self.frames_processed += 1
        elapsed = time.time() - t0
        if elapsed > 0:
            self.last_fps = 1.0 / elapsed
        # Throttle to the target FPS so we don't pin the GPU pointlessly.
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
```

- [ ] **Step 3: Commit, push, confirm GREEN** **[LAPTOP — Claude then JETSON — student]**

```bash
git add car_logger/services/pipeline.py tests/unit/test_pipeline_resilience.py
git commit -m "fix(pipeline): a failing tick logs and continues instead of killing the thread

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
**[JETSON]** `git pull && python3 -m pytest tests/ -v` → green.

**CHECKPOINT:** paste before Task 9.

---

### Task 9: Broker — fan-out on the loop, coalescing queues (finding 6; TDD)

**Files:**
- Modify: `car_logger/services/broker.py`
- Test: `tests/unit/test_broker.py` (append)

- [ ] **Step 1: Failing test** **[LAPTOP — Claude]**

Append to `tests/unit/test_broker.py`:
```python
@pytest.mark.asyncio
async def test_burst_publishes_coalesce_to_one_signal():
    # Signals carry no payload the client uses — "something changed" twice
    # is worth exactly one re-fetch. maxsize=1 + drop-on-full coalesces.
    broker = EventBroker()
    broker.set_loop(asyncio.get_event_loop())
    queue = await broker.subscribe()
    broker.publish("created")
    broker.publish("updated")
    broker.publish("deleted")
    await asyncio.sleep(0.05)
    assert queue.qsize() == 1
```

Commit/push (`test(broker): failing test for coalescing burst publishes` + trailer); **[JETSON]** confirm RED (`qsize() == 3`).

- [ ] **Step 2: Implement** **[LAPTOP — Claude]**

`car_logger/services/broker.py` — replace `subscribe` and `publish`, add `_fanout`:
```python
    async def subscribe(self):
        """One queue per SSE client. maxsize=1 + drop-on-full coalesces
        bursts: a client that already has an unread change-signal gains
        nothing from a second one (finding: unbounded queues on slow
        clients)."""
        queue = asyncio.Queue(maxsize=1)
        self._subscribers.add(queue)
        return queue

    def publish(self, data):
        """Thread-safe: hand the WHOLE fan-out to the loop thread. The
        subscriber set is then only ever touched on the loop (subscribe/
        unsubscribe already run there), so no cross-thread set mutation
        can race the iteration. No-op if no loop yet."""
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._fanout, data)

    def _fanout(self, data):
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                pass  # client already has an unread change-signal
```

- [ ] **Step 3: Commit, push, confirm GREEN** **[LAPTOP — Claude then JETSON — student]**

```bash
git add car_logger/services/broker.py tests/unit/test_broker.py
git commit -m "fix(broker): fan-out runs on the loop; per-client queues coalesce

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
**[JETSON]** `git pull && python3 -m pytest tests/unit/test_broker.py -v` → all pass (the 3 original tests must stay green).

**CHECKPOINT:** paste before Task 10.

---

### Task 10: ANPR worker shutdown — drain as skipped, close the client (finding 7; TDD)

**Files:**
- Modify: `car_logger/services/anpr_worker.py`, `car_logger/services/anpr_client.py` (add `close()`)
- Test: `tests/unit/test_anpr_worker.py` (append)

- [ ] **Step 1: Failing test** **[LAPTOP — Claude]**

Append to `tests/unit/test_anpr_worker.py`:
```python
def test_stop_drains_pending_jobs_as_skipped_and_closes_client():
    # codex finding 7: jobs queued at shutdown (daily 04:00 restart!) must
    # not leave their events 'pending' forever.
    calls = []

    class ClosableClient(object):
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    client = ClosableClient()
    worker = AnprWorker(
        client, lambda eid, res, crop: calls.append((eid, res.status)))
    worker.submit(1, b"a")
    worker.submit(2, b"b")
    worker.stop()  # never started: everything is still queued
    assert calls == [(1, "skipped"), (2, "skipped")]
    assert client.closed is True
```
(Ensure the file imports `AnprWorker` — it already tests the worker.)

Commit/push (`test(anpr): failing test - shutdown drains queue, closes client` + trailer); **[JETSON]** confirm RED (`calls == []`, `closed is False`).

- [ ] **Step 2: Implement** **[LAPTOP — Claude]**

`car_logger/services/anpr_client.py` — add to `AnprClient`:
```python
    def close(self):
        """Release the HTTP connection pool (called from worker shutdown)."""
        self._client.close()
```

`car_logger/services/anpr_worker.py` — replace `stop`:
```python
    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # Daily-restart reality: jobs still queued at shutdown would leave
        # their events 'pending' forever. Mark them skipped instead.
        while True:
            try:
                event_id, crop_bytes = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                self._on_result(
                    event_id, PlateResult(None, None, "skipped", None),
                    crop_bytes)
            except Exception:
                log.exception("drain: on_result raised for event %s",
                              event_id)
        self._client.close()
```

- [ ] **Step 3: Commit, push, confirm GREEN** **[LAPTOP — Claude then JETSON — student]**

```bash
git add car_logger/services/anpr_worker.py car_logger/services/anpr_client.py tests/unit/test_anpr_worker.py
git commit -m "fix(anpr): shutdown drains queued jobs as skipped and closes the client

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
**[JETSON]** `git pull && python3 -m pytest tests/ -v` → green.

**CHECKPOINT:** paste before Task 11.

---

### Task 11: Data-repair migration — normalize, merge, drop orphans

**Files:**
- Create: `alembic/versions/c9d3e58f2b41_normalize_and_merge_plates.py`

**Interfaces:**
- Consumes: revision `b21c47d1a9e0` (Task 3) as `down_revision`.
- Produces: clean live data. `downgrade()` is a documented no-op (spec).

- [ ] **Step 1: Write the migration** **[LAPTOP — Claude]**

```python
"""normalize plate texts, merge duplicate vehicles, drop orphans

Revision ID: c9d3e58f2b41
Revises: b21c47d1a9e0
Create Date: 2026-07-08
"""
from alembic import op
import sqlalchemy as sa

revision = "c9d3e58f2b41"
down_revision = "b21c47d1a9e0"
branch_labels = None
depends_on = None


def _normalize(text):
    # Deliberate inline copy of plate_rules.normalize_plate: a migration
    # must stay frozen even if the app code changes later.
    if text is None:
        return None
    return text.replace(" ", "").replace("-", "").upper()


def upgrade():
    conn = op.get_bind()

    # 1) normalize event plate texts in place
    for row in list(conn.execute(sa.text(
            "SELECT id, plate_text FROM events "
            "WHERE plate_text IS NOT NULL"))):
        normalized = _normalize(row[1])
        if normalized != row[1]:
            conn.execute(sa.text(
                "UPDATE events SET plate_text = :p WHERE id = :i"),
                {"p": normalized, "i": row[0]})

    # 2) group vehicles by normalized text; merge each group into its
    #    earliest member (events repointed first, so no orphan FKs)
    rows = list(conn.execute(sa.text(
        "SELECT id, plate_text, first_seen_at, last_seen_at, "
        "total_sightings FROM vehicles ORDER BY id")))
    groups = {}
    for vid, plate, first_seen, last_seen, sightings in rows:
        groups.setdefault(_normalize(plate), []).append(
            (vid, plate, first_seen, last_seen, sightings))
    for normalized, members in groups.items():
        survivor = members[0]
        for loser in members[1:]:
            conn.execute(sa.text(
                "UPDATE events SET vehicle_id = :s WHERE vehicle_id = :d"),
                {"s": survivor[0], "d": loser[0]})
            conn.execute(sa.text(
                "UPDATE vehicles SET "
                "total_sightings = total_sightings + :n, "
                "first_seen_at = MIN(first_seen_at, :f), "
                "last_seen_at = MAX(last_seen_at, :l) WHERE id = :s"),
                {"n": loser[4], "f": loser[2], "l": loser[3],
                 "s": survivor[0]})
            conn.execute(sa.text("DELETE FROM vehicles WHERE id = :d"),
                         {"d": loser[0]})
        # rename LAST: every colliding row is gone, so UNIQUE can't fire
        if survivor[1] != normalized:
            conn.execute(sa.text(
                "UPDATE vehicles SET plate_text = :p WHERE id = :i"),
                {"p": normalized, "i": survivor[0]})

    # 3) drop orphan vehicles (zero events) — today's phantoms
    conn.execute(sa.text(
        "DELETE FROM vehicles WHERE id NOT IN "
        "(SELECT DISTINCT vehicle_id FROM events "
        " WHERE vehicle_id IS NOT NULL)"))


def downgrade():
    # Irreversible by design: pre-normalization casing isn't stored
    # anywhere and merged vehicles can't be un-merged (spec). No-op.
    pass
```

- [ ] **Step 2: Commit and push** **[LAPTOP — Claude]**

```bash
git add alembic/versions/c9d3e58f2b41_normalize_and_merge_plates.py
git commit -m "feat(db): data migration - normalize plates, merge dupes, drop orphans

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

---

### Task 12: Deploy + live verification on the device

- [ ] **Step 1: Backup, migrate, restart** **[JETSON — student]**

```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -m pytest tests/ -v                 # full suite green FIRST
sudo systemctl stop car-logger
cp car_logger.db car_logger.db.bak-v12      # spec risk note: backup!
alembic upgrade head
sudo systemctl start car-logger
```

- [ ] **Step 2: Verify the cleanup** **[JETSON — student]**

```bash
sqlite3 car_logger.db "SELECT id, plate_text, total_sightings FROM vehicles;"
sqlite3 car_logger.db "SELECT id, plate_text, region, anpr_status FROM events ORDER BY id DESC LIMIT 10;"
```
Expected: vehicles show `MMM8748` (uppercase) and NO `elt4740`/`el4740`-style phantoms; every remaining vehicle has ≥1 event.

- [ ] **Step 3: Verify live behavior** **[JETSON/laptop — student]**

Dashboard: vehicles list is clean; open an event → drawer shows "Regiune". Optional (1 credit): one paper-photo detection → event appears with region set; a second low-quality showing that misreads should create an event but NO new vehicle.

**CHECKPOINT:** paste the two sqlite3 outputs + suite total. v1.2 done.

---

## Self-Review

**1. Spec coverage:** normalize at boundary (Task 2), region stored+displayed (Task 3), confidence gate + region-aware RO regex (Tasks 1, 5), `MIN_VEHICLE_CONFIDENCE` (Task 4), data migration normalize/merge/orphans + backup + no-op downgrade (Tasks 11-12), events always keep the reading (Task 5 tests). Codex findings: 1→T5, 2→T6, 3→rejected (documented), 4→T7, 5→T8, 6→T9, 7→T10, 8→T2, 9→T5, 10→T6. Deviation from spec (two revisions) documented in Global Constraints. ✓

**2. Placeholder scan:** every code step carries full code; no TBDs. ✓

**3. Type consistency:** `PlateResult(plate_text, confidence, status, region)` used with 4 args in client, worker, and all tests (Tasks 2, 5, 10). `should_create_vehicle(plate_text, confidence, region, min_confidence)` identical in Task 1 impl/tests and Task 5 caller. `update_event_anpr(..., vehicle_id=None, region=None)` matches Task 3 impl and Task 5 calls. `delete_event -> (bool, Optional[str])` matches route unpacking (Task 6). Revision chain `a714d2651be8 → b21c47d1a9e0 → c9d3e58f2b41`. ✓
