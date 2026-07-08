# v1.2 "Clean plate data" — Design

**Date:** 2026-07-08 · **Status:** approved by student · **Sequencing:** execute after v1.1 (dashboard animations), which itself is gated on the v1.0 tag.

## Problem

Live testing on 2026-07-08 produced five "vehicles" from one physical
(Czech) plate: `el147ad`, `ee142ad`, `elt4740`, `elt47a0`, `el4740` — OCR
misreads of the same sheet of paper. Vehicle identity is currently keyed on
the *exact* plate text, so every misread mints a new phantom vehicle. OCR
output is a hypothesis with a confidence score, not a fact; the system
must manage that uncertainty instead of trusting it blindly.

## Goals

1. Stop phantom vehicles at the source: a low-trust read must not create a
   vehicle identity.
2. Keep every reading visible: events always keep the raw(ly normalized)
   text, confidence, and region — we filter identity creation, never data.
3. Store the plate region the API already returns (today it is discarded).
4. Normalize plate text once, at the system boundary.
5. Clean up existing data (lowercase texts, today's phantoms) via an
   Alembic data migration.

## Non-goals (v2 material)

- Fuzzy matching / automatic merging of similar plates (edit distance 1 can
  be two real cars; wrong merges are irreversible, phantoms are visible).
- Manual "merge vehicles" UI.
- Format regexes for countries other than RO.
- Multi-read voting per pass (costs N× API credits).

## Student decisions (2026-07-08)

- RO format regex applies **only when the API says region is "ro"** —
  the CZ lesson: a Romanian regex applied blindly rejects correct foreign
  reads. Foreign regions are gated by confidence alone.
- Confidence threshold **0.85**, configurable as `MIN_VEHICLE_CONFIDENCE`
  in `.env`.
- Existing data is repaired by an **Alembic data migration** (normalize +
  merge collisions + delete orphan vehicles), not by hand.

## Design

### 1. New module: `car_logger/services/plate_rules.py` (pure functions, TDD)

- `normalize_plate(text)` → uppercase, strip spaces and dashes; `None`
  passes through. `"b 123-abc"` → `"B123ABC"`.
- `is_valid_ro_plate(text)` → regex on normalized text:
  `^(B\d{2,3}|[A-Z]{2}\d{2})[A-Z]{3}$` (Bucharest: B + 2-3 digits + 3
  letters; counties: 2 letters + 2 digits + 3 letters).
- `should_create_vehicle(plate_text, confidence, region, min_confidence)`
  → the single gate: `False` when text is missing, confidence is missing
  or `< min_confidence`, or (`region == "ro"` and not a valid RO format);
  `True` otherwise.

### 2. ANPR client learns the region

- `PlateResult` namedtuple gains a 4th field `region` (existing
  constructions updated to pass `None`).
- `_parse` extracts `results[0].region.code` (dict-safe: missing key →
  `None`) and returns `normalize_plate(best.get("plate"))` — everything
  downstream sees normalized text only.
- Existing `tests/unit/test_anpr_client.py` expectations updated for the
  new tuple shape.

### 3. Schema + API surface

- `events.region` — new nullable String column (Alembic schema migration).
- `EventCreate` / `EventRead` gain `region: Optional[str]`.
- `repositories.update_event_anpr(...)` gains `region=None` parameter.
- Event detail drawer shows "Regiune: RO/CZ/—" (uppercased code).

### 4. Wiring (`main.py`, `on_result`)

On ANPR success: event is always updated with text + confidence + region;
`upsert_vehicle_for_plate` runs **only if** `should_create_vehicle(...)`
approves (threshold from `settings.min_vehicle_confidence`). A gated read
is an event without a vehicle — visible, but identity-less.

### 5. Config

`min_vehicle_confidence: float = 0.85` in `Settings`; documented in
`.env.example` and README's configuration table.

### 6. Data migration (same Alembic revision as the schema change)

1. Normalize `events.plate_text` and `vehicles.plate_text` in place.
2. Collision handling: if two vehicles normalize to the same text, merge —
   repoint events to the survivor, sum `total_sightings`, keep earliest
   `first_seen_at` / latest `last_seen_at`, delete the duplicate.
3. Delete orphan vehicles (zero events) — removes today's phantoms.
4. `downgrade()` is a documented no-op: normalization is not reversible
   (the pre-normalization casing is not stored anywhere).

## Testing

- Unit, `plate_rules`: normalization cases; RO regex accept/reject; gate
  matrix (below threshold, ro+bad format, ro+good format, foreign region
  passing on confidence alone, missing text/confidence).
- Unit, `anpr_client._parse`: payload with region → extracted + text
  normalized; payload without region → `region=None`.
- Integration: event round-trip carries `region`.
- Migration: verified live on the Jetson against today's real phantom
  data — after `alembic upgrade head`, the phantoms are gone and
  `mmm8748` reads `MMM8748`.

## Risks

- The RO regex has rare exceptions (diplomatic, military, temporary red
  plates). Accepted: a missed vehicle identity still leaves the event
  intact; the regex can be loosened later without touching data.
- Merging vehicles in the migration touches real rows — the student backs
  up `car_logger.db` (file copy) before running it on the device.
