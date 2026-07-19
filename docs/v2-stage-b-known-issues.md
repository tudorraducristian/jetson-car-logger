# v2 Stage B — Known Issues (backlog for a later pass)

Findings from a Codex (gpt-5.5, read-only static review) pass over the
Stage B diff (`git diff 553af94..HEAD -- car_logger/ tests/`), triaged and
re-severitied for THIS appliance (single-user LAN, daily 04:00 systemd
restart, a stranded `pending` event is cosmetic, not a crash).

**None of these block the current deployment.** The appliance is correct
on the happy path and under the shipped config (`anpr_reads_per_track=3`).
This is a "when we do a hardening pass" list, ordered by real priority.
Each item is written to be executed TDD (write the failing test first).

Confirmed correct during the review (do NOT re-open): the 3-read vote
cases (2-of-3 wins, all-different → failed, `no_plate` abstains,
`winner_index` picks the max-confidence agreeing read), the detector's
plain-square coordinate scaling, and Python 3.6 compliance (no walrus /
`Literal` / dict-union found).

---

## 1. Vote accepts a tie when two texts each have ≥2 votes (LATENT)

- **Where:** `car_logger/services/plate_voting.py` — the
  `if len(indexes) >= 2 or len(votes) == 1:` branch.
- **Bug:** `best_text = max(votes, key=len)` picks one of the tied texts;
  the code then declares `success` on `len(indexes) >= 2` without checking
  whether another text has the SAME count. The stated rule is "no majority
  and ≥2 distinct texts → failed."
- **Failing scenario:** with `anpr_reads_per_track = 4`, reads
  `AAA111, AAA111, BBB222, BBB222` → returns `success` for whichever text
  `max()` sees first. Should be `failed` (2–2 tie, no majority).
- **Why it's latent, not active:** at the shipped default of **3** reads a
  2–2 tie is impossible (2+2 > 3), so the winning text is always a true
  majority. This only bites if `anpr_reads_per_track` is ever raised to 4+.
- **Priority:** highest of this list — it's the core error filter, and the
  fix is tiny. **Fix direction:** accept only when the top count is
  strictly greater than the second-best count, OR there is exactly one
  distinct text (`len(votes) == 1`). Test: `reads_per_track=4`, two texts
  ×2 each → `failed`; and a genuine 3-of-4 majority → `success`.

## 2. Shutdown can strand an in-flight event as `pending`

- **Where:** `car_logger/services/anpr_worker.py` `stop()` (2 s join then
  drain the QUEUE), and `car_logger/main.py` `_shutdown` (pipeline stop →
  collector.drain → worker stop), plus `pipeline.stop()`'s 2 s join.
- **Bug:** `stop()` joins the daemon worker for only 2 s, then drains the
  queue and closes the client. A job already dequeued and IN a
  `read_plate_multi` call (v2's 3-crop read is ~1 s, longer than v1's
  single read, and can exceed 2 s under load) is NOT in the queue, so it
  is neither completed nor drained-as-skipped — if the process then exits,
  the daemon thread is killed mid-read and that event stays `pending`.
- **Related race:** if `pipeline.stop()`'s 2 s join times out while a
  `_tick()` is blocked in a slow detector, `collector.drain()` can run,
  then the pipeline resumes and calls `collector.start()` again — a new
  `pending` collection registered after the worker is stopped, and the
  `_pending` dict mutated from two threads with no lock.
- **Failing scenario:** shutdown fires during a slow read/tick; that
  event never gets a terminal status.
- **Priority:** low-medium. Mitigated hard by the daily 04:00 restart and
  the fact that a stray `pending` row is cosmetic. **Fix direction:** track
  the in-flight job so `stop()` can mark it `skipped` if the join times
  out; and/or make the collector's `_pending` access lock-guarded and only
  `drain()` after the pipeline thread is provably stopped (join without a
  short timeout, or a stopped-flag the tick checks before `start()`).

## 3. Detector picks the highest-score row THEN rejects it, masking a valid box

- **Where:** `car_logger/services/onnx_engines.py` `best_detection()` —
  `best = rows[np.argmax(rows[:, 6])]` happens before the degenerate-box
  (`x2 <= x1 or y2 <= y1`) check.
- **Bug:** if the top-scoring row is degenerate but a lower-scoring row is
  a valid plate box, the function returns `None` (→ `no_plate`) instead of
  the valid detection.
- **Failing scenario:** rows include score 0.95 with `x1 == x2` and a valid
  plate at score 0.80 → `best_detection()` returns `None`.
- **Priority:** medium; low probability (unclear the YOLO end2end head ever
  emits a high-score degenerate box, but the logic doesn't guard it).
  **Fix direction:** filter out degenerate boxes BEFORE `argmax`, then pick
  the best of the survivors. Test with a degenerate high-score row + a valid
  lower-score row → returns the valid box.

## 4. Event created before crop #1 → `pending` if the crop path throws

- **Where:** `car_logger/main.py` `on_confirmed` (creates the `pending`
  event, then calls `collector.start()`), and
  `car_logger/services/crop_collector.py` `start()` (calls `crop_fn`, and
  fires `on_complete` synchronously in single-read mode) — neither catches
  a failure or marks the event terminal.
- **Bug:** the DB row is committed as `pending` first; if `crop_to_jpeg`
  raises (bad frame / OpenCV encode error), the pipeline's `_tick`
  try/except keeps the thread alive but the event is orphaned `pending`.
- **Failing scenario:** OpenCV raises during JPEG encode of crop #1.
- **Priority:** low; rare. **Fix direction:** wrap the crop/collection
  kickoff so a failure updates the event to `failed`/`skipped` instead of
  leaving it `pending` (mirror the worker's defense-in-depth at this
  boundary).

## 5. Plate search does not escape LIKE wildcards (cosmetic)

- **Where:** `car_logger/repositories.py` `list_events` —
  `Event.plate_text.like("%" + plate_text + "%")`.
- **Not a security bug:** SQLAlchemy parameterizes the value (no injection).
  The `q` string's `%` and `_` are treated as SQL wildcards, so searching a
  literal `_` is impossible and `%` matches every non-null plate.
- **Priority:** cosmetic — plates don't contain `%`/`_`. **Fix direction:**
  if ever wanted, escape `%`/`_` in `q` and pass `escape="\\"` to `like`.

---

_Source: Codex review 2026-07-19, thread `019f7ad2-bea1-71e0-b71c-8d256e42e0c7`._
_Codex could not run pytest (read-only sandbox), so this is static analysis only._
