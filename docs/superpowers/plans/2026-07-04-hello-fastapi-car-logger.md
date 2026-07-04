# Hello FastAPI on Jetson (car_logger stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A minimal FastAPI app (`/` and `/health`) running on the Jetson, reachable from a browser on the laptop, with the pinned dependency stack installed and 2 passing tests.

**Architecture:** First real code of the `car_logger` package — the embryo of the final app. `car_logger/main.py` holds the FastAPI `app` object that every later stage attaches to. Code is written on the laptop, flows to the Jetson via git (push → pull), and all runtime verification happens on the Jetson (Python 3.6.9).

**Tech Stack:** FastAPI 0.67.0, uvicorn 0.15.0, pytest 7.0.1 (full pinned stack from CLAUDE.md installed up front).

## Global Constraints

- **Target runtime is the Jetson: Python 3.6.9.** No Python 3.7+ syntax anywhere: no walrus (`:=`), no f-string `=`, no dict union (`|`), no `typing.Literal`.
- **Pinned versions from CLAUDE.md, verbatim.** Do not upgrade any package.
- **Sync `def` endpoints only** (no `async def` in this stage).
- **Split execution:** steps marked **[LAPTOP — Claude]** are done by Claude in this session (write files, commit, push). Steps marked **[JETSON — student]** are commands the student runs over SSH (`ssh tudor@192.168.0.232`, repo at `~/jetson-car-logger`, venv activated with `source venv/bin/activate`). **Claude cannot run anything on the Jetson** — after each Jetson step, the student pastes the output back before the plan continues.
- **The laptop cannot run this code** (Python 3.14 can't install the 3.6 pins). Never try to pytest/run it on the laptop.
- **Commit style:** English messages, ending with the `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer.
- **Do not touch** `experiments/lpr_batch/` or the docs.

---

### Task 1: Pinned `requirements.txt` + install on the Jetson

**Files:**
- Create: `requirements.txt` (repo root)

**Interfaces:**
- Produces: installed venv on the Jetson where `import fastapi` works; `requirements.txt` used by all later stages.

- [ ] **Step 1: Create the dependency file** **[LAPTOP — Claude]**

`requirements.txt`:
```
# Python 3.6 CONSTRAINT: many recent packages dropped 3.6. These versions
# are the last known-good. DO NOT upgrade without testing on the Jetson.

fastapi==0.67.0              # last 3.6-compatible release
uvicorn==0.15.0              # compatible with fastapi 0.67
pydantic==1.8.2              # v1 API, required by fastapi 0.67
sqlalchemy==1.3.24           # last 1.3 release, works with 3.6
alembic==1.7.7               # 3.6 compatible
python-multipart==0.0.5      # for file uploads in fastapi
jinja2==3.0.3                # last 3.0 release, 3.6 compatible
aiofiles==0.8.0              # for StaticFiles + templates
httpx==0.22.0                # for Plate Recognizer API calls
pyyaml==5.4.1                # for config
structlog==21.5.0            # structured logging
sse-starlette==0.10.3        # SSE support for FastAPI
pytest==7.0.1                # testing
pytest-asyncio==0.16.0       # async test support
requests==2.27.1             # needed by fastapi TestClient (last 3.6 release)
```

- [ ] **Step 2: Commit and push** **[LAPTOP — Claude]**

```bash
git add requirements.txt
git commit -m "build: pinned Python 3.6 dependency stack

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```

- [ ] **Step 3: Check swap before installing** **[JETSON — student]**

```bash
free -h
```
Look at the `Swap:` line. JetPack usually ships ~2GB of zram — enough for this install. If `Swap:` shows `0B`, stop and report before continuing.

- [ ] **Step 4: Pull and install** **[JETSON — student]**

```bash
cd ~/jetson-car-logger
source venv/bin/activate
git pull
pip install --upgrade pip
pip install -r requirements.txt
```
Expected: takes **10–20 minutes** (some packages compile on the Nano — this is normal, mostly waiting). Ends with `Successfully installed ...` listing the packages.

- [ ] **Step 5: Verify the install** **[JETSON — student]**

```bash
python3 -c "import fastapi, uvicorn, pytest; print(fastapi.__version__, uvicorn.__version__, pytest.__version__)"
```
Expected: `0.67.0 0.15.0 7.0.1`

**CHECKPOINT:** paste the output of Steps 3 and 5 back to Claude before Task 2.

---

### Task 2: Failing tests for `/` and `/health`

**Files:**
- Create: `car_logger/__init__.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `fastapi.testclient.TestClient` (from Task 1's install).
- Produces: the test contract for Task 3 — `car_logger.main` must expose `app` with `GET /` returning `{"message": "Car Logger is running", "version": "0.1.0"}` and `GET /health` returning `{"status": "ok"}`.

- [ ] **Step 1: Create the empty package marker** **[LAPTOP — Claude]**

`car_logger/__init__.py`:
```python
```
(empty file — it marks `car_logger/` as a Python package)

- [ ] **Step 2: Write the failing tests** **[LAPTOP — Claude]**

`tests/test_main.py`:
```python
from fastapi.testclient import TestClient

from car_logger.main import app

client = TestClient(app)


def test_root_returns_greeting_and_version():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Car Logger is running"
    assert body["version"] == "0.1.0"


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 3: Commit and push** **[LAPTOP — Claude]**

```bash
git add car_logger/__init__.py tests/test_main.py
git commit -m "test(api): failing tests for root and health endpoints

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```

- [ ] **Step 4: Run tests to verify they fail** **[JETSON — student]**

```bash
cd ~/jetson-car-logger
git pull
python3 -m pytest tests/ -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'car_logger.main'` — red confirmed, `main.py` doesn't exist yet.

**CHECKPOINT:** paste the pytest output back to Claude before Task 3.

---

### Task 3: Implement `car_logger/main.py` (make tests pass)

**Files:**
- Create: `car_logger/main.py`

**Interfaces:**
- Consumes: the test contract from Task 2.
- Produces: `app` — the FastAPI object that **every later stage** (events API, dashboard, SSE) attaches to; `GET /health` used by systemd/monitoring in the final appliance.

- [ ] **Step 1: Write the minimal implementation** **[LAPTOP — Claude]**

`car_logger/main.py`:
```python
"""Car Logger API entrypoint - the app object everything else attaches to."""

from fastapi import FastAPI

APP_VERSION = "0.1.0"

app = FastAPI(title="Car Logger", version=APP_VERSION)


@app.get("/")
def root():
    """Greeting endpoint - proves the server is reachable from the LAN."""
    return {"message": "Car Logger is running", "version": APP_VERSION}


@app.get("/health")
def health():
    """Liveness probe - used later by systemd and monitoring."""
    return {"status": "ok"}
```

- [ ] **Step 2: Commit and push** **[LAPTOP — Claude]**

```bash
git add car_logger/main.py
git commit -m "feat(api): minimal FastAPI app with root and health endpoints

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```

- [ ] **Step 3: Run tests to verify they pass** **[JETSON — student]**

```bash
cd ~/jetson-car-logger
git pull
python3 -m pytest tests/ -v
```
Expected: `2 passed`.

**CHECKPOINT:** paste the pytest output back to Claude before Task 4.

---

### Task 4: Run the server and verify from the laptop browser

**Files:** none (no code changes — this is the end-to-end proof).

**Interfaces:**
- Consumes: `car_logger.main:app` from Task 3.
- Produces: the working dev-run command used for the rest of the project.

- [ ] **Step 1: Start the server** **[JETSON — student]**

```bash
cd ~/jetson-car-logger
source venv/bin/activate
uvicorn car_logger.main:app --host 0.0.0.0 --port 8000 --reload
```
`--host 0.0.0.0` = listen for the whole LAN, not just the Jetson itself. `--reload` = auto-restart when files change (so future `git pull`s apply live). Leave this terminal running.

Expected output ends with: `Uvicorn running on http://0.0.0.0:8000`.

- [ ] **Step 2: Verify from the laptop browser** **[LAPTOP — student]**

Open in the browser, in order:
1. `http://192.168.0.232:8000/` → expect `{"message":"Car Logger is running","version":"0.1.0"}`
2. `http://192.168.0.232:8000/health` → expect `{"status":"ok"}`
3. `http://192.168.0.232:8000/docs` → expect Swagger UI listing both endpoints; expand `GET /health` → "Try it out" → "Execute" → response 200.

- [ ] **Step 3: Understanding checkpoint** **[student]**

Per PLAN.md's verification: state in one sentence, in your own words, what `uvicorn` does vs what FastAPI does. (If this is hard, re-read the restaurant analogy — waiter vs cook.)

**CHECKPOINT:** confirm what the browser showed. Stage 1 code is done when all three URLs work.

---

## Self-Review

**1. Spec coverage:**
- Pinned requirements.txt from CLAUDE.md stack: Task 1 (verbatim pins + `requests` for TestClient). ✓
- `car_logger/main.py` with `/` + `/health`, sync handlers: Task 3. ✓
- Reachable from laptop browser + `/docs` Swagger: Task 4. ✓
- ≥ 2 tests passing with pytest: Tasks 2–3. ✓
- First commit(s) on GitHub: every task commits and pushes. ✓
- `.gitignore` + README (PLAN.md 1.6): already exist in the repo — no task needed. ✓
- Swap check (PLAN.md 1.1): folded into Task 1 Step 3 as a pre-install check. Fan systemd service: deliberately deferred — matters when CV runs continuously (stage 3), not for a hello-world server.

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**3. Type consistency:** `car_logger.main` exposes `app`; tests import `from car_logger.main import app`; response bodies in Task 2 tests match Task 3 implementation exactly (`message`, `version` = `"0.1.0"`, `{"status": "ok"}`). Uvicorn target `car_logger.main:app` matches. ✓

## Notes for the executor

- **This plan has human-in-the-loop checkpoints.** Claude writes/commits/pushes on the laptop; the student runs every Jetson command over SSH and pastes output back. Do not proceed past a CHECKPOINT without the student's confirmation.
- The venv on the Jetson was created with `--system-site-packages` (so `jetson.inference` stays visible); it already exists — do not recreate it.
- If `pip install` fails on a specific package, stop and report the exact error — do not improvise different versions.