# Stage 2 — Database + CRUD API (car_logger) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A full CRUD API over SQLite — create an event, list events (paginated + plate filter), fetch one event — plus configuration, the ORM models, Pydantic DTOs, a repository layer, and an Alembic migration. No camera, no detection yet.

**Architecture:** The classic layered backend: `config` → `database` (engine/session/Base) → `models` (ORM) → `schemas` (Pydantic DTOs) → `repositories` (queries) → `api/routes_events` (HTTP), wired into the Stage 1 `app`. The route layer never touches the ORM directly — it goes through repositories, and speaks Pydantic schemas at the boundary. Alembic owns the real DB schema; tests use an in-memory SQLite built from `Base.metadata`.

**Tech Stack:** SQLAlchemy 1.3.24 (classic Query API), pydantic 1.8.2 (v1 `BaseSettings` + `orm_mode`), alembic 1.7.7, FastAPI 0.67.0 (sync `def` + `Depends`), pytest 7.0.1.

## Global Constraints

- **Python 3.6.9 target.** No 3.7+ syntax. Plain f-strings OK; no `f"{x=}"`.
- **Pydantic v1 only.** `BaseSettings` from `pydantic`; nested `class Config:`; `orm_mode = True`.
- **SQLAlchemy 1.3 only.** `db.query(Model).filter(...)`. No `select()`.
- **Sync `def` endpoints.**
- **SQLite cross-thread:** engine needs `connect_args={"check_same_thread": False}` (the pipeline threads in Stage 3 depend on it).
- **No secrets in git.** `.env` is gitignored; `.env.example` holds placeholders.
- **Split execution:** **[LAPTOP — Claude]** writes/commits/pushes; **[JETSON — student]** pulls and runs. Claude cannot run on the Jetson. Paste output back at each **CHECKPOINT**.
- **Commit trailer:** `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## File structure (what this stage creates)

- `car_logger/config.py` — Pydantic `Settings`, loaded from `.env`.
- `car_logger/database.py` — engine, `SessionLocal`, `Base`, `get_db` dependency.
- `car_logger/models.py` — `Vehicle` and `Event` ORM models.
- `car_logger/schemas.py` — `EventCreate`, `EventRead`, `VehicleRead`.
- `car_logger/repositories.py` — `create_event`, `get_event`, `list_events`.
- `car_logger/api/__init__.py`, `car_logger/api/routes_events.py` — the `/api/events` router.
- `car_logger/main.py` — modified to `include_router`.
- `.env.example` — placeholder settings.
- `alembic.ini`, `alembic/env.py`, `alembic/versions/*` — migration.
- `tests/conftest.py`, `tests/unit/test_config.py`, `tests/unit/test_repositories.py`, `tests/integration/test_api_events.py`.

---

### Task 1: Configuration (`config.py`)

**Files:**
- Create: `car_logger/config.py`
- Create: `.env.example`
- Test: `tests/unit/test_config.py`
- Create: `tests/__init__.py`, `tests/unit/__init__.py` (empty package markers)

**Interfaces:**
- Produces: `settings` — a `Settings` instance with fields `database_url: str`, `anpr_api_key: str`, `anpr_api_url: str`, `log_level: str`, `max_pipeline_fps: int`. Every later module imports `from car_logger.config import settings`.

- [ ] **Step 1: Write the config module** **[LAPTOP — Claude]**

`car_logger/config.py`:
```python
"""Application settings, loaded from environment / .env (Pydantic v1)."""

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Central config. Field names map to env vars case-insensitively,
    so `anpr_api_key` is filled from ANPR_API_KEY in .env."""

    database_url: str = "sqlite:///./car_logger.db"
    anpr_api_key: str = ""
    anpr_api_url: str = "https://api.platerecognizer.com/v1/plate-reader/"
    log_level: str = "INFO"
    max_pipeline_fps: int = 15

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
```

- [ ] **Step 2: Write the example env file** **[LAPTOP — Claude]**

`.env.example`:
```
# Copy to .env (gitignored) and fill in real values.
DATABASE_URL=sqlite:///./car_logger.db
ANPR_API_KEY=your-plate-recognizer-token-here
ANPR_API_URL=https://api.platerecognizer.com/v1/plate-reader/
LOG_LEVEL=INFO
MAX_PIPELINE_FPS=15
```

- [ ] **Step 3: Write the test** **[LAPTOP — Claude]**

`tests/__init__.py`, `tests/unit/__init__.py`: empty files.

`tests/unit/test_config.py`:
```python
from car_logger.config import Settings


def test_defaults_are_sane():
    s = Settings()
    assert s.database_url.startswith("sqlite")
    assert s.log_level == "INFO"
    assert s.max_pipeline_fps == 15


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("MAX_PIPELINE_FPS", "8")
    monkeypatch.setenv("ANPR_API_KEY", "test-token")
    s = Settings()
    assert s.max_pipeline_fps == 8
    assert s.anpr_api_key == "test-token"
```

- [ ] **Step 4: Commit and push** **[LAPTOP — Claude]**

```bash
git add car_logger/config.py .env.example tests/__init__.py tests/unit/__init__.py tests/unit/test_config.py
git commit -m "feat(config): pydantic settings loaded from .env

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

- [ ] **Step 5: Run the config tests** **[JETSON — student]**

```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -m pytest tests/unit/test_config.py -v
```
Expected: `2 passed`.

**CHECKPOINT:** paste the pytest output back before Task 2.

---

### Task 2: Database module (`database.py`)

**Files:**
- Create: `car_logger/database.py`

**Interfaces:**
- Consumes: `settings.database_url`.
- Produces: `engine`, `SessionLocal` (session factory), `Base` (declarative base every model subclasses), `get_db()` (FastAPI dependency yielding a session and closing it).

- [ ] **Step 1: Write the database module** **[LAPTOP — Claude]**

`car_logger/database.py`:
```python
"""SQLAlchemy engine, session factory, and declarative Base."""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from car_logger.config import settings

# check_same_thread=False: in Stage 3 the camera/pipeline run in background
# threads that also write to SQLite. SQLite otherwise refuses a connection
# used from a thread other than the one that created it. This is safe here
# because we never share a single Session across threads — each thread pulls
# its own from SessionLocal.
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()


def get_db():
    """FastAPI dependency: hand out a session, always close it afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

- [ ] **Step 2: Commit and push** **[LAPTOP — Claude]**

```bash
git add car_logger/database.py
git commit -m "feat(db): sqlalchemy engine, session factory, get_db dependency

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

- [ ] **Step 3: Smoke-check the import** **[JETSON — student]**

```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -c "from car_logger.database import engine, SessionLocal, Base, get_db; print('db ok', engine.url)"
```
Expected: `db ok sqlite:///./car_logger.db`.

**CHECKPOINT:** paste the output back before Task 3.

---

### Task 3: ORM models (`models.py`)

**Files:**
- Create: `car_logger/models.py`

**Interfaces:**
- Consumes: `Base` from `car_logger.database`.
- Produces: `Vehicle` and `Event` ORM classes. `Event` columns: `id, timestamp, vehicle_id, plate_text, plate_confidence, anpr_status, bbox_json, image_path, track_id`. `Vehicle` columns: `id, plate_text, first_seen_at, last_seen_at, total_sightings, notes`. These column names are the contract for schemas (Task 4) and repositories (Task 5).

- [ ] **Step 1: Write the models** **[LAPTOP — Claude]**

`car_logger/models.py`:
```python
"""ORM models: Vehicle (a unique plate) and Event (each detection sighting)."""

from datetime import datetime

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, String, Text
)
from sqlalchemy.orm import relationship

from car_logger.database import Base


class Vehicle(Base):
    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, index=True)
    plate_text = Column(String, unique=True, nullable=False, index=True)
    first_seen_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    total_sightings = Column(Integer, default=0, nullable=False)
    notes = Column(Text, nullable=True)

    events = relationship("Event", back_populates="vehicle")

    def __repr__(self):
        return "<Vehicle id={0} plate={1!r} sightings={2}>".format(
            self.id, self.plate_text, self.total_sightings
        )


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False,
                       index=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=True)
    # plate_text is denormalized onto the event for fast listing/filtering.
    plate_text = Column(String, nullable=True, index=True)
    plate_confidence = Column(Float, nullable=True)
    # anpr_status: pending | success | failed | skipped | throttled
    anpr_status = Column(String, nullable=False, default="pending")
    bbox_json = Column(Text, nullable=True)
    image_path = Column(String, nullable=True)
    track_id = Column(Integer, nullable=True)

    vehicle = relationship("Vehicle", back_populates="events")

    def __repr__(self):
        return "<Event id={0} plate={1!r} status={2}>".format(
            self.id, self.plate_text, self.anpr_status
        )
```

- [ ] **Step 2: Commit and push** **[LAPTOP — Claude]**

```bash
git add car_logger/models.py
git commit -m "feat(models): Vehicle and Event ORM models with indexes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

- [ ] **Step 3: Verify tables register on the metadata** **[JETSON — student]**

```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -c "from car_logger.database import Base; import car_logger.models; print(sorted(Base.metadata.tables.keys()))"
```
Expected: `['events', 'vehicles']`.

**CHECKPOINT:** paste the output back before Task 4.

---

### Task 4: Pydantic schemas (`schemas.py`)

**Files:**
- Create: `car_logger/schemas.py`

**Interfaces:**
- Produces: `EventCreate` (request body — everything optional except `anpr_status` default), `EventRead` (response, `orm_mode`), `VehicleRead` (response, `orm_mode`). Repositories (Task 5) accept `EventCreate`; routes (Task 6) respond with `EventRead`/`VehicleRead`.

- [ ] **Step 1: Write the schemas** **[LAPTOP — Claude]**

`car_logger/schemas.py`:
```python
"""Pydantic v1 DTOs — the API boundary, kept separate from the ORM models.

Why separate from models.py? The ORM class is about *persistence* (columns,
relationships, indexes). These schemas are about the *API contract* (what a
client may send, what we promise to return). Keeping them apart means the DB
can change without silently changing the public API, and vice versa.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class EventCreate(BaseModel):
    plate_text: Optional[str] = None
    plate_confidence: Optional[float] = None
    anpr_status: str = "pending"
    bbox_json: Optional[str] = None
    image_path: Optional[str] = None
    track_id: Optional[int] = None
    vehicle_id: Optional[int] = None


class EventRead(BaseModel):
    id: int
    timestamp: datetime
    vehicle_id: Optional[int]
    plate_text: Optional[str]
    plate_confidence: Optional[float]
    anpr_status: str
    bbox_json: Optional[str]
    image_path: Optional[str]
    track_id: Optional[int]

    class Config:
        orm_mode = True


class VehicleRead(BaseModel):
    id: int
    plate_text: str
    first_seen_at: datetime
    last_seen_at: datetime
    total_sightings: int
    notes: Optional[str]

    class Config:
        orm_mode = True
```

- [ ] **Step 2: Commit and push** **[LAPTOP — Claude]**

```bash
git add car_logger/schemas.py
git commit -m "feat(schemas): pydantic v1 DTOs for events and vehicles

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

- [ ] **Step 3: Verify the schemas import** **[JETSON — student]**

```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -c "from car_logger.schemas import EventCreate, EventRead, VehicleRead; print(EventCreate(anpr_status='pending').dict())"
```
Expected: a dict with all fields, `anpr_status='pending'`, the rest `None`.

**CHECKPOINT:** paste the output back before Task 5.

---

### Task 5: Repositories + unit tests (test-first)

**Files:**
- Create: `car_logger/repositories.py`
- Create: `tests/conftest.py`
- Test: `tests/unit/test_repositories.py`

**Interfaces:**
- Consumes: `Event` (models), `EventCreate` (schemas), a `Session`.
- Produces:
  - `create_event(db, event: EventCreate) -> Event`
  - `get_event(db, event_id: int) -> Optional[Event]` (returns `None` if not found — the **student's decision**: return None, let the route raise 404)
  - `list_events(db, skip=0, limit=50, plate_text=None) -> List[Event]` (newest first; `limit` capped at `MAX_LIST_LIMIT = 100` — **student's decision**)

- [ ] **Step 1: Write the in-memory DB fixture** **[LAPTOP — Claude]**

`tests/conftest.py`:
```python
"""Test fixtures: an isolated in-memory SQLite session + a TestClient whose
get_db dependency is overridden to use that same session."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from fastapi.testclient import TestClient

from car_logger.database import Base, get_db
import car_logger.models  # noqa: F401  (registers tables on Base.metadata)
from car_logger.main import app


@pytest.fixture
def db_session():
    # StaticPool + a single shared in-memory connection so every query in a
    # test sees the same database.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    testing_session = sessionmaker(bind=engine, autoflush=False,
                                   autocommit=False)
    session = testing_session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session):
    # Note: TestClient is NOT used as a context manager, so app startup/shutdown
    # events (which start the camera in Stage 3) do NOT fire during these tests.
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()
```

- [ ] **Step 2: Write the failing repository tests** **[LAPTOP — Claude]**

`tests/unit/test_repositories.py`:
```python
from car_logger import repositories, schemas


def _make(plate=None, status="pending"):
    return schemas.EventCreate(plate_text=plate, anpr_status=status)


def test_create_and_get_event(db_session):
    created = repositories.create_event(db_session, _make(plate="B123XYZ"))
    assert created.id is not None
    fetched = repositories.get_event(db_session, created.id)
    assert fetched.plate_text == "B123XYZ"


def test_get_missing_event_returns_none(db_session):
    assert repositories.get_event(db_session, 999) is None


def test_list_events_empty(db_session):
    assert repositories.list_events(db_session) == []


def test_list_events_newest_first(db_session):
    a = repositories.create_event(db_session, _make(plate="AAA"))
    b = repositories.create_event(db_session, _make(plate="BBB"))
    rows = repositories.list_events(db_session)
    assert [r.id for r in rows] == [b.id, a.id]


def test_list_events_plate_filter_is_partial(db_session):
    repositories.create_event(db_session, _make(plate="B123XYZ"))
    repositories.create_event(db_session, _make(plate="CJ99ABC"))
    rows = repositories.list_events(db_session, plate_text="123")
    assert len(rows) == 1
    assert rows[0].plate_text == "B123XYZ"


def test_list_events_caps_limit(db_session):
    for i in range(5):
        repositories.create_event(db_session, _make(plate="P" + str(i)))
    rows = repositories.list_events(db_session, limit=1000)
    assert len(rows) == 5  # all 5 returned, but limit was capped, not errored
    assert repositories.MAX_LIST_LIMIT == 100
```

- [ ] **Step 3: Commit tests, push, confirm RED** **[LAPTOP — Claude then JETSON — student]**

```bash
git add tests/conftest.py tests/unit/test_repositories.py
git commit -m "test(repositories): failing tests for event CRUD

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
Then **[JETSON — student]**:
```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -m pytest tests/unit/test_repositories.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'car_logger.repositories'`.

- [ ] **Step 4: Write the repository implementation** **[LAPTOP — Claude]**

`car_logger/repositories.py`:
```python
"""Data access layer — every DB query lives here, not in the API routes.

Separating this from the routes means the query logic is testable without HTTP
and reusable from the pipeline (Stage 3) which has no request at all."""

from typing import List, Optional

from sqlalchemy.orm import Session

from car_logger import schemas
from car_logger.models import Event

# Student's decision: hard ceiling on how many rows one list call returns, so a
# client can never ask for the whole table at once.
MAX_LIST_LIMIT = 100


def create_event(db: Session, event: schemas.EventCreate) -> Event:
    db_event = Event(**event.dict())
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    return db_event


def get_event(db: Session, event_id: int) -> Optional[Event]:
    """Return the Event, or None if the id doesn't exist (route raises 404)."""
    return db.query(Event).filter(Event.id == event_id).first()


def list_events(db: Session, skip: int = 0, limit: int = 50,
                plate_text: Optional[str] = None) -> List[Event]:
    capped = min(limit, MAX_LIST_LIMIT)
    query = db.query(Event)
    if plate_text:
        query = query.filter(Event.plate_text.like("%" + plate_text + "%"))
    return (query.order_by(Event.timestamp.desc(), Event.id.desc())
                 .offset(skip)
                 .limit(capped)
                 .all())
```

- [ ] **Step 5: Commit, push, confirm GREEN** **[LAPTOP — Claude then JETSON — student]**

```bash
git add car_logger/repositories.py
git commit -m "feat(repositories): create/get/list events with plate filter

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
Then **[JETSON — student]**:
```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -m pytest tests/unit/test_repositories.py -v
```
Expected: `6 passed`.

**CHECKPOINT:** paste the pytest output back before Task 6.

---

### Task 6: API routes + wire into `main.py` + integration tests

**Files:**
- Create: `car_logger/api/__init__.py` (empty)
- Create: `car_logger/api/routes_events.py`
- Modify: `car_logger/main.py`
- Test: `tests/integration/test_api_events.py`
- Create: `tests/integration/__init__.py` (empty)

**Interfaces:**
- Consumes: `repositories`, `schemas`, `get_db`.
- Produces HTTP endpoints:
  - `POST /api/events` → 200, body `EventRead`
  - `GET /api/events?skip=&limit=&plate=` → 200, `List[EventRead]` (`limit` max 100)
  - `GET /api/events/{id}` → 200 `EventRead` or 404

- [ ] **Step 1: Write the failing integration tests** **[LAPTOP — Claude]**

`tests/integration/__init__.py`: empty file.

`tests/integration/test_api_events.py`:
```python
def test_create_event_returns_id(client):
    resp = client.post("/api/events", json={"plate_text": "B123XYZ"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] > 0
    assert body["plate_text"] == "B123XYZ"
    assert body["anpr_status"] == "pending"


def test_get_event_roundtrip(client):
    created = client.post("/api/events", json={"plate_text": "CJ01AAA"}).json()
    resp = client.get("/api/events/" + str(created["id"]))
    assert resp.status_code == 200
    assert resp.json()["plate_text"] == "CJ01AAA"


def test_get_missing_event_is_404(client):
    resp = client.get("/api/events/9999")
    assert resp.status_code == 404


def test_list_events_empty(client):
    resp = client.get("/api/events")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_events_plate_filter(client):
    client.post("/api/events", json={"plate_text": "B123XYZ"})
    client.post("/api/events", json={"plate_text": "CJ99ABC"})
    resp = client.get("/api/events", params={"plate": "123"})
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_list_events_limit_over_100_rejected(client):
    resp = client.get("/api/events", params={"limit": 500})
    assert resp.status_code == 422  # Query(le=100) enforces the ceiling
```

- [ ] **Step 2: Commit tests, push, confirm RED** **[LAPTOP — Claude then JETSON — student]**

```bash
git add tests/integration/__init__.py tests/integration/test_api_events.py
git commit -m "test(api): failing tests for /api/events endpoints

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
Then **[JETSON — student]**:
```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -m pytest tests/integration/test_api_events.py -v
```
Expected: FAIL — routes return 404 (no `/api/events` yet).

- [ ] **Step 3: Write the events router** **[LAPTOP — Claude]**

`car_logger/api/__init__.py`: empty file.

`car_logger/api/routes_events.py`:
```python
"""/api/events endpoints — a thin HTTP layer over the repository."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from car_logger import repositories, schemas
from car_logger.database import get_db

router = APIRouter(prefix="/api/events", tags=["events"])


@router.post("", response_model=schemas.EventRead)
def create_event(event: schemas.EventCreate, db: Session = Depends(get_db)):
    return repositories.create_event(db, event)


@router.get("", response_model=List[schemas.EventRead])
def list_events(skip: int = 0,
                limit: int = Query(50, ge=1, le=100),
                plate: Optional[str] = None,
                db: Session = Depends(get_db)):
    return repositories.list_events(db, skip=skip, limit=limit,
                                    plate_text=plate)


@router.get("/{event_id}", response_model=schemas.EventRead)
def get_event(event_id: int, db: Session = Depends(get_db)):
    event = repositories.get_event(db, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return event
```

- [ ] **Step 4: Wire the router into `main.py`** **[LAPTOP — Claude]**

Modify `car_logger/main.py` to include the router (keep the existing `/` and `/health`):
```python
"""Car Logger API entrypoint - the app object everything else attaches to."""

from fastapi import FastAPI

from car_logger.api.routes_events import router as events_router

APP_VERSION = "0.2.0"

app = FastAPI(title="Car Logger", version=APP_VERSION)

app.include_router(events_router)


@app.get("/")
def root():
    """Greeting endpoint - proves the server is reachable from the LAN."""
    return {"message": "Car Logger is running", "version": APP_VERSION}


@app.get("/health")
def health():
    """Liveness probe - used later by systemd and monitoring."""
    return {"status": "ok"}
```

> Note: the Stage 1 test asserts `version == "0.1.0"`. Bumping to `"0.2.0"` here means `tests/test_main.py` must update too — do it in the same commit.

Modify `tests/test_main.py` line asserting the version:
```python
    assert body["version"] == "0.2.0"
```

- [ ] **Step 5: Commit, push, confirm GREEN (full suite)** **[LAPTOP — Claude then JETSON — student]**

```bash
git add car_logger/api/__init__.py car_logger/api/routes_events.py car_logger/main.py tests/test_main.py
git commit -m "feat(api): /api/events router wired into the app

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
Then **[JETSON — student]**:
```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
python3 -m pytest tests/ -v
```
Expected: all pass — Stage 1 (2) + config (2) + repositories (6) + api (6) = **16 passed**.

**CHECKPOINT:** paste the full pytest output back before Task 7.

---

### Task 7: Alembic migration (real DB schema)

**Files:**
- Create: `alembic.ini`, `alembic/env.py`, `alembic/versions/` (via `alembic init`, then edited)
- Modify: `.gitignore` if `car_logger.db` isn't already ignored

**Interfaces:**
- Consumes: `Base.metadata` (all models).
- Produces: a migration that creates `vehicles` and `events`; `alembic upgrade head` builds the real on-disk DB used by the running server.

- [ ] **Step 1: Initialise Alembic** **[JETSON — student]**

```bash
cd ~/jetson-car-logger && source venv/bin/activate
alembic init alembic
```
This creates `alembic.ini` and `alembic/`. Paste back the file list.

- [ ] **Step 2: Point Alembic at our models** **[LAPTOP — Claude]** *(after student pastes the generated `env.py`)*

Edit `alembic/env.py` — replace `target_metadata = None` with our metadata, and make the URL come from settings. Add near the top (after the existing imports):
```python
from car_logger.config import settings
from car_logger.database import Base
import car_logger.models  # noqa: F401  (imports register the tables)

target_metadata = Base.metadata
```
And set the URL programmatically so `.env` is the single source of truth. In both `run_migrations_offline()` and `run_migrations_online()`, before the migration runs, override the config URL:
```python
    config.set_main_option("sqlalchemy.url", settings.database_url)
```
(Place that line right after `config = context.config` is available in each function, or once at module import after `config = context.config`.)

- [ ] **Step 3: Ensure the app can find `car_logger`** **[LAPTOP — Claude]**

At the very top of `alembic/env.py`, guarantee the project root is importable:
```python
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

- [ ] **Step 4: Commit and push the alembic config** **[LAPTOP — Claude]**

```bash
git add alembic.ini alembic/env.py alembic/script.py.mako
git commit -m "build(alembic): wire migrations to car_logger models and settings

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```

- [ ] **Step 5: Autogenerate and apply the initial migration** **[JETSON — student]**

```bash
cd ~/jetson-car-logger && source venv/bin/activate && git pull
alembic revision --autogenerate -m "initial schema: vehicles and events"
alembic upgrade head
sqlite3 car_logger.db ".schema"
```
Expected: `.schema` shows `CREATE TABLE vehicles (...)` and `CREATE TABLE events (...)` with the indexes.

- [ ] **Step 6: Commit the generated migration file** **[JETSON — student]** then **[LAPTOP — Claude]** reviews

```bash
git add alembic/versions/*.py
git commit -m "build(alembic): initial schema migration

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push
```
(The student commits from the Jetson since the file was generated there; Claude reviews the diff on next pull.)

- [ ] **Step 7: Verify downgrade/upgrade round-trips** **[JETSON — student]**

```bash
alembic downgrade base && alembic upgrade head
```
Expected: both run clean, no errors.

**CHECKPOINT:** paste `.schema` output + the downgrade/upgrade result. Stage 2 is done when the full suite is green and this round-trips.

---

### Task 8: End-to-end manual verification via Swagger

**Files:** none.

- [ ] **Step 1: Run the server** **[JETSON — student]**

```bash
cd ~/jetson-car-logger && source venv/bin/activate
uvicorn car_logger.main:app --host 0.0.0.0 --port 8000 --reload
```

- [ ] **Step 2: Exercise the API from the laptop browser** **[LAPTOP — student]**

Open `http://192.168.0.232:8000/docs`. Then:
1. `POST /api/events` → "Try it out" → body `{"plate_text": "B123XYZ"}` → Execute → expect 200 with an `id`.
2. `GET /api/events/{id}` with that id → expect the same event.
3. `GET /api/events?plate=123` → expect the event in the list.
4. `GET /api/events/9999` → expect 404.

- [ ] **Step 3: Confirm the row hit disk** **[JETSON — student]**

```bash
sqlite3 car_logger.db "SELECT id, plate_text, anpr_status FROM events;"
```
Expected: your event row(s).

**CHECKPOINT:** confirm all four Swagger calls behaved. Stage 2 code is done.

---

## Self-Review

**1. Spec coverage** (against `PLAN.md` Week 2 + `CLAUDE.md` data model):
- Config via `pydantic.BaseSettings` v1 from `.env`, incl. `max_pipeline_fps`: Task 1. ✓
- `database.py` with engine/SessionLocal/Base/get_db + `check_same_thread`: Task 2. ✓
- Vehicle + Event ORM with indexes on `events.timestamp`, `vehicles.plate_text`, `__repr__`: Task 3. ✓
- Pydantic schemas with `orm_mode`: Task 4. ✓
- Repositories: create/get/list, `get_event`→None decision, `limit` cap decision: Task 5. ✓
- 3 event endpoints via `Depends`, `limit` max 100: Task 6. ✓
- Alembic init/env wiring/autogenerate/upgrade + downgrade round-trip: Task 7. ✓
- ≥ 8 tests: 2 config + 6 repo + 6 api = 14 new (+2 Stage 1). ✓
- Swagger POST→GET, `sqlite3 SELECT` verification: Task 8. ✓

**2. Placeholder scan:** every code step shows complete code; no TBD/TODO. ✓

**3. Type consistency:** `create_event`/`get_event`/`list_events` signatures identical in Task 5 impl, its tests, and the router (Task 6). `EventCreate`/`EventRead` fields match model columns. `MAX_LIST_LIMIT = 100` matches the router's `Query(le=100)`. `settings` field names match `.env.example` keys. ✓

## Notes for the executor

- **Human-in-the-loop checkpoints.** Do not proceed past a CHECKPOINT without the student's pasted output.
- The Jetson venv is `--system-site-packages` and already exists — do not recreate it.
- If `alembic revision --autogenerate` produces an empty migration, the models weren't imported in `env.py` — re-check Task 7 Step 2.
- Business-logic decisions in this stage (the `get_event`→None-vs-raise choice, the `MAX_LIST_LIMIT` value) are the student's: the plan ships defaults; confirm or change them deliberately.
