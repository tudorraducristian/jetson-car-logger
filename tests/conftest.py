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
