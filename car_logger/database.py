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
