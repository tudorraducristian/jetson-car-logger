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
