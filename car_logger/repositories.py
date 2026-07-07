"""Data access layer — every DB query lives here, not in the API routes.

Separating this from the routes means the query logic is testable without HTTP
and reusable from the pipeline (Stage 3) which has no request at all."""

from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy.orm import Session

from car_logger import schemas
from car_logger.models import Event, Vehicle

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


def get_or_create_vehicle(db: Session, plate_text: str) -> Vehicle:
    """Return the Vehicle for this plate, creating it on first sighting.

    Does not commit — the caller decides the transaction boundary, so the
    vehicle and the event update land in ONE commit (or neither does)."""
    vehicle = (db.query(Vehicle)
                 .filter(Vehicle.plate_text == plate_text)
                 .first())
    if vehicle is None:
        vehicle = Vehicle(plate_text=plate_text, total_sightings=0)
        db.add(vehicle)
        db.flush()  # assigns vehicle.id without committing
    return vehicle


def update_event_anpr(db: Session, event_id: int, status: str,
                      plate_text: Optional[str] = None,
                      confidence: Optional[float] = None,
                      image_path: Optional[str] = None) -> Optional[Event]:
    """Write an ANPR outcome onto an event; on success also upsert the Vehicle.

    Returns the updated Event, or None if the id vanished (shouldn't happen,
    but the worker must not crash on it)."""
    event = get_event(db, event_id)
    if event is None:
        return None

    event.anpr_status = status
    if image_path is not None:
        event.image_path = image_path

    if status == "success" and plate_text:
        event.plate_text = plate_text
        event.plate_confidence = confidence
        vehicle = get_or_create_vehicle(db, plate_text)
        vehicle.total_sightings += 1
        vehicle.last_seen_at = event.timestamp or datetime.utcnow()
        event.vehicle_id = vehicle.id

    db.commit()
    db.refresh(event)
    return event


def list_vehicles(db: Session, skip: int = 0,
                  limit: int = 50) -> List[Vehicle]:
    capped = min(limit, MAX_LIST_LIMIT)
    return (db.query(Vehicle)
              .order_by(Vehicle.last_seen_at.desc())
              .offset(skip)
              .limit(capped)
              .all())


def get_stats(db: Session) -> dict:
    """Counters for the dashboard stats panel — one cheap query each."""
    day_ago = datetime.utcnow() - timedelta(hours=24)
    return {
        "total_events": db.query(Event).count(),
        "events_last_24h": (db.query(Event)
                              .filter(Event.timestamp >= day_ago)
                              .count()),
        "total_vehicles": db.query(Vehicle).count(),
        "plates_read": (db.query(Event)
                          .filter(Event.anpr_status == "success")
                          .count()),
    }
