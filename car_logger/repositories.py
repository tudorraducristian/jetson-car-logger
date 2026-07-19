"""Data access layer — every DB query lives here, not in the API routes.

Separating this from the routes means the query logic is testable without HTTP
and reusable from the pipeline (Stage 3) which has no request at all."""

from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from car_logger import schemas
from car_logger.models import Event, Vehicle

# Student's decision: hard ceiling on how many rows one list call returns, so a
# client can never ask for the whole table at once.
MAX_LIST_LIMIT = 100


def create_event(db: Session, event: schemas.EventCreate) -> Event:
    """Insert a new Event and return it with its generated id."""
    db_event = Event(**event.dict())
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    return db_event


def get_event(db: Session, event_id: int) -> Optional[Event]:
    """Return the Event, or None if the id doesn't exist (route raises 404)."""
    return db.query(Event).filter(Event.id == event_id).first()


def list_events(db: Session, skip: int = 0, limit: int = 50,
                plate_text: Optional[str] = None,
                only_read: bool = False) -> List[Event]:
    """Newest-first page of events; optional plate substring filter.

    only_read narrows to plate-read events (anpr_status == "success") —
    the dashboard's default view since v2 (student decision 2026-07-19:
    filter in the UI, not in the data; everything stays persisted)."""
    capped = min(limit, MAX_LIST_LIMIT)
    query = db.query(Event)
    if plate_text:
        query = query.filter(Event.plate_text.like("%" + plate_text + "%"))
    if only_read:
        query = query.filter(Event.anpr_status == "success")
    return (query.order_by(Event.timestamp.desc(), Event.id.desc())
                 .offset(skip)
                 .limit(capped)
                 .all())


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


def upsert_vehicle_for_plate(db, plate_text):
    """Create the vehicle for this plate or bump its sighting counters."""
    now = datetime.utcnow()
    vehicle = db.query(Vehicle).filter(
        Vehicle.plate_text == plate_text
    ).first()
    if vehicle is None:
        vehicle = Vehicle(plate_text=plate_text, first_seen_at=now,
                          last_seen_at=now, total_sightings=1)
        db.add(vehicle)
    else:
        vehicle.last_seen_at = now
        vehicle.total_sightings += 1
    db.commit()
    db.refresh(vehicle)
    return vehicle


def list_vehicles(db, skip=0, limit=50):
    """Vehicles by most-recent sighting, newest first."""
    capped = min(limit, MAX_LIST_LIMIT)
    return (db.query(Vehicle)
              .order_by(Vehicle.last_seen_at.desc())
              .offset(skip).limit(capped).all())


def event_stats(db):
    """The three dashboard counters: events, plates read, unique vehicles."""
    total = db.query(Event).count()
    plates = db.query(Event).filter(Event.plate_text.isnot(None)).count()
    vehicles = db.query(Vehicle).count()
    return {
        "total_events": total,
        "plates_read": plates,
        "unique_vehicles": vehicles,
    }
