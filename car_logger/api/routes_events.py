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
