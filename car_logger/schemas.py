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
