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
    # region code the ANPR API detected for the plate ("ro", "cz"…) — used
    # by the identity gate and shown in the detail drawer.
    region = Column(String, nullable=True)
    # anpr_status: pending | success | failed | no_plate | skipped |
    # throttled (throttled: historical rows only — the v1 cloud rate limit)
    anpr_status = Column(String, nullable=False, default="pending")
    bbox_json = Column(Text, nullable=True)
    image_path = Column(String, nullable=True)
    track_id = Column(Integer, nullable=True)

    vehicle = relationship("Vehicle", back_populates="events")

    def __repr__(self):
        return "<Event id={0} plate={1!r} status={2}>".format(
            self.id, self.plate_text, self.anpr_status
        )
