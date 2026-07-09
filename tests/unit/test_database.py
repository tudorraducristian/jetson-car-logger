"""SQLite ships with foreign-key enforcement OFF; we turn it on per connection."""

import pytest
from sqlalchemy.exc import IntegrityError

from car_logger.models import Event


def test_sqlite_rejects_bogus_vehicle_fk(db_session):
    db_session.add(Event(vehicle_id=9999, anpr_status="pending"))
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()
