"""normalize plate texts, merge duplicate vehicles, drop orphans

Revision ID: c9d3e58f2b41
Revises: b21c47d1a9e0
Create Date: 2026-07-08
"""
from alembic import op
import sqlalchemy as sa

revision = "c9d3e58f2b41"
down_revision = "b21c47d1a9e0"
branch_labels = None
depends_on = None


def _normalize(text):
    # Deliberate inline copy of plate_rules.normalize_plate: a migration
    # must stay frozen even if the app code changes later.
    if text is None:
        return None
    return text.replace(" ", "").replace("-", "").upper()


def upgrade():
    conn = op.get_bind()

    # 1) normalize event plate texts in place
    for row in list(conn.execute(sa.text(
            "SELECT id, plate_text FROM events "
            "WHERE plate_text IS NOT NULL"))):
        normalized = _normalize(row[1])
        if normalized != row[1]:
            conn.execute(sa.text(
                "UPDATE events SET plate_text = :p WHERE id = :i"),
                {"p": normalized, "i": row[0]})

    # 2) group vehicles by normalized text; merge each group into its
    #    earliest member (events repointed first, so no orphan FKs)
    rows = list(conn.execute(sa.text(
        "SELECT id, plate_text, first_seen_at, last_seen_at, "
        "total_sightings FROM vehicles ORDER BY id")))
    groups = {}
    for vid, plate, first_seen, last_seen, sightings in rows:
        groups.setdefault(_normalize(plate), []).append(
            (vid, plate, first_seen, last_seen, sightings))
    for normalized, members in groups.items():
        survivor = members[0]
        for loser in members[1:]:
            conn.execute(sa.text(
                "UPDATE events SET vehicle_id = :s WHERE vehicle_id = :d"),
                {"s": survivor[0], "d": loser[0]})
            conn.execute(sa.text(
                "UPDATE vehicles SET "
                "total_sightings = total_sightings + :n, "
                "first_seen_at = MIN(first_seen_at, :f), "
                "last_seen_at = MAX(last_seen_at, :l) WHERE id = :s"),
                {"n": loser[4], "f": loser[2], "l": loser[3],
                 "s": survivor[0]})
            conn.execute(sa.text("DELETE FROM vehicles WHERE id = :d"),
                         {"d": loser[0]})
        # rename LAST: every colliding row is gone, so UNIQUE can't fire
        if survivor[1] != normalized:
            conn.execute(sa.text(
                "UPDATE vehicles SET plate_text = :p WHERE id = :i"),
                {"p": normalized, "i": survivor[0]})

    # 3) drop orphan vehicles (zero events) — today's phantoms
    conn.execute(sa.text(
        "DELETE FROM vehicles WHERE id NOT IN "
        "(SELECT DISTINCT vehicle_id FROM events "
        " WHERE vehicle_id IS NOT NULL)"))


def downgrade():
    # Irreversible by design: pre-normalization casing isn't stored
    # anywhere and merged vehicles can't be un-merged (spec). No-op.
    pass
