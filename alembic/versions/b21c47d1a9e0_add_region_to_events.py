"""add region column to events

Revision ID: b21c47d1a9e0
Revises: a714d2651be8
Create Date: 2026-07-08
"""
from alembic import op
import sqlalchemy as sa

revision = "b21c47d1a9e0"
down_revision = "a714d2651be8"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("events", sa.Column("region", sa.String(), nullable=True))


def downgrade():
    # batch mode: this SQLite (3.22) has no native DROP COLUMN
    with op.batch_alter_table("events") as batch_op:
        batch_op.drop_column("region")
