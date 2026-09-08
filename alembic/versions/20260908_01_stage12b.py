"""Stage 12B persistent vehicle watchlist."""
import sqlalchemy as sa

from alembic import op

revision="20260908_01"
down_revision="20260905_01"
branch_labels=None
depends_on=None


def upgrade():
    op.create_table("vehicle_watchlist",
        sa.Column("watchlist_id",sa.String(80),primary_key=True),
        sa.Column("plate_text",sa.String(32),nullable=False),
        sa.Column("reason",sa.String(300),nullable=False),
        sa.Column("severity",sa.String(16),nullable=False),
        sa.Column("source",sa.String(120),nullable=False),
        sa.Column("active",sa.Boolean(),nullable=False,server_default=sa.true()),
        sa.Column("created_at",sa.DateTime(timezone=True),nullable=False,server_default=sa.func.now()),
        sa.Column("expires_at",sa.DateTime(timezone=True),nullable=True),
        sa.Column("notes",sa.String(1000),nullable=True))
    op.create_index("ix_vehicle_watchlist_plate_text","vehicle_watchlist",["plate_text"])
    op.create_index("ix_vehicle_watchlist_active","vehicle_watchlist",["active"])


def downgrade():
    op.drop_index("ix_vehicle_watchlist_active",table_name="vehicle_watchlist")
    op.drop_index("ix_vehicle_watchlist_plate_text",table_name="vehicle_watchlist")
    op.drop_table("vehicle_watchlist")
