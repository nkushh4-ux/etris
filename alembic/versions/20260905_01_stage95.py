"""Stage 9.5 cameras and vehicle sightings."""

import sqlalchemy as sa

from alembic import op

revision = "20260905_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # PostGIS is intentionally NOT created here.
    # For the SIH MVP, latitude/longitude are stored as normal FLOAT columns.
    # PostGIS can be added later in a separate migration.

    op.create_table(
        "cameras",
        sa.Column("camera_id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("road_name", sa.String(160), nullable=True),
        sa.Column("sector", sa.String(100), nullable=True),
        sa.Column("direction", sa.String(60), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_table(
        "vehicle_sightings",
        sa.Column("sighting_id", sa.String(80), primary_key=True),
        sa.Column("plate_text", sa.String(32), nullable=False),
        sa.Column(
            "camera_id",
            sa.String(64),
            sa.ForeignKey("cameras.camera_id"),
            nullable=False,
        ),
        sa.Column(
            "observed_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "plate_confidence",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "source_track_id",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "direction",
            sa.String(60),
            nullable=True,
        ),
        sa.Column(
            "status",
            sa.String(32),
            nullable=False,
        ),
        sa.Column(
            "frame_id",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "source_type",
            sa.String(80),
            nullable=True,
        ),
        sa.Column(
            "latitude",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "longitude",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_index(
        "ix_sightings_plate_observed",
        "vehicle_sightings",
        ["plate_text", "observed_at"],
    )

    op.create_index(
        "ix_sightings_camera_observed",
        "vehicle_sightings",
        ["camera_id", "observed_at"],
    )

    op.create_index(
        "ix_vehicle_sightings_plate_text",
        "vehicle_sightings",
        ["plate_text"],
    )

    op.create_index(
        "ix_vehicle_sightings_camera_id",
        "vehicle_sightings",
        ["camera_id"],
    )

    op.create_index(
        "ix_vehicle_sightings_observed_at",
        "vehicle_sightings",
        ["observed_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_vehicle_sightings_observed_at",
        table_name="vehicle_sightings",
    )

    op.drop_index(
        "ix_vehicle_sightings_camera_id",
        table_name="vehicle_sightings",
    )

    op.drop_index(
        "ix_vehicle_sightings_plate_text",
        table_name="vehicle_sightings",
    )

    op.drop_index(
        "ix_sightings_camera_observed",
        table_name="vehicle_sightings",
    )

    op.drop_index(
        "ix_sightings_plate_observed",
        table_name="vehicle_sightings",
    )

    op.drop_table("vehicle_sightings")
    op.drop_table("cameras")