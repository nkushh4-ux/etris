from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.base import Base


class CameraRecord(Base):
    __tablename__ = "cameras"
    camera_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(160))
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    road_name: Mapped[str | None] = mapped_column(String(160))
    sector: Mapped[str | None] = mapped_column(String(100))
    direction: Mapped[str | None] = mapped_column(String(60))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class VehicleSightingRecord(Base):
    __tablename__ = "vehicle_sightings"
    __table_args__ = (
        UniqueConstraint("sighting_id", name="uq_vehicle_sighting_event"),
        Index("ix_sightings_plate_observed", "plate_text", "observed_at"),
        Index("ix_sightings_camera_observed", "camera_id", "observed_at"),
    )
    sighting_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    plate_text: Mapped[str] = mapped_column(String(32), index=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.camera_id"), index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    plate_confidence: Mapped[float] = mapped_column(Float)
    source_track_id: Mapped[int | None] = mapped_column(Integer)
    direction: Mapped[str | None] = mapped_column(String(60))
    status: Mapped[str] = mapped_column(String(32))
    frame_id: Mapped[int | None] = mapped_column(Integer)
    source_type: Mapped[str | None] = mapped_column(String(80))
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class WatchlistRecord(Base):
    __tablename__="vehicle_watchlist"
    watchlist_id:Mapped[str]=mapped_column(String(80),primary_key=True)
    plate_text:Mapped[str]=mapped_column(String(32),index=True)
    reason:Mapped[str]=mapped_column(String(300))
    severity:Mapped[str]=mapped_column(String(16))
    source:Mapped[str]=mapped_column(String(120))
    active:Mapped[bool]=mapped_column(Boolean,default=True,index=True)
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now())
    expires_at:Mapped[datetime|None]=mapped_column(DateTime(timezone=True))
    notes:Mapped[str|None]=mapped_column(String(1000))
