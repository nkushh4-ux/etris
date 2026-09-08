import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.alerts.blacklisted_vehicle import (
    BlacklistAlertIngestor,
    BlacklistedVehicleAlertHandler,
)
from backend.alerts.repository import InMemoryAlertRepository
from backend.alerts.service import AlertService, CongestionRuntimeIngestor
from backend.alerts.watchlist_repository import (
    InMemoryWatchlistRepository,
    SQLAlchemyWatchlistRepository,
)
from backend.api.alerts import router as alerts_router
from backend.api.analytics import router as analytics_router
from backend.api.anpr import router
from backend.api.signal_control import router as signal_control_router
from backend.api.stage9 import router as stage9_router
from backend.api.watchlist import router as watchlist_router
from backend.database.runtime import build_storage_runtime
from backend.streaming.anpr_stream import ANPRStreamService
from backend.streaming.factory import DEFAULT_VIDEO, production_pipeline_factory


def create_app(*, stream_service=None) -> FastAPI:
    app = FastAPI(title="ETRIS Ultimate AI Demo API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    max_frames_value = os.getenv("ETRIS_ANPR_MAX_FRAMES")
    app.state.storage = build_storage_runtime()
    app.state.alert_repository = InMemoryAlertRepository()
    app.state.alert_service = AlertService(app.state.alert_repository)
    app.state.congestion_ingestor = CongestionRuntimeIngestor(app.state.alert_service)
    app.state.watchlist_repository = (SQLAlchemyWatchlistRepository(app.state.storage.session_factory)
        if app.state.storage.session_factory is not None else InMemoryWatchlistRepository())
    app.state.blacklist_ingestor = BlacklistAlertIngestor(
        BlacklistedVehicleAlertHandler(app.state.alert_repository,app.state.watchlist_repository))
    def completed_sink(completed):
        app.state.storage.ingestor.submit(completed); app.state.blacklist_ingestor.submit(completed)
    app.state.anpr_stream = stream_service or ANPRStreamService(
        Path(os.getenv("ETRIS_ANPR_VIDEO", str(DEFAULT_VIDEO))),
        os.getenv("ETRIS_ANPR_CAMERA_ID", "CAM-DEMO-01"),
        production_pipeline_factory,
        loop=os.getenv("ETRIS_ANPR_LOOP", "1") == "1",
        max_frames=int(max_frames_value) if max_frames_value else None,
        completed_sink=completed_sink,
    )
    app.include_router(router)
    app.include_router(stage9_router)
    app.include_router(analytics_router)
    app.include_router(signal_control_router)
    app.include_router(alerts_router)
    app.include_router(watchlist_router)

    @app.on_event("shutdown")
    def stop_stream() -> None:
        app.state.anpr_stream.stop()
        app.state.storage.ingestor.stop()
        app.state.blacklist_ingestor.stop()
        if app.state.storage.engine is not None:
            app.state.storage.engine.dispose()

    return app


app = create_app()
