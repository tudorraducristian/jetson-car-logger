"""Car Logger API entrypoint - the app object everything else attaches to."""

from fastapi import FastAPI

from car_logger.api.routes_events import router as events_router
from car_logger.api.routes_status import router as status_router
from car_logger.config import settings
from car_logger.database import SessionLocal
from car_logger import repositories, schemas

APP_VERSION = "0.3.0"

app = FastAPI(title="Car Logger", version=APP_VERSION)

app.include_router(events_router)
app.include_router(status_router)


def _persist_event(event_dict):
    """on_event callback: open a short-lived session in the pipeline thread and
    write the event. A new session per event keeps thread ownership simple."""
    db = SessionLocal()
    try:
        repositories.create_event(db, schemas.EventCreate(**event_dict))
    finally:
        db.close()


@app.on_event("startup")
def _startup():
    if not settings.enable_pipeline:
        return
    # Imported here (not at module top) so importing main.py without a camera
    # (e.g. test collection) never needs cv2/jetson at import time.
    from car_logger.services.capture import CameraWorker
    from car_logger.services.detector import Detector
    from car_logger.services.tracker import IoUTracker
    from car_logger.services.pipeline import PipelineWorker

    camera = CameraWorker(device_index=settings.camera_index)
    camera.start()
    pipeline = PipelineWorker(
        camera=camera,
        detector=Detector(threshold=settings.detector_threshold),
        tracker=IoUTracker(),
        on_event=_persist_event,
        target_fps=settings.max_pipeline_fps,
    )
    pipeline.start()
    app.state.camera = camera
    app.state.pipeline = pipeline


@app.on_event("shutdown")
def _shutdown():
    pipeline = getattr(app.state, "pipeline", None)
    camera = getattr(app.state, "camera", None)
    if pipeline is not None:
        pipeline.stop()
    if camera is not None:
        camera.stop()


@app.get("/")
def root():
    """Greeting endpoint - proves the server is reachable from the LAN."""
    return {"message": "Car Logger is running", "version": APP_VERSION}


@app.get("/health")
def health():
    """Liveness probe - used later by systemd and monitoring."""
    return {"status": "ok"}
