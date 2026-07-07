"""Car Logger API entrypoint - the app object everything else attaches to."""

import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from car_logger.api.routes_dashboard import router as dashboard_router
from car_logger.api.routes_events import router as events_router
from car_logger.api.routes_status import router as status_router
from car_logger.config import settings
from car_logger.database import SessionLocal
from car_logger import repositories, schemas

APP_VERSION = "0.4.0"

app = FastAPI(title="Car Logger", version=APP_VERSION)

app.include_router(events_router)
app.include_router(status_router)
app.include_router(dashboard_router)

# The dashboard loads crops straight from disk: /data/plates/<event_id>.jpg
os.makedirs(settings.plates_dir, exist_ok=True)
app.mount("/data/plates", StaticFiles(directory=settings.plates_dir),
          name="plates")


def _persist_event(event_dict, crop):
    """on_event callback (runs on the pipeline thread): write the event, then
    hand its crop to the ANPR worker. A new session per event keeps thread
    ownership simple; JPEG encoding happens here because it runs per-event
    (a few per minute), not per-frame."""
    db = SessionLocal()
    try:
        event = repositories.create_event(db,
                                          schemas.EventCreate(**event_dict))
    finally:
        db.close()
    worker = getattr(app.state, "anpr_worker", None)
    if worker is None or crop is None:
        return
    import cv2  # local import: keeps main.py importable without OpenCV
    ok, buf = cv2.imencode(".jpg", crop)
    if ok:
        worker.submit(event.id, buf.tobytes())


@app.on_event("startup")
def _startup():
    if not settings.enable_pipeline:
        return
    # Imported here (not at module top) so importing main.py without a camera
    # (e.g. test collection) never needs cv2/jetson at import time.
    from car_logger.services.anpr_client import AnprClient
    from car_logger.services.anpr_worker import AnprWorker, cleanup_old_crops
    from car_logger.services.capture import CameraWorker
    from car_logger.services.detector import Detector
    from car_logger.services.tracker import IoUTracker
    from car_logger.services.pipeline import PipelineWorker

    cleanup_old_crops()  # daily sweep via the Stage 5 scheduled restart

    # ANPR worker first: the pipeline may emit as soon as it starts.
    anpr_worker = AnprWorker(AnprClient(), SessionLocal)
    anpr_worker.start()
    app.state.anpr_worker = anpr_worker

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
    anpr_worker = getattr(app.state, "anpr_worker", None)
    # Producers first, consumer last: stop the pipeline before the worker so
    # nothing submits into a queue nobody drains.
    if pipeline is not None:
        pipeline.stop()
    if camera is not None:
        camera.stop()
    if anpr_worker is not None:
        anpr_worker.stop()


@app.get("/health")
def health():
    """Liveness probe - used later by systemd and monitoring."""
    return {"status": "ok"}
