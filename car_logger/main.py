"""Car Logger API entrypoint - the app object everything else attaches to."""

import json
import os
import time

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from car_logger.api.routes_dashboard import router as dashboard_router
from car_logger.api.routes_events import router as events_router
from car_logger.api.routes_status import router as status_router
from car_logger.api.routes_stream import router as stream_router
from car_logger.config import settings
from car_logger.database import SessionLocal
from car_logger.logging_config import configure_logging, get_logger
from car_logger.services.broker import EventBroker
from car_logger import repositories, schemas

configure_logging(settings.log_level)
log = get_logger("car_logger")

APP_VERSION = "0.5.0"

PLATES_DIR = "data/plates"
CROP_RETENTION_DAYS = 30  # student decision: old crops are disk noise after a month

app = FastAPI(title="Car Logger", version=APP_VERSION)

broker = EventBroker()

app.include_router(events_router)
app.include_router(status_router)
app.include_router(dashboard_router)
app.include_router(stream_router)

# The dashboard loads crops straight from disk: /data/plates/<event_id>.jpg
os.makedirs(PLATES_DIR, exist_ok=True)
app.mount("/data/plates", StaticFiles(directory=PLATES_DIR), name="plates")


def _cleanup_old_crops(plates_dir=PLATES_DIR, max_age_days=CROP_RETENTION_DAYS):
    """Delete stored plate crops older than the retention window.

    Runs once at startup: the SD card, not the DB, is the scarce resource. A
    failed delete must never stop the app from starting."""
    if not os.path.isdir(plates_dir):
        return 0
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    for name in os.listdir(plates_dir):
        path = os.path.join(plates_dir, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
        except OSError:
            pass
    return removed


def _make_on_result(broker):
    """Build the ANPR result callback: save crop, update event, upsert vehicle.

    Student amendment (2026-07-07): the crop is saved for EVERY outcome, not
    only success — a failed read's image is the debugging evidence."""
    def on_result(event_id, plate_result, crop_bytes):
        db = SessionLocal()
        try:
            image_path = None
            if crop_bytes:
                os.makedirs(PLATES_DIR, exist_ok=True)
                image_path = os.path.join(PLATES_DIR, str(event_id) + ".jpg")
                with open(image_path, "wb") as fh:
                    fh.write(crop_bytes)
            if plate_result.status == "success" and plate_result.plate_text:
                vehicle = repositories.upsert_vehicle_for_plate(
                    db, plate_result.plate_text
                )
                repositories.update_event_anpr(
                    db, event_id, plate_result.plate_text,
                    plate_result.confidence, "success", image_path, vehicle.id,
                )
            else:
                repositories.update_event_anpr(
                    db, event_id, None, None, plate_result.status, image_path,
                )
            broker.publish("updated")
        finally:
            db.close()
    return on_result


@app.on_event("startup")
def _startup():
    _cleanup_old_crops()
    # Before the early return: /stream/events needs the broker even when the
    # pipeline is disabled (tests, laptop dev-runs without camera).
    app.state.broker = broker
    if not settings.enable_pipeline:
        return
    # Imported here (not at module top) so importing main.py without a camera
    # (e.g. test collection) never needs cv2/jetson at import time.
    from car_logger.services.capture import CameraWorker
    from car_logger.services.detector import Detector
    from car_logger.services.tracker import IoUTracker
    from car_logger.services.pipeline import PipelineWorker
    from car_logger.services.anpr_client import AnprClient
    from car_logger.services.anpr_worker import AnprWorker
    from car_logger.services.cropping import crop_to_jpeg

    camera = CameraWorker(device_index=settings.camera_index)
    camera.start()

    anpr_client = AnprClient(settings.anpr_api_url, settings.anpr_api_key)
    anpr_worker = AnprWorker(anpr_client, _make_on_result(app.state.broker))
    anpr_worker.start()

    def on_confirmed(track, frame):
        # 1) persist a pending event to get its id
        db = SessionLocal()
        try:
            event = repositories.create_event(db, schemas.EventCreate(
                bbox_json=json.dumps(list(track.box)),
                track_id=track.track_id,
                anpr_status="pending",
            ))
            event_id = event.id
        finally:
            db.close()
        app.state.broker.publish("created")
        # 2) crop and hand off to ANPR — pipeline does NOT wait for the network
        crop_bytes = crop_to_jpeg(frame, track.box)
        submitted = anpr_worker.submit(event_id, crop_bytes)
        if not submitted:
            db2 = SessionLocal()
            try:
                repositories.update_event_anpr(
                    db2, event_id, None, None, "skipped", None,
                )
            finally:
                db2.close()

    pipeline = PipelineWorker(
        camera=camera,
        detector=Detector(threshold=settings.detector_threshold),
        tracker=IoUTracker(),
        on_confirmed=on_confirmed,
        target_fps=settings.max_pipeline_fps,
    )
    pipeline.start()
    app.state.camera = camera
    app.state.pipeline = pipeline
    app.state.anpr_worker = anpr_worker
    log.info("pipeline_started", target_fps=settings.max_pipeline_fps)


@app.on_event("shutdown")
def _shutdown():
    # Stop order matters: pipeline first (no new submits), then the ANPR
    # worker, then the camera it was reading from.
    for name in ("pipeline", "anpr_worker", "camera"):
        worker = getattr(app.state, name, None)
        if worker is not None:
            worker.stop()
    log.info("app_shutdown")


@app.get("/health")
def health():
    """Liveness probe - used later by systemd and monitoring."""
    return {"status": "ok"}
