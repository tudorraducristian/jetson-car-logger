"""GET /api/status — pipeline health for monitoring and the dashboard."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["status"])


@router.get("/status")
def status(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    camera = getattr(request.app.state, "camera", None)
    anpr_worker = getattr(request.app.state, "anpr_worker", None)
    anpr_queue = anpr_worker.pending() if anpr_worker is not None else 0
    if pipeline is None:
        return {
            "pipeline_running": False,
            "fps": 0.0,
            "frames_processed": 0,
            "camera_ok": False,
            "last_event_at": None,
            "anpr_queue": anpr_queue,
        }
    return {
        "pipeline_running": True,
        "fps": round(pipeline.last_fps, 1),
        "frames_processed": pipeline.frames_processed,
        "camera_ok": camera is not None and camera.get_latest_frame() is not None,
        "last_event_at": pipeline.last_event_at,
        "anpr_queue": anpr_queue,
    }
