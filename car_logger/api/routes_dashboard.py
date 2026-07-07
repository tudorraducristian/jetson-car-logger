"""Dashboard routes: the full page plus the htmx partials.

The full page (GET /) is just a skeleton; every panel then pulls its own
fragment from /partials/* on a timer. The same repository functions feed
both the JSON API and these HTML fragments — only the presentation differs.
"""

import os

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from car_logger import repositories
from car_logger.database import get_db

TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

router = APIRouter(tags=["dashboard"], include_in_schema=False)


@router.get("/")
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@router.get("/partials/events-feed")
def events_feed(request: Request, q: str = "", db: Session = Depends(get_db)):
    events = repositories.list_events(db, limit=15, plate_text=(q or None))
    return templates.TemplateResponse(
        "partials/events_feed.html",
        {"request": request, "events": events})


@router.get("/partials/vehicles-list")
def vehicles_list(request: Request, db: Session = Depends(get_db)):
    vehicles = repositories.list_vehicles(db, limit=8)
    return templates.TemplateResponse(
        "partials/vehicles_list.html",
        {"request": request, "vehicles": vehicles})


@router.get("/partials/stats")
def stats(request: Request, db: Session = Depends(get_db)):
    pipeline = getattr(request.app.state, "pipeline", None)
    camera = getattr(request.app.state, "camera", None)
    anpr_worker = getattr(request.app.state, "anpr_worker", None)
    return templates.TemplateResponse("partials/stats.html", {
        "request": request,
        "stats": repositories.event_stats(db),
        "pipeline_running": pipeline is not None,
        "fps": round(pipeline.last_fps, 1) if pipeline is not None else 0.0,
        "camera_ok": (camera is not None
                      and camera.get_latest_frame() is not None),
        "anpr_queue": anpr_worker.pending() if anpr_worker is not None else 0,
    })


@router.get("/partials/event-detail")
def event_detail_empty(request: Request):
    """The drawer's resting state — also the target of its close button."""
    return templates.TemplateResponse(
        "partials/event_detail_empty.html", {"request": request})


@router.get("/partials/event/{event_id}")
def event_detail(event_id: int, request: Request,
                 db: Session = Depends(get_db)):
    event = repositories.get_event(db, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return templates.TemplateResponse(
        "partials/event_detail.html",
        {"request": request, "event": event})
