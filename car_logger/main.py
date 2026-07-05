"""Car Logger API entrypoint - the app object everything else attaches to."""

from fastapi import FastAPI

from car_logger.api.routes_events import router as events_router

APP_VERSION = "0.2.0"

app = FastAPI(title="Car Logger", version=APP_VERSION)

app.include_router(events_router)


@app.get("/")
def root():
    """Greeting endpoint - proves the server is reachable from the LAN."""
    return {"message": "Car Logger is running", "version": APP_VERSION}


@app.get("/health")
def health():
    """Liveness probe - used later by systemd and monitoring."""
    return {"status": "ok"}
