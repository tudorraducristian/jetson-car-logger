"""Car Logger API entrypoint - the app object everything else attaches to."""

from fastapi import FastAPI

APP_VERSION = "0.1.0"

app = FastAPI(title="Car Logger", version=APP_VERSION)


@app.get("/")
def root():
    """Greeting endpoint - proves the server is reachable from the LAN."""
    return {"message": "Car Logger is running", "version": APP_VERSION}


@app.get("/health")
def health():
    """Liveness probe - used later by systemd and monitoring."""
    return {"status": "ok"}
