"""Application settings, loaded from environment / .env (Pydantic v1)."""

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Central config. Field names map to env vars case-insensitively,
    so `anpr_api_key` is filled from ANPR_API_KEY in .env."""

    database_url: str = "sqlite:///./car_logger.db"
    anpr_api_key: str = ""
    anpr_api_url: str = "https://api.platerecognizer.com/v1/plate-reader/"
    # Student's decisions (Stage 4): 5s tolerates home Wi-Fi jitter; 2 retries
    # with exponential backoff bounds the worst case at ~12s per event.
    anpr_timeout_seconds: float = 5.0
    anpr_max_retries: int = 2
    # Bounded queue: if ANPR falls behind, we skip events instead of growing
    # memory without limit (4GB RAM budget).
    anpr_queue_maxsize: int = 32
    plates_dir: str = "data/plates"
    # Student's decision: 30 days of crops is plenty for a personal project.
    crop_retention_days: int = 30
    log_level: str = "INFO"
    max_pipeline_fps: int = 15
    detector_threshold: float = 0.5
    camera_index: int = 0
    enable_pipeline: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
