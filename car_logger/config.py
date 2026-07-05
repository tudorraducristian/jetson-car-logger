"""Application settings, loaded from environment / .env (Pydantic v1)."""

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Central config. Field names map to env vars case-insensitively,
    so `anpr_api_key` is filled from ANPR_API_KEY in .env."""

    database_url: str = "sqlite:///./car_logger.db"
    anpr_api_key: str = ""
    anpr_api_url: str = "https://api.platerecognizer.com/v1/plate-reader/"
    log_level: str = "INFO"
    max_pipeline_fps: int = 15

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
