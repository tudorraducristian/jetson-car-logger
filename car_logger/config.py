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
    detector_threshold: float = 0.5
    # identity gate: a plate reading below this confidence never creates a
    # Vehicle (the event still keeps the reading). Recalibrated for the
    # LOCAL engine from the bake-off distributions (student decision
    # 2026-07-18): 0.90 is a garbage floor ONLY — this model's confidence
    # does NOT separate correct from wrong reads; the multi-frame vote is
    # the real error filter.
    min_vehicle_confidence: float = 0.90
    camera_index: int = 0
    # camera self-healing (student decision 2026-07-15): no fresh frame for
    # this long => camera lost => camera_ok False + reopen. Reopen retries
    # this often until the device returns.
    camera_stale_after_s: float = 2.0
    camera_reopen_backoff_s: float = 2.0
    enable_pipeline: bool = True
    # v2 local ANPR (Stage B, spec 2026-07-19). Paths are relative to the
    # repo root — both `uvicorn` in dev and systemd's WorkingDirectory
    # run from there. Committed models: see models/anpr/README.md.
    anpr_detector_model_path: str = (
        "models/anpr/yolo-v9-t-384-license-plates-end2end-opset15.onnx")
    anpr_ocr_model_path: str = "models/anpr/cct_xs_v2_global.onnx"
    anpr_ocr_config_path: str = "models/anpr/cct_xs_v2_global_plate_config.yaml"
    # 0.4 = fast-alpr 0.4.0's default detector threshold — the setting the
    # bake-off accuracy (93.5% / 100%) was measured under.
    plate_detection_threshold: float = 0.4
    # The multi-frame vote (the REAL error filter — confidence is not,
    # per the bake-off calibration): 3 reads per track, >= 0.4 s apart.
    anpr_reads_per_track: int = 3
    anpr_read_spacing_s: float = 0.4

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
