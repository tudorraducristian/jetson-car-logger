from car_logger.config import Settings


def test_defaults_are_sane():
    s = Settings()
    assert s.database_url.startswith("sqlite")
    assert s.log_level == "INFO"
    assert s.max_pipeline_fps == 15


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("MAX_PIPELINE_FPS", "8")
    monkeypatch.setenv("ANPR_API_KEY", "test-token")
    s = Settings()
    assert s.max_pipeline_fps == 8
    assert s.anpr_api_key == "test-token"


def test_camera_healing_settings_have_defaults():
    s = Settings()
    assert s.camera_stale_after_s == 2.0
    assert s.camera_reopen_backoff_s == 2.0


def test_v2_local_anpr_defaults():
    from car_logger.config import Settings
    settings = Settings(_env_file=None)
    assert settings.anpr_detector_model_path == (
        "models/anpr/yolo-v9-t-384-license-plates-end2end-opset15.onnx")
    assert settings.anpr_ocr_model_path == "models/anpr/cct_xs_v2_global.onnx"
    assert settings.anpr_ocr_config_path == (
        "models/anpr/cct_xs_v2_global_plate_config.yaml")
    assert settings.plate_detection_threshold == 0.4
    assert settings.anpr_reads_per_track == 3
    assert settings.anpr_read_spacing_s == 0.4
    # Stage A verdict: 0.90 is the garbage floor for the LOCAL engine's
    # confidence scale (the 0.85 default was calibrated for the cloud API)
    assert settings.min_vehicle_confidence == 0.90
