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
