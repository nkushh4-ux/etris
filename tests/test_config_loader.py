from backend.config.loader import (
    load_base_config,
    load_camera_config,
    load_development_config,
    load_model_config,
)


def test_base_config_loads():
    config = load_base_config()

    assert config["project"]["name"] == "ETRIS Ultimate AI"
    assert config["project"]["version"] == "0.1.0"


def test_model_config_loads():
    config = load_model_config()

    assert config["vehicle_detection"]["enabled"] is True
    assert config["plate_detection"]["enabled"] is True
    assert config["ocr"]["enabled"] is True


def test_camera_config_loads():
    config = load_camera_config()

    assert config["cameras"] == []
    assert config["defaults"]["timezone"] == "Asia/Kolkata"


def test_development_config_loads():
    config = load_development_config()

    assert config["environment"] == "development"
    assert config["runtime"]["device"] == "cuda"
    assert config["runtime"]["batch_size"] == 1
    assert config["runtime"]["half_precision"] is True