from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


class ConfigurationError(Exception):
    """Raised when project configuration is invalid."""


def load_yaml(filename: str) -> dict[str, Any]:
    """Load one YAML configuration file."""
    path = CONFIG_DIR / filename

    if not path.is_file():
        raise ConfigurationError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Invalid YAML in {path}: {exc}"
        ) from exc

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Configuration root must be a mapping: {path}"
        )

    return data


def load_base_config() -> dict[str, Any]:
    """Load base project configuration."""
    return load_yaml("base.yaml")


def load_model_config() -> dict[str, Any]:
    """Load model configuration."""
    return load_yaml("models.yaml")


def load_camera_config() -> dict[str, Any]:
    """Load camera configuration."""
    return load_yaml("cameras.yaml")


def load_development_config() -> dict[str, Any]:
    """Load development configuration."""
    return load_yaml("development.yaml")