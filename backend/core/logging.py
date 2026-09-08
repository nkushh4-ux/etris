import logging
from pathlib import Path

DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def configure_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
) -> None:
    """Configure application-wide logging."""

    numeric_level = getattr(logging, level.upper(), None)

    if not isinstance(numeric_level, int):
        raise TypeError(f"Invalid logging level: {level}")

    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
    ]

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format=DEFAULT_LOG_FORMAT,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger for an application component."""
    return logging.getLogger(name)