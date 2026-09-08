import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def configured_database_url() -> str | None:
    return os.getenv("DATABASE_URL") or None


def create_session_factory(url: str):
    engine = create_engine(url, pool_pre_ping=True)
    return engine, sessionmaker(engine, expire_on_commit=False)


def probe_database(engine) -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
