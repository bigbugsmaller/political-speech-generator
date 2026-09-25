"""SQLAlchemy engine/session for Postgres persistence.

Note: LanceDB (vector store) uses LANCEDB_URL / LANCEDB_PATH separately.
DATABASE_URL here is the Postgres connection string.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()

# Shortcut: create_all() instead of Alembic migrations — replace with Alembic later.
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/speechgen",
)

# If someone still has the old LanceDB path in DATABASE_URL, fall back to default Postgres URL.
if DATABASE_URL and not DATABASE_URL.startswith(("postgresql", "postgres")):
    DATABASE_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/speechgen"


class Base(DeclarativeBase):
    pass


engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Create tables if they do not exist (no Alembic yet — intentional shortcut)."""
    # Import models so metadata is registered.
    import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
