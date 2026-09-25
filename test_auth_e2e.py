"""
End-to-end auth + speech persistence test against Postgres.

By default uses DATABASE_URL from .env (Docker speechgen-pg).
Fallback: set USE_PGSERVER=1 to use embedded pgserver instead.

Usage:
    py test_auth_e2e.py
    USE_PGSERVER=1 py test_auth_e2e.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("JWT_SECRET_KEY", "e2e-test-jwt-secret")

USE_PGSERVER = os.environ.get("USE_PGSERVER", "").strip().lower() in ("1", "true", "yes")

if USE_PGSERVER:
    from pgserver import get_server

    PG_DIR = Path(__file__).resolve().parent / "data" / "pgdata"
    print("=== Starting embedded Postgres (pgserver) ===")
    pg = get_server(PG_DIR)
    raw_uri = pg.get_uri()
    DATABASE_URL = raw_uri.replace("postgresql://", "postgresql+psycopg2://", 1)
    os.environ["DATABASE_URL"] = DATABASE_URL
else:
    DATABASE_URL = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/speechgen",
    )
    if not DATABASE_URL.startswith(("postgresql", "postgres")):
        raise SystemExit(
            f"DATABASE_URL must be a Postgres URL for this test, got: {DATABASE_URL!r}\n"
            "Start Docker Postgres or set USE_PGSERVER=1"
        )
    print("=== Using Postgres from DATABASE_URL (Docker / real instance) ===")

print(f"DATABASE_URL={DATABASE_URL}")

# Wait briefly for Docker Postgres to accept connections.
from sqlalchemy import create_engine, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
for attempt in range(1, 31):
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"=== Postgres reachable (attempt {attempt}) ===")
        break
    except OperationalError as e:
        if attempt == 30:
            raise SystemExit(
                f"Could not connect to Postgres at {DATABASE_URL}\n"
                f"Last error: {e}\n"
                "Start the container:\n"
                "  docker run --name speechgen-pg -e POSTGRES_PASSWORD=postgres "
                "-e POSTGRES_DB=speechgen -p 5432:5432 -d postgres:16\n"
                "Or: docker start speechgen-pg"
            ) from e
        print(f"Waiting for Postgres... attempt {attempt}/30")
        time.sleep(2)

import db as db_mod

db_mod.DATABASE_URL = DATABASE_URL
db_mod.engine = engine
db_mod.SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

import models  # noqa: F401
from models import Speech, User

# Keep existing rows in Docker DB unless RESET_DB=1 (default: reset for clean e2e).
if os.environ.get("RESET_DB", "1").strip().lower() in ("1", "true", "yes"):
    db_mod.Base.metadata.drop_all(bind=engine)
db_mod.init_db()
print("=== Tables ensured via create_all() ===")

from app import app

app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "e2e-test-jwt-secret")

FAKE_SPEECH = {
    "speech": "Friends of Lucknow, together we will build opportunity and dignity.",
    "key_themes": ["Jobs", "Youth", "Fair exams"],
    "sentiment": {"category": "Inspirational", "explanation": "Forward-looking tone."},
}

SAMPLE_INPUT = {
    "candidate-name": "Akhilesh Yadav",
    "political-party": "Samajwadi Party (SP)",
    "geographic-location": "Lucknow, Uttar Pradesh",
    "policy-points": "internship stipends, MSP guarantee",
    "speech-length": "Short (5 minutes)",
}


def main() -> int:
    client = app.test_client()
    # Unique email so re-runs without RESET_DB still work.
    email = os.environ.get("E2E_EMAIL", f"e2e_user_{int(time.time())}@example.com")
    password = "testpass123"

    print("\n=== 1) REGISTER ===")
    r = client.post(
        "/api/auth/register",
        json={"email": email, "password": password},
    )
    print(f"status={r.status_code}")
    print(json.dumps(r.get_json(), indent=2))
    assert r.status_code == 201, r.get_json()
    register_body = r.get_json()
    assert "access_token" in register_body
    user_id = register_body["user_id"]

    print("\n=== 2) LOGIN ===")
    r = client.post(
        "/api/auth/login",
        json={"email": email, "password": password},
    )
    print(f"status={r.status_code}")
    print(json.dumps(r.get_json(), indent=2))
    assert r.status_code == 200, r.get_json()
    token = r.get_json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    print("\n=== 3) /process WITHOUT JWT (expect 401) ===")
    r = client.post("/process", json=SAMPLE_INPUT)
    print(f"status={r.status_code}")
    assert r.status_code == 401

    print("\n=== 4) /process WITH JWT (mocked generate_response) ===")
    with patch("app.generate_response", return_value=dict(FAKE_SPEECH)):
        r = client.post("/process", json=SAMPLE_INPUT, headers=headers)
    print(f"status={r.status_code}")
    body = r.get_json()
    print(json.dumps(body, indent=2)[:800])
    assert r.status_code == 200, body
    assert body.get("speech") == FAKE_SPEECH["speech"]
    speech_id = body.get("speech_id")
    assert speech_id, "speech_id missing — persistence may have failed"
    print(f"speech_id={speech_id}")

    print("\n=== 5) Direct Postgres check (user_id + speech row) ===")
    with db_mod.SessionLocal() as session:
        row = session.get(Speech, speech_id)
        assert row is not None
        print(
            f"Speech row: id={row.id} user_id={row.user_id} "
            f"themes={row.key_themes} speech_preview={row.generated_speech[:60]!r}"
        )
        assert row.user_id == user_id
        user = session.get(User, user_id)
        print(f"User row: id={user.id} email={user.email}")
        n = len(session.scalars(select(Speech).where(Speech.user_id == user_id)).all())
        print(f"speeches for user={n}")

    print("\n=== 6) GET /api/speeches ===")
    r = client.get("/api/speeches", headers=headers)
    print(f"status={r.status_code}")
    print(json.dumps(r.get_json(), indent=2)[:1200])
    assert r.status_code == 200
    speeches = r.get_json()["speeches"]
    assert any(s["id"] == speech_id for s in speeches)

    print("\n=== 7) GET /api/speeches/<id> ===")
    r = client.get(f"/api/speeches/{speech_id}", headers=headers)
    print(f"status={r.status_code}")
    print(json.dumps(r.get_json(), indent=2)[:1200])
    assert r.status_code == 200
    assert r.get_json()["user_id"] == user_id

    print("\nALL E2E CHECKS PASSED")
    print(f"Data persisted in: {DATABASE_URL}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
