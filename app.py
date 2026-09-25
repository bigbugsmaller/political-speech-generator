import os
import time
import uuid
from datetime import timedelta

from flask import Flask, g, jsonify, request, send_from_directory
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    get_jwt_identity,
    jwt_required,
)
from sqlalchemy import select

from auth_utils import hash_password, verify_password
from database import ping_db
from db import SessionLocal, get_session, init_db
from llm import generate_response
from logger import clear_request_context, logger, set_request_context
from models import Speech, User

app = Flask(__name__)

app.config["JWT_SECRET_KEY"] = os.environ.get(
    "JWT_SECRET_KEY", "dev-only-change-me-in-production"
)
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(
    seconds=int(os.environ.get("JWT_ACCESS_TOKEN_EXPIRES", str(60 * 60 * 24)))
)

jwt = JWTManager(app)

# create_all() shortcut — replace with Alembic migrations later.
try:
    init_db()
    logger.info("Postgres tables ensured via create_all()")
except Exception as e:
    logger.warning(f"Postgres init_db deferred/failed at import: {e}")


@app.before_request
def bind_request_context():
    g.request_start = time.perf_counter()
    g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    set_request_context(g.request_id, request.endpoint or request.path)


@app.after_request
def log_request(response):
    if hasattr(g, "request_start") and hasattr(g, "request_id"):
        latency_ms = (time.perf_counter() - g.request_start) * 1000
        set_request_context(g.request_id, request.endpoint or request.path, latency_ms)
        logger.info(f"{request.method} {request.path} -> {response.status_code}")
    clear_request_context()
    return response


@app.route("/health")
def health():
    try:
        ping_db()
        db_status = "connected"
    except Exception as e:
        logger.error(f"Health check database ping failed: {e}")
        db_status = "error"

    pg_status = "unknown"
    try:
        with SessionLocal() as session:
            session.execute(select(1))
        pg_status = "connected"
    except Exception as e:
        logger.error(f"Health check Postgres ping failed: {e}")
        pg_status = "error"

    return jsonify({"status": "ok", "db": db_status, "postgres": pg_status})


@app.route("/")
def home():
    return send_from_directory("static", "page1.html")


@app.route("/login")
def login_page():
    return send_from_directory("static", "login.html")


@app.route("/page2")
def page2():
    return send_from_directory("static", "page2.html")


@app.route("/page3")
def page3():
    return send_from_directory("static", "page3.html")


@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "ERR_INVALID_INPUT", "message": "email and password required"}), 400
    if len(password) < 6:
        return jsonify({"error": "ERR_INVALID_INPUT", "message": "password must be at least 6 characters"}), 400

    try:
        with get_session() as session:
            existing = session.scalar(select(User).where(User.email == email))
            if existing:
                return jsonify({"error": "ERR_CONFLICT", "message": "email already registered"}), 409

            user = User(email=email, hashed_password=hash_password(password))
            session.add(user)
            session.flush()
            user_id = user.id
    except ValueError as e:
        return jsonify({"error": "ERR_INVALID_INPUT", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"Register failed: {e}")
        return jsonify({"error": "ERR_DB_FAILURE", "message": str(e)}), 500

    token = create_access_token(identity=str(user_id))
    return jsonify({"access_token": token, "user_id": user_id, "email": email}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "ERR_INVALID_INPUT", "message": "email and password required"}), 400

    try:
        with SessionLocal() as session:
            user = session.scalar(select(User).where(User.email == email))
            if not user or not verify_password(password, user.hashed_password):
                return jsonify({"error": "ERR_UNAUTHORIZED", "message": "invalid email or password"}), 401
            user_id = user.id
            user_email = user.email
    except Exception as e:
        logger.error(f"Login failed: {e}")
        return jsonify({"error": "ERR_DB_FAILURE", "message": str(e)}), 500

    token = create_access_token(identity=str(user_id))
    return jsonify({"access_token": token, "user_id": user_id, "email": user_email})


@app.route("/process", methods=["POST"])
@jwt_required()
def process_prompt():
    data = request.get_json()
    logger.info("Received speech generation request")

    user_id = int(get_jwt_identity())
    response = generate_response(data)

    # Persist successful generations only (keep error responses unchanged for clients).
    if isinstance(response, dict) and "speech" in response and "error" not in response:
        try:
            with get_session() as session:
                speech = Speech(
                    user_id=user_id,
                    input_data=data if isinstance(data, dict) else {},
                    generated_speech=response.get("speech") or "",
                    key_themes=response.get("key_themes"),
                    sentiment=response.get("sentiment"),
                )
                session.add(speech)
                session.flush()
                response = dict(response)
                response["speech_id"] = speech.id
            logger.info(f"Saved speech id={response.get('speech_id')} for user_id={user_id}")
        except Exception as e:
            logger.error(f"Failed to persist speech for user_id={user_id}: {e}")
            # Generation succeeded; persistence failure should not wipe the speech payload.

    return jsonify(response)


@app.route("/api/speeches", methods=["GET"])
@jwt_required()
def list_speeches():
    user_id = int(get_jwt_identity())
    with SessionLocal() as session:
        rows = session.scalars(
            select(Speech)
            .where(Speech.user_id == user_id)
            .order_by(Speech.created_at.desc())
        ).all()
        payload = [
            {
                "id": s.id,
                "user_id": s.user_id,
                "generated_speech": s.generated_speech,
                "key_themes": s.key_themes,
                "sentiment": s.sentiment,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in rows
        ]
    return jsonify({"speeches": payload, "count": len(payload)})


@app.route("/api/speeches/<int:speech_id>", methods=["GET"])
@jwt_required()
def get_speech(speech_id: int):
    user_id = int(get_jwt_identity())
    with SessionLocal() as session:
        speech = session.scalar(
            select(Speech).where(Speech.id == speech_id, Speech.user_id == user_id)
        )
        if not speech:
            return jsonify({"error": "ERR_NOT_FOUND", "message": "speech not found"}), 404
        return jsonify(
            {
                "id": speech.id,
                "user_id": speech.user_id,
                "input_data": speech.input_data,
                "generated_speech": speech.generated_speech,
                "key_themes": speech.key_themes,
                "sentiment": speech.sentiment,
                "created_at": speech.created_at.isoformat() if speech.created_at else None,
            }
        )


if __name__ == "__main__":
    app.run(debug=False)
