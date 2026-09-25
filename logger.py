import contextvars
import json
import logging
import os
from datetime import datetime, timezone

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
route_name_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "route_name", default=None
)
latency_ms_var: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "latency_ms", default=None
)

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.environ.get("LOG_FILE", "app.log")


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "filename": record.filename,
            "lineno": record.lineno,
        }

        request_id = request_id_var.get()
        if request_id is not None:
            log_record["request_id"] = request_id

        route_name = route_name_var.get()
        if route_name is not None:
            log_record["route_name"] = route_name

        latency_ms = latency_ms_var.get()
        if latency_ms is not None:
            log_record["latency_ms"] = round(latency_ms, 2)

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record, ensure_ascii=False)


def configure_logging() -> logging.Logger:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return logging.getLogger("AppLogger")

    formatter = JSONFormatter()

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root_logger.setLevel(LOG_LEVEL)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    return logging.getLogger("AppLogger")


def set_request_context(
    request_id: str,
    route_name: str,
    latency_ms: float | None = None,
) -> None:
    request_id_var.set(request_id)
    route_name_var.set(route_name)
    latency_ms_var.set(latency_ms)


def clear_request_context() -> None:
    request_id_var.set(None)
    route_name_var.set(None)
    latency_ms_var.set(None)


logger = configure_logging()
