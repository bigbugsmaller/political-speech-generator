"""Langfuse observability for the speech agent (optional; fails open)."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Iterator

from logger import logger

_MAX_PAYLOAD_CHARS = 12_000


def _truncate(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        if len(value) <= _MAX_PAYLOAD_CHARS:
            return value
        return value[:_MAX_PAYLOAD_CHARS] + f"...[truncated, {len(value) - _MAX_PAYLOAD_CHARS} more chars]"
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) <= _MAX_PAYLOAD_CHARS:
        return value
    return text[:_MAX_PAYLOAD_CHARS] + "...[truncated]"


def _credentials_configured() -> bool:
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
        and os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    )


def flush_langfuse() -> None:
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as e:
        logger.warning(f"Langfuse flush failed (ignored): {e}")


class _NoOpObservation:
    trace_id: str | None = None

    def update(self, **_: Any) -> None:
        return None


@contextmanager
def speech_generation_trace(
    *,
    candidate: str,
    party: str,
    location: str,
    llm_provider: str,
    model: str,
) -> Iterator[tuple[Any, _NoOpObservation]]:
    """
    Root trace for one agent run. Yields (session, root_observation).
    Session exposes nested span context managers; root is updated on finalize().
    """
    session = AgentTraceSession()
    root: Any = _NoOpObservation()

    if not _credentials_configured():
        yield session, root
        return

    try:
        from langfuse import get_client, propagate_attributes

        client = get_client()
        tags = []
        if candidate:
            tags.append(f"candidate:{candidate}")
        if party:
            tags.append(f"party:{party}")
        if location:
            tags.append(f"location:{location}")

        base_metadata = {
            "llm_provider": llm_provider,
            "model": model,
            "candidate_name": candidate,
            "political_party": party,
            "geographic_location": location,
        }

        with propagate_attributes(
            trace_name="speech_generation",
            tags=tags or None,
            metadata=base_metadata,
        ):
            with client.start_as_current_observation(
                as_type="span",
                name="speech_generation",
                input={
                    "candidate": candidate,
                    "party": party,
                    "location": location,
                },
            ) as root_span:
                session.activate(client)
                root = root_span
                try:
                    yield session, root
                finally:
                    session.apply_root_update(root)
    except Exception as e:
        logger.warning(f"Langfuse tracing disabled for this run: {e}")
        yield session, root


class AgentTraceSession:
    """Nested Langfuse observations; all methods fail open."""

    def __init__(self) -> None:
        self._client: Any = None
        self._pending_root: dict[str, Any] = {}

    def activate(self, client: Any) -> None:
        self._client = client

    @property
    def active(self) -> bool:
        return self._client is not None

    def set_run_outcome(
        self,
        *,
        total_tool_calls: int,
        total_latency_ms: float,
        hit_tool_cap: bool,
        success: bool,
        error_code: str | None = None,
        llm_provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self._pending_root.update(
            {
                "total_tool_calls": total_tool_calls,
                "total_latency_ms": round(total_latency_ms, 2),
                "hit_tool_cap": hit_tool_cap,
                "success": success,
                "error_code": error_code,
                "llm_provider": llm_provider,
                "model": model,
            }
        )

    def apply_root_update(self, root: Any) -> None:
        if root is None or isinstance(root, _NoOpObservation):
            return
        try:
            root.update(
                metadata=self._pending_root,
                output={
                    "success": self._pending_root.get("success"),
                    "total_tool_calls": self._pending_root.get("total_tool_calls"),
                    "hit_tool_cap": self._pending_root.get("hit_tool_cap"),
                },
            )
        except Exception as e:
            logger.warning(f"Langfuse root span update failed (ignored): {e}")

    @contextmanager
    def llm_turn(
        self,
        *,
        turn_index: int,
        messages: list[dict[str, Any]],
        request_kwargs: dict[str, Any],
    ) -> Iterator[_NoOpObservation]:
        if not self.active:
            yield _NoOpObservation()
            return
        name = f"llm_turn_{turn_index}"
        try:
            with self._client.start_as_current_observation(
                as_type="generation",
                name=name,
                model=request_kwargs.get("model"),
                input=_truncate(
                    {
                        "messages": messages,
                        "use_tools": "tools" in request_kwargs,
                        "tool_choice": request_kwargs.get("tool_choice"),
                        "response_format": request_kwargs.get("response_format"),
                        "max_tokens": request_kwargs.get("max_tokens"),
                    }
                ),
            ) as gen:
                yield gen
        except Exception as e:
            logger.warning(f"Langfuse LLM span failed (ignored): {e}")
            yield _NoOpObservation()

    @contextmanager
    def tool_call(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Iterator[_NoOpObservation]:
        if not self.active:
            yield _NoOpObservation()
            return
        try:
            with self._client.start_as_current_observation(
                as_type="tool",
                name=tool_name,
                input=_truncate(arguments),
            ) as tool_span:
                yield tool_span
        except Exception as e:
            logger.warning(f"Langfuse tool span failed (ignored): {e}")
            yield _NoOpObservation()

    @contextmanager
    def parse_response(self, *, raw_content: str | None) -> Iterator[_NoOpObservation]:
        if not self.active:
            yield _NoOpObservation()
            return
        try:
            with self._client.start_as_current_observation(
                as_type="span",
                name="parse_model_response",
                input=_truncate({"raw_content_chars": len(raw_content or ""), "preview": (raw_content or "")[:500]}),
                metadata={"module": "text_processing.parse_model_response"},
            ) as span:
                yield span
        except Exception as e:
            logger.warning(f"Langfuse parse span failed (ignored): {e}")
            yield _NoOpObservation()
