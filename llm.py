"""Speech generation via LangGraph-orchestrated OpenAI tool-calling agent."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    ContentFilterFinishReasonError,
    InternalServerError,
    LengthFinishReasonError,
    NotFoundError,
    OpenAI,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
    _AmbiguousModuleClientUsageError,
)

from config import (
    EVAL_USER_PREAMBLE,
    LLM_PROVIDER,
    MAX_OUTPUT_TOKENS,
    MODEL,
    MODEL_URL,
    OPENAI_API,
    is_agent_eval_mode,
    resolve_system_prompt,
    resolve_user_template,
)
from langfuse_tracing import (
    AgentTraceSession,
    _NoOpObservation,
    flush_langfuse,
    speech_generation_trace,
)
from logger import logger
from text_processing import substitute_template
from tools import AGENT_TOOLS, execute_tool

client_openai = OpenAI(api_key=OPENAI_API, base_url=MODEL_URL)

MAX_TOOL_CALLS = 3

FINAL_ANSWER_INSTRUCTION = (
    "You have reached the maximum of 3 tool calls. You must now provide your "
    "final speech based on available information. Do not call any more tools. "
    "Respond only with the required JSON object containing speech, key_themes, "
    "and sentiment."
)

FORCE_JSON_INSTRUCTION = (
    "Provide your final answer now as a valid JSON object with keys "
    "speech, key_themes, and sentiment. Do not call tools."
)

ToolExecutor = Callable[[str, dict[str, Any]], tuple[str, dict[str, Any]]]


def _api_error(code: str, message: str) -> dict:
    return {"error": code, "message": message}


def _serialize_assistant_message(message, tool_calls=None) -> dict[str, Any]:
    """Convert an OpenAI assistant message into a plain dict for history.

    If tool_calls is provided, use that list (e.g. first-call-only) instead of
    the full message.tool_calls — required when we discard parallel extras so
    the API is not left expecting tool results for ignored call ids.
    """
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": message.content,
    }
    calls = tool_calls if tool_calls is not None else getattr(message, "tool_calls", None)
    if calls:
        payload["tool_calls"] = [
            {
                "id": tc.id,
                "type": getattr(tc, "type", "function") or "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in calls
        ]
    return payload


def _parse_tool_arguments(raw_args: str | None) -> dict[str, Any]:
    if not raw_args:
        return {}
    try:
        parsed = json.loads(raw_args)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        logger.warning(f"Could not parse tool arguments as JSON: {raw_args!r}")
        return {}


def _dump_tool_calls_raw(tool_calls) -> list[dict[str, Any]]:
    """Plain-dict view of tool_calls for logging / traces."""
    out = []
    for tc in tool_calls or []:
        out.append(
            {
                "id": getattr(tc, "id", None),
                "type": getattr(tc, "type", "function"),
                "function": {
                    "name": getattr(getattr(tc, "function", None), "name", None),
                    "arguments": getattr(
                        getattr(tc, "function", None), "arguments", None
                    ),
                },
            }
        )
    return out


def _build_chat_kwargs(
    messages: list[dict[str, Any]],
    *,
    use_tools: bool,
    force_final: bool = False,
    require_tool: bool = False,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "messages": messages,
        "temperature": 0.0,
    }

    if use_tools and not force_final:
        kwargs["tools"] = AGENT_TOOLS
        kwargs["tool_choice"] = "required" if require_tool else "auto"
        kwargs["parallel_tool_calls"] = False
    else:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


def _update_langfuse_generation(gen: Any, message: Any, response: Any) -> None:
    if isinstance(gen, _NoOpObservation):
        return
    try:
        tool_calls = getattr(message, "tool_calls", None) or []
        gen.update(
            output={
                "content": message.content,
                "tool_calls": _dump_tool_calls_raw(tool_calls),
            },
            usage_details=_extract_usage(response),
        )
    except Exception as e:
        logger.warning(f"Langfuse generation update failed (ignored): {e}")


def _extract_usage(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage", None)
    if not usage:
        return None
    details: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        val = getattr(usage, key, None)
        if val is not None:
            details[key] = int(val)
    return details or None


def _chat_completion(
    messages: list[dict[str, Any]],
    *,
    use_tools: bool,
    force_final: bool = False,
    require_tool: bool = False,
    client: Any = None,
    lf: AgentTraceSession | None = None,
    turn_index: int = 0,
):
    kwargs = _build_chat_kwargs(
        messages,
        use_tools=use_tools,
        force_final=force_final,
        require_tool=require_tool,
    )

    active_client = client or client_openai
    logger.info(
        f"[llm] BEFORE chat.completions.create model={kwargs['model']} "
        f"use_tools={use_tools} force_final={force_final} "
        f"require_tool={require_tool} "
        f"parallel_tool_calls={kwargs.get('parallel_tool_calls')} "
        f"messages={len(messages)}"
    )
    started = time.perf_counter()

    def _invoke():
        return active_client.chat.completions.create(**kwargs)

    if lf is not None and lf.active:
        with lf.llm_turn(
            turn_index=turn_index,
            messages=messages,
            request_kwargs=kwargs,
        ) as gen:
            response = _invoke()
            message = response.choices[0].message
            _update_langfuse_generation(gen, message, response)
    else:
        response = _invoke()
        message = response.choices[0].message

    logger.info(
        f"[llm] AFTER chat.completions.create "
        f"elapsed_ms={(time.perf_counter() - started) * 1000:.1f}"
    )
    return response


def _handle_openai_exception(e: Exception) -> dict:
    if isinstance(e, RateLimitError):
        logger.error(f"Rate limit exceeded: {e}")
        return _api_error(
            "ERR_API_FAILURE",
            f"Rate limit exceeded. Please try again later: {e}",
        )
    if isinstance(e, APITimeoutError):
        logger.error(f"API request timed out: {e}")
        return _api_error(
            "ERR_API_FAILURE",
            f"Request to OpenAI API timed out: {e}",
        )
    if isinstance(e, APIConnectionError):
        logger.error(f"Connection error with OpenAI API: {e}")
        return _api_error(
            "ERR_API_FAILURE",
            f"Failed to connect to OpenAI API. Check your network connection: {e}",
        )
    if isinstance(e, AuthenticationError):
        logger.error(f"Authentication error with OpenAI API: {e}")
        return _api_error(
            "ERR_API_FAILURE",
            f"Authentication failed. Check your API key: {e}",
        )
    if isinstance(e, PermissionDeniedError):
        logger.error(f"Permission denied by OpenAI API: {e}")
        return _api_error(
            "ERR_API_FAILURE",
            f"Permission denied to access the requested resource: {e}",
        )
    if isinstance(e, BadRequestError):
        logger.error(f"Bad request to OpenAI API: {e}")
        return _api_error(
            "ERR_API_FAILURE",
            f"Invalid request parameters sent to OpenAI API: {e}",
        )
    if isinstance(e, NotFoundError):
        logger.error(f"Resource not found in OpenAI API: {e}")
        return _api_error(
            "ERR_API_FAILURE",
            f"The requested resource was not found. Check model name: {e}",
        )
    if isinstance(e, ConflictError):
        logger.error(f"Conflict error with OpenAI API: {e}")
        return _api_error(
            "ERR_API_FAILURE",
            f"Request conflicts with current state of the server: {e}",
        )
    if isinstance(e, InternalServerError):
        logger.error(f"Internal server error from OpenAI API: {e}")
        return _api_error(
            "ERR_API_FAILURE",
            f"OpenAI API experienced an internal error. Try again later: {e}",
        )
    if isinstance(e, UnprocessableEntityError):
        logger.error(f"Unprocessable entity error from OpenAI API: {e}")
        return _api_error(
            "ERR_API_FAILURE",
            f"The request was well-formed but unable to be processed: {e}",
        )
    if isinstance(e, ContentFilterFinishReasonError):
        logger.error(f"Content filter triggered in OpenAI API: {e}")
        return _api_error(
            "ERR_API_FAILURE",
            f"Response was filtered due to content safety policies: {e}",
        )
    if isinstance(e, LengthFinishReasonError):
        logger.error(f"Response length limit reached in OpenAI API: {e}")
        return _api_error(
            "ERR_API_FAILURE",
            f"Response was truncated due to token limit constraints: {e}",
        )
    if isinstance(e, APIResponseValidationError):
        logger.error(f"API response validation error from OpenAI: {e}")
        return _api_error(
            "ERR_API_FAILURE",
            f"OpenAI API response failed validation: {e}",
        )
    if isinstance(e, APIStatusError):
        logger.error(f"API status error from OpenAI: {e}")
        return _api_error(
            "ERR_API_FAILURE",
            f"OpenAI API returned an unexpected status code: {e}",
        )
    if isinstance(e, _AmbiguousModuleClientUsageError):
        logger.error(f"Ambiguous module client usage with OpenAI API: {e}")
        return _api_error(
            "ERR_API_FAILURE",
            f"Ambiguous usage of OpenAI client: {e}",
        )
    if isinstance(e, OpenAIError):
        logger.error(f"General OpenAI API error: {e}")
        return _api_error(
            "ERR_API_FAILURE",
            f"An error occurred with the OpenAI API: {e}",
        )
    logger.error(f"Unexpected error occurred while generating response: {e}")
    return _api_error(
        "ERR_API_FAILURE",
        f"Failed to get response from OpenAI API: {e}",
    )


def run_agent(
    data: dict,
    client: Any = None,
    tool_executor: ToolExecutor | None = None,
    *,
    eval_mode: bool | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Run the LangGraph tool-calling workflow.

    Returns (result_dict, tool_trace).
    result_dict matches the /process response shape on success, or an error object.

    Optional hooks (for tests / later wiring):
    - client: OpenAI-compatible chat client
    - tool_executor: callable(name, args) -> (result_text, meta_dict)
    """
    trace: list[dict[str, Any]] = []
    run_tool = tool_executor or execute_tool

    if not isinstance(data, dict):
        logger.error("Invalid input: data must be a dictionary")
        return (
            _api_error("ERR_INVALID_INPUT", "Input data must be a dictionary"),
            trace,
        )

    working = dict(data)

    if working.get("political-party") == "other":
        working["political-party"] = working.get("other-party", "")
        if not working["political-party"]:
            logger.warning("Other party selected but no party name provided")

    if not working.get("candidate-name"):
        logger.error("Missing required field: candidate-name")
        return (
            _api_error("ERR_MISSING_FIELD", "Candidate name is required"),
            trace,
        )

    # Agent gathers context via tools; seed retrieved_info as empty.
    working.setdefault("retrieved_info", "")

    use_eval = eval_mode if eval_mode is not None else is_agent_eval_mode()
    system_prompt = resolve_system_prompt(eval_mode=use_eval)
    user_template = resolve_user_template(eval_mode=use_eval)

    try:
        formatted_prompt = substitute_template(working, template_string=user_template)
        logger.info("Formatted prompt successfully")
        logger.debug(f"Full Prompt:\n {formatted_prompt}")
    except Exception as e:
        logger.error(f"Template substitution failed: {e}")
        return (
            _api_error(
                "ERR_TEMPLATE_FAILED",
                f"Failed to format prompt: {e}",
            ),
            trace,
        )

    candidate = working.get("candidate-name", "")
    party = working.get("political-party", "")
    location = working.get("geographic-location", "")
    if use_eval:
        user_preamble = EVAL_USER_PREAMBLE.format(
            candidate=candidate, party=party, location=location
        ) + formatted_prompt
    else:
        user_preamble = (
            "Before writing the speech, gather context with tools when needed.\n"
            "Rules:\n"
            "- Call at most ONE tool per turn. Do not request multiple tools at once.\n"
            "- After each tool result, decide: enough info to write the final speech JSON, "
            "or call exactly one more tool.\n"
            "- Prefer search_vector_db first. Skip search_web if vector results are already "
            "sufficient. Skip fact_check_claim unless you plan to state a specific "
            "statistic/date/policy figure.\n"
            "- At most 3 tool calls total. You may stop after 1 or 2 if you have enough.\n"
            "When ready, return only the final JSON speech object.\n\n"
            f"Suggested starting query: {candidate} {party} {location}\n\n"
            f"{formatted_prompt}"
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_preamble},
    ]

    tool_calls_used = 0
    loop_started = time.perf_counter()
    agent_success = False
    agent_error: str | None = None
    logger.info(
        f"[agent] START candidate={candidate!r} party={party!r} "
        f"location={location!r}"
    )

    with speech_generation_trace(
        candidate=candidate,
        party=party,
        location=location,
        llm_provider=LLM_PROVIDER,
        model=MODEL,
    ) as (lf, langfuse_root):
        try:
            from agent_graph import invoke_speech_agent

            result, trace, final_state = invoke_speech_agent(
                messages=messages,
                client=client,
                tool_executor=run_tool,
                lf=lf,
            )
            agent_success = bool(final_state.get("agent_success"))
            agent_error = final_state.get("agent_error")
            tool_calls_used = int(final_state.get("tool_calls_used") or 0)
            logger.info(
                f"[agent] LangGraph finished tool_calls={tool_calls_used} "
                f"trace_steps={len(trace)} "
                f"elapsed_ms={(time.perf_counter() - loop_started) * 1000:.1f}"
            )
            return result, trace
        except Exception as e:
            logger.error(f"[agent] Unhandled exception: {e}")
            agent_error = "ERR_API_FAILURE"
            return _handle_openai_exception(e), trace
        finally:
            lf.set_run_outcome(
                total_tool_calls=tool_calls_used,
                total_latency_ms=(time.perf_counter() - loop_started) * 1000,
                hit_tool_cap=tool_calls_used >= MAX_TOOL_CALLS,
                success=agent_success,
                error_code=agent_error,
                llm_provider=LLM_PROVIDER,
                model=MODEL,
            )
            flush_langfuse()
            trace_id = getattr(langfuse_root, "trace_id", None)
            if trace_id:
                logger.info(
                    f"Langfuse trace_id={trace_id} "
                    "(cloud.langfuse.com → Tracing → Traces)"
                )


def generate_response(data: dict) -> dict:
    """
    Public entry used by Flask /process.
    Same request/response contract as before.
    """
    result, _trace = run_agent(data)
    return result
