"""LangGraph workflow for the political speech tool-calling agent."""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from langfuse_tracing import AgentTraceSession, _NoOpObservation
from logger import logger
from text_processing import parse_model_response
from tools import execute_tool as default_execute_tool

from llm import (
    FINAL_ANSWER_INSTRUCTION,
    FORCE_JSON_INSTRUCTION,
    MAX_TOOL_CALLS,
    ToolExecutor,
    _chat_completion,
    _dump_tool_calls_raw,
    _parse_tool_arguments,
    _serialize_assistant_message,
)

Route = Literal["tools", "parse", "force_json"]


class SpeechAgentState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    tool_calls_used: int
    turn_index: int
    trace: list[dict[str, Any]]
    result: dict[str, Any] | None
    agent_success: bool
    agent_error: str | None
    force_final: bool
    final_content: str | None
    pending_tool_call: dict[str, Any] | None
    last_raw_tool_calls: list[dict[str, Any]]
    last_assistant_content: str | None
    runtime: dict[str, Any]


def _runtime(state: SpeechAgentState, config: RunnableConfig | None) -> dict[str, Any]:
    rt = state.get("runtime") or {}
    cfg = _cfg(config)
    return {
        "client": cfg.get("client") if cfg.get("client") is not None else rt.get("client"),
        "tool_executor": cfg.get("tool_executor") or rt.get("tool_executor"),
        "lf": cfg.get("lf") if cfg.get("lf") is not None else rt.get("lf"),
    }


def _cfg(config: RunnableConfig | None) -> dict[str, Any]:
    if config is None:
        return {}
    configurable = config.get("configurable")
    if isinstance(configurable, dict):
        return configurable
    return {}


def agent_llm_node(
    state: SpeechAgentState, config: RunnableConfig | None = None
) -> SpeechAgentState:
    """LLM turn: optional cap instruction, chat completion, append assistant message."""
    rt = _runtime(state, config)
    client = rt.get("client")
    lf: AgentTraceSession | None = rt.get("lf")

    messages = list(state["messages"])
    tool_calls_used = int(state.get("tool_calls_used") or 0)
    turn_index = int(state.get("turn_index") or 0)

    force_final = tool_calls_used >= MAX_TOOL_CALLS
    use_tools = not force_final
    require_tool = use_tools and tool_calls_used == 0

    if force_final:
        messages.append({"role": "user", "content": FINAL_ANSWER_INSTRUCTION})
        logger.info(
            f"[agent] Tool-call cap reached ({MAX_TOOL_CALLS}); forcing final answer"
        )

    turn_index += 1
    logger.info(
        f"[agent] LOOP turn={turn_index} tool_calls_used={tool_calls_used} "
        f"force_final={force_final} require_tool={require_tool}"
    )

    response = _chat_completion(
        messages,
        use_tools=use_tools,
        force_final=force_final,
        require_tool=require_tool,
        client=client,
        lf=lf,
        turn_index=turn_index,
    )
    message = response.choices[0].message
    raw_tool_calls = list(getattr(message, "tool_calls", None) or [])
    raw_dump = _dump_tool_calls_raw(raw_tool_calls)
    logger.info(
        f"[agent] RAW turn={turn_index} tool_calls_in_message="
        f"{len(raw_tool_calls)} content_chars="
        f"{len(message.content or '')} raw_tool_calls="
        f"{json.dumps(raw_dump, ensure_ascii=False)}"
    )

    if len(raw_tool_calls) > 1:
        discarded = [tc.function.name for tc in raw_tool_calls[1:]]
        logger.warning(
            f"[agent] parallel_tool_calls safeguard: got "
            f"{len(raw_tool_calls)} tool_calls; executing only the first "
            f"({raw_tool_calls[0].function.name}); discarding={discarded}"
        )
    tool_calls = raw_tool_calls[:1]
    messages.append(_serialize_assistant_message(message, tool_calls=tool_calls or None))

    logger.info(
        f"[agent] Model returned tool_calls={len(raw_tool_calls)} "
        f"(executing={len(tool_calls)}) "
        f"content_chars={len(message.content or '')}"
    )

    pending_tool_call = None
    if tool_calls:
        tc = tool_calls[0]
        pending_tool_call = {
            "id": tc.id,
            "name": tc.function.name,
            "arguments": _parse_tool_arguments(tc.function.arguments),
        }

    return {
        **state,
        "messages": messages,
        "turn_index": turn_index,
        "force_final": force_final,
        "final_content": message.content,
        "pending_tool_call": pending_tool_call,
        "last_raw_tool_calls": raw_dump,
        "last_assistant_content": message.content,
    }


def route_after_llm(state: SpeechAgentState) -> Route:
    force_final = bool(state.get("force_final"))
    pending = state.get("pending_tool_call")
    content = state.get("final_content") or ""

    if force_final or not pending:
        if not content and not force_final:
            return "force_json"
        return "parse"
    return "tools"


def execute_tool_node(
    state: SpeechAgentState, config: RunnableConfig | None = None
) -> SpeechAgentState:
    """Run one tool call and append the tool result to message history."""
    rt = _runtime(state, config)
    run_tool: ToolExecutor = rt.get("tool_executor") or default_execute_tool
    lf: AgentTraceSession | None = rt.get("lf")

    messages = list(state["messages"])
    trace = list(state.get("trace") or [])
    tool_calls_used = int(state.get("tool_calls_used") or 0)
    turn_index = int(state.get("turn_index") or 0)
    pending = state.get("pending_tool_call") or {}
    raw_dump = state.get("last_raw_tool_calls") or []

    tool_name = pending.get("name", "")
    tool_id = pending.get("id", "")
    args = pending.get("arguments") or {}

    if tool_calls_used >= MAX_TOOL_CALLS:
        skip_msg = (
            f"{tool_name} failed: tool-call limit of "
            f"{MAX_TOOL_CALLS} reached, continue with available context"
        )
        logger.warning(f"[agent] Skipping tool due to cap: {tool_name}")
        messages.append({"role": "tool", "tool_call_id": tool_id, "content": skip_msg})
        trace.append(
            {
                "step": len(trace) + 1,
                "turn": turn_index,
                "tool_name": tool_name,
                "arguments": args,
                "result_summary": skip_msg,
                "latency_ms": 0.0,
                "ok": False,
                "skipped": True,
                "assistant_content": state.get("last_assistant_content"),
                "raw_turn_tool_calls_count": len(raw_dump),
                "raw_turn_tool_calls": raw_dump,
                "executed_this_turn": 0,
            }
        )
        return {
            **state,
            "messages": messages,
            "trace": trace,
            "pending_tool_call": None,
        }

    logger.info(f"[agent] BEFORE tool={tool_name} args={args}")
    tool_started = time.perf_counter()

    def _run_tool_call() -> tuple[str, dict[str, Any]]:
        try:
            return run_tool(tool_name, args)
        except Exception as e:
            result_text = f"{tool_name} failed: {e}, continue with available context"
            meta = {
                "tool_name": tool_name,
                "arguments": args,
                "result_summary": result_text,
                "latency_ms": 0.0,
                "ok": False,
            }
            logger.error(
                f"[agent] tool exception name={tool_name} args={args} error={e}"
            )
            return result_text, meta

    if lf is not None:
        with lf.tool_call(tool_name=tool_name, arguments=args) as tool_span:
            result_text, meta = _run_tool_call()
            if not isinstance(tool_span, _NoOpObservation):
                tool_span.update(
                    output=meta.get("result_summary", (result_text or "")[:240])
                )
    else:
        result_text, meta = _run_tool_call()

    logger.info(
        f"[agent] AFTER tool={tool_name} ok={meta.get('ok')} "
        f"elapsed_ms={(time.perf_counter() - tool_started) * 1000:.1f} "
        f"result_chars={len(result_text or '')}"
    )

    tool_calls_used += 1
    trace.append(
        {
            "step": len(trace) + 1,
            "turn": turn_index,
            "tool_name": meta.get("tool_name", tool_name),
            "arguments": meta.get("arguments", args),
            "result_summary": meta.get("result_summary", result_text[:240]),
            "latency_ms": round(float(meta.get("latency_ms", 0.0)), 2),
            "ok": bool(meta.get("ok", True)),
            "assistant_content": state.get("last_assistant_content"),
            "raw_turn_tool_calls_count": len(raw_dump),
            "raw_turn_tool_calls": raw_dump,
            "executed_this_turn": 1,
            "discarded_parallel": [c["function"]["name"] for c in raw_dump[1:]],
        }
    )

    messages.append({"role": "tool", "tool_call_id": tool_id, "content": result_text})

    return {
        **state,
        "messages": messages,
        "trace": trace,
        "tool_calls_used": tool_calls_used,
        "pending_tool_call": None,
    }


def force_json_node(
    state: SpeechAgentState, config: RunnableConfig | None = None
) -> SpeechAgentState:
    """Follow-up LLM call when the model returned empty content without hitting tool cap."""
    rt = _runtime(state, config)
    client = rt.get("client")
    lf: AgentTraceSession | None = rt.get("lf")
    turn_index = int(state.get("turn_index") or 0)

    messages = list(state["messages"])
    logger.info("[agent] Empty content; forcing JSON follow-up")
    messages.append({"role": "user", "content": FORCE_JSON_INSTRUCTION})

    response = _chat_completion(
        messages,
        use_tools=False,
        force_final=True,
        client=client,
        lf=lf,
        turn_index=turn_index + 100,
    )
    content = response.choices[0].message.content
    logger.debug(f"Raw forced JSON response: {content}")

    return {**state, "messages": messages, "final_content": content}


def parse_response_node(
    state: SpeechAgentState, config: RunnableConfig | None = None
) -> SpeechAgentState:
    """Parse final JSON speech from LLM content."""
    rt = _runtime(state, config)
    lf: AgentTraceSession | None = rt.get("lf")
    content = state.get("final_content") or ""
    logger.debug(f"Raw final response: {content}")

    if not content:
        logger.error("[agent] Received empty response from API")
        return {
            **state,
            "result": {
                "error": "ERR_EMPTY_RESPONSE",
                "message": "Received empty response from API",
            },
            "agent_success": False,
            "agent_error": "ERR_EMPTY_RESPONSE",
        }

    def _parse() -> SpeechAgentState:
        try:
            parsed = parse_model_response(content)
            tool_calls_used = int(state.get("tool_calls_used") or 0)
            trace = state.get("trace") or []
            logger.info(
                f"[agent] DONE tool_calls={tool_calls_used} "
                f"trace_steps={len(trace)}"
            )
            return {
                **state,
                "result": parsed,
                "agent_success": True,
                "agent_error": None,
            }
        except Exception as e:
            logger.error(f"[agent] Failed to parse model response: {e}")
            return {
                **state,
                "result": {
                    "error": "ERR_PARSING_FAILURE",
                    "message": f"Failed to parse model response: {e}",
                },
                "agent_success": False,
                "agent_error": "ERR_PARSING_FAILURE",
            }

    if lf is not None:
        with lf.parse_response(raw_content=content) as parse_span:
            outcome = _parse()
            if not isinstance(parse_span, _NoOpObservation):
                if outcome.get("agent_success"):
                    parsed = outcome.get("result") or {}
                    parse_span.update(
                        output={
                            "success": True,
                            "speech_chars": len(parsed.get("speech") or ""),
                            "key_themes": parsed.get("key_themes"),
                            "sentiment_category": (parsed.get("sentiment") or {}).get(
                                "category"
                            ),
                        }
                    )
                else:
                    parse_span.update(
                        output={
                            "success": False,
                            "error": (outcome.get("result") or {}).get("message"),
                        }
                    )
            return outcome

    return _parse()


def build_speech_agent_graph():
    """Compile the LangGraph workflow: agent_llm ↔ tools → parse."""
    graph = StateGraph(SpeechAgentState)
    graph.add_node("agent_llm", agent_llm_node)
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_node("force_json", force_json_node)
    graph.add_node("parse_response", parse_response_node)

    graph.add_edge(START, "agent_llm")
    graph.add_conditional_edges(
        "agent_llm",
        route_after_llm,
        {
            "tools": "execute_tool",
            "parse": "parse_response",
            "force_json": "force_json",
        },
    )
    graph.add_edge("execute_tool", "agent_llm")
    graph.add_edge("force_json", "parse_response")
    graph.add_edge("parse_response", END)

    return graph.compile()


_COMPILED_GRAPH = None


def get_speech_agent_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_speech_agent_graph()
    return _COMPILED_GRAPH


def invoke_speech_agent(
    *,
    messages: list[dict[str, Any]],
    client: Any = None,
    tool_executor: ToolExecutor | None = None,
    lf: AgentTraceSession | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], SpeechAgentState]:
    """
    Run the compiled LangGraph agent.

    Returns (result_dict, tool_trace, final_state).
    """
    initial: SpeechAgentState = {
        "messages": messages,
        "tool_calls_used": 0,
        "turn_index": 0,
        "trace": [],
        "result": None,
        "agent_success": False,
        "agent_error": None,
        "force_final": False,
        "final_content": None,
        "pending_tool_call": None,
        "last_raw_tool_calls": [],
        "last_assistant_content": None,
        "runtime": {
            "client": client,
            "tool_executor": tool_executor,
            "lf": lf,
        },
    }

    graph = get_speech_agent_graph()
    final_state = graph.invoke(
        initial,
        config={
            "recursion_limit": 25,
            "configurable": {
                "client": client,
                "tool_executor": tool_executor,
                "lf": lf,
            },
        },
    )

    result = final_state.get("result") or {
        "error": "ERR_EMPTY_RESPONSE",
        "message": "Agent graph finished without a result",
    }
    trace = final_state.get("trace") or []
    return result, trace, final_state
