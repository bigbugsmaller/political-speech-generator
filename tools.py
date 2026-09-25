"""Tool implementations and OpenAI-compatible schemas for the speech agent."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from config import EVAL_MAX_TOOL_RESULT_CHARS, is_agent_eval_mode
from database import insert_text_into_db, table
from logger import logger
from retriever import search_with_threshold
from serper_api import fetch_additional_results

MAX_TOOL_RESULT_CHARS = 4000


def _tool_result_char_limit() -> int:
    if is_agent_eval_mode():
        return EVAL_MAX_TOOL_RESULT_CHARS
    return MAX_TOOL_RESULT_CHARS

# Contradiction cues used for a lightweight fact-check heuristic (no extra LLM call).
_CONTRADICTION_PATTERNS = re.compile(
    r"\b("
    r"false|incorrect|inaccurate|debunked|myth|misleading|not true|untrue|"
    r"denied|refuted|disputed|no evidence|never happened|fabricated|"
    r"contradicts|contrary to|overstated|exaggerated"
    r")\b",
    re.IGNORECASE,
)


AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_vector_db",
            "description": (
                "Search the local LanceDB vector store for background context about a "
                "candidate, party, location, policy, or event. Use this first before "
                "search_web. Returns concatenated relevant text chunks, or an empty "
                "result if nothing meets the similarity threshold."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query for the vector database.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the live web via Serper, scrape paragraph text from top results, "
                "optionally index new pages into LanceDB, and return the extracted text. "
                "Use when the vector DB has little or no useful context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Web search query (candidate, policy, local issue, etc.).",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fact_check_claim",
            "description": (
                "Verify a specific factual claim you plan to include in the speech "
                "(statistic, date, policy figure, historical fact). Re-runs retrieval "
                "against that claim and returns supporting evidence, contradicting "
                "evidence, or unverified."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": "The exact factual claim to verify.",
                    }
                },
                "required": ["claim"],
            },
        },
    },
]


def _truncate(text: str, limit: int | None = None) -> str:
    if limit is None:
        limit = _tool_result_char_limit()
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text) - limit} more chars]"


def _summarize_result(result: str, limit: int = 240) -> str:
    compact = " ".join((result or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def search_vector_db(query: str) -> str:
    """Vector-only LanceDB search (no automatic Serper fallback)."""
    started = time.perf_counter()
    logger.info(f"[tool:search_vector_db] START query={query!r}")
    if not query or not str(query).strip():
        return "No results: empty query."

    # max_recursion=0 keeps this tool distinct from search_web.
    result = search_with_threshold(
        table,
        str(query).strip(),
        threshold=0.75,
        max_recursion=0,
    )
    logger.info(
        f"[tool:search_vector_db] DONE chars={len(result or '')} "
        f"elapsed_ms={(time.perf_counter() - started) * 1000:.1f}"
    )
    if not result:
        return "No relevant chunks found in the vector database for this query."
    return _truncate(result)


def search_web(query: str) -> str:
    """Serper search + scrape; indexes new pages into LanceDB when found."""
    started = time.perf_counter()
    logger.info(f"[tool:search_web] START query={query!r}")
    if not query or not str(query).strip():
        return "No results: empty query."

    query = str(query).strip()
    logger.info("[tool:search_web] BEFORE fetch_additional_results")
    scraped = fetch_additional_results(table, query, min_results=5)
    logger.info(
        f"[tool:search_web] AFTER fetch_additional_results "
        f"sources={len(scraped or {})}"
    )
    if not scraped:
        return "No web results found or no page text could be extracted."

    try:
        logger.info("[tool:search_web] BEFORE insert_text_into_db")
        insert_text_into_db(scraped)
        logger.info("[tool:search_web] AFTER insert_text_into_db")
    except Exception as e:
        logger.warning(f"[tool:search_web] failed to index scraped pages: {e}")

    parts = []
    for source_id, text in scraped.items():
        snippet = _truncate(text, limit=1200)
        parts.append(f"Source: {source_id}\n{snippet}")

    combined = "\n\n---\n\n".join(parts)
    logger.info(
        f"[tool:search_web] DONE chars={len(combined)} "
        f"elapsed_ms={(time.perf_counter() - started) * 1000:.1f}"
    )
    return _truncate(combined)


def fact_check_claim(claim: str) -> str:
    """
    Re-retrieve evidence for a specific claim and classify as
    supporting / contradicting / unverified.
    """
    if not claim or not str(claim).strip():
        return json.dumps(
            {
                "status": "unverified",
                "claim": claim,
                "evidence": "",
                "reason": "Empty claim provided.",
            }
        )

    claim = str(claim).strip()
    started = time.perf_counter()
    logger.info(f"[tool:fact_check_claim] START claim={claim!r}")
    evidence_parts: list[str] = []

    try:
        logger.info("[tool:fact_check_claim] BEFORE vector search")
        vector_hits = search_with_threshold(
            table, claim, threshold=0.70, max_recursion=0
        )
        logger.info(
            f"[tool:fact_check_claim] AFTER vector search "
            f"chars={len(vector_hits or '')}"
        )
        if vector_hits:
            evidence_parts.append(f"[vector_db]\n{_truncate(vector_hits, 2000)}")
    except Exception as e:
        logger.warning(f"[tool:fact_check_claim] vector search failed: {e}")

    if not evidence_parts:
        try:
            logger.info("[tool:fact_check_claim] BEFORE web fallback")
            scraped = fetch_additional_results(table, claim, min_results=3)
            logger.info(
                f"[tool:fact_check_claim] AFTER web fallback "
                f"sources={len(scraped or {})}"
            )
            if scraped:
                try:
                    insert_text_into_db(scraped)
                except Exception as e:
                    logger.warning(f"[tool:fact_check_claim] index failed: {e}")
                for source_id, text in list(scraped.items())[:3]:
                    evidence_parts.append(
                        f"[web:{source_id}]\n{_truncate(text, 1000)}"
                    )
        except Exception as e:
            logger.warning(f"[tool:fact_check_claim] web search failed: {e}")

    if not evidence_parts:
        payload = {
            "status": "unverified",
            "claim": claim,
            "evidence": "",
            "reason": "No relevant evidence found in the vector DB or on the web.",
        }
        return json.dumps(payload)

    evidence_text = "\n\n".join(evidence_parts)
    if _CONTRADICTION_PATTERNS.search(evidence_text):
        status = "contradicting"
        reason = (
            "Retrieved evidence contains contradiction/debunking language "
            "relative to the claim topic; review carefully before using the claim."
        )
    else:
        status = "supporting"
        reason = (
            "Relevant evidence was retrieved that appears topically related "
            "to the claim. Prefer citing only details grounded in this evidence."
        )

    payload = {
        "status": status,
        "claim": claim,
        "evidence": _truncate(evidence_text, 3500),
        "reason": reason,
    }
    logger.info(
        f"[tool:fact_check_claim] DONE status={status} "
        f"elapsed_ms={(time.perf_counter() - started) * 1000:.1f}"
    )
    return json.dumps(payload)


TOOL_HANDLERS: dict[str, Callable[..., str]] = {
    "search_vector_db": search_vector_db,
    "search_web": search_web,
    "fact_check_claim": fact_check_claim,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Execute a named tool and return (result_text, log_meta).
    Never raises — failures become an error string for the model.
    """
    started = time.perf_counter()
    meta: dict[str, Any] = {
        "tool_name": name,
        "arguments": arguments,
        "result_summary": "",
        "latency_ms": 0.0,
        "ok": True,
    }

    logger.info(f"[execute_tool] BEFORE name={name} args={arguments}")
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        result = f"{name} failed: unknown tool"
        meta["ok"] = False
        meta["result_summary"] = result
        meta["latency_ms"] = (time.perf_counter() - started) * 1000
        logger.error(
            f"[execute_tool] unknown tool name={name} args={arguments} "
            f"latency_ms={meta['latency_ms']:.2f}"
        )
        return result, meta

    try:
        if name == "fact_check_claim":
            result = handler(claim=arguments.get("claim", ""))
        else:
            result = handler(query=arguments.get("query", ""))
    except Exception as e:
        result = f"{name} failed: {e}, continue with available context"
        meta["ok"] = False

    meta["result_summary"] = _summarize_result(result)
    meta["latency_ms"] = (time.perf_counter() - started) * 1000
    logger.info(
        f"[execute_tool] AFTER name={meta['tool_name']} ok={meta['ok']} "
        f"result_summary={meta['result_summary']} "
        f"latency_ms={meta['latency_ms']:.2f}"
    )
    return result, meta
