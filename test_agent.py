"""
Run the tool-calling agent against sample form inputs and print tool-call traces.

Usage:
    python test_agent.py           # live LLM + real tools (needs working MODEL_URL credits)
    python test_agent.py --mock    # scripted LLM + stubbed tools (no network / no billing)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import uuid
from types import SimpleNamespace
from typing import Any

from config import set_agent_eval_mode
from llm import run_agent
from logger import clear_request_context, set_request_context
from main import test_cases


def _safe_print(text: str = "") -> None:
    """Avoid Windows cp1252 crashes on ₹ and other unicode in tool results."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def _print_trace(label: str, case: dict, trace: list[dict], result: dict) -> None:
    _safe_print("\n" + "=" * 80)
    _safe_print(f"TEST CASE: {label}")
    _safe_print("=" * 80)
    _safe_print(f"Candidate:     {case.get('candidate-name')}")
    _safe_print(f"Party:         {case.get('political-party')}")
    _safe_print(f"Policy points: {case.get('policy-points')}")
    _safe_print(f"Location:      {case.get('geographic-location')}")

    used = [e for e in trace if not e.get("skipped")]
    hit_cap = len(used) >= 3
    _safe_print("\n--- Tool-call sequence ---")
    if not used:
        _safe_print("(none — model answered without tools)")
    else:
        for entry in used:
            _safe_print(
                f"\n  [{entry.get('step')}] turn={entry.get('turn')} {entry.get('tool_name')}"
            )
            _safe_print(
                f"      Args:      {json.dumps(entry.get('arguments', {}), ensure_ascii=False)}"
            )
            _safe_print(f"      OK:        {entry.get('ok')}")
            _safe_print(f"      Latency:   {entry.get('latency_ms')} ms")
            _safe_print(
                f"      Raw turn tool_calls count (API message): "
                f"{entry.get('raw_turn_tool_calls_count')}"
            )
            _safe_print(
                f"      Raw turn tool_calls: "
                f"{json.dumps(entry.get('raw_turn_tool_calls'), ensure_ascii=False, indent=2)}"
            )
            discarded = entry.get("discarded_parallel") or []
            if discarded:
                _safe_print(f"      Discarded parallel (not executed): {discarded}")
            _safe_print(f"      Result:    {entry.get('result_summary')}")
            reasoning = entry.get("assistant_content")
            if reasoning and str(reasoning).strip():
                _safe_print(f"      Why/note:  {str(reasoning).strip()[:400]}")
            else:
                _safe_print(
                    "      Why/note:  (none visible — model issued tool call(s) "
                    "without a separate reasoning message)"
                )

    _safe_print("\n--- Cap / early-stop ---")
    _safe_print(f"Tool calls used: {len(used)} / 3")
    if hit_cap:
        _safe_print(
            "Stopped:         Used all 3 calls (hit cap; final answer forced after)."
        )
    else:
        _safe_print(
            "Stopped:         Early — final answer before hitting the 3-call cap."
        )

    _safe_print("\n--- Final parsed output ---")
    if result.get("error"):
        _safe_print(
            f"PARSE/API ERROR: {result.get('error')}: {result.get('message')}"
        )
    else:
        speech = result.get("speech")
        themes = result.get("key_themes")
        sentiment = result.get("sentiment")
        shape_ok = (
            isinstance(speech, str)
            and isinstance(themes, list)
            and isinstance(sentiment, dict)
            and "category" in sentiment
        )
        _safe_print(
            f"Shape OK:        {shape_ok}  "
            "(expects speech:str, key_themes:list, sentiment:dict)"
        )
        _safe_print(f"speech chars:    {len(speech or '')}")
        _safe_print(f"key_themes:      {themes}")
        _safe_print(f"sentiment:       {sentiment}")
        preview = (speech or "").replace("\n", " ").strip()
        _safe_print(f"speech preview:  {preview[:350]}...")


def _tool_call(name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"call_{uuid.uuid4().hex[:12]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _assistant_message(
    *,
    content: str | None = None,
    tool_calls: list | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=tool_calls or None)


class ScriptedCompletions:
    """Fake chat.completions that drives a multi-tool agent path."""

    def __init__(self, candidate: str, party: str, location: str):
        self.candidate = candidate
        self.party = party
        self.location = location
        self.turn = 0

    def create(self, **kwargs):
        self.turn += 1
        force_json = "response_format" in kwargs and not kwargs.get("tools")

        if force_json or self.turn >= 4:
            content = json.dumps(
                {
                    "speech": (
                        f"Friends of {self.location}, I am {self.candidate} of "
                        f"{self.party}. Together we will build opportunity and dignity."
                    ),
                    "key_themes": [
                        "Local development",
                        "Opportunity",
                        "Community trust",
                    ],
                    "sentiment": {
                        "category": "Inspirational",
                        "explanation": "Optimistic call to collective progress.",
                    },
                }
            )
            message = _assistant_message(content=content)
        elif self.turn == 1:
            message = _assistant_message(
                content=(
                    f"I will search the vector DB for {self.candidate} and {self.party}."
                ),
                tool_calls=[
                    _tool_call(
                        "search_vector_db",
                        {"query": f"{self.candidate} {self.party} {self.location}"},
                    )
                ],
            )
        elif self.turn == 2:
            message = _assistant_message(
                content="Vector context may be thin; searching the web for local facts.",
                tool_calls=[
                    _tool_call(
                        "search_web",
                        {
                            "query": (
                                f"{self.candidate} {self.location} recent development "
                                "policies"
                            )
                        },
                    )
                ],
            )
        else:
            message = _assistant_message(
                content="Fact-checking a key claim before drafting the speech.",
                tool_calls=[
                    _tool_call(
                        "fact_check_claim",
                        {
                            "claim": (
                                f"{self.candidate} has launched major education and "
                                f"healthcare initiatives relevant to {self.location}"
                            )
                        },
                    )
                ],
            )

        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class ScriptedClient:
    def __init__(self, candidate: str, party: str, location: str):
        self.chat = SimpleNamespace(
            completions=ScriptedCompletions(candidate, party, location)
        )


def stub_tool_executor(name: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Fast deterministic tool results for --mock (no LanceDB / Serper / scrape)."""
    if name == "search_vector_db":
        result = (
            f"[stub vector] Background notes for query={arguments.get('query')}: "
            "candidate career highlights, party platform, regional development themes."
        )
    elif name == "search_web":
        result = (
            f"[stub web] Recent coverage for query={arguments.get('query')}: "
            "local infrastructure announcements, employment schemes, heritage tourism."
        )
    elif name == "fact_check_claim":
        result = json.dumps(
            {
                "status": "supporting",
                "claim": arguments.get("claim", ""),
                "evidence": "[stub] Related policy references found in indexed notes.",
                "reason": "Stub executor returning supporting evidence for loop testing.",
            }
        )
    else:
        result = f"{name} failed: unknown tool, continue with available context"

    meta = {
        "tool_name": name,
        "arguments": arguments,
        "result_summary": result[:240],
        "latency_ms": 0.5,
        "ok": not result.startswith(f"{name} failed"),
    }
    print(
        f"  [stub tool] name={name} args={json.dumps(arguments, ensure_ascii=False)} "
        f"summary={meta['result_summary']}"
    )
    return result, meta


def run_cases(
    *,
    mock: bool,
    eval_mode: bool = False,
    case_indices: list[int] | None = None,
) -> int:
    if eval_mode:
        set_agent_eval_mode(True)
        print("\n>>> Eval mode ON (EVAL_SYSTEMPROMPT + compact template + shorter tool results)")
    labels_by_idx = {
        0: "Narendra Modi / BJP",
        1: "Rahul Gandhi / INC",
        3: "Arvind Kejriwal / AAP",
    }
    indices = case_indices if case_indices is not None else [0, 1, 3]
    selected = [(idx, labels_by_idx.get(idx, f"case-{idx}")) for idx in indices]

    summaries = []

    for idx, label in selected:
        case = dict(test_cases[idx])
        case["retrieved_info"] = ""
        case["speech-length"] = "Short (5 minutes)"
        # Keep live Groq requests under free-tier TPM by truncating huge bios.
        bio = case.get("bio") or ""
        if len(bio) > 600:
            case["bio"] = bio[:600] + "..."

        request_id = f"test-agent-{'mock-' if mock else ''}{idx}"
        set_request_context(request_id, "test_agent")
        mode = "MOCK" if mock else "LIVE"
        print(f"\n>>> [{mode}] Running agent for: {label} (request_id={request_id})")
        if not mock and summaries:
            time.sleep(8)  # respect Groq TPM between cases

        client = None
        tool_executor = None
        if mock:
            client = ScriptedClient(
                candidate=case.get("candidate-name", ""),
                party=case.get("political-party", ""),
                location=case.get("geographic-location", ""),
            )
            tool_executor = stub_tool_executor

        try:
            result, trace = run_agent(
                case, client=client, tool_executor=tool_executor, eval_mode=eval_mode
            )
        except Exception:
            traceback.print_exc()
            clear_request_context()
            return 1

        clear_request_context()
        _print_trace(label, case, trace, result)

        tools_order = [e.get("tool_name") for e in trace if not e.get("skipped")]
        summaries.append(
            {
                "label": label,
                "tools_order": tools_order,
                "distinct": sorted(set(tools_order)),
                "used_count": len(tools_order),
                "hit_cap": len(tools_order) >= 3,
                "ok": "error" not in result,
                "has_fact_check": "fact_check_claim" in tools_order,
            }
        )

    print("\n" + "#" * 80)
    print("CROSS-RUN SUMMARY")
    print("#" * 80)
    all_distinct = set()
    any_fact_check = False
    any_early_stop = False
    for s in summaries:
        print(
            f"- {s['label']}: tools={s['tools_order']} "
            f"used={s['used_count']}/3 hit_cap={s['hit_cap']} ok={s['ok']}"
        )
        all_distinct.update(s["distinct"])
        any_fact_check = any_fact_check or s["has_fact_check"]
        any_early_stop = any_early_stop or (not s["hit_cap"])

    print(f"\nDistinct tools seen across all runs: {sorted(all_distinct)}")
    if any_fact_check:
        print("fact_check_claim: CALLED in at least one test case.")
    else:
        print(
            "fact_check_claim: NEVER called in ANY of the test cases "
            "(model never chose it)."
        )
    if any_early_stop:
        print("Early stop: at least one case finished before using all 3 tool calls.")
    else:
        print("Early stop: NONE — every case used all 3 tool calls (hit the cap).")

    if len(all_distinct) < 2:
        print(
            "WARNING: Expected more than one tool type across runs. "
            "For LIVE mode, fix LLM credentials and re-run without --mock."
        )
        return 2

    print("OK: Multiple tool types observed across runs.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify speech agent tool-calling")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Scripted LLM + stubbed tools (no billing / no network tools)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Use compact EVAL prompts/template (saves Groq TPD; not used by /process)",
    )
    parser.add_argument(
        "--token-budget",
        action="store_true",
        help="Print token estimates and exit (see token_budget.py)",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Comma-separated test_cases indices only (e.g. 1,3)",
    )
    args = parser.parse_args()
    if args.token_budget:
        from token_budget import print_report

        if args.eval:
            set_agent_eval_mode(True)
        print_report(eval_mode=args.eval)
        return 0
    case_indices = None
    if args.cases:
        case_indices = [int(x.strip()) for x in args.cases.split(",") if x.strip()]
    return run_cases(
        mock=args.mock, eval_mode=args.eval, case_indices=case_indices
    )


if __name__ == "__main__":
    sys.exit(main())
