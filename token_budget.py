"""
Estimate Groq/LLM prompt tokens for agent test cases (chars/4 heuristic).

Usage:
    python token_budget.py
    python token_budget.py --eval
"""

from __future__ import annotations

import argparse
import json
import sys

from config import (
    EVAL_MAX_TOOL_RESULT_CHARS,
    EVAL_USER_PREAMBLE,
    MAX_OUTPUT_TOKENS,
    resolve_system_prompt,
    resolve_user_template,
    set_agent_eval_mode,
)
from main import test_cases
from text_processing import substitute_template
from tools import AGENT_TOOLS, MAX_TOOL_RESULT_CHARS

# Groq reported ~1941 prompt_tokens on turn-1 with ~2300 char-est; scale heuristic.
CHARS_PER_TOKEN = 4.0


def est_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def _prepare_case(idx: int) -> dict:
    case = dict(test_cases[idx])
    case["retrieved_info"] = ""
    case["speech-length"] = "Short (5 minutes)"
    bio = case.get("bio") or ""
    if len(bio) > 600:
        case["bio"] = bio[:600] + "..."
    return case


def build_turn1_user_message(case: dict, *, eval_mode: bool) -> str:
    template = resolve_user_template(eval_mode=eval_mode)
    formatted = substitute_template(case, template_string=template)
    candidate = case.get("candidate-name", "")
    party = case.get("political-party", "")
    location = case.get("geographic-location", "")
    if eval_mode:
        return EVAL_USER_PREAMBLE.format(
            candidate=candidate, party=party, location=location
        ) + formatted
    return (
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
        f"{formatted}"
    )


def estimate_sequential_case(
    case: dict, *, eval_mode: bool, tool_calls: int = 3
) -> dict[str, int]:
    """Rough cumulative input tokens across sequential turns (re-sent history)."""
    sys_p = resolve_system_prompt(eval_mode=eval_mode)
    user0 = build_turn1_user_message(case, eval_mode=eval_mode)
    tools = json.dumps(AGENT_TOOLS)
    tool_cap = EVAL_MAX_TOOL_RESULT_CHARS if eval_mode else 4000

    turn_prompts: list[int] = []
    history_text = ""
    for turn in range(1, tool_calls + 2):
        if turn <= tool_calls:
            assistant_stub = "x" * 80
            tool_result = "x" * tool_cap
            chunk = history_text + assistant_stub + tool_result
            prompt = (
                est_tokens(sys_p)
                + est_tokens(user0)
                + est_tokens(tools)
                + est_tokens(chunk)
            )
            turn_prompts.append(prompt)
            history_text += assistant_stub + tool_result
        else:
            # final JSON turn (no tools)
            prompt = est_tokens(sys_p) + est_tokens(user0) + est_tokens(history_text) + 200
            turn_prompts.append(prompt)

    return {
        "turn1_prompt_est": turn_prompts[0] if turn_prompts else 0,
        "final_turn_prompt_est": turn_prompts[-1] if turn_prompts else 0,
        "sum_all_turn_prompts_est": sum(turn_prompts),
        "max_output_tokens_cap": MAX_OUTPUT_TOKENS,
        "estimated_case_total_est": sum(turn_prompts) + MAX_OUTPUT_TOKENS * len(turn_prompts),
    }


def print_report(*, eval_mode: bool) -> None:
    mode = "EVAL" if eval_mode else "PRODUCTION (Groq)"
    sys_p = resolve_system_prompt(eval_mode=eval_mode)
    tools_json = json.dumps(AGENT_TOOLS)

    print("=" * 72)
    print(f"TOKEN BUDGET REPORT - {mode}")
    print("=" * 72)
    print(f"System prompt:     {len(sys_p):>6} chars  ~{est_tokens(sys_p):>5} tokens")
    print(f"Tool schemas JSON: {len(tools_json):>6} chars  ~{est_tokens(tools_json):>5} tokens")
    if eval_mode:
        print(
            f"Tool result cap:   {EVAL_MAX_TOOL_RESULT_CHARS} chars/result "
            f"(prod {MAX_TOOL_RESULT_CHARS})"
        )
    else:
        print(f"Tool result cap:   {MAX_TOOL_RESULT_CHARS} chars/result")
    print(f"Max output/turn:   {MAX_OUTPUT_TOKENS} tokens")
    print()

    indices = [0, 1, 3]
    labels = ["Modi/BJP", "Rahul/INC", "Kejriwal/AAP"]
    for idx, label in zip(indices, labels):
        case = _prepare_case(idx)
        user_msg = build_turn1_user_message(case, eval_mode=eval_mode)
        formatted_only = substitute_template(
            case, template_string=resolve_user_template(eval_mode=eval_mode)
        )
        seq = estimate_sequential_case(case, eval_mode=eval_mode, tool_calls=3)
        print(f"--- Test case {idx} ({label}) ---")
        print(f"  Candidate fields template: {len(formatted_only):>5} chars  ~{est_tokens(formatted_only):>4} tok")
        print(f"  Full user message turn-1:  {len(user_msg):>5} chars  ~{est_tokens(user_msg):>4} tok")
        print(
            f"  Turn-1 API prompt (sys+user+tools): ~{seq['turn1_prompt_est']} tokens "
            f"(Groq logs prompt_tokens on each call)"
        )
        print(
            f"  Sequential 3-tool run: ~{seq['sum_all_turn_prompts_est']} input tokens "
            f"across {3 + 1} LLM calls (history re-sent each turn)"
        )
        print(
            f"  + completion budget up to {MAX_OUTPUT_TOKENS} tok/call -> "
            f"rough case ceiling ~{seq['estimated_case_total_est']} tokens"
        )
        print()

    print("Notes:")
    print("- System prompt on Groq is already compact (~450 tok); user template is larger.")
    print("- Tool results dominate history; eval mode truncates to EVAL_MAX_TOOL_RESULT_CHARS.")
    print("- Groq free tier TPD is 100k; divide by per-case estimate for max runs/day.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate agent token usage per test case")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Use EVAL_SYSTEMPROMPT + EVAL_TEMPLATE (test_agent --eval)",
    )
    args = parser.parse_args()
    if args.eval:
        set_agent_eval_mode(True)
    print_report(eval_mode=args.eval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
