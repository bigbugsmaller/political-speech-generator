"""
RAGAS evaluation for the political speech agent.

Runs each sample through the same agent path as Flask /process (run_agent / generate_response
logic with eval_mode=False), captures retrieval tool output, scores with RAGAS, and writes
eval_results.json.

Usage:
    set LLM_PROVIDER=openrouter
  rem If the default :free OpenRouter slug 404s, set a working model, e.g.:
    set OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct
    py eval.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Default provider for eval runs (override via env).
os.environ.setdefault("LLM_PROVIDER", "openrouter")

from openai import APIStatusError, RateLimitError  # noqa: E402

from eval_ragas_llm import build_ragas_embeddings, build_ragas_llm  # noqa: E402
from llm import run_agent  # noqa: E402
from logger import clear_request_context, logger, set_request_context  # noqa: E402
from tools import execute_tool  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "eval_dataset.json"
RESULTS_PATH = ROOT / "eval_results.json"

RETRIEVAL_TOOLS = frozenset({"search_vector_db", "search_web", "fact_check_claim"})


def _safe_print(text: str = "") -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"No samples in {path}")
    return samples


def build_user_input(form: dict[str, Any]) -> str:
    """Effective user request for RAGAS (question / user_input)."""
    lines = [
        (
            f"Generate a {form.get('speech-type', 'campaign')} speech for "
            f"{form.get('candidate-name')} ({form.get('political-party')}) "
            f"in {form.get('geographic-location')}."
        ),
        f"Office sought: {form.get('office-sought', '')}",
        f"Speech length: {form.get('speech-length', '')}",
        f"Tone / speech-tone: {form.get('tone', '')} | {form.get('speech-tone', '')}",
        f"Slogan: {form.get('slogan', '')}",
        f"Main message: {form.get('main-message', '')}",
        f"Policy points: {form.get('policy-points', '')}",
        f"Key messages: {form.get('key-messages', '')}",
        f"Primary objective: {form.get('primary-objective', '')}",
        f"Language / dialect: {form.get('language-dialect', '')}",
        f"Call to action: {form.get('call-to-action', '')}",
    ]
    return "\n".join(line for line in lines if line.strip())


def contexts_from_trace(trace: list[dict[str, Any]]) -> list[str]:
    chunks: list[str] = []
    for entry in trace:
        if entry.get("skipped"):
            continue
        name = entry.get("tool_name")
        if name not in RETRIEVAL_TOOLS:
            continue
        args = entry.get("arguments") or {}
        summary = entry.get("result_summary") or ""
        label = name
        if name in ("search_vector_db", "search_web"):
            label = f"{name}(query={args.get('query', '')!r})"
        elif name == "fact_check_claim":
            label = f"fact_check_claim(claim={args.get('claim', '')!r})"
        chunks.append(f"[{label}]\n{summary}")
    return chunks


def make_capturing_executor(store: list[str]):
    def _executor(name: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        result_text, meta = execute_tool(name, arguments)
        if name in RETRIEVAL_TOOLS:
            if name in ("search_vector_db", "search_web"):
                q = arguments.get("query", "")
                store.append(f"[{name}] query={q}\n{result_text}")
            else:
                store.append(f"[{name}]\n{result_text}")
        return result_text, meta

    return _executor


def is_rate_limit_error(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError) and getattr(exc, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return "rate limit" in text or "429" in text


def is_rate_limit_result(result: dict[str, Any]) -> bool:
    if result.get("error") != "ERR_API_FAILURE":
        return False
    msg = (result.get("message") or "").lower()
    return "rate limit" in msg or "429" in msg


def run_generation(
    sample_id: str,
    form: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str], list[dict[str, Any]], str | None]:
    """
    Returns (result_dict, full_context_chunks, trace, skip_reason).
    skip_reason set on rate limit or hard failure.
    """
    case = dict(form)
    case["retrieved_info"] = ""
    bio = case.get("bio") or ""
    if len(bio) > 800:
        case["bio"] = bio[:800] + "..."

    captured: list[str] = []
    set_request_context(f"eval-{sample_id}", "eval.py")
    try:
        result, trace = run_agent(
            case,
            tool_executor=make_capturing_executor(captured),
            eval_mode=False,
        )
    except Exception as e:
        if is_rate_limit_error(e):
            return None, captured, [], f"rate_limit: {e}"
        raise
    finally:
        clear_request_context()

    if result.get("error"):
        if is_rate_limit_result(result):
            return None, captured, trace, f"rate_limit: {result.get('message')}"
        return None, captured, trace, f"agent_error: {result.get('error')}: {result.get('message')}"

    if not captured:
        captured = contexts_from_trace(trace)
    if not captured:
        captured = ["(No retrieval tool returned context for this run.)"]

    return result, captured, trace, None


def score_with_ragas(rows: list[dict[str, Any]]) -> list[dict[str, float | None]]:
    from ragas import evaluate
    from ragas.metrics import (
        LLMContextPrecisionWithoutReference,
        answer_relevancy,
        faithfulness,
    )

    from datasets import Dataset

    llm = build_ragas_llm()
    embeddings = build_ragas_embeddings()
    context_precision_metric = LLMContextPrecisionWithoutReference()

    dataset = Dataset.from_list(rows)
    try:
        eval_result = evaluate(
            dataset,
            metrics=[faithfulness, context_precision_metric, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
            show_progress=True,
        )
    except Exception as e:
        if is_rate_limit_error(e):
            raise
        raise

    df = eval_result.to_pandas()
    scores: list[dict[str, float | None]] = []
    for _, row in df.iterrows():
        scores.append(
            {
                "faithfulness": _float_or_none(row.get("faithfulness")),
                "context_precision": _float_or_none(
                    row.get("llm_context_precision_without_reference")
                ),
                "answer_relevancy": _float_or_none(row.get("answer_relevancy")),
            }
        )
    return scores


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def average_metric(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def print_summary_table(
    completed: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    averages: dict[str, float | None],
    worst: dict[str, Any] | None,
) -> None:
    _safe_print("\n" + "=" * 88)
    _safe_print("RAGAS EVAL SUMMARY")
    _safe_print("=" * 88)
    _safe_print(f"Completed: {len(completed)}   Skipped: {len(skipped)}")
    _safe_print(
        f"Averages (completed only): "
        f"faithfulness={averages.get('faithfulness')}  "
        f"context_precision={averages.get('context_precision')}  "
        f"answer_relevancy={averages.get('answer_relevancy')}"
    )
    if worst:
        _safe_print(
            f"Lowest faithfulness: {worst.get('faithfulness')} — "
            f"id={worst.get('id')} ({worst.get('label')})"
        )
    _safe_print("\nPer-sample scores:")
    _safe_print(f"{'id':<28} {'faith':>8} {'ctx_prec':>10} {'ans_rel':>10}  status")
    _safe_print("-" * 88)
    for row in completed:
        sc = row.get("scores") or {}
        _safe_print(
            f"{row.get('id', '')[:28]:<28} "
            f"{_fmt(sc.get('faithfulness')):>8} "
            f"{_fmt(sc.get('context_precision')):>10} "
            f"{_fmt(sc.get('answer_relevancy')):>10}  ok"
        )
    for row in skipped:
        _safe_print(
            f"{row.get('id', '')[:28]:<28} "
            f"{'—':>8} {'—':>10} {'—':>10}  SKIP: {row.get('reason', '')[:30]}"
        )
    _safe_print("=" * 88)


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS eval for speech agent")
    parser.add_argument(
        "--sleep",
        type=float,
        default=8.0,
        help="Seconds between agent runs (rate-limit cushion)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only first N samples (0 = all)",
    )
    args = parser.parse_args()

    samples = load_dataset(DATASET_PATH)
    if args.limit > 0:
        samples = samples[: args.limit]

    completed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    ragas_rows: list[dict[str, Any]] = []
    ragas_index_map: list[int] = []

    for i, sample in enumerate(samples):
        sample_id = sample.get("id", f"sample-{i}")
        label = sample.get("label", sample_id)
        form = sample.get("input")
        if not isinstance(form, dict):
            skipped.append({"id": sample_id, "label": label, "reason": "invalid input"})
            continue

        if i > 0 and args.sleep > 0:
            time.sleep(args.sleep)

        _safe_print(f"\n>>> [{i + 1}/{len(samples)}] Generating speech: {label}")
        try:
            result, contexts, trace, skip_reason = run_generation(sample_id, form)
        except Exception as e:
            if is_rate_limit_error(e):
                reason = f"rate_limit: {e}"
                logger.error(f"Eval skip {sample_id}: {reason}")
                skipped.append({"id": sample_id, "label": label, "reason": reason})
                continue
            logger.error(f"Eval failed {sample_id}: {e}\n{traceback.format_exc()}")
            skipped.append({"id": sample_id, "label": label, "reason": str(e)})
            continue

        if skip_reason:
            logger.error(f"Eval skip {sample_id}: {skip_reason}")
            skipped.append({"id": sample_id, "label": label, "reason": skip_reason})
            continue

        speech = (result or {}).get("speech") or ""
        user_input = build_user_input(form)
        entry = {
            "id": sample_id,
            "label": label,
            "user_input": user_input,
            "retrieved_contexts": contexts,
            "response": speech,
            "agent_output": result,
            "tool_trace_summary": [
                {
                    "tool_name": t.get("tool_name"),
                    "arguments": t.get("arguments"),
                    "ok": t.get("ok"),
                }
                for t in trace
                if not t.get("skipped")
            ],
        }
        completed.append(entry)
        ragas_rows.append(
            {
                "user_input": user_input,
                "retrieved_contexts": contexts,
                "response": speech,
            }
        )
        ragas_index_map.append(len(completed) - 1)

    if ragas_rows:
        _safe_print(f"\n>>> Scoring {len(ragas_rows)} samples with RAGAS...")
        try:
            metric_scores = score_with_ragas(ragas_rows)
        except Exception as e:
            if is_rate_limit_error(e):
                logger.error(f"RAGAS scoring hit rate limit: {e}")
                _safe_print(f"\nRAGAS scoring failed (rate limit):\n{e}")
                for entry in completed:
                    entry["scores"] = None
                    entry["scores_error"] = str(e)
            else:
                _safe_print(f"\nRAGAS scoring failed:\n{traceback.format_exc()}")
                raise
        else:
            for idx, scores in zip(ragas_index_map, metric_scores):
                completed[idx]["scores"] = scores

    faith_vals = [
        e["scores"]["faithfulness"]
        for e in completed
        if e.get("scores") and e["scores"].get("faithfulness") is not None
    ]
    ctx_vals = [
        e["scores"]["context_precision"]
        for e in completed
        if e.get("scores") and e["scores"].get("context_precision") is not None
    ]
    rel_vals = [
        e["scores"]["answer_relevancy"]
        for e in completed
        if e.get("scores") and e["scores"].get("answer_relevancy") is not None
    ]
    averages = {
        "faithfulness": average_metric(faith_vals),
        "context_precision": average_metric(ctx_vals),
        "answer_relevancy": average_metric(rel_vals),
    }

    worst = None
    scored = [
        e
        for e in completed
        if e.get("scores") and e["scores"].get("faithfulness") is not None
    ]
    if scored:
        worst = min(scored, key=lambda e: e["scores"]["faithfulness"])
        worst = {
            "id": worst["id"],
            "label": worst["label"],
            "faithfulness": worst["scores"]["faithfulness"],
        }

    output = {
        "ragas_version": "0.4.3",
        "llm_provider": os.environ.get("LLM_PROVIDER"),
        "metrics": {
            "faithfulness": "ragas.metrics.faithfulness",
            "context_precision": "ragas.metrics.LLMContextPrecisionWithoutReference (no reference answer required)",
            "answer_relevancy": "ragas.metrics.answer_relevancy",
        },
        "completed_count": len(completed),
        "skipped_count": len(skipped),
        "averages": averages,
        "worst_faithfulness": worst,
        "completed": completed,
        "skipped": skipped,
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print_summary_table(completed, skipped, averages, worst)
    _safe_print(f"\nWrote {RESULTS_PATH}")
    return 0 if not skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
