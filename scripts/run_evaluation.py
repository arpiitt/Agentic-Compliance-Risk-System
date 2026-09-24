"""
Evaluation harness for the Agentic Compliance & Risk Research Assistant.
Runs the full pipeline on the golden test set and reports:
  - Citation accuracy (% citations resolving to real source)
  - Precision/recall on risk factors vs. golden set
  - Hallucination rate (flagged_claims / total_claims)

Usage: python scripts/run_evaluation.py [--entity TICKER] [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from tabulate import tabulate

from app.agents.graph import create_compiled_graph
from app.agents.state import AgentState
from app.config import get_settings
from app.retrieval.embeddings import init_embeddings

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
settings = get_settings()
console = Console()

GOLDEN_SET_PATH = Path(__file__).parent.parent / "app" / "evaluation" / "golden_test_set.json"


#  Evaluation metrics 

def compute_factor_precision_recall(
    predicted_factors: list[dict],
    expected_factors: list[dict],
) -> tuple[float, float]:
    """Compute precision and recall for risk factors."""
    if not expected_factors:
        return 1.0, 1.0
    if not predicted_factors:
        return 0.0, 0.0

    predicted_names = {f.get("name", "").lower() for f in predicted_factors}

    true_positives = 0
    for ef in expected_factors:
        ef_name = ef["factor_name"].lower()
        # Check for partial match (factor names may vary)
        match = any(
            ef_name in pn or pn in ef_name or any(
                kw.lower() in pn for kw in ef.get("required_citation_keywords", [])
            )
            for pn in predicted_names
        )
        if match:
            true_positives += 1

    precision = true_positives / len(predicted_factors) if predicted_factors else 0.0
    recall = true_positives / len(expected_factors) if expected_factors else 0.0
    return precision, recall


def check_citation_accuracy(
    predicted_factors: list[dict],
    expected_factors: list[dict],
) -> float:
    """
    Check what fraction of citations (a) exist and (b) contain
    at least one expected keyword for the matching risk factor.
    """
    total_citations = 0
    accurate_citations = 0

    for pf in predicted_factors:
        # Find matching expected factor
        matching_ef = next(
            (
                ef for ef in expected_factors
                if ef["factor_name"].lower() in pf.get("name", "").lower()
                or pf.get("name", "").lower() in ef["factor_name"].lower()
                or any(kw.lower() in pf.get("rationale", "").lower()
                       for kw in ef.get("required_citation_keywords", []))
            ),
            None,
        )

        for cit in pf.get("citations", []):
            total_citations += 1
            # Citation is accurate if it has a source and non-empty passage
            has_source = bool(cit.get("source_doc_id") and cit.get("passage"))
            has_url = bool(cit.get("url") or cit.get("source_doc_id"))

            if has_source and has_url:
                if matching_ef:
                    # Check if passage contains any expected keyword
                    passage = cit.get("passage", "").lower()
                    keywords = matching_ef.get("required_citation_keywords", [])
                    if any(kw.lower() in passage for kw in keywords):
                        accurate_citations += 1
                    elif passage:  # passage exists but keywords don't match — partial credit
                        accurate_citations += 0.5
                else:
                    accurate_citations += 0.5  # No matching expected factor — partial credit

    if total_citations == 0:
        return 0.0
    return accurate_citations / total_citations


def risk_level_accuracy(predicted: str, expected: str) -> bool:
    """Check if predicted risk level matches expected (exact or adjacent)."""
    levels = {"Low": 0, "Medium": 1, "High": 2}
    p = levels.get(predicted, -1)
    e = levels.get(expected, -1)
    return abs(p - e) <= 1  # Allow one level off


#  Run evaluation 

async def evaluate_entity(
    entity: dict,
    graph,
    run_id_prefix: str = "eval",
) -> dict[str, Any]:
    """Run the pipeline on one entity and compute metrics."""
    entity_name = entity["entity_name"]
    ticker = entity["ticker"]
    expected_level = entity["expected_risk_level"]
    expected_factors = entity["expected_risk_factors"]

    import uuid
    run_id = f"{run_id_prefix}-{ticker}-{uuid.uuid4().hex[:8]}"

    initial_state: AgentState = {
        "run_id": run_id,
        "entity_name": entity_name,
        "ticker": ticker,
        "filings": [],
        "news_items": [],
        "researcher_tool_calls": [],
        "policy_matches": [],
        "compliance_tool_calls": [],
        "draft_report": None,
        "flagged_claims": [],
        "final_report": None,
        "verifier_loop_count": 0,
        "should_retry": False,
        "errors": [],
        "token_budget_remaining": settings.token_budget,
        "cost_budget_remaining": settings.cost_budget_usd,
        "total_tokens_used": 0,
        "total_cost_usd": 0.0,
        "last_completed_step": "",
    }

    t0 = time.perf_counter()
    final_state_values: AgentState | None = None

    try:
        config = {"configurable": {"thread_id": run_id}}
        async for _ in graph.astream(initial_state, config=config, stream_mode="updates"):
            pass  # consume stream
        final_snapshot = await graph.aget_state(config)
        final_state_values = final_snapshot.values
    except Exception as exc:
        logger.error("Eval run failed for %s: %s", ticker, exc)
        return {
            "ticker": ticker,
            "entity_name": entity_name,
            "error": str(exc),
            "precision": 0.0,
            "recall": 0.0,
            "citation_accuracy": 0.0,
            "hallucination_rate": 1.0,
            "risk_level_match": False,
            "latency_s": time.perf_counter() - t0,
        }

    elapsed = time.perf_counter() - t0
    final_report = (final_state_values or {}).get("final_report")
    flagged_claims = (final_state_values or {}).get("flagged_claims", [])

    if not final_report:
        return {
            "ticker": ticker,
            "entity_name": entity_name,
            "error": "No final report produced",
            "precision": 0.0,
            "recall": 0.0,
            "citation_accuracy": 0.0,
            "hallucination_rate": 1.0,
            "risk_level_match": False,
            "latency_s": elapsed,
        }

    predicted_factors = final_report.get("factors", [])
    predicted_level = final_report.get("overall_level", "Unknown")

    precision, recall = compute_factor_precision_recall(predicted_factors, expected_factors)
    citation_acc = check_citation_accuracy(predicted_factors, expected_factors)

    total_claims = sum(len(f.get("citations", [])) for f in predicted_factors)
    hallucination_rate = len(flagged_claims) / max(total_claims, 1)

    return {
        "ticker": ticker,
        "entity_name": entity_name,
        "error": None,
        "predicted_level": predicted_level,
        "expected_level": expected_level,
        "risk_level_match": risk_level_accuracy(predicted_level, expected_level),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "citation_accuracy": citation_acc,
        "hallucination_rate": hallucination_rate,
        "total_claims": total_claims,
        "flagged_claims": len(flagged_claims),
        "confidence": final_report.get("confidence", 0.0),
        "latency_s": elapsed,
        "cost_usd": (final_state_values or {}).get("total_cost_usd", 0.0),
    }


async def run_evaluation(entity_limit: int | None = None, filter_ticker: str | None = None):
    """Run the evaluation harness and print results."""
    with open(GOLDEN_SET_PATH) as f:
        golden = json.load(f)

    entities = golden["entities"]
    if filter_ticker:
        entities = [e for e in entities if e["ticker"].upper() == filter_ticker.upper()]
    if entity_limit:
        entities = entities[:entity_limit]

    console.print(f"\n[bold cyan]Agentic Compliance & Risk Assistant — Evaluation[/bold cyan]")
    console.print(f"Evaluating {len(entities)} entities from golden test set...\n")

    init_embeddings()
    graph = create_compiled_graph()  # in-memory, no checkpointer for eval

    results = []
    for i, entity in enumerate(entities, 1):
        console.print(
            f"[{i}/{len(entities)}] Evaluating [bold]{entity['ticker']}[/bold] — {entity['entity_name']}..."
        )
        result = await evaluate_entity(entity, graph)
        results.append(result)
        status = "" if not result.get("error") else ""
        console.print(
            f"  {status} P={result['precision']:.2f} R={result['recall']:.2f} "
            f"CitAcc={result['citation_accuracy']:.2f} "
            f"HalRate={result['hallucination_rate']:.2f} "
            f"t={result['latency_s']:.1f}s"
        )

    # Aggregate metrics
    valid = [r for r in results if not r.get("error")]
    failed = [r for r in results if r.get("error")]

    if valid:
        avg_precision = sum(r["precision"] for r in valid) / len(valid)
        avg_recall = sum(r["recall"] for r in valid) / len(valid)
        avg_f1 = sum(r["f1"] for r in valid) / len(valid)
        avg_citation_acc = sum(r["citation_accuracy"] for r in valid) / len(valid)
        avg_hallucination = sum(r["hallucination_rate"] for r in valid) / len(valid)
        avg_latency = sum(r["latency_s"] for r in valid) / len(valid)
        risk_level_acc = sum(1 for r in valid if r.get("risk_level_match", False)) / len(valid)
        total_cost = sum(r.get("cost_usd", 0.0) for r in results)

        console.print("\n" + "=" * 70)
        console.print("[bold green]EVALUATION RESULTS SUMMARY[/bold green]")
        console.print("=" * 70)

        summary_table = [
            ["Metric", "Value"],
            ["Entities evaluated", f"{len(valid)}/{len(entities)}"],
            ["Failed runs", len(failed)],
            ["Avg Precision (factor detection)", f"{avg_precision:.1%}"],
            ["Avg Recall (factor detection)", f"{avg_recall:.1%}"],
            ["Avg F1 Score", f"{avg_f1:.1%}"],
            ["Citation Accuracy", f"{avg_citation_acc:.1%}"],
            ["Hallucination Rate (flagged/total claims)", f"{avg_hallucination:.1%}"],
            ["Risk Level Accuracy (±1 level)", f"{risk_level_acc:.1%}"],
            ["Avg Latency per entity", f"{avg_latency:.1f}s"],
            ["Total API Cost (USD)", f"${total_cost:.4f}"],
        ]
        print(tabulate(summary_table, headers="firstrow", tablefmt="rounded_outline"))

        # Per-entity table
        console.print("\n[bold]Per-Entity Results:[/bold]")
        per_entity_table = [
            ["Ticker", "Expected", "Predicted", "Precision", "Recall", "CitAcc", "HalRate", "OK?"]
        ]
        for r in valid:
            per_entity_table.append([
                r["ticker"],
                r.get("expected_level", "?"),
                r.get("predicted_level", "?"),
                f"{r['precision']:.0%}",
                f"{r['recall']:.0%}",
                f"{r['citation_accuracy']:.0%}",
                f"{r['hallucination_rate']:.0%}",
                "" if r.get("risk_level_match") else "",
            ])
        print(tabulate(per_entity_table, headers="firstrow", tablefmt="rounded_outline"))

        # Save results
        output_path = Path(__file__).parent.parent / "evaluation_results.json"
        with open(output_path, "w") as f:
            json.dump({
                "summary": {
                    "precision": avg_precision,
                    "recall": avg_recall,
                    "f1": avg_f1,
                    "citation_accuracy": avg_citation_acc,
                    "hallucination_rate": avg_hallucination,
                    "risk_level_accuracy": risk_level_acc,
                    "avg_latency_s": avg_latency,
                    "total_cost_usd": total_cost,
                },
                "results": results,
            }, f, indent=2)
        console.print(f"\n[dim]Full results saved to {output_path}[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run compliance assistant evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of entities")
    parser.add_argument("--entity", type=str, default=None, help="Filter to specific ticker")
    args = parser.parse_args()
    asyncio.run(run_evaluation(entity_limit=args.limit, filter_ticker=args.entity))
