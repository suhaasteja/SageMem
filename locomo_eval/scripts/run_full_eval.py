#!/usr/bin/env python
"""Main entrypoint for running the LoCoMo evaluation pipeline."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.progress import track

sys.path.insert(0, str(Path(__file__).parents[1]))
load_dotenv(Path(__file__).parents[1] / ".env")

from src.config import (
    ANSWER_MODEL,
    DATASET_PATH,
    DEFAULT_RUNS,
    EVAL_CATEGORIES,
    JUDGE_MODEL,
    RESULTS_DIR,
    TOP_K,
)
from src.data import filter_qa, load_dataset
from src.pipeline.evaluate import evaluate_predictions
from src.pipeline.ingest import ingest_conversation
from src.pipeline.query import query_conversation
from src.reporting.aggregate import aggregate_runs

console = Console()


def make_adapter(name: str, client: OpenAI):
    if name == "full_context":
        from src.adapters.full_context import FullContextAdapter
        return FullContextAdapter()
    elif name == "naive_rag":
        from src.adapters.naive_rag import NaiveRAGAdapter
        return NaiveRAGAdapter(client)
    elif name == "my_system":
        from src.adapters.my_system import SageMemAdapter
        return SageMemAdapter()
    else:
        raise ValueError(f"Unknown adapter: {name}")


def run_single(adapter, conversations, client, answer_model, judge_model, top_k, run_dir: Path, max_qa: int | None = None):
    all_evaluated = []
    for conv in track(conversations, description="Conversations"):
        qa_items = filter_qa(conv.qa, EVAL_CATEGORIES)
        if not qa_items:
            continue
        if max_qa is not None:
            qa_items = qa_items[:max_qa]
        ingest_conversation(adapter, conv)
        predictions = query_conversation(
            adapter, conv, qa_items,
            client=client, answer_model=answer_model, top_k=top_k,
        )
        evaluated = evaluate_predictions(predictions, client=client, judge_model=judge_model)
        all_evaluated.extend(evaluated)

    run_dir.mkdir(parents=True, exist_ok=True)
    preds_path = run_dir / "predictions.json"
    preds_path.write_text(json.dumps([p.to_dict() for p in all_evaluated], indent=2))
    console.print(f"  Saved {len(all_evaluated)} predictions → {preds_path}")
    return all_evaluated


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoCoMo evaluation")
    parser.add_argument("--adapter", default="my_system",
                        choices=["my_system", "full_context", "naive_rag"])
    parser.add_argument("--answer-model", default=ANSWER_MODEL)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--output-dir", type=Path,
                        default=RESULTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--max-conversations", type=int, default=None,
                        help="Limit to first N conversations (for smoke tests)")
    parser.add_argument("--max-qa", type=int, default=None,
                        help="Limit to first N QA items per conversation (for smoke tests)")
    args = parser.parse_args()

    if not args.dataset.exists():
        console.print(f"[red]Dataset not found: {args.dataset}[/red]")
        console.print("Run: python scripts/verify_dataset.py")
        sys.exit(1)

    smoke = args.max_conversations or args.max_qa
    console.print(
        f"[bold]LoCoMo Eval[/bold] — adapter={args.adapter}, runs={args.runs}"
        + (" [yellow](smoke test)[/yellow]" if smoke else "")
    )
    client = OpenAI()
    conversations = load_dataset(args.dataset)
    if args.max_conversations:
        conversations = conversations[: args.max_conversations]
    adapter = make_adapter(args.adapter, client)

    all_runs = []
    for i in range(args.runs):
        console.print(f"\n[bold]Run {i + 1}/{args.runs}[/bold]")
        run_dir = args.output_dir / f"run_{i + 1}"
        evaluated = run_single(
            adapter, conversations, client,
            args.answer_model, args.judge_model, args.top_k, run_dir,
            max_qa=args.max_qa,
        )
        all_runs.append(evaluated)

    agg = aggregate_runs(
        all_runs,
        adapter_name=args.adapter,
        answer_model=args.answer_model,
        judge_model=args.judge_model,
    )
    agg_path = args.output_dir / "aggregate.json"
    agg_path.write_text(json.dumps(agg, indent=2))
    console.print(f"\n[green]Aggregate saved → {agg_path}[/green]")

    overall = agg["overall"]
    console.print(
        f"\n[bold]Overall J={overall['j_mean']:.3f}  "
        f"F1={overall['f1_mean']:.3f}  "
        f"BLEU-1={overall['bleu1_mean']:.3f}[/bold]"
    )


if __name__ == "__main__":
    main()
