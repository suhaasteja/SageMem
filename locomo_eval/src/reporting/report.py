"""Generate report.md and results.csv from aggregate JSON files."""
import json
from pathlib import Path


def generate_report(aggregate_files: list[Path], output_dir: Path) -> None:
    """Read N aggregate.json files (one per adapter) and write report.md."""
    aggregates = []
    for f in aggregate_files:
        aggregates.append(json.loads(f.read_text()))

    if not aggregates:
        return

    # Collect all category keys
    all_cats = sorted({cat for agg in aggregates for cat in agg["per_category"]})
    adapters = [agg["adapter"] for agg in aggregates]

    lines = ["# LoCoMo Evaluation Report\n"]
    lines.append(f"**Judge model:** {aggregates[0]['judge_model']}  ")
    lines.append(f"**Answer model:** {aggregates[0]['answer_model']}  ")
    lines.append(f"**Filter:** categories 1–4 (adversarial excluded)  ")
    lines.append(f"**Runs:** {aggregates[0]['runs']}\n")

    # J-score table
    lines.append("## J Score (mean ± std)\n")
    header = "| Category | " + " | ".join(adapters)
    if len(adapters) > 1:
        header += f" | Δ vs full_context"
    header += " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(adapters) + 2))

    full_ctx = next((a for a in aggregates if a["adapter"] == "full_context"), None)

    for cat in all_cats:
        row = f"| {cat} |"
        cat_vals = []
        for agg in aggregates:
            cat_data = agg["per_category"].get(cat, {})
            j_mean = cat_data.get("j_mean", 0.0)
            j_std = cat_data.get("j_std", 0.0)
            row += f" {j_mean:.3f} ± {j_std:.3f} |"
            cat_vals.append(j_mean)
        if len(adapters) > 1 and full_ctx:
            fc_j = full_ctx["per_category"].get(cat, {}).get("j_mean", 0.0)
            delta = cat_vals[0] - fc_j if cat_vals else 0.0
            sign = "+" if delta >= 0 else ""
            row += f" {sign}{delta:.3f} |"
        lines.append(row)

    # Overall row
    row = "| **overall** |"
    for agg in aggregates:
        j = agg["overall"]["j_mean"]
        row += f" **{j:.3f}** |"
    if len(adapters) > 1 and full_ctx:
        fc_j = full_ctx["overall"]["j_mean"]
        my_j = aggregates[0]["overall"]["j_mean"]
        delta = my_j - fc_j
        sign = "+" if delta >= 0 else ""
        row += f" {sign}{delta:.3f} |"
    lines.append(row)

    # Token + latency table
    lines.append("\n## Tokens & Latency\n")
    lines.append("| Adapter | Avg retrieval tokens | Avg prompt tokens | Avg search ms | Avg gen ms |")
    lines.append("|---|---|---|---|---|")
    for agg in aggregates:
        t = agg["tokens"]
        l = agg["latency"]
        lines.append(
            f"| {agg['adapter']} "
            f"| {t['avg_retrieval']:.0f} "
            f"| {t['avg_prompt']:.0f} "
            f"| {l['avg_search_ms']:.1f} "
            f"| {l['avg_generation_ms']:.1f} |"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Report written to {output_dir / 'report.md'}")
