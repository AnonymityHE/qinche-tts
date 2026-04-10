"""Analyze existing evaluation results from archive/eval_from_project/ directory.

Loads all eval_results.json and comparison.json files, aggregates them into
a unified comparison table, useful for the final report.

Usage:
    python -m src.eval.analyze_existing
    python -m src.eval.analyze_existing --eval-dir archive/eval_from_project --output docs/eval_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import click


def analyze_existing_results(
    eval_dir: str | Path = "archive/eval_from_project",
    output_path: str | Path | None = None,
) -> dict:
    """Scan eval_dir for eval_results.json files and build a summary table."""
    eval_dir = Path(eval_dir)
    if not eval_dir.exists():
        print(f"Directory {eval_dir} not found")
        return {}

    all_summaries: dict[str, dict] = {}

    for result_file in sorted(eval_dir.rglob("eval_results.json")):
        name = result_file.parent.name
        with open(result_file, encoding="utf-8") as f:
            data = json.load(f)

        entry = {"source_file": str(result_file)}
        for key in ("model", "checkpoint", "avg_sim_gt", "avg_sim_ref", "avg_wer", "avg_rtf", "num_samples"):
            if key in data:
                entry[key] = data[key]

        if "per_sample" in data:
            entry["num_per_sample"] = len(data["per_sample"])
        elif "per_sentence" in data:
            entry["num_per_sample"] = len(data["per_sentence"])

        all_summaries[name] = entry

    for comp_file in sorted(eval_dir.rglob("*comparison*.json")):
        with open(comp_file, encoding="utf-8") as f:
            comp_data = json.load(f)
        for key, val in comp_data.items():
            comp_name = f"{comp_file.stem}/{key}"
            if comp_name not in all_summaries:
                all_summaries[comp_name] = {"source_file": str(comp_file), **val}

    _print_summary_table(all_summaries)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2)
        print(f"\nSummary saved to {output_path}")

    return all_summaries


def _print_summary_table(summaries: dict[str, dict]) -> None:
    print("\n" + "=" * 85)
    print(f"{'Name':<40s} {'SIM_gt':>8s} {'SIM_ref':>8s} {'WER':>8s} {'RTF':>8s} {'N':>5s}")
    print("-" * 85)

    for name, s in summaries.items():
        sim_gt = f"{s['avg_sim_gt']:.4f}" if "avg_sim_gt" in s else "   N/A"
        sim_ref = f"{s['avg_sim_ref']:.4f}" if "avg_sim_ref" in s else "   N/A"
        wer_val = f"{s['avg_wer']:.4f}" if "avg_wer" in s else "   N/A"
        rtf = f"{s['avg_rtf']:.3f}" if "avg_rtf" in s else "  N/A"
        n = str(s.get("num_samples", s.get("num_per_sample", "?")))

        display_name = name[:39]
        print(f"{display_name:<40s} {sim_gt:>8s} {sim_ref:>8s} {wer_val:>8s} {rtf:>8s} {n:>5s}")

    print("=" * 85)


@click.command()
@click.option("--eval-dir", default="archive/eval_from_project", help="Directory with eval results")
@click.option("--output", default=None, help="Output summary JSON path")
def main(eval_dir, output):
    """Analyze existing evaluation results."""
    analyze_existing_results(eval_dir=eval_dir, output_path=output)


if __name__ == "__main__":
    main()
