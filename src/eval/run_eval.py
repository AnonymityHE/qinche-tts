"""Run comparative evaluation across all TTS backend conditions.

Usage:
    python -m src.eval.run_eval                         # evaluate all conditions
    python -m src.eval.run_eval --conditions qwen,clone
    python -m src.eval.run_eval --skip-emotion           # skip emotion2vec (slow)

Reads pipeline_results.json from each condition directory to match generated audio
with reference text and target emotion labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm


def _load_pipeline_results(output_dir: Path) -> dict | None:
    """Load the pipeline_results.json if it exists."""
    results_path = output_dir / "pipeline_results.json"
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _find_ref_audios(ref_audio_dir: Path) -> list[Path]:
    """Find reference audio files (ref_00.wav .. ref_04.wav)."""
    refs = sorted(ref_audio_dir.glob("ref_*.wav"))
    if not refs:
        refs = sorted(ref_audio_dir.glob("*.wav"))
    return refs


def run_evaluation(
    output_dir: str | Path = "output",
    ref_audio_dir: str | Path = "data/ref_audio",
    test_manifest: str | Path = "data/test_manifest.jsonl",
    conditions: list[str] | None = None,
    skip_emotion: bool = False,
    report_path: str | Path = "output/eval_report.json",
) -> dict:
    """Run full evaluation suite across specified conditions.

    For each generated wav, computes:
    - SIM_ref: speaker similarity against reference audio mean
    - SIM_gt: speaker similarity against ground truth audio (if available)
    - WER: character-level word error rate via ASR
    - Emotion Acc: emotion classification match (optional)
    """
    from .sim import compute_sim, compute_sim_against_mean
    from .wer import compute_wer

    output_dir = Path(output_dir)
    ref_audio_dir = Path(ref_audio_dir)
    conditions = conditions or ["qwen", "clone", "fish", "baseline"]

    ref_audios = _find_ref_audios(ref_audio_dir)
    if not ref_audios:
        print("WARNING: No reference audio found. SIM_ref will be skipped.")

    gt_manifest = _load_test_manifest(test_manifest)

    pipeline_results = _load_pipeline_results(output_dir)

    emotion_fn = None
    if not skip_emotion:
        try:
            from .emotion_acc import compute_emotion_accuracy

            emotion_fn = compute_emotion_accuracy
        except ImportError as e:
            print(f"WARNING: emotion2vec not available ({e}), skipping emotion eval")

    all_results: dict[str, list[dict]] = {}

    for condition in conditions:
        condition_dir = output_dir / condition
        if not condition_dir.exists():
            print(f"Skipping {condition}: directory {condition_dir} not found")
            continue

        wav_files = sorted(condition_dir.glob("*.wav"))
        if not wav_files:
            print(f"Skipping {condition}: no wav files found")
            continue

        print(f"\nEvaluating {condition} ({len(wav_files)} files)...")
        entries = []

        for wav in tqdm(wav_files, desc=condition):
            entry: dict = {"file": wav.name, "path": str(wav)}

            ref_text, target_emotion, gt_audio = _match_metadata(
                wav, pipeline_results, gt_manifest, condition
            )

            if ref_audios:
                try:
                    entry["sim_ref"] = compute_sim_against_mean(wav, ref_audios)
                except Exception as e:
                    entry["sim_ref_error"] = str(e)

            if gt_audio and Path(gt_audio).exists():
                try:
                    entry["sim_gt"] = compute_sim(wav, gt_audio)
                except Exception as e:
                    entry["sim_gt_error"] = str(e)

            if ref_text:
                try:
                    entry["wer"] = compute_wer(wav, ref_text)
                    entry["ref_text"] = ref_text
                except Exception as e:
                    entry["wer_error"] = str(e)

            if emotion_fn and target_emotion:
                try:
                    emo_result = emotion_fn(wav, target_emotion)
                    entry["emotion"] = emo_result
                except Exception as e:
                    entry["emotion_error"] = str(e)

            entries.append(entry)

        all_results[condition] = entries

    summary = _aggregate(all_results)

    report = {"summary": summary, "details": {k: v for k, v in all_results.items()}}
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _print_table(summary)
    print(f"\nFull report saved to {report_path}")
    return summary


def _load_test_manifest(manifest_path: str | Path) -> list[dict]:
    """Load test manifest JSONL (each line has audio_filepath + text)."""
    path = Path(manifest_path)
    if not path.exists():
        return []
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _match_metadata(
    wav_path: Path,
    pipeline_results: dict | None,
    gt_manifest: list[dict],
    condition: str,
) -> tuple[str | None, str | None, str | None]:
    """Try to find reference text, target emotion, and ground truth audio for a generated wav.

    First checks pipeline_results.json, then falls back to index-based manifest lookup.
    """
    ref_text = None
    target_emotion = None
    gt_audio = None

    stem = wav_path.stem
    idx = _parse_index_from_filename(stem)

    if pipeline_results:
        for entry in pipeline_results:
            if entry.get("index") == idx:
                ref_text = entry.get("text")
                emo_data = entry.get("emotion", {})
                target_emotion = emo_data.get("ref_emotion_category")
                break

    if gt_manifest and idx is not None and idx < len(gt_manifest):
        manifest_entry = gt_manifest[idx]
        if not ref_text:
            ref_text = manifest_entry.get("text")
        gt_audio = manifest_entry.get("audio_filepath")

    return ref_text, target_emotion, gt_audio


def _parse_index_from_filename(stem: str) -> int | None:
    """Extract numeric index from filenames like 'line_003' or 'gen_05'."""
    import re

    m = re.search(r"(\d+)$", stem)
    if m:
        return int(m.group(1))
    return None


def _aggregate(results: dict[str, list[dict]]) -> dict:
    """Compute mean metrics per condition."""
    summary = {}
    for condition, entries in results.items():
        if not entries:
            summary[condition] = {"count": 0}
            continue

        sim_refs = [e["sim_ref"] for e in entries if "sim_ref" in e]
        sim_gts = [e["sim_gt"] for e in entries if "sim_gt" in e]
        wers = [e["wer"] for e in entries if "wer" in e]
        matches = [
            e["emotion"]["match"]
            for e in entries
            if "emotion" in e and isinstance(e["emotion"], dict)
        ]

        summary[condition] = {
            "count": len(entries),
            "sim_ref_mean": _safe_mean(sim_refs),
            "sim_gt_mean": _safe_mean(sim_gts),
            "wer_mean": _safe_mean(wers),
            "emotion_acc": _safe_mean([float(m) for m in matches]),
        }
    return summary


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(np.mean(values)), 4)


def _print_table(summary: dict) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 72)
    print(
        f"{'Condition':<18} {'Count':>6} {'SIM_ref':>8} {'SIM_gt':>8} "
        f"{'WER':>8} {'Emo Acc':>8}"
    )
    print("-" * 72)
    for condition, m in summary.items():
        count = m.get("count", 0)
        sr = f"{m['sim_ref_mean']:.4f}" if m.get("sim_ref_mean") is not None else "N/A"
        sg = f"{m['sim_gt_mean']:.4f}" if m.get("sim_gt_mean") is not None else "N/A"
        w = f"{m['wer_mean']:.4f}" if m.get("wer_mean") is not None else "N/A"
        ea = f"{m['emotion_acc']:.4f}" if m.get("emotion_acc") is not None else "N/A"
        print(f"{condition:<18} {count:>6} {sr:>8} {sg:>8} {w:>8} {ea:>8}")
    print("=" * 72)


@click.command()
@click.option("--output-dir", default="output", help="Base output directory")
@click.option("--ref-audio-dir", default="data/ref_audio", help="Reference audio directory")
@click.option("--test-manifest", default="data/test_manifest.jsonl", help="Test manifest JSONL path")
@click.option(
    "--conditions",
    default="qwen,clone,fish,baseline",
    help="Comma-separated condition names (must match pipeline backend output dirs)",
)
@click.option("--skip-emotion", is_flag=True, help="Skip emotion2vec evaluation")
@click.option("--report-path", default="output/eval_report.json", help="Output report JSON path")
def main(output_dir, ref_audio_dir, test_manifest, conditions, skip_emotion, report_path):
    """Run comparative evaluation across TTS conditions."""
    cond_list = [c.strip() for c in conditions.split(",") if c.strip()]
    run_evaluation(
        output_dir=output_dir,
        ref_audio_dir=ref_audio_dir,
        test_manifest=test_manifest,
        conditions=cond_list,
        skip_emotion=skip_emotion,
        report_path=report_path,
    )


if __name__ == "__main__":
    main()
