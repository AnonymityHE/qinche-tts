"""Batch-classify 664 training samples into 6 emotion buckets using GPT-4o.

Usage:
    # Step 1: Classify all samples (needs OPENAI_API_KEY in .env)
    python -m src.context_engine.classify_samples classify \
        --manifest data/train_manifest.jsonl \
        --output data/emotion_classifications.json

    # Step 2: Generate emotion_buckets.json (pick top-N per bucket)
    python -m src.context_engine.classify_samples buckets \
        --classifications data/emotion_classifications.json \
        --output data/emotion_buckets.json \
        --top-n 5

    # Or run both steps in one go:
    python -m src.context_engine.classify_samples run-all \
        --manifest data/train_manifest.jsonl
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import click
from openai import OpenAI
from tqdm import tqdm

_CLASSIFY_PROMPT = """\
你是一位游戏配音情感分析专家。请分析以下游戏角色"秦彻"的台词，判断其情感类别。

角色背景：秦彻是《恋与深空》中的男主角之一，N109区暗点组织首领，表面冷酷自信、内心温柔深情。

情感类别（只能选一个）：
- tender（温柔）：关心、安慰、心疼、温柔低语
- calm（沉稳）：日常对话、理性分析、平静陈述
- playful（俏皮）：调侃、轻松打趣、偶尔的幽默
- intense（激烈）：战斗、愤怒、紧张、严厉警告
- cold（冷淡）：疏离、威严、拒人于千里之外
- intimate（亲密）：私密场景、深情告白、低语呢喃

严格输出 JSON：
{
  "emotion": "类别名",
  "intensity": 0.0-1.0,
  "description": "简短中文描述该句的情感特征",
  "confidence": 0.0-1.0
}
"""

VALID_EMOTIONS = {"tender", "calm", "playful", "intense", "cold", "intimate"}


def classify_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    model: str = "gpt-4o",
    api_key: str | None = None,
    batch_delay: float = 0.05,
    resume: bool = True,
) -> list[dict]:
    """Read a JSONL manifest, classify each sample via LLM, and write results.

    Supports resuming from a partially-completed output file.
    """
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)

    client = OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL"),
    )

    samples = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    existing: dict[str, dict] = {}
    if resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for entry in json.load(f):
                existing[entry.get("file", "")] = entry
        print(f"Resuming: {len(existing)} already classified, {len(samples) - len(existing)} remaining")

    results = []
    for sample in tqdm(samples, desc="Classifying emotions"):
        text = sample.get("text", "")
        audio_path = sample.get("audio_filepath", sample.get("audio", ""))

        if audio_path in existing:
            results.append(existing[audio_path])
            continue

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _CLASSIFY_PROMPT},
                    {"role": "user", "content": f"台词：{text}"},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            if data.get("emotion") not in VALID_EMOTIONS:
                data["emotion"] = "calm"
                data["_note"] = "auto-corrected invalid emotion to calm"
            data["file"] = audio_path
            data["text"] = text
            results.append(data)
        except Exception as e:
            results.append({
                "file": audio_path,
                "text": text,
                "emotion": "unknown",
                "intensity": 0.0,
                "description": f"Error: {e}",
                "confidence": 0.0,
            })

        if batch_delay > 0:
            time.sleep(batch_delay)

        if len(results) % 50 == 0:
            _save_checkpoint(results, output_path)

    _save_checkpoint(results, output_path)
    _print_summary(results)
    return results


def _save_checkpoint(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def generate_emotion_buckets(
    classifications_path: str | Path,
    output_path: str | Path = "data/emotion_buckets.json",
    top_n: int = 5,
    min_confidence: float = 0.7,
) -> dict[str, list[dict]]:
    """From full classification results, pick top-N highest-confidence samples per bucket.

    Output format (for pipeline consumption):
    [
      {"file": "normalized/qinche_01/xxx.wav", "emotion": "tender", "intensity": 0.8, ...},
      ...
    ]
    Also produces a per-bucket summary.
    """
    classifications_path = Path(classifications_path)
    output_path = Path(output_path)

    with open(classifications_path, encoding="utf-8") as f:
        all_results = json.load(f)

    buckets: dict[str, list[dict]] = {e: [] for e in VALID_EMOTIONS}
    for entry in all_results:
        emotion = entry.get("emotion", "unknown")
        if emotion in VALID_EMOTIONS:
            buckets[emotion].append(entry)

    selected: list[dict] = []
    bucket_summary: dict[str, dict] = {}

    for emotion, entries in buckets.items():
        entries.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        qualified = [e for e in entries if e.get("confidence", 0) >= min_confidence]
        picked = qualified[:top_n] if qualified else entries[:top_n]
        selected.extend(picked)
        bucket_summary[emotion] = {
            "total": len(entries),
            "qualified": len(qualified),
            "selected": len(picked),
            "files": [p["file"] for p in picked],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    print(f"\n=== Emotion Buckets Summary ===")
    print(f"{'Emotion':<12s} {'Total':>6s} {'Qualified':>10s} {'Selected':>9s}")
    print("-" * 40)
    for emotion, info in bucket_summary.items():
        print(f"{emotion:<12s} {info['total']:>6d} {info['qualified']:>10d} {info['selected']:>9d}")
    print(f"\nTotal selected: {len(selected)} -> {output_path}")

    return bucket_summary


def _print_summary(results: list[dict]) -> None:
    counts = Counter(r.get("emotion", "unknown") for r in results)
    total = len(results)
    print(f"\n=== Emotion Classification Summary ({total} samples) ===")
    for emotion, count in counts.most_common():
        pct = count / total * 100 if total > 0 else 0
        print(f"  {emotion:12s}: {count:4d} ({pct:.1f}%)")


@click.group()
def cli():
    """Emotion classification tools for training samples."""
    pass


@cli.command()
@click.option("--manifest", default="data/train_manifest.jsonl", help="Input JSONL manifest")
@click.option("--output", default="data/emotion_classifications.json", help="Output JSON path")
@click.option("--model", default="gpt-4o", help="LLM model name")
@click.option("--no-resume", is_flag=True, help="Start fresh, ignore existing output")
def classify(manifest, output, model, no_resume):
    """Classify all training samples using GPT-4o."""
    from dotenv import load_dotenv
    load_dotenv()
    classify_manifest(manifest, output, model=model, resume=not no_resume)


@cli.command()
@click.option("--classifications", default="data/emotion_classifications.json", help="Input classifications JSON")
@click.option("--output", default="data/emotion_buckets.json", help="Output buckets JSON")
@click.option("--top-n", default=5, help="Number of samples per bucket")
@click.option("--min-confidence", default=0.7, type=float, help="Minimum confidence threshold")
def buckets(classifications, output, top_n, min_confidence):
    """Generate emotion_buckets.json from classification results."""
    generate_emotion_buckets(classifications, output, top_n=top_n, min_confidence=min_confidence)


@cli.command(name="run-all")
@click.option("--manifest", default="data/train_manifest.jsonl", help="Input JSONL manifest")
@click.option("--model", default="gpt-4o", help="LLM model name")
@click.option("--top-n", default=5, help="Samples per bucket")
def run_all(manifest, model, top_n):
    """Run classify + buckets in one go."""
    from dotenv import load_dotenv
    load_dotenv()

    classifications_path = "data/emotion_classifications.json"
    classify_manifest(manifest, classifications_path, model=model)
    generate_emotion_buckets(classifications_path, "data/emotion_buckets.json", top_n=top_n)


if __name__ == "__main__":
    cli()
