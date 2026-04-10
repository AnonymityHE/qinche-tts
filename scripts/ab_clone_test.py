"""A/B test: clone (Base + ref audio) vs sft (CustomVoice + instruct) vs baseline.

Generates speech for a set of test sentences using all three backends,
saves to output/ab_clone_test/<backend>/ for side-by-side listening.

Usage:
    conda activate qinche
    python scripts/ab_clone_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.context_engine.models import EmotionAnnotation, EmotionCategory

OUTPUT_DIR = Path("output/ab_clone_test")
BACKENDS = ["auto", "auto_quality", "clone_blend", "clone_xvec", "baseline"]

TEST_CASES = [
    {
        "text": "慌什么，肩上的伤早就愈合了，听我的声音，这不是好得很。",
        "emotion": EmotionAnnotation(
            emotion="gentle_reassuring",
            intensity=0.7,
            pace="slow",
            style="用温柔且故作轻松的语气说",
            ref_emotion_category=EmotionCategory.TENDER,
        ),
    },
    {
        "text": "别妄自菲薄，你比你想象中有用得多。",
        "emotion": EmotionAnnotation(
            emotion="gentle_encouraging",
            intensity=0.7,
            pace="slow",
            style="用温柔且坚定的语气说",
            ref_emotion_category=EmotionCategory.TENDER,
        ),
    },
    {
        "text": "想抽取我的能量那就用你们的血来偿还。",
        "emotion": EmotionAnnotation(
            emotion="cold_threatening",
            intensity=0.9,
            pace="normal",
            style="用冷酷且威胁的语气说",
            ref_emotion_category=EmotionCategory.INTENSE,
        ),
    },
    {
        "text": "需要我帮你给走失的声带贴一张寻物告示吗?",
        "emotion": EmotionAnnotation(
            emotion="playful_teasing",
            intensity=0.7,
            pace="normal",
            style="用调侃且轻松的语气说",
            ref_emotion_category=EmotionCategory.PLAYFUL,
        ),
    },
    {
        "text": "直到现在,她终于成了我的夫人。",
        "emotion": EmotionAnnotation(
            emotion="deep_intimate",
            intensity=0.9,
            pace="slow",
            style="用深情且低沉的语气说",
            ref_emotion_category=EmotionCategory.INTIMATE,
        ),
    },
]


def main():
    from src.tts.synthesizer import synthesize
    from src.tts.clone_backend import precompute_all_prompts, compute_avg_xvec_prompts

    print("Pre-computing voice clone prompts (ICL mode)...")
    n_icl = precompute_all_prompts()
    print(f"  → {n_icl} ICL prompts cached")

    print("Computing averaged x-vector prompts per emotion...")
    n_xvec = compute_avg_xvec_prompts()
    print(f"  → {n_xvec} emotion categories averaged")

    results = []

    for i, case in enumerate(TEST_CASES):
        text = case["text"]
        emotion = case["emotion"]

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(TEST_CASES)}] {text}")
        print(f"  emotion: {emotion.ref_emotion_category.value} | style: {emotion.style}")
        print(f"{'='*60}")

        entry = {"index": i, "text": text, "emotion_category": emotion.ref_emotion_category.value, "audio": {}}

        for backend in BACKENDS:
            out_path = OUTPUT_DIR / backend / f"test_{i:02d}.wav"
            try:
                result = synthesize(
                    text=text,
                    emotion=emotion,
                    backend=backend,
                    output_path=out_path,
                )
                entry["audio"][backend] = str(result)
                print(f"  [{backend}] -> {result}")
            except Exception as e:
                entry["audio"][backend] = f"ERROR: {e}"
                print(f"  [{backend}] ERROR: {e}")

        results.append(entry)

    results_path = OUTPUT_DIR / "ab_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nA/B test complete. Results: {results_path}")
    print(f"Audio files in: {OUTPUT_DIR}/")
    for backend in BACKENDS:
        count = len(list((OUTPUT_DIR / backend).glob("*.wav")))
        print(f"  {backend}: {count} files")


if __name__ == "__main__":
    main()
