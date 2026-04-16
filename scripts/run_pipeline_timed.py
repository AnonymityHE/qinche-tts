"""Run pipeline with detailed timing instrumentation.

Measures:
  - Total pipeline wall time
  - Per-backend generation times (auto, clone_xvec, baseline)
  - Per-line generation times
  - LLM emotion analysis times
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.context_engine.analyzer import EmotionAnalyzer
from src.context_engine.models import EmotionAnnotation, EmotionCategory, GameScript
from src.emotion_tracker.tracker import EmotionArcTracker
from src.rag.retriever import CharacterRetriever
from src.tts.synthesizer import synthesize
from src.pipeline import load_emotion_buckets, select_ref_audio


def run_pipeline_timed(
    script_path: str | Path,
    backends: list[str] | None = None,
    output_dir: str | Path = "output",
) -> dict:
    backends = backends or ["auto", "clone_xvec", "baseline"]
    output_dir = Path(output_dir)

    t0_total = time.time()

    script = GameScript.model_validate_json(Path(script_path).read_text(encoding="utf-8"))

    t0_init = time.time()
    retriever = CharacterRetriever()
    analyzer = EmotionAnalyzer()
    tracker = EmotionArcTracker()
    buckets = load_emotion_buckets()
    t_init = time.time() - t0_init
    print(f"\n[TIMING] Init (retriever + analyzer + tracker): {t_init:.2f}s")

    _rag_cache: dict[str, list[str]] = {}
    results = []
    timing: dict[str, list[float]] = {b: [] for b in backends}
    timing["emotion_analysis"] = []
    timing["rag_retrieval"] = []

    for i, line in enumerate(script.dialogue):
        if line.speaker != "秦彻":
            tracker.update(
                EmotionAnnotation(
                    emotion="other_speaker",
                    intensity=0.0,
                    pace="normal",
                    style="",
                    ref_emotion_category=EmotionCategory.CALM,
                ),
                line.text,
            )
            continue

        print(f"\n[{i + 1}/{len(script.dialogue)}] {line.speaker}: {line.text}")

        # RAG
        t0_rag = time.time()
        if script.scene not in _rag_cache:
            _rag_cache[script.scene] = retriever.retrieve(script.scene)
        character_ctx = _rag_cache[script.scene]
        t_rag = time.time() - t0_rag
        timing["rag_retrieval"].append(t_rag)

        # Emotion analysis
        t0_emo = time.time()
        emotion = analyzer.analyze(
            line=line.text,
            scene_description=script.scene,
            emotion_history=tracker.history,
            character_context=character_ctx,
        )
        t_emo = time.time() - t0_emo
        timing["emotion_analysis"].append(t_emo)
        print(f"  [TIMING] emotion analysis: {t_emo:.2f}s")
        print(f"  → emotion: {emotion.emotion} ({emotion.ref_emotion_category.value})")

        tracker.update(emotion, line.text)

        ref_audio = None
        try:
            ref_audio = select_ref_audio(
                emotion.ref_emotion_category, buckets, intensity=emotion.intensity,
            )
        except FileNotFoundError as e:
            print(f"  ⚠ {e}")

        entry = {
            "index": i,
            "text": line.text,
            "emotion": emotion.model_dump(),
            "audio": {},
            "timing": {},
        }

        for backend in backends:
            t0_gen = time.time()
            try:
                audio_path = synthesize(
                    text=line.text,
                    emotion=emotion,
                    backend=backend,
                    ref_audio_path=ref_audio,
                    output_path=output_dir / backend / f"line_{i:03d}.wav",
                )
                entry["audio"][backend] = str(audio_path)
                t_gen = time.time() - t0_gen
                entry["timing"][backend] = t_gen
                timing[backend].append(t_gen)
                print(f"  → [{backend}] {audio_path} ({t_gen:.2f}s)")
            except Exception as e:
                t_gen = time.time() - t0_gen
                entry["audio"][backend] = f"SKIPPED: {type(e).__name__}: {e}"
                entry["timing"][backend] = t_gen
                timing[backend].append(t_gen)
                print(f"  → [{backend}] SKIPPED: {type(e).__name__}: {e} ({t_gen:.2f}s)")

        results.append(entry)

    t_total = time.time() - t0_total

    # Aggregate timing
    timing_summary = {"total_pipeline_s": t_total, "init_s": t_init}
    for key, values in timing.items():
        if values:
            timing_summary[f"{key}_total_s"] = sum(values)
            timing_summary[f"{key}_mean_s"] = sum(values) / len(values)
            timing_summary[f"{key}_count"] = len(values)

    print(f"\n{'=' * 60}")
    print(f"TIMING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total pipeline time: {t_total:.2f}s")
    print(f"Init time: {t_init:.2f}s")
    for key, values in timing.items():
        if values:
            print(f"{key}: total={sum(values):.2f}s, mean={sum(values)/len(values):.2f}s, n={len(values)}")
    print(f"{'=' * 60}")

    # Save results
    output = {
        "pipeline_results": results,
        "timing_summary": timing_summary,
    }
    results_path = output_dir / "pipeline_results_timed.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Also save pipeline_results.json for eval compatibility
    pr_path = output_dir / "pipeline_results.json"
    with open(pr_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {results_path}")
    return output


if __name__ == "__main__":
    import sys

    script_file = sys.argv[1] if len(sys.argv) > 1 else "data/scripts/test_scene_01.json"
    backends_str = sys.argv[2] if len(sys.argv) > 2 else "auto,clone_xvec,baseline"
    backends = [b.strip() for b in backends_str.split(",")]

    run_pipeline_timed(script_file, backends=backends)
