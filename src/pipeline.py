"""End-to-end pipeline: Game Script → Emotion Analysis → TTS Generation."""

from __future__ import annotations

import json
import random
from pathlib import Path

from dotenv import load_dotenv

from src.context_engine.analyzer import EmotionAnalyzer
from src.context_engine.models import EmotionAnnotation, EmotionCategory, GameScript
from src.emotion_tracker.tracker import EmotionArcTracker
from src.rag.retriever import CharacterRetriever
from src.tts.synthesizer import synthesize

load_dotenv()

_EMOTION_BUCKETS_PATH = Path("data/emotion_buckets.json")
_DEFAULT_REF_AUDIO = Path("data/ref_audio/default.wav")


def load_emotion_buckets(path: Path = _EMOTION_BUCKETS_PATH) -> dict[str, list[dict]]:
    """Load emotion bucket mapping: category -> list of entry dicts.

    Each entry has at least {file, text, emotion, intensity, confidence}.
    """
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    buckets: dict[str, list[dict]] = {}
    for entry in data:
        emotion = entry.get("emotion", "calm")
        if emotion not in buckets:
            buckets[emotion] = []
        buckets[emotion].append(entry)
    return buckets


def select_ref_audio(
    category: EmotionCategory,
    buckets: dict[str, list[dict]],
    intensity: float = 0.5,
) -> Path:
    """Select a reference audio clip from the matching emotion bucket.

    Uses intensity-weighted selection: candidates whose bucket intensity
    is closest to the requested intensity are preferred.
    """
    candidates = buckets.get(category.value, [])
    if not candidates:
        all_entries = [e for entries in buckets.values() for e in entries]
        candidates = all_entries

    valid = [(e, Path(e["file"])) for e in candidates if Path(e["file"]).exists()]
    if not valid:
        if _DEFAULT_REF_AUDIO.exists():
            return _DEFAULT_REF_AUDIO
        raise FileNotFoundError(
            f"No reference audio found for category '{category.value}' "
            f"and no default ref audio at {_DEFAULT_REF_AUDIO}"
        )

    valid.sort(key=lambda ep: abs(ep[0].get("intensity", 0.5) - intensity))
    top_n = valid[: min(3, len(valid))]
    return random.choice(top_n)[1]


def run_pipeline(
    script_path: str | Path,
    backends: list[str] | None = None,
    output_dir: str | Path = "output",
) -> list[dict]:
    """Run the full pipeline on a game script JSON file.

    Args:
        script_path: Path to the game script JSON.
        backends: List of TTS backends to use.
        output_dir: Base directory for generated audio.

    Returns:
        List of result dicts, one per dialogue line.
    """
    backends = backends or ["auto", "clone_xvec", "baseline"]
    output_dir = Path(output_dir)

    script = GameScript.model_validate_json(Path(script_path).read_text(encoding="utf-8"))

    retriever = CharacterRetriever()
    analyzer = EmotionAnalyzer()
    tracker = EmotionArcTracker()
    buckets = load_emotion_buckets()
    _rag_cache: dict[str, list[str]] = {}

    results = []

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

        if script.scene not in _rag_cache:
            _rag_cache[script.scene] = retriever.retrieve(script.scene)
        character_ctx = _rag_cache[script.scene]

        emotion = analyzer.analyze(
            line=line.text,
            scene_description=script.scene,
            emotion_history=tracker.history,
            character_context=character_ctx,
        )

        print(f"  → emotion: {emotion.emotion} ({emotion.ref_emotion_category.value})")
        print(f"  → style: {emotion.style}")
        print(f"  → intensity: {emotion.intensity}, pace: {emotion.pace}")

        if not tracker.check_transition(emotion.ref_emotion_category):
            print(f"  ⚠ Abrupt emotion transition detected, may want to smooth")

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
        }

        for backend in backends:
            try:
                audio_path = synthesize(
                    text=line.text,
                    emotion=emotion,
                    backend=backend,
                    ref_audio_path=ref_audio,
                    output_path=output_dir / backend / f"line_{i:03d}.wav",
                )
                entry["audio"][backend] = str(audio_path)
                print(f"  → [{backend}] {audio_path}")
            except Exception as e:
                entry["audio"][backend] = f"SKIPPED: {type(e).__name__}: {e}"
                print(f"  → [{backend}] SKIPPED: {type(e).__name__}: {e}")

        results.append(entry)

    results_path = output_dir / "pipeline_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nPipeline complete. Results saved to {results_path}")
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline <script.json> [--backends qwen,fish,baseline]")
        sys.exit(1)

    script_file = sys.argv[1]
    run_pipeline(script_file)
