"""Auto-routing TTS backend — intelligently selects the best strategy per line.

Routing logic:
  1. calm emotion OR intensity < 0.3  → baseline (SFT, stable quality)
  2. everything else                  → clone_blend (x-vector interpolation)

This gives baseline's natural voice for neutral/subtle cases and
clone_blend's emotional colouring for expressive lines.

The `auto_quality` variant generates both candidates and picks the one
whose speaker embedding is closer to the canonical speaker x-vector,
acting as a quality gate against degraded voice-clone outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.context_engine.models import EmotionAnnotation

from ._common import safe_filename

_INTENSITY_THRESHOLD = 0.3


def generate_speech(
    text: str,
    emotion: EmotionAnnotation,
    ref_audio_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Smart route: baseline for calm/low-intensity, clone_blend otherwise."""
    output_path = Path(output_path or f"output/auto/{safe_filename(text)}.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    category = emotion.ref_emotion_category.value

    if category == "calm" or emotion.intensity < _INTENSITY_THRESHOLD:
        from .baseline_backend import generate_speech as _baseline
        print(f"  [auto] → baseline (cat={category}, intensity={emotion.intensity:.2f})")
        return _baseline(text, ref_audio_path, output_path)

    from .clone_backend import generate_speech_blend
    print(f"  [auto] → clone_blend (cat={category}, intensity={emotion.intensity:.2f})")
    return generate_speech_blend(text, emotion, output_path)


def generate_speech_quality_gate(
    text: str,
    emotion: EmotionAnnotation,
    ref_audio_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Dual-generate + quality gate: pick the output closer to the speaker.

    Generates with both baseline and clone_blend, then computes cosine
    similarity of each output's speaker embedding against the canonical
    calm x-vector.  The output with higher similarity wins — this prevents
    heavily distorted emotional outputs from reaching the user.
    """
    output_path = Path(output_path or f"output/auto_quality/{safe_filename(text)}.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import shutil
    import tempfile

    import torch

    from .clone_backend import (
        _AVG_XVEC_PROMPTS,
        _load_model as _load_base_model,
        generate_speech_blend,
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="auto_q_"))
    path_baseline = tmp_dir / "baseline.wav"
    path_blend = tmp_dir / "blend.wav"

    from .baseline_backend import generate_speech as _baseline_gen
    _baseline_gen(text, ref_audio_path, path_baseline)

    category = emotion.ref_emotion_category.value
    if category == "calm" or emotion.intensity < _INTENSITY_THRESHOLD:
        shutil.copy2(path_baseline, output_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"  [auto_quality] → baseline (calm/low-intensity shortcut)")
        return output_path

    generate_speech_blend(text, emotion, path_blend)

    calm_prompt = _AVG_XVEC_PROMPTS.get("calm")
    if not calm_prompt or not path_baseline.exists() or not path_blend.exists():
        winner = path_blend if path_blend.exists() else path_baseline
        shutil.copy2(winner, output_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return output_path

    canon_emb = calm_prompt[0].ref_spk_embedding

    model = _load_base_model()
    scores = {}
    for label, wav_path in [("baseline", path_baseline), ("blend", path_blend)]:
        try:
            items = model.create_voice_clone_prompt(
                ref_audio=str(wav_path),
                x_vector_only_mode=True,
            )
            out_emb = items[0].ref_spk_embedding
            sim = torch.nn.functional.cosine_similarity(
                canon_emb.unsqueeze(0), out_emb.unsqueeze(0)
            ).item()
            scores[label] = sim
        except Exception:
            scores[label] = -1.0

    winner_label = max(scores, key=scores.get)  # type: ignore[arg-type]
    winner_path = path_baseline if winner_label == "baseline" else path_blend

    shutil.copy2(winner_path, output_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"  [auto_quality] → {winner_label} "
          f"(baseline={scores.get('baseline', -1):.3f}, "
          f"blend={scores.get('blend', -1):.3f})")
    return output_path
