"""Unified TTS synthesis interface dispatching to different backends."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.context_engine.models import EmotionAnnotation


def synthesize(
    text: str,
    emotion: EmotionAnnotation,
    backend: str = "auto",
    ref_audio_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Generate speech for *text* using the specified backend.

    Supported backends:
        auto          – smart route: baseline for calm/low-intensity, clone_blend otherwise
        auto_quality  – dual-generate + speaker-similarity quality gate
        clone_blend   – x-vector interpolation between calm and emotion embeddings
        clone_xvec    – per-emotion averaged x-vector (Method B)
        baseline      – SFT CustomVoice, no emotion control
        qwen          – SFT CustomVoice + text instruct
        clone         – single-ref ICL voice clone (deprecated)
        fish          – Fish Audio S2 Pro
    """
    if backend == "auto":
        from .auto_backend import generate_speech
        return generate_speech(text, emotion, ref_audio_path, output_path)
    elif backend == "auto_quality":
        from .auto_backend import generate_speech_quality_gate
        return generate_speech_quality_gate(text, emotion, ref_audio_path, output_path)
    elif backend == "clone_blend":
        from .clone_backend import generate_speech_blend
        return generate_speech_blend(text, emotion, output_path)
    elif backend == "clone_xvec":
        from .clone_backend import generate_speech_xvec
        return generate_speech_xvec(text, emotion, output_path)
    elif backend == "baseline":
        from .baseline_backend import generate_speech
        return generate_speech(text, ref_audio_path, output_path)
    elif backend == "qwen":
        from .qwen_backend import generate_speech
        return generate_speech(text, emotion, ref_audio_path, output_path)
    elif backend == "clone":
        from .clone_backend import generate_speech
        return generate_speech(text, emotion, ref_audio_path, output_path)
    elif backend == "fish":
        from .fish_backend import generate_speech
        return generate_speech(text, emotion, ref_audio_path, output_path)
    else:
        raise ValueError(f"Unknown TTS backend: {backend}")
