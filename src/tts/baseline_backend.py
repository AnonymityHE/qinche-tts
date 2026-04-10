"""Baseline TTS backend — no emotion control.

Uses the fine-tuned CustomVoice model without any instruct parameter,
serving as the no-context baseline for A/B comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import soundfile as sf

from ._common import load_qwen_model, mock_generate, safe_filename

_MODEL = None
_MODEL_PATH = Path("models/qwen3-tts")
_SPEAKER = "qinche"


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    try:
        from .qwen_backend import _MODEL as _shared
        if _shared is not None:
            _MODEL = _shared
            return _MODEL
    except ImportError:
        pass

    _MODEL = load_qwen_model(_MODEL_PATH, label="Baseline")
    return _MODEL


def generate_speech(
    text: str,
    ref_audio_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Generate speech with NO emotion control — flat prosody baseline."""
    output_path = Path(output_path or f"output/baseline/{safe_filename(text)}.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not _MODEL_PATH.exists() or not any(_MODEL_PATH.iterdir()):
        return mock_generate(text, output_path, tag="baseline")

    import torch

    model = _load_model()

    with torch.no_grad():
        wavs, sr = model.generate_custom_voice(
            text=text,
            speaker=_SPEAKER,
            language="Chinese",
            max_new_tokens=2048,
            temperature=0.7,
            top_k=35,
            top_p=0.9,
        )

    sf.write(str(output_path), wavs[0], sr)
    return output_path


