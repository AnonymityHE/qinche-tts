"""Qwen3-TTS backend — SFT CustomVoice pathway with emotion-aware instruct.

Uses generate_custom_voice(text, language, speaker, instruct) with the
fine-tuned "qinche" speaker. Runs in mock mode when checkpoint is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import soundfile as sf

from src.context_engine.models import EmotionAnnotation

from ._common import load_qwen_model, mock_generate, safe_filename

_MODEL = None
_MODEL_PATH = Path("models/qwen3-tts")
_SPEAKER = "qinche"


def _load_model():
    """Lazy-load the Qwen3-TTS model from local checkpoint."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    _MODEL = load_qwen_model(_MODEL_PATH, label="Qwen3-TTS SFT")
    return _MODEL


def _format_instruct(style: str) -> str:
    """Normalize LLM style text to Qwen3-TTS imperative instruct format.

    Ensures the instruct starts with '用' and ends with '说'/'口吻' pattern.
    If already in correct format, returns as-is. Caps length at 30 chars.
    """
    if not style:
        return ""
    style = style.strip()
    if style.startswith("用") and ("说" in style or "口吻" in style):
        return style[:30]
    return f"用{style}的语气说"[:30]


def generate_speech(
    text: str,
    emotion: EmotionAnnotation,
    ref_audio_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Generate speech using Qwen3-TTS CustomVoice with emotion control.

    The fine-tuned model has the "qinche" speaker embedding baked in,
    so we always use generate_custom_voice + instruct for emotion control.
    ref_audio_path is ignored — CustomVoice models don't support voice clone.
    """
    output_path = Path(output_path or f"output/qwen/{safe_filename(text)}.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not _MODEL_PATH.exists() or not any(_MODEL_PATH.iterdir()):
        return mock_generate(text, output_path, tag="qwen_backend")

    import torch

    model = _load_model()

    instruct = _format_instruct(emotion.style)

    with torch.no_grad():
        wavs, sr = model.generate_custom_voice(
            text=text,
            speaker=_SPEAKER,
            language="Chinese",
            instruct=instruct,
            max_new_tokens=2048,
            temperature=0.7,
            top_k=35,
            top_p=0.9,
        )

    sf.write(str(output_path), wavs[0], sr)
    return output_path


