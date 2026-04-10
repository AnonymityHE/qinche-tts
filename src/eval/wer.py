"""Word Error Rate evaluation using WhisperX ASR + jiwer.

Adapted from scripts_from_project/eval_sim_wer.py and eval_finetuned.py.
Includes Chinese-aware text normalization (T2S conversion, punctuation stripping).
"""

from __future__ import annotations

import re
from pathlib import Path

from jiwer import wer as _jiwer_wer

_ASR_MODEL = None
_ASR_DEVICE = None


def _normalize_text(text: str) -> str:
    """Normalize for CER-style WER: optional T2S conversion and strip punctuation."""
    try:
        from opencc import OpenCC

        text = OpenCC("t2s").convert(text)
    except ImportError:
        pass
    text = re.sub(
        r"[，。！？、；：\u201c\u201d\u2018\u2019《》【】（）\s.!?,;:\"'()\-\u2014\u2026]",
        "",
        text,
    )
    return text


def _get_asr_model():
    """Lazy-load WhisperX (or fallback to standard whisper) ASR model."""
    global _ASR_MODEL, _ASR_DEVICE
    if _ASR_MODEL is not None:
        return _ASR_MODEL, _ASR_DEVICE

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
    except ImportError:
        device = "cpu"
        compute_type = "int8"

    try:
        import whisperx

        _ASR_MODEL = whisperx.load_model(
            "large-v3", device, compute_type=compute_type, language="zh"
        )
        _ASR_DEVICE = device
        return _ASR_MODEL, _ASR_DEVICE
    except ImportError:
        import whisper

        _ASR_MODEL = whisper.load_model("base")
        _ASR_DEVICE = device
        return _ASR_MODEL, _ASR_DEVICE


def transcribe(audio_path: str | Path) -> str:
    """Transcribe audio to text. Uses WhisperX if available, else standard Whisper."""
    model, device = _get_asr_model()
    audio_path = str(audio_path)

    try:
        import whisperx

        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=4, language="zh")
        segments = result.get("segments", [])
        return "".join(seg.get("text", "") for seg in segments).strip()
    except (ImportError, AttributeError):
        result = model.transcribe(audio_path, language="zh")
        return result.get("text", "").strip()


def compute_wer(generated_path: str | Path, reference_text: str) -> float:
    """Transcribe generated audio and compute character-level WER against reference text."""
    hyp_text = transcribe(str(generated_path))
    return compute_wer_from_texts(reference_text, hyp_text)


def compute_wer_from_texts(reference_text: str, hypothesis_text: str) -> float:
    """Compute character-level WER between two text strings."""
    ref_chars = " ".join(_normalize_text(reference_text))
    hyp_chars = " ".join(_normalize_text(hypothesis_text))

    if not ref_chars.strip():
        return 0.0 if not hyp_chars.strip() else 1.0

    return float(_jiwer_wer(ref_chars, hyp_chars))
