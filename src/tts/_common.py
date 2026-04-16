"""Shared utilities for TTS backends."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

# Workaround for cuDNN CUDNN_STATUS_NOT_INITIALIZED error on this machine
import torch
torch.backends.cudnn.enabled = False


def select_device() -> tuple[str, "torch.dtype"]:
    """Return (device, dtype) for model loading."""
    import torch

    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def safe_filename(text: str, max_len: int = 30) -> str:
    """Sanitize text into a safe filename fragment."""
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in text)
    return safe[:max_len]


def mock_generate(text: str, output_path: Path, tag: str = "mock") -> Path:
    """Generate a silent placeholder WAV when model is not available."""
    sample_rate = 24000
    duration_s = max(1.0, len(text) * 0.15)
    silence = np.zeros(int(sample_rate * duration_s), dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), silence, sample_rate)
    print(f"  [{tag} MOCK] {output_path}")
    return output_path


def load_qwen_model(model_path: str | Path, label: str = "Qwen3-TTS"):
    """Load a Qwen3TTSModel from checkpoint with device auto-detection.

    Returns the loaded model, or raises FileNotFoundError.
    """
    model_path = Path(model_path)
    if not model_path.exists() or not any(model_path.iterdir()):
        raise FileNotFoundError(f"{label} checkpoint not found at {model_path}")

    from qwen_tts import Qwen3TTSModel

    device, dtype = select_device()

    try:
        return Qwen3TTSModel.from_pretrained(
            str(model_path),
            device_map=device,
            dtype=dtype,
            attn_implementation="flash_attention_2",
        )
    except (ImportError, RuntimeError):
        return Qwen3TTSModel.from_pretrained(
            str(model_path),
            device_map=device,
            dtype=dtype,
            attn_implementation="sdpa",
        )
