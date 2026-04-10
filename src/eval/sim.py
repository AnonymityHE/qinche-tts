"""Speaker similarity evaluation using pyannote embedding cosine distance.

Supports both single-pair comparison and batch evaluation with caching.
Adapted from scripts_from_project/eval_sim_wer.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_SPEAKER_MODEL = None


def _get_speaker_model():
    """Lazy-load and cache the pyannote speaker embedding inference model."""
    global _SPEAKER_MODEL
    if _SPEAKER_MODEL is not None:
        return _SPEAKER_MODEL

    from pyannote.audio import Model, Inference

    model = Model.from_pretrained(
        "pyannote/embedding",
        use_auth_token=os.environ.get("HF_TOKEN", ""),
    )
    _SPEAKER_MODEL = Inference(model, window="whole")
    return _SPEAKER_MODEL


def get_embedding(audio_path: str | Path) -> np.ndarray:
    """Extract a normalized speaker embedding from an audio file."""
    inference = _get_speaker_model()

    try:
        emb = inference(str(audio_path)).reshape(-1)
    except Exception:
        import torchaudio

        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        emb = inference({"waveform": waveform, "sample_rate": 16000}).reshape(-1)

    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def compute_sim(generated_path: str | Path, reference_path: str | Path) -> float:
    """Compute cosine similarity between speaker embeddings of two audio files."""
    emb_gen = get_embedding(generated_path)
    emb_ref = get_embedding(reference_path)
    return float(np.dot(emb_gen, emb_ref))


def compute_sim_against_mean(
    generated_path: str | Path,
    ref_audio_paths: list[str | Path],
) -> float:
    """Compute cosine similarity between generated audio and the mean of multiple references.

    This is useful when you have several reference clips (e.g. ref_00..ref_04)
    and want a single aggregate similarity score.
    """
    emb_gen = get_embedding(generated_path)
    ref_embs = [get_embedding(p) for p in ref_audio_paths]
    ref_mean = np.mean(ref_embs, axis=0)
    ref_mean = ref_mean / np.linalg.norm(ref_mean)
    return float(np.dot(emb_gen, ref_mean))
