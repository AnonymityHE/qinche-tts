"""Emotion accuracy evaluation using emotion2vec.

Classifies the emotion of generated audio and compares against LLM-predicted target labels.
Uses a coarse mapping to account for vocabulary mismatch between emotion2vec and our 6-class system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

_EMOTION_MODEL = None


def _get_model(model_dir: Optional[str] = None):
    """Lazy-load and cache the emotion2vec model."""
    global _EMOTION_MODEL
    if _EMOTION_MODEL is not None:
        return _EMOTION_MODEL

    from funasr import AutoModel

    model_dir = model_dir or "models/emotion2vec"
    _EMOTION_MODEL = AutoModel(model="iic/emotion2vec_plus_large", cache_dir=model_dir)
    return _EMOTION_MODEL


EMOTION_MAP: dict[str, str] = {
    # emotion2vec outputs -> coarse category
    "happy": "positive",
    "excited": "positive",
    "surprise": "positive",
    # our 6-class -> coarse
    "tender": "positive",
    "playful": "positive",
    "intimate": "positive",
    # negative
    "sad": "negative",
    "angry": "negative",
    "fearful": "negative",
    "disgusted": "negative",
    "intense": "negative",
    # neutral
    "calm": "neutral",
    "cold": "neutral",
    "neutral": "neutral",
}


def compute_emotion_accuracy(
    generated_path: str | Path,
    target_emotion: str,
    model_dir: Optional[str] = None,
) -> dict:
    """Classify the emotion of generated audio and compare against the target label.

    Returns a dict with predicted emotion, confidence, coarse match boolean,
    and the coarse categories used for comparison.
    """
    model = _get_model(model_dir)
    result = model.generate(str(generated_path))

    if result and len(result) > 0:
        pred = result[0]
        labels = pred.get("labels", [])
        scores = pred.get("scores", [])
        predicted_label = labels[0] if labels else "unknown"
        confidence = float(scores[0]) if scores else 0.0
    else:
        predicted_label = "unknown"
        confidence = 0.0

    target_coarse = EMOTION_MAP.get(target_emotion, target_emotion)
    predicted_coarse = EMOTION_MAP.get(predicted_label, predicted_label)

    return {
        "predicted": predicted_label,
        "predicted_coarse": predicted_coarse,
        "target": target_emotion,
        "target_coarse": target_coarse,
        "confidence": confidence,
        "match": target_coarse == predicted_coarse,
    }
