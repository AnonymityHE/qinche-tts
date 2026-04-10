"""Voice Clone TTS backend — Audio-as-Emotion-Prompt.

Uses Qwen3-TTS Base model with emotion-matched reference audio from the
original training data. Instead of relying on text instruct for emotion,
the reference audio naturally carries voice identity + emotional prosody.

Flow: GPT-4o emotion → select matching ref clip → Base model voice clone → speech
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import soundfile as sf

from src.context_engine.models import EmotionAnnotation, EmotionCategory

from ._common import load_qwen_model, mock_generate, safe_filename


@dataclass
class _SyntheticXvecPrompt:
    """Mimics VoiceClonePromptItem for averaged x-vector prompts."""
    ref_code: object = None
    ref_spk_embedding: object = None
    x_vector_only_mode: bool = True
    icl_mode: bool = False
    ref_text: object = None

_MODEL = None
_MODEL_PATH = Path("models/qwen3-tts-base")
_EMOTION_BUCKETS_PATH = Path("data/emotion_buckets.json")

_PROMPT_CACHE: dict[str, object] = {}
_AVG_XVEC_PROMPTS: dict[str, object] = {}
_BUCKETS_CACHE: dict[str, list[dict]] | None = None


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    _MODEL = load_qwen_model(_MODEL_PATH, label="Qwen3-TTS Base (clone)")
    return _MODEL


def _load_emotion_buckets() -> dict[str, list[dict]]:
    """Load emotion buckets as {category: [{file, text, ...}, ...]}. Cached."""
    global _BUCKETS_CACHE
    if _BUCKETS_CACHE is not None:
        return _BUCKETS_CACHE

    if not _EMOTION_BUCKETS_PATH.exists():
        return {}

    with open(_EMOTION_BUCKETS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    buckets: dict[str, list[dict]] = {}
    for entry in data:
        emotion = entry.get("emotion", "calm")
        if emotion not in buckets:
            buckets[emotion] = []
        buckets[emotion].append(entry)
    _BUCKETS_CACHE = buckets
    return buckets


def _select_ref_entry(category: EmotionCategory) -> dict | None:
    """Select a reference audio entry matching the emotion category."""
    buckets = _load_emotion_buckets()
    candidates = buckets.get(category.value, [])
    if not candidates:
        all_entries = [e for entries in buckets.values() for e in entries]
        if all_entries:
            candidates = all_entries

    for entry in candidates:
        path = Path(entry.get("file", ""))
        if path.exists():
            return entry

    return None


def _find_entry_by_path(ref_audio_path: str | Path) -> dict | None:
    """Look up the emotion bucket entry for a specific audio file path."""
    buckets = _load_emotion_buckets()
    ref_str = str(ref_audio_path)
    for entries in buckets.values():
        for entry in entries:
            if entry.get("file") == ref_str:
                return entry
    return None


def _get_or_create_prompt(ref_audio_path: str, ref_text: str):
    """Get cached VoiceClonePromptItem or create one."""
    cache_key = ref_audio_path
    if cache_key in _PROMPT_CACHE:
        return _PROMPT_CACHE[cache_key]

    model = _load_model()
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        x_vector_only_mode=False,
    )
    _PROMPT_CACHE[cache_key] = prompt_items
    return prompt_items


def precompute_all_prompts():
    """Pre-compute VoiceClonePromptItem for every clip in emotion_buckets.json.

    Call once at startup to avoid latency during generation.
    Returns the number of prompts pre-computed.
    """
    buckets = _load_emotion_buckets()
    count = 0
    for category, entries in buckets.items():
        for entry in entries:
            ref_path = entry.get("file", "")
            ref_text = entry.get("text", "")
            if Path(ref_path).exists() and ref_text:
                _get_or_create_prompt(ref_path, ref_text)
                count += 1
                print(f"  [clone] cached prompt: {category} → {Path(ref_path).name}")
    return count


def compute_avg_xvec_prompts() -> int:
    """Method B: compute per-emotion averaged x-vector prompts.

    For each emotion category, extracts x-vectors from all available reference
    clips, averages them, and builds a synthetic prompt with
    x_vector_only_mode=True.  Returns count of categories processed.
    """
    import torch

    model = _load_model()
    buckets = _load_emotion_buckets()
    count = 0

    for category, entries in buckets.items():
        embeddings = []
        for entry in entries:
            ref_path = entry.get("file", "")
            if not Path(ref_path).exists():
                continue
            items = model.create_voice_clone_prompt(
                ref_audio=ref_path,
                x_vector_only_mode=True,
            )
            embeddings.append(items[0].ref_spk_embedding)

        if not embeddings:
            continue

        avg_emb = torch.stack(embeddings).mean(dim=0)
        _AVG_XVEC_PROMPTS[category] = [_SyntheticXvecPrompt(ref_spk_embedding=avg_emb)]
        count += 1
        print(f"  [clone] avg x-vec: {category} ({len(embeddings)} clips)")

    return count


def generate_speech_xvec(
    text: str,
    emotion: EmotionAnnotation,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Method B: generate speech using averaged x-vector per emotion category."""
    output_path = Path(output_path or f"output/clone_xvec/{safe_filename(text)}.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import torch

    category = emotion.ref_emotion_category.value
    if category not in _AVG_XVEC_PROMPTS:
        print(f"  [clone_xvec] no avg prompt for {category}, falling back to ICL")
        return generate_speech(text, emotion, output_path=output_path)

    model = _load_model()
    prompt = _AVG_XVEC_PROMPTS[category]

    with torch.no_grad():
        wavs, sr = model.generate_voice_clone(
            text=text,
            language="Chinese",
            voice_clone_prompt=prompt,
            max_new_tokens=2048,
            temperature=0.7,
            top_k=35,
            top_p=0.9,
        )

    sf.write(str(output_path), wavs[0], sr)
    print(f"  [clone_xvec] {output_path} (avg x-vec: {category})")
    return output_path


def generate_speech_blend(
    text: str,
    emotion: EmotionAnnotation,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Method C: interpolate calm and emotion x-vectors weighted by intensity.

    final_xvec = (1 - intensity) * calm_xvec + intensity * emotion_xvec

    This produces a smooth gradient from the speaker's neutral timbre
    to a fully emotion-coloured voice.  When the target IS calm, delegates
    directly to generate_speech_xvec to avoid a no-op interpolation.
    """
    output_path = Path(output_path or f"output/clone_blend/{safe_filename(text)}.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import torch

    category = emotion.ref_emotion_category.value

    if category == "calm":
        return generate_speech_xvec(text, emotion, output_path=output_path)

    calm_prompt = _AVG_XVEC_PROMPTS.get("calm")
    emotion_prompt = _AVG_XVEC_PROMPTS.get(category)

    if not calm_prompt or not emotion_prompt:
        print(f"  [clone_blend] missing x-vec (calm={bool(calm_prompt)}, "
              f"{category}={bool(emotion_prompt)}), fallback to xvec")
        return generate_speech_xvec(text, emotion, output_path=output_path)

    alpha = max(0.0, min(1.0, emotion.intensity))
    calm_emb = calm_prompt[0].ref_spk_embedding
    emo_emb = emotion_prompt[0].ref_spk_embedding
    blend_emb = (1.0 - alpha) * calm_emb + alpha * emo_emb

    blended_prompt = [_SyntheticXvecPrompt(ref_spk_embedding=blend_emb)]

    model = _load_model()
    with torch.no_grad():
        wavs, sr = model.generate_voice_clone(
            text=text,
            language="Chinese",
            voice_clone_prompt=blended_prompt,
            max_new_tokens=2048,
            temperature=0.7,
            top_k=35,
            top_p=0.9,
        )

    sf.write(str(output_path), wavs[0], sr)
    print(f"  [clone_blend] {output_path} (calm↔{category}, α={alpha:.2f})")
    return output_path


def generate_speech(
    text: str,
    emotion: EmotionAnnotation,
    ref_audio_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Generate speech via Base model voice clone with emotion-matched reference.

    The reference audio is selected from emotion_buckets.json based on
    emotion.ref_emotion_category. If ref_audio_path is explicitly provided
    and exists, it takes precedence.
    """
    output_path = Path(output_path or f"output/clone/{safe_filename(text)}.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not _MODEL_PATH.exists() or not any(_MODEL_PATH.iterdir()):
        return mock_generate(text, output_path, tag="clone")

    import torch

    ref_entry = _select_ref_entry(emotion.ref_emotion_category)

    if ref_audio_path and Path(ref_audio_path).exists():
        matched = _find_entry_by_path(ref_audio_path)
        actual_ref = str(ref_audio_path)
        actual_text = (matched or ref_entry or {}).get("text", "")
    elif ref_entry:
        actual_ref = ref_entry["file"]
        actual_text = ref_entry["text"]
    else:
        print(f"  [clone] no ref audio for {emotion.ref_emotion_category.value}, "
              "falling back to mock")
        return mock_generate(text, output_path, tag="clone")

    model = _load_model()
    prompt_items = _get_or_create_prompt(actual_ref, actual_text)

    with torch.no_grad():
        wavs, sr = model.generate_voice_clone(
            text=text,
            language="Chinese",
            voice_clone_prompt=prompt_items,
            max_new_tokens=2048,
            temperature=0.7,
            top_k=35,
            top_p=0.9,
        )

    sf.write(str(output_path), wavs[0], sr)
    print(f"  [clone] {output_path} (ref: {Path(actual_ref).name})")
    return output_path


