"""Fish Audio S2 Pro backend — comparison pathway using inline emotion tags.

Adapted from archive/scripts_from_project/eval_fish_zeroshot.py for the pipeline interface.

Deployment (on A100):
    git clone https://github.com/fishaudio/fish-speech
    huggingface-cli download fishaudio/fish-speech-1.5 --local-dir models/fish-speech

Local dev: code runs in mock mode when model weights are absent.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import soundfile as sf

from src.context_engine.models import EmotionAnnotation

from ._common import mock_generate, safe_filename

_MODEL_PATH = Path("models/fish-speech")
_CODEC = None
_LLM_MODEL = None
_DECODE_ONE_TOKEN = None
_REF_CODES = None
_REF_TEXT = None


def _load_model(compile: bool = False):
    """Load Fish Audio S2 Pro: LLM + Codec + reference audio encoding."""
    global _LLM_MODEL, _DECODE_ONE_TOKEN, _CODEC, _REF_CODES, _REF_TEXT

    if _LLM_MODEL is not None:
        return

    import torch

    if not _MODEL_PATH.exists() or not any(_MODEL_PATH.iterdir()):
        raise FileNotFoundError(
            f"Fish Audio S2 Pro weights not found at {_MODEL_PATH}. "
            "Download with:\n"
            "  huggingface-cli download fishaudio/fish-speech-1.5 "
            f"--local-dir {_MODEL_PATH}"
        )

    fish_speech_dir = os.environ.get("FISH_SPEECH_DIR", "")
    if fish_speech_dir and fish_speech_dir not in sys.path:
        sys.path.insert(0, fish_speech_dir)

    from fish_speech.models.text2semantic.inference import (
        init_model,
        load_codec_model,
        encode_audio,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = torch.bfloat16 if device == "cuda" else torch.float32

    _LLM_MODEL, _DECODE_ONE_TOKEN = init_model(
        checkpoint_path=_MODEL_PATH,
        device=device,
        precision=precision,
        compile=compile,
    )

    with torch.device(device):
        _LLM_MODEL.setup_caches(
            max_batch_size=1,
            max_seq_len=_LLM_MODEL.config.max_seq_len,
            dtype=next(_LLM_MODEL.parameters()).dtype,
        )

    codec_path = _MODEL_PATH / "codec.pth"
    _CODEC = load_codec_model(
        codec_checkpoint_path=codec_path,
        device=device,
        precision=precision,
    )

    ref_audio_path = Path("data/ref_audio/default.wav")
    if ref_audio_path.exists():
        _REF_CODES = encode_audio(str(ref_audio_path), _CODEC, device)

    ref_manifest = Path("data/ref_manifest.json")
    if ref_manifest.exists():
        with open(ref_manifest, encoding="utf-8") as f:
            refs = json.load(f)
        _REF_TEXT = refs[0].get("text", "") if refs else ""


def generate_speech(
    text: str,
    emotion: EmotionAnnotation,
    ref_audio_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Generate speech using Fish Audio S2 Pro with inline emotion tags.

    Uses emotion.fish_audio_tags which contains text with embedded
    [tag] annotations, e.g. "[温柔][轻松]你还好吗？"
    """
    output_path = Path(output_path or f"output/fish/{safe_filename(text)}.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tagged_text = emotion.fish_audio_tags or text

    if not _MODEL_PATH.exists() or not any(_MODEL_PATH.iterdir()):
        return mock_generate(tagged_text, output_path, tag="fish_backend")

    import torch

    _load_model()

    from fish_speech.models.text2semantic.inference import (
        generate_long,
        decode_to_audio,
        encode_audio,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ref_codes = _REF_CODES
    ref_text = _REF_TEXT or ""

    if ref_audio_path and Path(ref_audio_path).exists():
        ref_codes = encode_audio(str(ref_audio_path), _CODEC, device)

    if ref_codes is None:
        raise FileNotFoundError("No reference audio available for Fish Audio voice cloning")

    codes_list = []
    for resp in generate_long(
        model=_LLM_MODEL,
        device=device,
        decode_one_token=_DECODE_ONE_TOKEN,
        text=tagged_text,
        max_new_tokens=2048,
        prompt_text=[ref_text] if ref_text else None,
        prompt_tokens=[ref_codes],
        temperature=0.7,
        top_p=0.8,
        top_k=30,
        compile=False,
    ):
        if resp.action == "sample" and resp.codes is not None:
            codes_list.append(resp.codes)

    if not codes_list:
        raise RuntimeError(f"Fish Audio generated no codes for: {tagged_text[:50]}...")

    merged_codes = torch.cat(codes_list, dim=1).to(device)
    audio_tensor = decode_to_audio(merged_codes, _CODEC)

    audio_np = audio_tensor.cpu().float().numpy()
    sample_rate = _CODEC.sample_rate
    sf.write(str(output_path), audio_np, sample_rate)

    return output_path


