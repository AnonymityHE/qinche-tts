"""Benchmark: native Qwen3-TTS vs FasterQwen3TTS (CUDA Graph) inference speed.

Compares the SFT CustomVoice model with and without CUDA Graph acceleration.
Also tests clone_xvec with the Base model.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

torch.backends.cudnn.enabled = False

TEST_TEXTS = [
    "前面的红绿灯路口执行，不错，你的反应速度比梅菲斯特快0.1秒。",
    "慌什么，肩上的伤早就愈合了，听我的声音，这不是好得很。",
    "别妄自菲薄，你比你想象中有用得多，这几天他都会跟着你出门。",
    "好吧，林空试试，你的地盘你做主，有事给我打电话。",
]

SFT_CKPT = "models/qwen3-tts"
BASE_CKPT = "models/qwen3-tts-base"


def benchmark_native_sft():
    """Benchmark native Qwen3-TTS SFT CustomVoice inference."""
    from qwen_tts import Qwen3TTSModel

    print("=" * 60)
    print("Native SFT CustomVoice (no CUDA Graph)")
    print("=" * 60)

    print("Loading model...")
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        SFT_CKPT, device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s")

    # Warmup
    print("Warmup...")
    with torch.no_grad():
        model.generate_custom_voice(
            text="测试", speaker="qinche", language="Chinese", max_new_tokens=256,
        )

    times = []
    durations = []
    for text in TEST_TEXTS:
        t0 = time.time()
        with torch.no_grad():
            wavs, sr = model.generate_custom_voice(
                text=text, speaker="qinche", language="Chinese",
                max_new_tokens=2048, temperature=0.7, top_k=35, top_p=0.9,
            )
        gen_time = time.time() - t0
        audio_dur = len(wavs[0]) / sr
        rtf = gen_time / audio_dur
        times.append(gen_time)
        durations.append(audio_dur)
        print(f"  {text[:20]}... → {gen_time:.2f}s, audio={audio_dur:.2f}s, RTF={rtf:.3f}")

    mean_rtf = sum(t / d for t, d in zip(times, durations)) / len(times)
    print(f"\nNative SFT: mean_gen={np.mean(times):.2f}s, mean_RTF={mean_rtf:.3f}")

    del model
    torch.cuda.empty_cache()
    return {"method": "native_sft", "times": times, "durations": durations, "mean_rtf": mean_rtf}


def benchmark_cuda_graph_sft():
    """Benchmark FasterQwen3TTS with CUDA Graph."""
    from faster_qwen3_tts import FasterQwen3TTS

    print("\n" + "=" * 60)
    print("FasterQwen3TTS (CUDA Graph)")
    print("=" * 60)

    print("Loading model...")
    t0 = time.time()
    model = FasterQwen3TTS.from_pretrained(SFT_CKPT)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s")

    # Warmup (triggers CUDA graph capture)
    print("Warmup + CUDA Graph capture...")
    t0 = time.time()
    model.generate_custom_voice(
        text="测试", speaker="qinche", language="Chinese", max_new_tokens=256,
    )
    warmup_time = time.time() - t0
    print(f"Warmup done in {warmup_time:.2f}s")

    times = []
    durations = []
    for text in TEST_TEXTS:
        t0 = time.time()
        wavs, sr = model.generate_custom_voice(
            text=text, speaker="qinche", language="Chinese",
            max_new_tokens=2048, temperature=0.7, top_k=35, top_p=0.9,
        )
        gen_time = time.time() - t0
        audio_dur = len(wavs[0]) / sr
        rtf = gen_time / audio_dur
        times.append(gen_time)
        durations.append(audio_dur)
        print(f"  {text[:20]}... → {gen_time:.2f}s, audio={audio_dur:.2f}s, RTF={rtf:.3f}")

    mean_rtf = sum(t / d for t, d in zip(times, durations)) / len(times)
    print(f"\nCUDA Graph SFT: mean_gen={np.mean(times):.2f}s, mean_RTF={mean_rtf:.3f}")

    del model
    torch.cuda.empty_cache()
    return {"method": "cuda_graph_sft", "times": times, "durations": durations, "mean_rtf": mean_rtf}


def benchmark_clone_xvec():
    """Benchmark Base model voice clone (x-vector mode)."""
    from qwen_tts import Qwen3TTSModel

    print("\n" + "=" * 60)
    print("Base Model Clone (x-vector, native)")
    print("=" * 60)

    print("Loading base model...")
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        BASE_CKPT, device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s")

    # Create x-vector from ref audio
    print("Computing x-vector from ref audio...")
    ref_files = sorted(Path("data/ref_audio").glob("ref_*.wav"))
    embeddings = []
    for ref in ref_files[:3]:
        items = model.create_voice_clone_prompt(ref_audio=str(ref), x_vector_only_mode=True)
        embeddings.append(items[0].ref_spk_embedding)
    avg_emb = torch.stack(embeddings).mean(dim=0)

    class _SyntheticPrompt:
        def __init__(self, emb):
            self.ref_code = None
            self.ref_spk_embedding = emb
            self.x_vector_only_mode = True
            self.icl_mode = False
            self.ref_text = None

    prompt = [_SyntheticPrompt(avg_emb)]

    # Warmup
    print("Warmup...")
    with torch.no_grad():
        model.generate_voice_clone(
            text="测试", language="Chinese", voice_clone_prompt=prompt, max_new_tokens=256,
        )

    times = []
    durations = []
    for text in TEST_TEXTS:
        t0 = time.time()
        with torch.no_grad():
            wavs, sr = model.generate_voice_clone(
                text=text, language="Chinese", voice_clone_prompt=prompt,
                max_new_tokens=2048, temperature=0.7, top_k=35, top_p=0.9,
            )
        gen_time = time.time() - t0
        audio_dur = len(wavs[0]) / sr
        rtf = gen_time / audio_dur
        times.append(gen_time)
        durations.append(audio_dur)
        print(f"  {text[:20]}... → {gen_time:.2f}s, audio={audio_dur:.2f}s, RTF={rtf:.3f}")

    mean_rtf = sum(t / d for t, d in zip(times, durations)) / len(times)
    print(f"\nClone x-vec: mean_gen={np.mean(times):.2f}s, mean_RTF={mean_rtf:.3f}")

    del model
    torch.cuda.empty_cache()
    return {"method": "clone_xvec", "times": times, "durations": durations, "mean_rtf": mean_rtf}


if __name__ == "__main__":
    import json

    results = []

    r1 = benchmark_native_sft()
    results.append(r1)

    r2 = benchmark_cuda_graph_sft()
    results.append(r2)

    r3 = benchmark_clone_xvec()
    results.append(r3)

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    for r in results:
        print(f"{r['method']:25s}  mean_RTF={r['mean_rtf']:.3f}")
    print("=" * 60)

    with open("output/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Results saved to output/benchmark_results.json")
