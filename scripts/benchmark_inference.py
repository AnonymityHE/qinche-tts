"""Benchmark: baseline Qwen3TTSModel vs FasterQwen3TTS (CUDA graphs).

Runs the same test sentences through both backends and reports RTF comparison.
Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_inference.py \
        --ckpt output/qinche_sft_v2/checkpoint-epoch-3 \
        --mode custom_voice \
        --num-samples 5
"""
import argparse
import json
import os
import time

import numpy as np
import soundfile as sf
import torch


def get_test_sentences(manifest_path, num_samples=5):
    items = []
    with open(manifest_path) as f:
        for line in f:
            items.append(json.loads(line.strip()))
    return items[:num_samples]


def benchmark_baseline(ckpt_path, sentences, speaker, language, device):
    """Benchmark using original Qwen3TTSModel."""
    from qwen_tts import Qwen3TTSModel

    print(f"\n{'='*60}")
    print("BASELINE: Qwen3TTSModel (dynamic cache)")
    print(f"{'='*60}")

    torch.set_float32_matmul_precision("high")

    try:
        model = Qwen3TTSModel.from_pretrained(
            ckpt_path, device_map=device, dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        print("  Using: flash_attention_2")
    except (ImportError, ValueError):
        model = Qwen3TTSModel.from_pretrained(
            ckpt_path, device_map=device, dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        print("  Using: sdpa")

    print("  Warming up...")
    with torch.no_grad():
        _ = model.generate_custom_voice(
            text="测试。", speaker=speaker, language=language, max_new_tokens=32,
        )
    del _
    torch.cuda.synchronize()
    print("  Warmup done.")

    rtfs = []
    for i, item in enumerate(sentences):
        text = item["text"]
        print(f"  [{i+1}/{len(sentences)}] {text[:40]}...")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        wavs, sr = model.generate_custom_voice(
            text=text, speaker=speaker, language=language, max_new_tokens=2048,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        audio = wavs[0]
        audio_dur = len(audio) / sr
        rtf = elapsed / audio_dur if audio_dur > 0 else 999
        rtfs.append(rtf)
        print(f"    elapsed={elapsed:.2f}s  audio={audio_dur:.2f}s  RTF={rtf:.3f}")

    del model
    torch.cuda.empty_cache()

    avg_rtf = float(np.mean(rtfs))
    print(f"\n  >> Baseline avg RTF: {avg_rtf:.3f}")
    return rtfs, avg_rtf


def benchmark_faster(ckpt_path, sentences, speaker, language, device):
    """Benchmark using FasterQwen3TTS (CUDA graphs)."""
    from faster_qwen3_tts import FasterQwen3TTS

    print(f"\n{'='*60}")
    print("FASTER: FasterQwen3TTS (CUDA graphs)")
    print(f"{'='*60}")

    torch.set_float32_matmul_precision("high")

    model = FasterQwen3TTS.from_pretrained(
        ckpt_path, device=device, dtype=torch.bfloat16,
    )
    print(f"  Loaded from {ckpt_path}")

    print("  Warming up (CUDA graph capture)...")
    wavs, sr = model.generate_custom_voice(
        text="测试。", speaker=speaker, language=language, max_new_tokens=64,
    )
    torch.cuda.synchronize()
    print("  Warmup done.")

    rtfs = []
    for i, item in enumerate(sentences):
        text = item["text"]
        print(f"  [{i+1}/{len(sentences)}] {text[:40]}...")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        wavs, sr = model.generate_custom_voice(
            text=text, speaker=speaker, language=language, max_new_tokens=2048,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        audio = wavs[0]
        audio_dur = len(audio) / sr
        rtf = elapsed / audio_dur if audio_dur > 0 else 999
        rtfs.append(rtf)
        print(f"    elapsed={elapsed:.2f}s  audio={audio_dur:.2f}s  RTF={rtf:.3f}")

    del model
    torch.cuda.empty_cache()

    avg_rtf = float(np.mean(rtfs))
    print(f"\n  >> FasterQwen3TTS avg RTF: {avg_rtf:.3f}")
    return rtfs, avg_rtf


def benchmark_faster_streaming(ckpt_path, sentences, speaker, language, device):
    """Benchmark FasterQwen3TTS streaming mode."""
    from faster_qwen3_tts import FasterQwen3TTS

    print(f"\n{'='*60}")
    print("FASTER STREAMING: FasterQwen3TTS (CUDA graphs + streaming)")
    print(f"{'='*60}")

    torch.set_float32_matmul_precision("high")

    model = FasterQwen3TTS.from_pretrained(
        ckpt_path, device=device, dtype=torch.bfloat16,
    )

    print("  Warming up...")
    wavs, sr = model.generate_custom_voice(
        text="测试。", speaker=speaker, language=language, max_new_tokens=64,
    )
    torch.cuda.synchronize()
    print("  Warmup done.")

    rtfs = []
    ttfas = []
    for i, item in enumerate(sentences):
        text = item["text"]
        print(f"  [{i+1}/{len(sentences)}] {text[:40]}...")

        chunks = []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        t_first = None
        for audio_chunk, sr, timing in model.generate_custom_voice_streaming(
            text=text, speaker=speaker, language=language,
            max_new_tokens=2048, chunk_size=8,
        ):
            if t_first is None:
                t_first = time.perf_counter()
            chunks.append(audio_chunk)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        ttfa = (t_first - t0) if t_first else elapsed

        audio = np.concatenate(chunks) if chunks else np.array([])
        audio_dur = len(audio) / sr if sr > 0 else 0
        rtf = elapsed / audio_dur if audio_dur > 0 else 999
        rtfs.append(rtf)
        ttfas.append(ttfa)
        print(f"    elapsed={elapsed:.2f}s  audio={audio_dur:.2f}s  RTF={rtf:.3f}  TTFA={ttfa*1000:.0f}ms")

    del model
    torch.cuda.empty_cache()

    avg_rtf = float(np.mean(rtfs))
    avg_ttfa = float(np.mean(ttfas)) * 1000
    print(f"\n  >> Streaming avg RTF: {avg_rtf:.3f}  avg TTFA: {avg_ttfa:.0f}ms")
    return rtfs, avg_rtf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--mode", default="custom_voice", choices=["custom_voice", "voice_clone"])
    parser.add_argument("--speaker", default="qinche")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--manifest", default="/home/ubuntu/yunlin/TTS/data/dataset/test_manifest.jsonl")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-faster", action="store_true")
    parser.add_argument("--skip-streaming", action="store_true")
    args = parser.parse_args()

    sentences = get_test_sentences(args.manifest, args.num_samples)
    print(f"Benchmark: {len(sentences)} sentences from {args.manifest}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Device: {args.device}")

    results = {}

    if not args.skip_baseline:
        baseline_rtfs, baseline_avg = benchmark_baseline(
            args.ckpt, sentences, args.speaker, args.language, args.device,
        )
        results["baseline"] = {"rtfs": baseline_rtfs, "avg_rtf": baseline_avg}

    if not args.skip_faster:
        try:
            faster_rtfs, faster_avg = benchmark_faster(
                args.ckpt, sentences, args.speaker, args.language, args.device,
            )
            results["faster"] = {"rtfs": faster_rtfs, "avg_rtf": faster_avg}
        except Exception as e:
            print(f"\n  ERROR with FasterQwen3TTS: {e}")
            import traceback
            traceback.print_exc()

    if not args.skip_streaming:
        try:
            stream_rtfs, stream_avg = benchmark_faster_streaming(
                args.ckpt, sentences, args.speaker, args.language, args.device,
            )
            results["streaming"] = {"rtfs": stream_rtfs, "avg_rtf": stream_avg}
        except Exception as e:
            print(f"\n  ERROR with streaming: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        print(f"  {name:20s}  avg_rtf={r['avg_rtf']:.3f}")

    if "baseline" in results and "faster" in results:
        speedup = results["baseline"]["avg_rtf"] / results["faster"]["avg_rtf"]
        print(f"\n  Speedup (CUDA graphs): {speedup:.2f}x")

    if "baseline" in results and "streaming" in results:
        speedup = results["baseline"]["avg_rtf"] / results["streaming"]["avg_rtf"]
        print(f"  Speedup (streaming):   {speedup:.2f}x")

    out_path = os.path.join(os.path.dirname(args.ckpt), "..", "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
