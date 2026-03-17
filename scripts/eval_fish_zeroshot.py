"""Evaluate Fish Audio S2 Pro zero-shot voice cloning on the same test set
used for Qwen3-TTS evaluation.

Usage:
  CUDA_VISIBLE_DEVICES=5 conda run -n fish-speech python scripts/eval_fish_zeroshot.py
  CUDA_VISIBLE_DEVICES=5 conda run -n fish-speech python scripts/eval_fish_zeroshot.py --compile
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

FISH_SPEECH_DIR = "/home/ubuntu/yunlin/TTS/fish-speech"
CHECKPOINT_PATH = Path("/home/ubuntu/yunlin/TTS/models/fish-speech-s2-pro")
TEST_MANIFEST = "/home/ubuntu/yunlin/TTS/data/dataset/test_manifest.jsonl"
REF_MANIFEST = "/home/ubuntu/yunlin/TTS/data/dataset/ref_manifest.json"
REF_AUDIO = "/home/ubuntu/yunlin/TTS/data/ref_audio/ref_00.wav"

DEVICE = "cuda"
PRECISION = torch.bfloat16

sys.path.insert(0, FISH_SPEECH_DIR)


def load_test_data():
    items = []
    with open(TEST_MANIFEST) as f:
        for line in f:
            items.append(json.loads(line.strip()))
    return items


def load_ref_text():
    with open(REF_MANIFEST) as f:
        refs = json.load(f)
    return refs[0]["text"]


# ── Evaluation helpers (same as Qwen3 eval) ──

def build_speaker_model():
    from pyannote.audio import Model, Inference
    model = Model.from_pretrained(
        "pyannote/embedding",
        use_auth_token=os.environ.get("HF_TOKEN", ""),
    )
    return Inference(model, window="whole")


def get_embedding(inference, wav_path):
    import torchaudio
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    emb = inference({"waveform": waveform, "sample_rate": 16000}).reshape(-1)
    return emb / np.linalg.norm(emb)


def build_asr():
    import whisperx
    return whisperx.load_model("large-v3", "cuda", compute_type="float16", language="zh")


def transcribe(asr_model, wav_path):
    import whisperx
    audio = whisperx.load_audio(wav_path)
    result = asr_model.transcribe(audio, batch_size=4, language="zh")
    return "".join(seg["text"] for seg in result["segments"]).strip()


def _normalize_text(text):
    import re
    try:
        from opencc import OpenCC
        text = OpenCC('t2s').convert(text)
    except ImportError:
        pass
    text = re.sub(r"[，。！？、；：\u201c\u201d\u2018\u2019《》【】（）\s.!?,;:\"'()\-—…]", "", text)
    return text


def compute_wer(ref_text, hyp_text):
    from jiwer import wer
    ref_chars = " ".join(_normalize_text(ref_text))
    hyp_chars = " ".join(_normalize_text(hyp_text))
    if not ref_chars.strip():
        return 0.0 if not hyp_chars.strip() else 1.0
    return wer(ref_chars, hyp_chars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for decode_one_token")
    args = parser.parse_args()

    use_compile = args.compile
    EVAL_DIR = f"/home/ubuntu/yunlin/TTS/eval/fish_s2_zeroshot{'_compile' if use_compile else ''}"
    os.makedirs(EVAL_DIR, exist_ok=True)

    test_data = load_test_data()
    ref_text = load_ref_text()
    print(f"Test samples: {len(test_data)}")
    print(f"Reference audio: {REF_AUDIO}")
    print(f"Reference text: {ref_text[:60]}...")
    print(f"torch.compile: {use_compile}")

    # ── Load evaluation tools ──
    print("Building speaker embedding model...")
    spk_inf = build_speaker_model()

    print("Computing ground truth embeddings...")
    gt_embeddings = {}
    for item in test_data:
        gt_embeddings[item["audio_filepath"]] = get_embedding(spk_inf, item["audio_filepath"])

    print("Computing reference mean embedding...")
    ref_audios = json.load(open(REF_MANIFEST))
    ref_embs = [get_embedding(spk_inf, r["audio_filepath"]) for r in ref_audios]
    ref_mean_emb = np.mean(ref_embs, axis=0)
    ref_mean_emb /= np.linalg.norm(ref_mean_emb)

    print("Building ASR model...")
    asr_model = build_asr()

    # ── Load Fish S2 Pro ──
    print(f"\nLoading Fish Audio S2 Pro (compile={use_compile})...")
    from fish_speech.models.text2semantic.inference import (
        init_model,
        load_codec_model,
        encode_audio,
        decode_to_audio,
        generate_long,
    )

    model, decode_one_token = init_model(
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
        precision=PRECISION,
        compile=use_compile,
    )
    with torch.device(DEVICE):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )

    codec = load_codec_model(
        codec_checkpoint_path=CHECKPOINT_PATH / "codec.pth",
        device=DEVICE,
        precision=PRECISION,
    )

    print("Encoding reference audio to VQ tokens...")
    ref_codes = encode_audio(REF_AUDIO, codec, DEVICE)
    print(f"  Reference VQ shape: {ref_codes.shape}")

    # Warmup (extra important with compile — first call triggers compilation)
    print("Warming up (this may take 30-60s with --compile)...")
    warmup_t0 = time.time()
    for resp in generate_long(
        model=model,
        device=DEVICE,
        decode_one_token=decode_one_token,
        text="测试一下编译后的推理速度",
        max_new_tokens=128,
        prompt_text=[ref_text],
        prompt_tokens=[ref_codes],
        compile=use_compile,
    ):
        pass
    warmup_elapsed = time.time() - warmup_t0
    print(f"Warmup done in {warmup_elapsed:.1f}s.\n")

    # ── Run evaluation ──
    sims_gt, sims_ref, wers, rtfs = [], [], [], []

    for i, item in enumerate(test_data):
        text = item["text"]
        gt_path = item["audio_filepath"]

        print(f"  [{i+1}/{len(test_data)}] {text[:50]}...")

        torch.cuda.synchronize()
        t0 = time.time()

        codes_list = []
        try:
            for resp in generate_long(
                model=model,
                device=DEVICE,
                decode_one_token=decode_one_token,
                text=text,
                max_new_tokens=2048,
                prompt_text=[ref_text],
                prompt_tokens=[ref_codes],
                temperature=0.7,
                top_p=0.8,
                top_k=30,
                compile=use_compile,
            ):
                if resp.action == "sample" and resp.codes is not None:
                    codes_list.append(resp.codes)
        except Exception as e:
            print(f"    FAILED: {e}")
            continue

        if not codes_list:
            print(f"    FAILED: no codes generated")
            continue

        merged_codes = torch.cat(codes_list, dim=1).to(DEVICE)
        audio_tensor = decode_to_audio(merged_codes, codec)

        torch.cuda.synchronize()
        elapsed = time.time() - t0

        audio_np = audio_tensor.cpu().float().numpy()
        sample_rate = codec.sample_rate
        audio_dur = len(audio_np) / sample_rate
        rtf = elapsed / audio_dur if audio_dur > 0 else 999

        out_path = os.path.join(EVAL_DIR, f"gen_{i:02d}.wav")
        sf.write(out_path, audio_np, sample_rate)

        gen_emb = get_embedding(spk_inf, out_path)
        gt_emb = gt_embeddings[gt_path]
        sim_gt = float(np.dot(gen_emb, gt_emb))
        sim_ref = float(np.dot(gen_emb, ref_mean_emb))

        hyp_text = transcribe(asr_model, out_path)
        wer_val = compute_wer(text, hyp_text)

        sims_gt.append(sim_gt)
        sims_ref.append(sim_ref)
        wers.append(wer_val)
        rtfs.append(rtf)

        print(f"    SIM_gt={sim_gt:.3f} SIM_ref={sim_ref:.3f} WER={wer_val:.3f} RTF={rtf:.3f}")

    # ── Summary ──
    avg_sim_gt = np.mean(sims_gt) if sims_gt else 0
    avg_sim_ref = np.mean(sims_ref) if sims_ref else 0
    avg_wer = np.mean(wers) if wers else 0
    avg_rtf = np.mean(rtfs) if rtfs else 0

    print(f"\n  >> Fish S2 Pro Zero-shot: SIM_gt={avg_sim_gt:.4f} SIM_ref={avg_sim_ref:.4f} WER={avg_wer:.4f} RTF={avg_rtf:.3f}")

    results = {
        "model": "Fish-Audio-S2-Pro",
        "mode": f"zero-shot{' (compiled)' if use_compile else ''}",
        "compile": use_compile,
        "ref_audio": REF_AUDIO,
        "sim_gt": float(avg_sim_gt),
        "sim_ref": float(avg_sim_ref),
        "wer": float(avg_wer),
        "rtf": float(avg_rtf),
        "num_samples": len(sims_gt),
        "per_sample": [
            {"sim_gt": float(s), "sim_ref": float(r), "wer": float(w), "rtf": float(t)}
            for s, r, w, t in zip(sims_gt, sims_ref, wers, rtfs)
        ],
    }
    with open(os.path.join(EVAL_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {EVAL_DIR}/results.json")


if __name__ == "__main__":
    main()
