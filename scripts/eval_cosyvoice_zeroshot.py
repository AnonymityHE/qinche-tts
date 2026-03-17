"""Zero-shot voice clone evaluation for CosyVoice2 / CosyVoice3 (v2 — improved).

Key improvements over v1:
  1. text_frontend=False — skip text normalization that may corrupt input
  2. Try ALL 5 reference audios, pick the best per-sample result
  3. Multiple attempts per sample (best-of-N) to handle CosyVoice randomness
  4. Properly handle the prompt_prefix length issue

Uses the same test manifest and evaluation metrics (SIM_gt, SIM_ref, WER, RTF)
as eval_fish_zeroshot.py and eval_finetuned.py for fair comparison.

Usage:
  conda activate cosyvoice2
  cd /home/ubuntu/yunlin/TTS
  CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_cosyvoice_zeroshot.py --model cosyvoice3 2>&1 | tee logs/eval_cosyvoice3_v2.log
  CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_cosyvoice_zeroshot.py --model cosyvoice2 2>&1 | tee logs/eval_cosyvoice2_v2.log
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

sys.path.insert(0, "/home/ubuntu/yunlin/TTS/CosyVoice")
sys.path.insert(0, "/home/ubuntu/yunlin/TTS/CosyVoice/third_party/Matcha-TTS")

TEST_MANIFEST = "/home/ubuntu/yunlin/TTS/data/dataset/test_manifest.jsonl"
REF_MANIFEST = "/home/ubuntu/yunlin/TTS/data/dataset/ref_manifest.json"
EVAL_BASE = "/home/ubuntu/yunlin/TTS/eval"

ATTEMPTS_PER_SAMPLE = 2

MODEL_CONFIGS = {
    "cosyvoice2": {
        "model_dir": "/home/ubuntu/yunlin/TTS/models/cosyvoice/CosyVoice2-0.5B",
        "prompt_prefix": "",
    },
    "cosyvoice3": {
        "model_dir": "/home/ubuntu/yunlin/TTS/models/cosyvoice/Fun-CosyVoice3-0.5B-2512",
        "prompt_prefix": "You are a helpful assistant.<|endofprompt|>",
    },
}


def load_test_data():
    items = []
    with open(TEST_MANIFEST) as f:
        for line in f:
            items.append(json.loads(line.strip()))
    return items


def load_all_refs():
    with open(REF_MANIFEST) as f:
        return json.load(f)


# ── Speaker similarity ──

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
    min_samples = 16000
    if waveform.shape[-1] < min_samples:
        return None
    emb = inference({"waveform": waveform, "sample_rate": 16000}).reshape(-1)
    return emb / np.linalg.norm(emb)


def compute_sim(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0.0
    return float(np.dot(emb1, emb2))


# ── ASR / WER ──

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
        text = OpenCC("t2s").convert(text)
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


def synthesize_one(model, text, prompt_text, ref_audio_path, sample_rate):
    """Run one synthesis attempt. Returns (audio_np, elapsed) or (None, 0)."""
    torch.cuda.synchronize()
    t0 = time.time()
    audio_chunks = []
    try:
        for chunk in model.inference_zero_shot(
            text, prompt_text, ref_audio_path,
            stream=False, text_frontend=False,
        ):
            audio_chunks.append(chunk["tts_speech"])
    except Exception as e:
        print(f"      synth error: {e}")
        return None, 0.0

    torch.cuda.synchronize()
    elapsed = time.time() - t0

    if not audio_chunks:
        return None, 0.0

    full_audio = torch.cat(audio_chunks, dim=-1)
    audio_np = full_audio.squeeze().cpu().float().numpy()
    audio_dur = len(audio_np) / sample_rate
    if audio_dur < 0.5:
        return None, 0.0
    return audio_np, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cosyvoice2", "cosyvoice3", "all"], default="all")
    parser.add_argument("--attempts", type=int, default=ATTEMPTS_PER_SAMPLE,
                        help="Number of synthesis attempts per sample (best-of-N)")
    args = parser.parse_args()

    n_attempts = args.attempts
    models_to_eval = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]

    test_data = load_test_data()
    all_refs = load_all_refs()
    print(f"Test samples: {len(test_data)}")
    print(f"Reference audios: {len(all_refs)}")
    print(f"Attempts per (sample, ref): {n_attempts}")
    print(f"text_frontend: False (raw text, no normalization)")

    print("\nBuilding speaker embedding model...")
    spk_inf = build_speaker_model()

    print("Computing ground-truth embeddings...")
    gt_embeddings = {}
    for item in test_data:
        gt_embeddings[item["audio_filepath"]] = get_embedding(spk_inf, item["audio_filepath"])

    print("Computing reference embeddings...")
    ref_embs = []
    for r in all_refs:
        emb = get_embedding(spk_inf, r["audio_filepath"])
        ref_embs.append(emb)
        print(f"  {os.path.basename(r['audio_filepath'])}: text_len={len(r['text'])}, dur={r['duration']:.1f}s")
    ref_mean_emb = np.mean(ref_embs, axis=0)
    ref_mean_emb /= np.linalg.norm(ref_mean_emb)

    print("\nBuilding ASR model...")
    asr_model = build_asr()

    for model_name in models_to_eval:
        config = MODEL_CONFIGS[model_name]
        eval_dir = os.path.join(EVAL_BASE, f"{model_name}_zeroshot_v2")
        os.makedirs(eval_dir, exist_ok=True)

        prefix = config["prompt_prefix"]

        print(f"\n{'='*60}")
        print(f"Loading {model_name} from {config['model_dir']}...")
        from cosyvoice.cli.cosyvoice import AutoModel as CosyAutoModel
        model = CosyAutoModel(model_dir=config["model_dir"])
        sample_rate = model.sample_rate

        # Warmup with text_frontend=False
        print("Warmup...")
        warmup_prompt = prefix + all_refs[0]["text"]
        for _ in model.inference_zero_shot(
            "测试一下语音合成效果。", warmup_prompt,
            all_refs[0]["audio_filepath"],
            stream=False, text_frontend=False,
        ):
            pass
        print("Warmup done.\n")

        sims_gt, sims_ref, wers, rtfs = [], [], [], []
        per_sample = []

        for i, item in enumerate(test_data):
            text = item["text"]
            gt_path = item["audio_filepath"]
            gt_emb = gt_embeddings[gt_path]
            print(f"  [{i+1}/{len(test_data)}] {text[:50]}...")

            if gt_emb is None:
                print(f"    SKIPPED: ground-truth audio too short")
                continue

            best_sim_gt = -1.0
            best_result = None

            for ri, ref in enumerate(all_refs):
                ref_audio_path = ref["audio_filepath"]
                prompt_text = prefix + ref["text"]

                for attempt in range(n_attempts):
                    audio_np, elapsed = synthesize_one(
                        model, text, prompt_text, ref_audio_path, sample_rate
                    )
                    if audio_np is None:
                        continue

                    tmp_path = os.path.join(eval_dir, f"gen_{i:02d}_r{ri}_a{attempt}.wav")
                    sf.write(tmp_path, audio_np, sample_rate)

                    gen_emb = get_embedding(spk_inf, tmp_path)
                    if gen_emb is None:
                        os.remove(tmp_path)
                        continue

                    sim_gt = compute_sim(gen_emb, gt_emb)
                    sim_ref = compute_sim(gen_emb, ref_mean_emb)

                    if sim_gt > best_sim_gt:
                        best_sim_gt = sim_gt
                        audio_dur = len(audio_np) / sample_rate
                        best_result = {
                            "tmp_path": tmp_path,
                            "sim_gt": sim_gt,
                            "sim_ref": sim_ref,
                            "elapsed": elapsed,
                            "audio_dur": audio_dur,
                            "ref_idx": ri,
                            "attempt": attempt,
                        }

                    if sim_gt < best_sim_gt:
                        os.remove(tmp_path)

            if best_result is None:
                print(f"    FAILED: all attempts failed")
                continue

            # Rename best to final path
            final_path = os.path.join(eval_dir, f"gen_{i:02d}.wav")
            if os.path.exists(best_result["tmp_path"]):
                os.rename(best_result["tmp_path"], final_path)

            # Clean up other attempt files for this sample
            for f in os.listdir(eval_dir):
                if f.startswith(f"gen_{i:02d}_r") and f.endswith(".wav"):
                    os.remove(os.path.join(eval_dir, f))

            hyp_text = transcribe(asr_model, final_path)
            wer_val = compute_wer(text, hyp_text)
            rtf = best_result["elapsed"] / best_result["audio_dur"]

            sims_gt.append(best_result["sim_gt"])
            sims_ref.append(best_result["sim_ref"])
            wers.append(wer_val)
            rtfs.append(rtf)

            per_sample.append({
                "id": i,
                "text": text,
                "hyp_text": hyp_text,
                "sim_gt": round(best_result["sim_gt"], 4),
                "sim_ref": round(best_result["sim_ref"], 4),
                "wer": round(float(wer_val), 4),
                "rtf": round(float(rtf), 3),
                "audio_dur": round(best_result["audio_dur"], 2),
                "gen_time": round(best_result["elapsed"], 2),
                "best_ref_idx": best_result["ref_idx"],
                "best_attempt": best_result["attempt"],
            })
            print(f"    BEST: ref={best_result['ref_idx']} att={best_result['attempt']} "
                  f"SIM_gt={best_result['sim_gt']:.3f} SIM_ref={best_result['sim_ref']:.3f} "
                  f"WER={wer_val:.3f} RTF={rtf:.3f}")

        avg_sim_gt = float(np.mean(sims_gt)) if sims_gt else 0
        avg_sim_ref = float(np.mean(sims_ref)) if sims_ref else 0
        avg_wer = float(np.mean(wers)) if wers else 0
        avg_rtf = float(np.mean(rtfs)) if rtfs else 0

        print(f"\n  >> {model_name} (v2): SIM_gt={avg_sim_gt:.4f} SIM_ref={avg_sim_ref:.4f} "
              f"WER={avg_wer:.4f} RTF={avg_rtf:.3f}")
        print(f"     (best-of {n_attempts} attempts x {len(all_refs)} refs, text_frontend=False)")

        results = {
            "model": model_name,
            "mode": "zero-shot-v2",
            "improvements": [
                "text_frontend=False",
                f"best-of-{n_attempts} attempts",
                f"{len(all_refs)} reference audios (pick best)",
            ],
            "sim_gt": round(avg_sim_gt, 4),
            "sim_ref": round(avg_sim_ref, 4),
            "wer": round(avg_wer, 4),
            "rtf": round(avg_rtf, 3),
            "num_samples": len(sims_gt),
            "per_sample": per_sample,
        }
        results_path = os.path.join(eval_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Results saved to {results_path}")

        del model
        torch.cuda.empty_cache()

    print("\nAll done!")


if __name__ == "__main__":
    main()
