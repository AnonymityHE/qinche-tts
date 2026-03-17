"""Unified evaluation: Speaker Similarity (SIM) + WER for Qwen3-TTS zero-shot outputs."""
import os, json, glob, torch, numpy as np, soundfile as sf
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

EVAL_DIR = "/home/ubuntu/yunlin/TTS/eval"
REF_AUDIOS = sorted(glob.glob("/home/ubuntu/yunlin/TTS/data/ref_audio/ref_*.wav"))
MODELS = ["qwen_0.6b", "qwen_1.7b"]

def build_speaker_model():
    from pyannote.audio import Model, Inference
    model = Model.from_pretrained(
        "pyannote/embedding",
        use_auth_token=os.environ.get("HF_TOKEN", ""),
    )
    return Inference(model, window="whole")

def get_embedding(inference, wav_path):
    return inference(wav_path).reshape(-1)

def compute_sim(emb1, emb2):
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

def build_asr():
    import whisperx
    device = "cuda"
    model = whisperx.load_model("large-v3", device, compute_type="float16", language="zh")
    return model, device

def transcribe(asr_model, device, wav_path):
    import whisperx
    audio = whisperx.load_audio(wav_path)
    result = asr_model.transcribe(audio, batch_size=4, language="zh")
    text = "".join(seg["text"] for seg in result["segments"])
    return text.strip()

def compute_wer(ref_text, hyp_text):
    from jiwer import wer
    import re
    ref_chars = " ".join(re.sub(r"\s+", "", ref_text))
    hyp_chars = " ".join(re.sub(r"\s+", "", hyp_text))
    if not ref_chars.strip():
        return 0.0 if not hyp_chars.strip() else 1.0
    return wer(ref_chars, hyp_chars)

def main():
    print("Building speaker embedding model...")
    spk_inf = build_speaker_model()

    print(f"Computing reference embeddings from {len(REF_AUDIOS)} ref audios...")
    ref_embs = [get_embedding(spk_inf, p) for p in REF_AUDIOS]
    ref_mean = np.mean(ref_embs, axis=0)
    ref_mean = ref_mean / np.linalg.norm(ref_mean)

    print("Building ASR model...")
    asr_model, asr_device = build_asr()

    all_results = {}

    for model_name in MODELS:
        model_dir = os.path.join(EVAL_DIR, model_name)
        results_path = os.path.join(model_dir, "results.json")
        if not os.path.exists(results_path):
            print(f"[SKIP] {model_name}: no results.json")
            continue

        with open(results_path) as f:
            data = json.load(f)

        print(f"\n{'='*60}")
        print(f"Evaluating {model_name} ({data['model']})")
        print(f"{'='*60}")

        sims, wers = [], []
        for r in data["results"]:
            wav_path = r["audio_path"]
            ref_text = r["text"]

            emb = get_embedding(spk_inf, wav_path)
            emb = emb / np.linalg.norm(emb)
            sim = compute_sim(ref_mean, emb)
            sims.append(sim)

            hyp_text = transcribe(asr_model, asr_device, wav_path)
            w = compute_wer(ref_text, hyp_text)
            wers.append(w)

            print(f"  [{r['id']}] SIM={sim:.3f}  WER={w:.3f}  RTF={r['rtf']:.3f}")
            print(f"       REF: {ref_text}")
            print(f"       HYP: {hyp_text}")

        avg_sim = np.mean(sims)
        avg_wer = np.mean(wers)
        avg_rtf = data["avg_rtf"]

        summary = {
            "model": data["model"],
            "avg_sim": round(float(avg_sim), 4),
            "avg_wer": round(float(avg_wer), 4),
            "avg_rtf": avg_rtf,
            "per_sentence": []
        }
        for i, r in enumerate(data["results"]):
            summary["per_sentence"].append({
                "id": r["id"],
                "text": r["text"],
                "sim": round(float(sims[i]), 4),
                "wer": round(float(wers[i]), 4),
                "rtf": r["rtf"],
            })

        out_path = os.path.join(model_dir, "eval_results.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n  >> {model_name}: SIM={avg_sim:.4f}  WER={avg_wer:.4f}  RTF={avg_rtf:.3f}")
        all_results[model_name] = summary

    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':20s} {'SIM':>8s} {'WER':>8s} {'RTF':>8s}")
    print("-" * 48)
    for name, s in all_results.items():
        print(f"{name:20s} {s['avg_sim']:8.4f} {s['avg_wer']:8.4f} {s['avg_rtf']:8.3f}")

    comparison_path = os.path.join(EVAL_DIR, "zeroshot_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nComparison saved to {comparison_path}")

if __name__ == "__main__":
    main()
