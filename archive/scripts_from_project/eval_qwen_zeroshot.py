"""Zero-shot voice clone evaluation for Qwen3-TTS Base models.
Uses the same test set and metrics (SIM_gt, SIM_ref, WER, RTF) as eval_finetuned.py."""
import os, json, time, glob, torch, numpy as np, soundfile as sf
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

MODELS_DIR = "/home/ubuntu/yunlin/TTS/models/qwen3-tts"
EVAL_DIR = "/home/ubuntu/yunlin/TTS/eval"
TEST_MANIFEST = "/home/ubuntu/yunlin/TTS/data/dataset/test_manifest.jsonl"
REF_AUDIOS = sorted(glob.glob("/home/ubuntu/yunlin/TTS/data/ref_audio/ref_*.wav"))
REF_AUDIO = REF_AUDIOS[0]

VARIANTS = [
    ("Qwen3-TTS-12Hz-1.7B-Base", "qwen_1.7b_zs"),
]


def load_test_data():
    items = []
    with open(TEST_MANIFEST) as f:
        for line in f:
            items.append(json.loads(line.strip()))
    return items


def build_speaker_model():
    from pyannote.audio import Model, Inference
    model = Model.from_pretrained(
        "pyannote/embedding",
        use_auth_token=os.environ.get("HF_TOKEN", ""),
    )
    return Inference(model, window="whole")


def get_embedding(inference, wav_path):
    emb = inference(wav_path).reshape(-1)
    return emb / np.linalg.norm(emb)


def build_asr():
    import whisperx
    model = whisperx.load_model("large-v3", "cuda", compute_type="float16", language="zh")
    return model


def transcribe(asr_model, wav_path):
    import whisperx
    audio = whisperx.load_audio(wav_path)
    result = asr_model.transcribe(audio, batch_size=4, language="zh")
    return "".join(seg["text"] for seg in result["segments"]).strip()


def compute_wer(ref_text, hyp_text):
    from jiwer import wer
    import re
    ref_chars = " ".join(re.sub(r"\s+", "", ref_text))
    hyp_chars = " ".join(re.sub(r"\s+", "", hyp_text))
    if not ref_chars.strip():
        return 0.0 if not hyp_chars.strip() else 1.0
    return wer(ref_chars, hyp_chars)


def get_ref_text():
    """Get the transcription for ref_00.wav from ref_manifest.json."""
    ref_manifest = os.path.join(os.path.dirname(TEST_MANIFEST), "ref_manifest.json")
    if os.path.exists(ref_manifest):
        with open(ref_manifest) as f:
            refs = json.load(f)
        if refs:
            return refs[0].get("text", "")
    return ""


def eval_model(model_name, out_subdir, test_data, spk_inf, gt_embeddings, ref_mean_emb, asr_model):
    from qwen_tts import Qwen3TTSModel

    model_path = os.path.join(MODELS_DIR, model_name)
    out_dir = os.path.join(EVAL_DIR, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Loading {model_name}...")
    try:
        model = Qwen3TTSModel.from_pretrained(
            model_path, device_map="cuda:0", dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    except ImportError:
        model = Qwen3TTSModel.from_pretrained(
            model_path, device_map="cuda:0", dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    ref_text = get_ref_text()
    print(f"  Creating voice clone prompt (ref={REF_AUDIO})...")
    prompt = model.create_voice_clone_prompt(
        ref_audio=REF_AUDIO,
        ref_text=ref_text if ref_text else None,
    )

    print("  Warming up...")
    with torch.no_grad():
        _ = model.generate_voice_clone(
            text="测试。",
            language="Chinese",
            voice_clone_prompt=prompt,
            max_new_tokens=32,
        )
    del _
    torch.cuda.synchronize()
    print("  Warmup done.")

    sims_gt, sims_ref, wers, rtfs = [], [], [], []

    for i, item in enumerate(test_data):
        text = item["text"]
        gt_path = item["audio_filepath"]

        print(f"  [{i+1}/{len(test_data)}] {text[:40]}...")

        torch.cuda.synchronize()
        t0 = time.time()
        wavs, sr = model.generate_voice_clone(
            text=text,
            language="Chinese",
            voice_clone_prompt=prompt,
        )
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        audio = wavs[0]
        audio_dur = len(audio) / sr
        rtf = elapsed / audio_dur if audio_dur > 0 else 999

        out_path = os.path.join(out_dir, f"gen_{i:02d}.wav")
        sf.write(out_path, audio, sr)

        gen_emb = get_embedding(spk_inf, out_path)
        sim_gt = float(np.dot(gen_emb, gt_embeddings[i]))
        sims_gt.append(sim_gt)

        sim_ref = float(np.dot(gen_emb, ref_mean_emb))
        sims_ref.append(sim_ref)

        hyp = transcribe(asr_model, out_path)
        w = compute_wer(text, hyp)
        wers.append(w)
        rtfs.append(rtf)

        print(f"    SIM_gt={sim_gt:.3f} SIM_ref={sim_ref:.3f} WER={w:.3f} RTF={rtf:.3f}")

    summary = {
        "model": model_name,
        "num_samples": len(test_data),
        "avg_sim_gt": round(float(np.mean(sims_gt)), 4),
        "avg_sim_ref": round(float(np.mean(sims_ref)), 4),
        "avg_wer": round(float(np.mean(wers)), 4),
        "avg_rtf": round(float(np.mean(rtfs)), 3),
        "per_sample": [
            {
                "id": i, "text": test_data[i]["text"],
                "sim_gt": round(float(sims_gt[i]), 4),
                "sim_ref": round(float(sims_ref[i]), 4),
                "wer": round(float(wers[i]), 4),
                "rtf": round(float(rtfs[i]), 3),
            }
            for i in range(len(test_data))
        ],
    }

    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n  >> {model_name}: SIM_gt={summary['avg_sim_gt']:.4f} SIM_ref={summary['avg_sim_ref']:.4f} WER={summary['avg_wer']:.4f} RTF={summary['avg_rtf']:.3f}")

    del model
    torch.cuda.empty_cache()
    return summary


if __name__ == "__main__":
    test_data = load_test_data()
    print(f"Test samples: {len(test_data)}")

    print("Building speaker embedding model...")
    spk_inf = build_speaker_model()

    print("Computing ground truth embeddings...")
    gt_embeddings = [get_embedding(spk_inf, item["audio_filepath"]) for item in test_data]

    print(f"Computing reference mean embedding from {len(REF_AUDIOS)} ref audios...")
    ref_embs = [get_embedding(spk_inf, p) for p in REF_AUDIOS]
    ref_mean_emb = np.mean(ref_embs, axis=0)
    ref_mean_emb = ref_mean_emb / np.linalg.norm(ref_mean_emb)

    print("Building ASR model...")
    asr_model = build_asr()

    all_results = {}
    for model_name, out_subdir in VARIANTS:
        try:
            s = eval_model(model_name, out_subdir, test_data, spk_inf, gt_embeddings, ref_mean_emb, asr_model)
            all_results[out_subdir] = s
        except Exception as e:
            print(f"ERROR with {model_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("ZERO-SHOT EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"{'Model':30s} {'SIM_gt':>8s} {'SIM_ref':>8s} {'WER':>8s} {'RTF':>8s}")
    print("-" * 66)
    for name, s in all_results.items():
        print(f"{s['model']:30s} {s['avg_sim_gt']:8.4f} {s['avg_sim_ref']:8.4f} {s['avg_wer']:8.4f} {s['avg_rtf']:8.3f}")

    comp_path = os.path.join(EVAL_DIR, "zeroshot_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {comp_path}")
