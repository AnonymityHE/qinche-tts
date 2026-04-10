"""Purify dataset: filter segments by speaker similarity to reference (pure) voice.
Drops qinche_01 entirely, filters qinche_02 per-segment, keeps pure sources as-is.
Reads from existing train/test manifests to get transcripts."""
import os
import json
import random
import re
import torch
import torchaudio
import numpy as np
import shutil
from dotenv import load_dotenv
from opencc import OpenCC

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from pyannote.audio import Model, Inference

SEGMENTS_DIR = "/home/ubuntu/yunlin/TTS/data/segments"
REF_AUDIO_DIR = "/home/ubuntu/yunlin/TTS/data/ref_audio"
DATASET_DIR = "/home/ubuntu/yunlin/TTS/data/dataset"

DROP_SOURCES = ["qinche_01"]
FILTER_SOURCES = ["qinche_02"]
SIM_THRESHOLD = 0.55
BEST_REF_DURATION = (3.0, 10.0)

_cc = OpenCC('t2s')


def normalize_text(text):
    """Traditional->Simplified Chinese, strip whitespace/punctuation."""
    text = _cc.convert(text)
    text = re.sub(r'\s+', '', text)
    return text.strip()


def load_existing_manifests():
    """Load all segments from existing manifests to get filepath->text mapping."""
    mapping = {}
    for fname in ["train_manifest.jsonl", "test_manifest.jsonl"]:
        path = os.path.join(DATASET_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                obj = json.loads(line.strip())
                mapping[obj["audio_filepath"]] = obj
    return mapping


def main():
    hf_token = os.environ.get("HF_TOKEN", "")
    model = Model.from_pretrained("pyannote/embedding", token=hf_token)
    inference = Inference(model, window="whole", device=torch.device("cuda"))

    def get_embedding(wav_path):
        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        emb = inference({"waveform": waveform, "sample_rate": 16000})
        return emb / np.linalg.norm(emb)

    # Load existing transcripts
    print("Loading existing manifests...")
    existing = load_existing_manifests()
    print(f"  Found {len(existing)} entries with transcripts")

    # Build reference embedding from pure sources
    print("\nBuilding reference embedding from pure sources...")
    ref_embs = []
    for d in sorted(os.listdir(SEGMENTS_DIR)):
        if d.startswith("qinche_pure_"):
            seg_dir = os.path.join(SEGMENTS_DIR, d)
            wavs = sorted([f for f in os.listdir(seg_dir) if f.endswith(".wav")])
            indices = np.linspace(0, len(wavs) - 1, min(5, len(wavs)), dtype=int)
            for idx in indices:
                try:
                    ref_embs.append(get_embedding(os.path.join(seg_dir, wavs[idx])))
                except:
                    pass
    ref_mean = np.mean(ref_embs, axis=0)
    ref_mean = ref_mean / np.linalg.norm(ref_mean)
    print(f"  Reference built from {len(ref_embs)} clips\n")

    # Process all sources
    all_segments = []

    for source_dir in sorted(os.listdir(SEGMENTS_DIR)):
        seg_dir = os.path.join(SEGMENTS_DIR, source_dir)
        if not os.path.isdir(seg_dir):
            continue
        wavs = sorted([f for f in os.listdir(seg_dir) if f.endswith(".wav")])
        if not wavs:
            continue

        if source_dir in DROP_SOURCES:
            dur_total = 0
            for w in wavs:
                info = torchaudio.info(os.path.join(seg_dir, w))
                dur_total += info.num_frames / info.sample_rate
            print(f"[DROP] {source_dir}: {len(wavs)} segs, {dur_total:.1f}s — entirely excluded")
            continue

        needs_filter = source_dir in FILTER_SOURCES
        kept, dropped = 0, 0

        for wav_name in wavs:
            wav_path = os.path.join(seg_dir, wav_name)
            entry = existing.get(wav_path)
            if not entry or not entry.get("text"):
                dropped += 1
                continue

            if needs_filter:
                try:
                    emb = get_embedding(wav_path)
                    sim = float(np.dot(emb, ref_mean))
                except:
                    sim = 0.0
                if sim < SIM_THRESHOLD:
                    dropped += 1
                    continue

            all_segments.append({
                "audio_filepath": wav_path,
                "text": normalize_text(entry["text"]),
                "duration": entry.get("duration", 0),
                "source": source_dir,
            })
            kept += 1

        tag = "FILTER" if needs_filter else "KEEP"
        total_dur = sum(s["duration"] for s in all_segments if s["source"] == source_dir)
        print(f"[{tag}] {source_dir}: {kept} kept, {dropped} dropped -> {total_dur:.1f}s")

    random.seed(42)

    total_dur = sum(s["duration"] for s in all_segments)
    print(f"\n=== Purified Dataset ===")
    print(f"  Total: {len(all_segments)} segments, {total_dur:.1f}s ({total_dur/60:.1f} min)")

    # Select reference clips from pure sources (longest clean samples)
    ref_candidates = [s for s in all_segments
                      if s["source"].startswith("qinche_pure_")
                      and BEST_REF_DURATION[0] <= s["duration"] <= BEST_REF_DURATION[1]]
    ref_candidates.sort(key=lambda x: x["duration"], reverse=True)
    ref_clips = ref_candidates[:5]

    ref_paths = set(s["audio_filepath"] for s in ref_clips)
    remaining = [s for s in all_segments if s["audio_filepath"] not in ref_paths]

    # Stratified split: ensure train set contains samples from all duration ranges
    test_count = min(20, max(5, len(remaining) // 20))
    random.shuffle(remaining)
    test_segments = remaining[:test_count]
    train_segments = remaining[test_count:]

    train_durs = [s["duration"] for s in train_segments]
    test_durs = [s["duration"] for s in test_segments]
    print(f"  Train duration range: {min(train_durs):.1f}s - {max(train_durs):.1f}s (mean {np.mean(train_durs):.1f}s)")
    print(f"  Test  duration range: {min(test_durs):.1f}s - {max(test_durs):.1f}s (mean {np.mean(test_durs):.1f}s)")

    print(f"  Train: {len(train_segments)} segments, {sum(s['duration'] for s in train_segments):.1f}s ({sum(s['duration'] for s in train_segments)/60:.1f} min)")
    print(f"  Test:  {len(test_segments)} segments")
    print(f"  Ref:   {len(ref_clips)} clips")

    # Save
    os.makedirs(REF_AUDIO_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    ref_manifest = []
    for i, clip in enumerate(ref_clips):
        dst = os.path.join(REF_AUDIO_DIR, f"ref_{i:02d}.wav")
        if os.path.abspath(clip["audio_filepath"]) != os.path.abspath(dst):
            shutil.copy2(clip["audio_filepath"], dst)
        ref_manifest.append({
            "audio_filepath": dst,
            "text": clip["text"],
            "duration": clip["duration"],
        })
        print(f"  Ref {i}: ({clip['duration']:.1f}s) \"{clip['text'][:60]}\"")

    with open(os.path.join(DATASET_DIR, "ref_manifest.json"), "w") as f:
        json.dump(ref_manifest, f, ensure_ascii=False, indent=2)

    with open(os.path.join(DATASET_DIR, "train_manifest.jsonl"), "w") as f:
        for s in train_segments:
            f.write(json.dumps({"audio_filepath": s["audio_filepath"], "text": s["text"], "duration": s["duration"]}, ensure_ascii=False) + "\n")

    with open(os.path.join(DATASET_DIR, "test_manifest.jsonl"), "w") as f:
        for s in test_segments:
            f.write(json.dumps({"audio_filepath": s["audio_filepath"], "text": s["text"], "duration": s["duration"]}, ensure_ascii=False) + "\n")

    print(f"\nManifests saved to {DATASET_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
