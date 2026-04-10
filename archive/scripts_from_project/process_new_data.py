"""Process newly downloaded audio: demucs → VAD → ASR → merge into dataset.

This is a one-shot script for the qinche_call_p01..p06 files.
It reuses the same pipeline logic as preprocess_audio.py but only for new files,
then merges results into the existing train/test manifests.
"""
import os
import json
import time
import glob
import traceback

import torch
import torchaudio
import numpy as np
import soundfile as sf
import subprocess
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

RAW_DIR = "/home/ubuntu/yunlin/TTS/data/raw"
DEMUCS_DIR = "/home/ubuntu/yunlin/TTS/data/demucs_output/htdemucs"
SEGMENTS_DIR = "/home/ubuntu/yunlin/TTS/data/segments"
DATASET_DIR = "/home/ubuntu/yunlin/TTS/data/dataset"

TARGET_SR = 24000
MIN_DURATION = 1.0
MAX_DURATION = 15.0

NEW_SOURCES = [f"qinche_call_p{i:02d}" for i in range(1, 7)]


def run_demucs(name):
    wav_path = os.path.join(RAW_DIR, f"{name}.wav")
    out_dir = os.path.join(DEMUCS_DIR, name)
    if os.path.exists(os.path.join(out_dir, "vocals.wav")):
        print(f"  [{name}] Demucs output already exists, skipping.")
        return
    print(f"  [{name}] Running demucs...")
    subprocess.run(
        ["python", "-m", "demucs", "-n", "htdemucs",
         "--two-stems", "vocals",
         "-o", os.path.dirname(DEMUCS_DIR),
         wav_path],
        check=True,
    )
    print(f"  [{name}] Demucs done.")


def load_vocal(name):
    path = os.path.join(DEMUCS_DIR, name, "vocals.wav")
    waveform, sr = torchaudio.load(path)
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).numpy(), TARGET_SR


def vad_segment(audio_np, sr):
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad",
        force_reload=False, onnx=False
    )
    get_speech_timestamps = utils[0]

    audio_tensor = torch.from_numpy(audio_np).float()
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio_16k = resampler(audio_tensor.unsqueeze(0)).squeeze(0)
    else:
        audio_16k = audio_tensor

    timestamps = get_speech_timestamps(audio_16k, model, sampling_rate=16000,
                                        min_speech_duration_ms=500,
                                        min_silence_duration_ms=800,
                                        threshold=0.5)

    scale = sr / 16000
    segments = []
    for ts in timestamps:
        start_sample = int(ts["start"] * scale)
        end_sample = int(ts["end"] * scale)
        duration = (end_sample - start_sample) / sr
        if MIN_DURATION <= duration <= MAX_DURATION:
            segments.append({
                "start_sample": start_sample,
                "end_sample": end_sample,
                "duration": duration,
            })
        elif duration > MAX_DURATION:
            chunk_samples = int(MAX_DURATION * sr)
            for cs in range(start_sample, end_sample, chunk_samples):
                ce = min(cs + chunk_samples, end_sample)
                d = (ce - cs) / sr
                if d >= MIN_DURATION:
                    segments.append({
                        "start_sample": cs,
                        "end_sample": ce,
                        "duration": d,
                    })
    return segments


def save_segments(audio_np, sr, segments, name):
    seg_dir = os.path.join(SEGMENTS_DIR, name)
    os.makedirs(seg_dir, exist_ok=True)
    for i, seg in enumerate(segments):
        chunk = audio_np[seg["start_sample"]:seg["end_sample"]]
        filename = f"{name}_{i:04d}.wav"
        filepath = os.path.join(seg_dir, filename)
        sf.write(filepath, chunk, sr)
        seg["filepath"] = filepath
        seg["filename"] = filename
    print(f"  [{name}] Saved {len(segments)} segments")
    return segments


def transcribe_segments(segments, name):
    import whisperx

    print(f"  [{name}] Loading whisperx model...")
    model = whisperx.load_model("large-v3", "cuda", compute_type="float16")

    for i, seg in enumerate(segments):
        try:
            audio = whisperx.load_audio(seg["filepath"])
            result = model.transcribe(audio, batch_size=16, language="zh")
            text_parts = [s["text"].strip() for s in result.get("segments", [])]
            seg["text"] = "".join(text_parts)
        except Exception as e:
            print(f"  [{name}] Error transcribing {seg.get('filename', i)}: {e}")
            seg["text"] = ""

    del model
    torch.cuda.empty_cache()

    transcribed = [s for s in segments if s.get("text") and len(s["text"]) > 1]
    print(f"  [{name}] Transcribed {len(transcribed)}/{len(segments)} segments")
    return transcribed


def process_source(name):
    print(f"\nProcessing {name}...")

    existing_dir = os.path.join(SEGMENTS_DIR, name)
    existing_wavs = sorted(glob.glob(os.path.join(existing_dir, f"{name}_*.wav"))) if os.path.exists(existing_dir) else []

    if existing_wavs:
        print(f"  [{name}] Found {len(existing_wavs)} existing segments, skipping demucs+VAD.")
        segments = []
        for fp in existing_wavs:
            info = sf.info(fp)
            if info.duration >= MIN_DURATION:
                segments.append({"filepath": fp, "filename": os.path.basename(fp), "duration": info.duration})
    else:
        run_demucs(name)
        audio_np, sr = load_vocal(name)
        print(f"  [{name}] Audio: {len(audio_np)/sr:.1f}s, sr={sr}")

        segments = vad_segment(audio_np, sr)
        print(f"  [{name}] Found {len(segments)} speech segments")
        segments = save_segments(audio_np, sr, segments, name)

    segments = transcribe_segments(segments, name)
    return segments


def main():
    all_new_segments = []
    for name in NEW_SOURCES:
        try:
            segs = process_source(name)
            all_new_segments.extend(segs)
        except Exception as e:
            print(f"[{name}] ERROR: {e}")
            traceback.print_exc()

    total_dur = sum(s["duration"] for s in all_new_segments)
    print(f"\nNew data: {len(all_new_segments)} segments, {total_dur:.1f}s ({total_dur/60:.1f} min)")

    # Load existing manifests
    train_path = os.path.join(DATASET_DIR, "train_manifest.jsonl")
    existing = []
    existing_paths = set()
    if os.path.exists(train_path):
        with open(train_path) as f:
            for line in f:
                obj = json.loads(line.strip())
                existing.append(obj)
                existing_paths.add(obj["audio_filepath"])

    # Append new segments (avoid duplicates)
    added = 0
    for seg in all_new_segments:
        if seg["filepath"] not in existing_paths:
            existing.append({
                "audio_filepath": seg["filepath"],
                "text": seg["text"],
                "duration": seg["duration"],
            })
            added += 1

    print(f"\nAdded {added} new entries to train manifest (total: {len(existing)})")

    with open(train_path, "w") as f:
        for entry in existing:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Updated: {train_path}")
    print("Done!")


if __name__ == "__main__":
    main()
