"""Download new Bilibili videos and process through the full pipeline.

Steps:
1. Download audio from Bilibili (multi-page support)
2. Demucs vocal separation
3. VAD segmentation (silero-vad)
4. ASR transcription (whisperx)
5. Deduplicate against existing training data (text + embedding similarity)
6. RMS normalization
7. Merge into existing train_manifest.jsonl

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/download_and_process_new.py
"""
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import soundfile as sf
import torch
import torchaudio
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Paths ──
BASE_DIR = "/home/ubuntu/yunlin/TTS"
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
DEMUCS_DIR = os.path.join(BASE_DIR, "data/demucs_output/htdemucs")
SEGMENTS_DIR = os.path.join(BASE_DIR, "data/segments")
NORM_DIR = os.path.join(BASE_DIR, "data/normalized")
DATASET_DIR = os.path.join(BASE_DIR, "data/dataset")
REF_AUDIO_DIR = os.path.join(BASE_DIR, "data/ref_audio")

for d in [RAW_DIR, SEGMENTS_DIR, NORM_DIR]:
    os.makedirs(d, exist_ok=True)

TARGET_SR = 24000
MIN_DURATION = 1.0
MAX_DURATION = 15.0
TARGET_RMS = 0.10
MIN_RMS = 1e-5
TAIL_SILENCE_SEC = 1.0

DEDUP_TEXT_THRESHOLD = 0.85
DEDUP_EMB_THRESHOLD = 0.92

VIDEOS = [
    {"bvid": "BV1h5m4ByEjL", "prefix": "qinche_voice", "pages": list(range(16))},
]


# ═══════════════════════════════════════
# Step 1: Download
# ═══════════════════════════════════════
async def download_all():
    from bilibili_api import video, HEADERS
    import httpx

    print("\n" + "="*60)
    print("STEP 1: Download audio from Bilibili")
    print("="*60)

    for vid in VIDEOS:
        bvid = vid["bvid"]
        prefix = vid["prefix"]
        pages = vid.get("pages", [0])

        v = video.Video(bvid=bvid)
        info = await v.get_info()
        all_pages = info.get("pages", [])
        print(f"\n[{bvid}] Title: {info.get('title', '?')}")
        print(f"  Total pages: {len(all_pages)}")

        for p in pages:
            name = f"{prefix}_p{p+1:02d}"
            wav_path = os.path.join(RAW_DIR, f"{name}.wav")
            if os.path.exists(wav_path):
                print(f"  [{name}] Already exists, skipping.")
                continue

            if p >= len(all_pages):
                print(f"  [{name}] Page {p} out of range, skipping.")
                continue

            page_title = all_pages[p]["part"]
            page_dur = all_pages[p]["duration"]
            print(f"  [{name}] Page {p}: {page_title} ({page_dur}s)")

            try:
                download_url_data = await v.get_download_url(p)
                audio_streams = download_url_data.get("dash", {}).get("audio", [])
                if not audio_streams:
                    print(f"  [{name}] No audio streams, skipping.")
                    continue

                audio_streams.sort(key=lambda x: x.get("bandwidth", 0), reverse=True)
                audio_url = audio_streams[0]["baseUrl"]

                headers = dict(HEADERS)
                headers["Referer"] = f"https://www.bilibili.com/video/{bvid}/"

                m4a_path = os.path.join(RAW_DIR, f"{name}.m4a")
                async with httpx.AsyncClient(
                    headers=headers, follow_redirects=True,
                    timeout=httpx.Timeout(30, read=180)
                ) as client:
                    async with client.stream("GET", audio_url) as resp:
                        resp.raise_for_status()
                        with open(m4a_path, "wb") as f:
                            async for chunk in resp.aiter_bytes(65536):
                                f.write(chunk)

                print(f"  [{name}] Downloaded {os.path.getsize(m4a_path)/1024/1024:.1f} MB")

                subprocess.run(
                    ["ffmpeg", "-y", "-i", m4a_path, "-ar", "24000", "-ac", "1", wav_path],
                    check=True, capture_output=True,
                )
                os.remove(m4a_path)
                dur = sf.info(wav_path).duration
                print(f"  [{name}] Converted: {dur:.1f}s")

            except Exception as e:
                print(f"  [{name}] Error: {e}")
                traceback.print_exc()

            await asyncio.sleep(2)


# ═══════════════════════════════════════
# Step 2: Demucs vocal separation
# ═══════════════════════════════════════
def run_demucs():
    print("\n" + "="*60)
    print("STEP 2: Demucs vocal separation")
    print("="*60)

    new_wavs = []
    for vid in VIDEOS:
        prefix = vid["prefix"]
        for p in vid.get("pages", [0]):
            name = f"{prefix}_p{p+1:02d}"
            wav_path = os.path.join(RAW_DIR, f"{name}.wav")
            vocals_path = os.path.join(DEMUCS_DIR, name, "vocals.wav")
            if os.path.exists(vocals_path):
                print(f"  [{name}] Vocals already exist, skipping.")
                continue
            if os.path.exists(wav_path):
                new_wavs.append((name, wav_path))

    if not new_wavs:
        print("  No new files to process.")
        return

    for name, wav_path in new_wavs:
        print(f"  [{name}] Running demucs...")
        try:
            subprocess.run(
                ["python", "-m", "demucs", "--two-stems", "vocals",
                 "-n", "htdemucs", "-o", os.path.dirname(DEMUCS_DIR),
                 wav_path],
                check=True, capture_output=True, text=True,
            )
            stem = os.path.splitext(os.path.basename(wav_path))[0]
            src_dir = os.path.join(DEMUCS_DIR, stem)
            target_dir = os.path.join(DEMUCS_DIR, name)
            if src_dir != target_dir and os.path.isdir(src_dir):
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                shutil.move(src_dir, target_dir)
            vocals_path = os.path.join(target_dir, "vocals.wav")
            if os.path.exists(vocals_path):
                dur = sf.info(vocals_path).duration
                print(f"  [{name}] Vocals extracted: {dur:.1f}s")
            else:
                print(f"  [{name}] WARNING: vocals.wav not found after demucs")
        except subprocess.CalledProcessError as e:
            print(f"  [{name}] Demucs failed: {e.stderr[:500] if e.stderr else e}")


# ═══════════════════════════════════════
# Step 3: VAD + ASR
# ═══════════════════════════════════════
def vad_segment(audio_np, sr):
    model, utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", force_reload=False, onnx=False
    )
    get_speech_timestamps = utils[0]

    audio_tensor = torch.from_numpy(audio_np).float()
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio_16k = resampler(audio_tensor.unsqueeze(0)).squeeze(0)
    else:
        audio_16k = audio_tensor

    timestamps = get_speech_timestamps(
        audio_16k, model, sampling_rate=16000,
        min_speech_duration_ms=500, min_silence_duration_ms=800, threshold=0.5,
    )

    scale = sr / 16000
    segments = []
    for ts in timestamps:
        start = int(ts["start"] * scale)
        end = int(ts["end"] * scale)
        dur = (end - start) / sr
        if MIN_DURATION <= dur <= MAX_DURATION:
            segments.append({"start": start, "end": end, "duration": dur})
        elif dur > MAX_DURATION:
            chunk = int(MAX_DURATION * sr)
            for cs in range(start, end, chunk):
                ce = min(cs + chunk, end)
                d = (ce - cs) / sr
                if d >= MIN_DURATION:
                    segments.append({"start": cs, "end": ce, "duration": d})
    return segments


def _t2s(text):
    try:
        from opencc import OpenCC
        text = OpenCC('t2s').convert(text)
    except ImportError:
        pass
    return re.sub(r'\s+', '', text).strip()


def process_vad_asr():
    print("\n" + "="*60)
    print("STEP 3: VAD segmentation + ASR transcription")
    print("="*60)

    import whisperx

    all_new_segments = []

    for vid in VIDEOS:
        prefix = vid["prefix"]
        for p in vid.get("pages", [0]):
            name = f"{prefix}_p{p+1:02d}"
            vocals_path = os.path.join(DEMUCS_DIR, name, "vocals.wav")
            seg_dir = os.path.join(SEGMENTS_DIR, name)

            if not os.path.exists(vocals_path):
                print(f"  [{name}] No vocals, skipping.")
                continue

            existing_segs = sorted(
                [f for f in os.listdir(seg_dir) if f.endswith(".wav")]
            ) if os.path.isdir(seg_dir) else []

            if existing_segs:
                print(f"  [{name}] {len(existing_segs)} segments already exist, skipping VAD.")
                seg_paths = [os.path.join(seg_dir, f) for f in existing_segs]
            else:
                print(f"  [{name}] Loading vocals...")
                waveform, sr = torchaudio.load(vocals_path)
                if sr != TARGET_SR:
                    waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                audio_np = waveform.squeeze(0).numpy()

                print(f"  [{name}] Running VAD... ({len(audio_np)/TARGET_SR:.1f}s)")
                segments = vad_segment(audio_np, TARGET_SR)
                print(f"  [{name}] Found {len(segments)} segments")

                os.makedirs(seg_dir, exist_ok=True)
                seg_paths = []
                for i, seg in enumerate(segments):
                    chunk = audio_np[seg["start"]:seg["end"]]
                    fpath = os.path.join(seg_dir, f"{name}_{i:04d}.wav")
                    sf.write(fpath, chunk, TARGET_SR)
                    seg_paths.append(fpath)

                del audio_np, waveform
                torch.cuda.empty_cache()

            all_new_segments.extend(seg_paths)

    if not all_new_segments:
        print("  No new segments to transcribe.")
        return []

    print(f"\n  Transcribing {len(all_new_segments)} segments with whisperx...")
    asr_model = whisperx.load_model("large-v3", "cuda", compute_type="float16")

    results = []
    for i, fpath in enumerate(all_new_segments):
        try:
            audio = whisperx.load_audio(fpath)
            result = asr_model.transcribe(audio, batch_size=16, language="zh")
            text = _t2s("".join(s["text"].strip() for s in result.get("segments", [])))
            if text and len(text) > 1:
                info = sf.info(fpath)
                results.append({
                    "audio_filepath": fpath,
                    "text": text,
                    "duration": info.duration,
                })
            if (i + 1) % 100 == 0:
                print(f"  Transcribed {i+1}/{len(all_new_segments)}...")
        except Exception as e:
            print(f"  Error transcribing {os.path.basename(fpath)}: {e}")

    del asr_model
    torch.cuda.empty_cache()

    print(f"  Transcribed {len(results)}/{len(all_new_segments)} segments successfully")
    return results


# ═══════════════════════════════════════
# Step 4: Deduplicate
# ═══════════════════════════════════════
def text_similarity(a, b):
    """Character-level Jaccard similarity for Chinese text."""
    a_clean = re.sub(r'[，。！？、；：\u201c\u201d\u2018\u2019《》【】（）\s.,!?;:"\'\(\)\-\u2014\u2026]', '', a)
    b_clean = re.sub(r'[，。！？、；：\u201c\u201d\u2018\u2019《》【】（）\s.,!?;:"\'\(\)\-\u2014\u2026]', '', b)
    if not a_clean or not b_clean:
        return 0.0
    set_a = set(a_clean)
    set_b = set(b_clean)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    jaccard = intersection / union if union > 0 else 0

    from difflib import SequenceMatcher
    seq_sim = SequenceMatcher(None, a_clean, b_clean).ratio()

    return max(jaccard, seq_sim)


def deduplicate(new_segments, existing_manifest_path):
    print("\n" + "="*60)
    print("STEP 4: Deduplicate against existing training data")
    print("="*60)

    existing_texts = []
    if os.path.exists(existing_manifest_path):
        with open(existing_manifest_path) as f:
            for line in f:
                obj = json.loads(line.strip())
                existing_texts.append(obj["text"])
    print(f"  Existing training samples: {len(existing_texts)}")
    print(f"  New candidate samples: {len(new_segments)}")

    kept = []
    duplicates = 0
    for seg in new_segments:
        is_dup = False
        for ex_text in existing_texts:
            sim = text_similarity(seg["text"], ex_text)
            if sim > DEDUP_TEXT_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            for prev in kept:
                sim = text_similarity(seg["text"], prev["text"])
                if sim > DEDUP_TEXT_THRESHOLD:
                    is_dup = True
                    break

        if is_dup:
            duplicates += 1
        else:
            kept.append(seg)

    print(f"  Duplicates removed: {duplicates}")
    print(f"  New unique samples: {len(kept)}")
    total_dur = sum(s["duration"] for s in kept)
    print(f"  New unique duration: {total_dur/60:.1f} min")
    return kept


# ═══════════════════════════════════════
# Step 5: Normalize + Merge
# ═══════════════════════════════════════
def rms_normalize(audio, target=TARGET_RMS):
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < MIN_RMS:
        return audio
    gain = target / rms
    normalized = audio * gain
    peak = np.max(np.abs(normalized))
    if peak > 0.99:
        normalized = normalized * (0.99 / peak)
    return normalized


def normalize_and_merge(new_segments, existing_manifest_path):
    print("\n" + "="*60)
    print("STEP 5: RMS normalize and merge into training set")
    print("="*60)

    existing_entries = []
    if os.path.exists(existing_manifest_path):
        with open(existing_manifest_path) as f:
            for line in f:
                existing_entries.append(json.loads(line.strip()))

    existing_dur = sum(e["duration"] for e in existing_entries)
    print(f"  Existing: {len(existing_entries)} samples, {existing_dur/60:.1f} min")

    new_norm_entries = []
    for seg in new_segments:
        src = seg["audio_filepath"]
        y, sr = sf.read(src)
        y_norm = rms_normalize(y, TARGET_RMS)
        silence = np.zeros(int(sr * TAIL_SILENCE_SEC), dtype=y_norm.dtype)
        y_norm = np.concatenate([y_norm, silence])

        source_dir = os.path.basename(os.path.dirname(src))
        out_subdir = os.path.join(NORM_DIR, source_dir)
        os.makedirs(out_subdir, exist_ok=True)
        out_path = os.path.join(out_subdir, os.path.basename(src))
        sf.write(out_path, y_norm, sr)

        new_norm_entries.append({
            "audio_filepath": out_path,
            "text": seg["text"],
            "duration": len(y_norm) / sr,
        })

    merged_manifest = existing_manifest_path
    backup = merged_manifest + ".bak"
    if os.path.exists(merged_manifest):
        shutil.copy2(merged_manifest, backup)
        print(f"  Backed up existing manifest to {backup}")

    with open(merged_manifest, "w", encoding="utf-8") as f:
        for entry in existing_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        for entry in new_norm_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total = len(existing_entries) + len(new_norm_entries)
    total_dur = existing_dur + sum(e["duration"] for e in new_norm_entries)
    new_dur = sum(e["duration"] for e in new_norm_entries)

    print(f"\n  New samples added: {len(new_norm_entries)} ({new_dur/60:.1f} min)")
    print(f"  Total training set: {total} samples, {total_dur/60:.1f} min")
    print(f"  Manifest: {merged_manifest}")

    return total, total_dur


# ═══════════════════════════════════════
# Main
# ═══════════════════════════════════════
def main():
    print("="*60)
    print("  New Data Pipeline: Download → Demucs → VAD → ASR → Dedup → Merge")
    print("="*60)

    existing_manifest = os.path.join(DATASET_DIR, "train_manifest.jsonl")

    print("\n--- Step 1: Download ---")
    asyncio.run(download_all())

    print("\n--- Step 2: Demucs ---")
    run_demucs()

    print("\n--- Step 3: VAD + ASR ---")
    new_segments = process_vad_asr()

    if not new_segments:
        print("\nNo new segments found. Exiting.")
        return

    print("\n--- Step 4: Deduplicate ---")
    unique_segments = deduplicate(new_segments, existing_manifest)

    if not unique_segments:
        print("\nAll segments are duplicates. Exiting.")
        return

    print("\n--- Step 5: Normalize + Merge ---")
    total, total_dur = normalize_and_merge(unique_segments, existing_manifest)

    print("\n" + "="*60)
    print(f"  DONE! Training set: {total} samples, {total_dur/60:.1f} min")
    print("="*60)


if __name__ == "__main__":
    main()
