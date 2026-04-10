"""
Audio preprocessing pipeline V2 for TTS fine-tuning:
1. Load demucs-separated vocals
2. Speaker diarization (pyannote via whisperx) — skip for pure single-speaker files
3. Filter to dominant speaker (= target voice)
4. VAD segmentation (silero-vad)
5. ASR transcription (whisperx)
6. Data cleaning, ref audio selection, train/test split

Supports resume: skips VAD if segments already exist on disk.
HF_TOKEN loaded from TTS/.env via python-dotenv.
"""
import os
import glob
import json
import time
import traceback
import shutil
import torch
import torchaudio
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DEMUCS_DIR = "/home/ubuntu/yunlin/TTS/data/demucs_output/htdemucs"
SEGMENTS_DIR = "/home/ubuntu/yunlin/TTS/data/segments"
REF_AUDIO_DIR = "/home/ubuntu/yunlin/TTS/data/ref_audio"
DATASET_DIR = "/home/ubuntu/yunlin/TTS/data/dataset"
TEST_DIR = "/home/ubuntu/yunlin/TTS/data/test"

for d in [SEGMENTS_DIR, REF_AUDIO_DIR, DATASET_DIR, TEST_DIR]:
    os.makedirs(d, exist_ok=True)

TARGET_SR = 24000
MIN_DURATION = 1.0
MAX_DURATION = 15.0
BEST_REF_DURATION = (2.0, 10.0)
WHISPERX_MAX_RETRIES = 3

PURE_VOICE_PREFIXES = ["qinche_pure_"]

SOURCES = {
    "mixed": ["qinche_01", "qinche_02"],
    "pure": [f"qinche_pure_p{i:02d}" for i in range(1, 12)],
}


def is_pure_voice(name):
    return any(name.startswith(p) for p in PURE_VOICE_PREFIXES)


def load_vocal(name):
    path = os.path.join(DEMUCS_DIR, name, "vocals.wav")
    waveform, sr = torchaudio.load(path)
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).numpy(), TARGET_SR


def diarize_vocals(name, audio_np, sr):
    """Run speaker diarization and return (diarize_df, dominant_speaker_id)."""
    from whisperx.diarize import DiarizationPipeline

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print(f"  [{name}] WARNING: HF_TOKEN not set, skipping diarization")
        return None, None

    print(f"  [{name}] Running speaker diarization...")
    pipeline = DiarizationPipeline(token=hf_token, device="cuda")

    import whisperx
    audio_for_diarize = whisperx.load_audio(
        os.path.join(DEMUCS_DIR, name, "vocals.wav")
    )
    diarize_df = pipeline(audio_for_diarize)

    del pipeline
    torch.cuda.empty_cache()

    speaker_durations = {}
    for _, row in diarize_df.iterrows():
        spk = row["speaker"]
        dur = row["end"] - row["start"]
        speaker_durations[spk] = speaker_durations.get(spk, 0) + dur

    print(f"  [{name}] Speaker durations:")
    for spk, dur in sorted(speaker_durations.items(), key=lambda x: -x[1]):
        print(f"    {spk}: {dur:.1f}s ({dur/60:.1f}min)")

    dominant = max(speaker_durations, key=speaker_durations.get)
    print(f"  [{name}] Dominant speaker: {dominant} ({speaker_durations[dominant]:.1f}s)")
    return diarize_df, dominant


def filter_segments_by_speaker(segments, diarize_df, target_speaker, sr):
    """Keep only segments that overlap predominantly with the target speaker."""
    if diarize_df is None:
        return segments

    filtered = []
    for seg in segments:
        seg_start_s = seg["start_sample"] / sr
        seg_end_s = seg["end_sample"] / sr

        overlap_target = 0.0
        overlap_other = 0.0
        for _, row in diarize_df.iterrows():
            ov_start = max(seg_start_s, row["start"])
            ov_end = min(seg_end_s, row["end"])
            if ov_end > ov_start:
                if row["speaker"] == target_speaker:
                    overlap_target += ov_end - ov_start
                else:
                    overlap_other += ov_end - ov_start

        if overlap_target > overlap_other and overlap_target > 0.3:
            filtered.append(seg)

    print(f"  Speaker filter: {len(filtered)}/{len(segments)} segments kept for {target_speaker}")
    return filtered


def vad_segment(audio_np, sr):
    """Use silero-vad to find speech segments."""
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
    paths = []
    for i, seg in enumerate(segments):
        chunk = audio_np[seg["start_sample"]:seg["end_sample"]]
        filename = f"{name}_{i:04d}.wav"
        filepath = os.path.join(seg_dir, filename)
        sf.write(filepath, chunk, sr)
        paths.append(filepath)
        seg["filepath"] = filepath
        seg["filename"] = filename
    print(f"  [{name}] Saved {len(paths)} segments to {seg_dir}")
    return segments


def load_existing_segments(name):
    seg_dir = os.path.join(SEGMENTS_DIR, name)
    wav_files = sorted(glob.glob(os.path.join(seg_dir, f"{name}_*.wav")))
    if not wav_files:
        return None
    segments = []
    for filepath in wav_files:
        info = sf.info(filepath)
        if info.duration < MIN_DURATION:
            continue
        segments.append({
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "duration": info.duration,
        })
    return segments


def _t2s(text):
    """Convert Traditional Chinese to Simplified Chinese and strip extra whitespace."""
    import re
    try:
        from opencc import OpenCC
        text = OpenCC('t2s').convert(text)
    except ImportError:
        pass
    text = re.sub(r'\s+', '', text)
    return text.strip()


def transcribe_segments(segments, name):
    """Transcribe segments using whisperx with retry logic."""
    import whisperx

    device = "cuda"
    compute_type = "float16"

    model = None
    for attempt in range(1, WHISPERX_MAX_RETRIES + 1):
        try:
            print(f"  [{name}] Loading whisperx model (attempt {attempt}/{WHISPERX_MAX_RETRIES})...")
            model = whisperx.load_model("large-v3", device, compute_type=compute_type)
            break
        except Exception as e:
            print(f"  [{name}] Failed to load model: {e}")
            traceback.print_exc()
            if attempt < WHISPERX_MAX_RETRIES:
                wait = 10 * attempt
                print(f"  [{name}] Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [{name}] All {WHISPERX_MAX_RETRIES} attempts failed, skipping.")
                return []

    for i, seg in enumerate(segments):
        try:
            audio = whisperx.load_audio(seg["filepath"])
            result = model.transcribe(audio, batch_size=16, language="zh")
            text_parts = [s["text"].strip() for s in result.get("segments", [])]
            seg["text"] = _t2s("".join(text_parts))
            if (i + 1) % 50 == 0:
                print(f"  [{name}] Transcribed {i+1}/{len(segments)}...")
        except Exception as e:
            print(f"  [{name}] Error transcribing {seg.get('filename', i)}: {e}")
            seg["text"] = ""

    del model
    torch.cuda.empty_cache()

    transcribed = [s for s in segments if s.get("text") and len(s["text"]) > 1]
    print(f"  [{name}] Transcribed {len(transcribed)}/{len(segments)} segments")
    return transcribed


def select_ref_and_split(all_segments):
    all_segments.sort(key=lambda s: s["duration"], reverse=True)

    ref_candidates = [
        s for s in all_segments
        if BEST_REF_DURATION[0] <= s["duration"] <= BEST_REF_DURATION[1]
        and len(s.get("text", "")) > 5
    ]

    ref_clips = ref_candidates[:5] if len(ref_candidates) >= 5 else ref_candidates
    ref_set = set(s["filepath"] for s in ref_clips)

    for i, ref in enumerate(ref_clips):
        dst = os.path.join(REF_AUDIO_DIR, f"ref_{i:02d}.wav")
        shutil.copy2(ref["filepath"], dst)
        ref["ref_path"] = dst
        print(f"  Reference audio {i}: {dst} ({ref['duration']:.1f}s) text: {ref['text'][:60]}...")

    remaining = [s for s in all_segments if s["filepath"] not in ref_set]
    n_test = min(20, max(5, len(remaining) // 10))
    test_segments = remaining[:n_test]
    train_segments = remaining[n_test:]

    for seg in test_segments:
        dst = os.path.join(TEST_DIR, os.path.basename(seg["filepath"]))
        shutil.copy2(seg["filepath"], dst)

    print(f"  Split: {len(train_segments)} train, {len(test_segments)} test, {len(ref_clips)} ref")
    return train_segments, test_segments, ref_clips


def save_dataset(train_segments, test_segments, ref_clips):
    manifest_path = os.path.join(DATASET_DIR, "train_manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for seg in train_segments:
            f.write(json.dumps({
                "audio_filepath": seg["filepath"],
                "text": seg["text"],
                "duration": seg["duration"],
            }, ensure_ascii=False) + "\n")

    test_manifest_path = os.path.join(DATASET_DIR, "test_manifest.jsonl")
    with open(test_manifest_path, "w", encoding="utf-8") as f:
        for seg in test_segments:
            f.write(json.dumps({
                "audio_filepath": seg["filepath"],
                "text": seg["text"],
                "duration": seg["duration"],
            }, ensure_ascii=False) + "\n")

    ref_manifest_path = os.path.join(DATASET_DIR, "ref_manifest.json")
    with open(ref_manifest_path, "w", encoding="utf-8") as f:
        json.dump([{
            "audio_filepath": r.get("ref_path", r["filepath"]),
            "text": r["text"],
            "duration": r["duration"],
        } for r in ref_clips], f, ensure_ascii=False, indent=2)

    total_duration = sum(s["duration"] for s in train_segments)
    print(f"\n=== Dataset Summary ===")
    print(f"  Train: {len(train_segments)} segments, {total_duration/60:.1f} min")
    print(f"  Test:  {len(test_segments)} segments")
    print(f"  Ref:   {len(ref_clips)} clips")
    print(f"  Manifests saved to: {DATASET_DIR}")


def process_source(name):
    """Process one vocal source: diarize (if mixed) -> VAD -> transcribe."""
    print(f"\nProcessing {name}...")

    existing = load_existing_segments(name)
    if existing:
        print(f"  [RESUME] Found {len(existing)} existing segments on disk, skipping VAD.")
        segments = existing
    else:
        print(f"  Loading vocals...")
        audio_np, sr = load_vocal(name)
        print(f"  Audio: {len(audio_np)/sr:.1f}s ({len(audio_np)/sr/60:.1f}min), sr={sr}")

        if is_pure_voice(name):
            print(f"  [PURE] Single-speaker file, skipping diarization.")
            diarize_df, dominant = None, None
        else:
            diarize_df, dominant = diarize_vocals(name, audio_np, sr)

        print(f"  Running VAD...")
        segments = vad_segment(audio_np, sr)
        print(f"  Found {len(segments)} speech segments")

        if diarize_df is not None and dominant is not None:
            segments = filter_segments_by_speaker(segments, diarize_df, dominant, sr)

        segments = save_segments(audio_np, sr, segments, name)

    print(f"  Transcribing {len(segments)} segments...")
    segments = transcribe_segments(segments, name)
    return segments


def main():
    vocal_dirs = sorted([
        d for d in os.listdir(DEMUCS_DIR)
        if os.path.isdir(os.path.join(DEMUCS_DIR, d))
    ])
    print(f"Found vocal sources: {vocal_dirs}")
    print(f"Pure voice prefixes: {PURE_VOICE_PREFIXES}")

    all_segments = []
    for name in vocal_dirs:
        try:
            segments = process_source(name)
            all_segments.extend(segments)
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")
            traceback.print_exc()

    print(f"\nTotal segments across all sources: {len(all_segments)}")
    total_dur = sum(s["duration"] for s in all_segments)
    print(f"Total audio duration: {total_dur/60:.1f} min")

    print("\nSelecting reference audio and splitting dataset...")
    train_segs, test_segs, ref_clips = select_ref_and_split(all_segments)
    save_dataset(train_segs, test_segs, ref_clips)

    print("\nDone!")


if __name__ == "__main__":
    main()
