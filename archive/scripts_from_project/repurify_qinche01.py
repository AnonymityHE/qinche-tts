"""Re-process qinche_01: full diarization on raw vocals -> identify 秦彻 speaker ->
extract only his segments -> re-do VAD + transcription."""
import os
import json
import torch
import torchaudio
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

VOCALS_PATH = "/home/ubuntu/yunlin/TTS/data/demucs_output/htdemucs/qinche_01/vocals.wav"
SEGMENTS_DIR = "/home/ubuntu/yunlin/TTS/data/segments"
OUT_DIR = os.path.join(SEGMENTS_DIR, "qinche_01_purified")


def build_reference():
    """Build a robust reference embedding from pure sources (long clips only)."""
    from pyannote.audio import Model, Inference
    hf_token = os.environ.get("HF_TOKEN", "")
    model = Model.from_pretrained("pyannote/embedding", token=hf_token)
    inference = Inference(model, window="whole", device=torch.device("cuda"))

    def get_emb(waveform, sr=16000):
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        e = inference({"waveform": waveform, "sample_rate": sr})
        return e / np.linalg.norm(e)

    ref_embs = []
    for d in sorted(os.listdir(SEGMENTS_DIR)):
        if d.startswith("qinche_pure_"):
            sd = os.path.join(SEGMENTS_DIR, d)
            wavs = sorted([f for f in os.listdir(sd) if f.endswith(".wav")])
            for wn in wavs:
                info = torchaudio.info(os.path.join(sd, wn))
                dur = info.num_frames / info.sample_rate
                if dur >= 3.0:
                    try:
                        w, sr = torchaudio.load(os.path.join(sd, wn))
                        if sr != 16000:
                            w = torchaudio.transforms.Resample(sr, 16000)(w)
                        ref_embs.append(get_emb(w))
                    except:
                        pass
    ref_mean = np.mean(ref_embs, axis=0)
    ref_mean /= np.linalg.norm(ref_mean)
    print(f"Reference: {len(ref_embs)} long pure clips")
    return ref_mean, inference


def main():
    hf_token = os.environ.get("HF_TOKEN", "")

    # Step 1: Full diarization
    print("=== Step 1: Full diarization on qinche_01 ===")
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    audio = whisperx.load_audio(VOCALS_PATH)
    pipeline = DiarizationPipeline(token=hf_token, device="cuda")
    diarize_df = pipeline(audio)

    speakers = {}
    for _, row in diarize_df.iterrows():
        spk = row["speaker"]
        dur = row["end"] - row["start"]
        if spk not in speakers:
            speakers[spk] = {"duration": 0, "count": 0, "segments": []}
        speakers[spk]["duration"] += dur
        speakers[spk]["count"] += 1
        speakers[spk]["segments"].append((row["start"], row["end"]))

    print(f"Found {len(speakers)} speakers:")
    for spk in sorted(speakers, key=lambda x: speakers[x]["duration"], reverse=True):
        s = speakers[spk]
        print(f"  {spk}: {s['count']} segs, {s['duration']:.1f}s ({s['duration']/60:.1f}min)")

    # Step 2: Identify 秦彻 by speaker embedding comparison
    print("\n=== Step 2: Match speakers to 秦彻 reference ===")
    ref_mean, inference = build_reference()

    def get_emb(waveform, sr=16000):
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        e = inference({"waveform": waveform, "sample_rate": sr})
        return e / np.linalg.norm(e)

    # Load full audio at 16kHz
    full_wav, orig_sr = torchaudio.load(VOCALS_PATH)
    if full_wav.shape[0] > 1:
        full_wav = full_wav.mean(0, keepdim=True)
    if orig_sr != 16000:
        full_wav = torchaudio.transforms.Resample(orig_sr, 16000)(full_wav)
    sr = 16000

    # Also load at original SR for saving segments
    full_wav_orig, orig_sr = torchaudio.load(VOCALS_PATH)
    if full_wav_orig.shape[0] > 1:
        full_wav_orig = full_wav_orig.mean(0, keepdim=True)
    # Resample to 24kHz for consistency with other segments
    if orig_sr != 24000:
        full_wav_24k = torchaudio.transforms.Resample(orig_sr, 24000)(full_wav_orig)
    else:
        full_wav_24k = full_wav_orig

    speaker_sims = {}
    for spk in speakers:
        segs = speakers[spk]["segments"]
        # Use longer segments for more stable embeddings
        long_segs = [(s, e) for s, e in segs if e - s >= 2.0]
        if len(long_segs) < 3:
            long_segs = sorted(segs, key=lambda x: x[1] - x[0], reverse=True)[:5]

        sample_indices = np.linspace(0, len(long_segs) - 1, min(10, len(long_segs)), dtype=int)
        sims = []
        for idx in sample_indices:
            start, end = long_segs[idx]
            s_frame = int(start * sr)
            e_frame = int(end * sr)
            chunk = full_wav[:, s_frame:e_frame]
            if chunk.shape[1] < sr * 0.5:
                continue
            try:
                emb = get_emb(chunk)
                sims.append(float(np.dot(emb, ref_mean)))
            except:
                pass

        avg_sim = np.mean(sims) if sims else 0
        speaker_sims[spk] = avg_sim
        is_qc = " *** 秦彻 ***" if avg_sim >= 0.50 else ""
        print(f"  {spk}: avg_SIM={avg_sim:.3f} ({len(sims)} samples, {speakers[spk]['duration']:.1f}s){is_qc}")

    # Find best matching speaker
    best_spk = max(speaker_sims, key=speaker_sims.get)
    best_sim = speaker_sims[best_spk]
    print(f"\nBest match: {best_spk} (avg_SIM={best_sim:.3f})")

    if best_sim < 0.40:
        print("Warning: low similarity, might not be 秦彻")

    # Step 3: Extract 秦彻's segments + VAD + transcription
    print(f"\n=== Step 3: Extract {best_spk}'s audio and re-segment ===")
    qc_segs = speakers[best_spk]["segments"]

    # Merge close segments (< 0.5s gap) for longer context
    merged = []
    for start, end in sorted(qc_segs):
        if merged and start - merged[-1][1] < 0.5:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    print(f"  Merged {len(qc_segs)} diarization segs -> {len(merged)} continuous blocks")

    # Run VAD on each block and save segments
    vad_model, vad_utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
    get_speech_timestamps = vad_utils[0]

    os.makedirs(OUT_DIR, exist_ok=True)
    seg_idx = 0
    all_segments = []

    for block_start, block_end in merged:
        s24 = int(block_start * 24000)
        e24 = int(block_end * 24000)
        block_audio = full_wav_24k[:, s24:e24].squeeze(0)

        if block_audio.shape[0] < 24000 * 0.5:
            continue

        # Resample to 16kHz for VAD
        block_16k = torchaudio.transforms.Resample(24000, 16000)(block_audio.unsqueeze(0)).squeeze(0)

        try:
            stamps = get_speech_timestamps(
                block_16k, vad_model,
                sampling_rate=16000,
                min_silence_duration_ms=500,
                speech_pad_ms=100,
                min_speech_duration_ms=500,
            )
        except:
            stamps = []

        for st in stamps:
            vad_start = st["start"] / 16000
            vad_end = st["end"] / 16000
            dur = vad_end - vad_start
            if dur < 0.8 or dur > 15.0:
                continue

            # Convert back to 24kHz frames
            s = int(vad_start * 24000)
            e = int(vad_end * 24000)
            seg_audio = block_audio[s:e].numpy()

            out_name = f"qinche_01_purified_{seg_idx:04d}.wav"
            out_path = os.path.join(OUT_DIR, out_name)
            sf.write(out_path, seg_audio, 24000)

            all_segments.append({
                "file": out_name,
                "filepath": out_path,
                "duration": dur,
                "global_start": block_start + vad_start,
                "global_end": block_start + vad_end,
            })
            seg_idx += 1

    total_dur = sum(s["duration"] for s in all_segments)
    print(f"  Extracted {len(all_segments)} segments, total {total_dur:.1f}s ({total_dur/60:.1f}min)")

    # Step 4: Verify with embedding
    print(f"\n=== Step 4: Verify purified segments ===")
    verified = []
    for seg in all_segments:
        try:
            w, wsr = torchaudio.load(seg["filepath"])
            if w.shape[0] > 1:
                w = w.mean(0, keepdim=True)
            if wsr != 16000:
                w = torchaudio.transforms.Resample(wsr, 16000)(w)
            emb = get_emb(w)
            sim = float(np.dot(emb, ref_mean))
        except:
            sim = 0
        seg["sim"] = sim
        if sim >= 0.40:
            verified.append(seg)

    ver_dur = sum(s["duration"] for s in verified)
    print(f"  Verified (SIM>=0.40): {len(verified)}/{len(all_segments)} segs, {ver_dur:.1f}s ({ver_dur/60:.1f}min)")

    # Remove unverified segments
    verified_files = set(s["file"] for s in verified)
    for seg in all_segments:
        if seg["file"] not in verified_files:
            try:
                os.remove(seg["filepath"])
            except:
                pass

    # Step 5: Transcribe verified segments
    print(f"\n=== Step 5: Transcribe {len(verified)} segments ===")
    import whisperx

    asr_model = whisperx.load_model("large-v3", device="cuda", compute_type="float16")

    for i, seg in enumerate(verified):
        try:
            audio_np = whisperx.load_audio(seg["filepath"])
            result = asr_model.transcribe(audio_np, batch_size=8)
            text = "".join([s.get("text", "") for s in result.get("segments", [])])
            seg["text"] = text.strip()
        except Exception as e:
            seg["text"] = ""
            print(f"  Transcription failed for {seg['file']}: {e}")

        if (i + 1) % 10 == 0:
            print(f"  Transcribed {i+1}/{len(verified)}...")

    with_text = [s for s in verified if s.get("text")]
    print(f"  Transcribed: {len(with_text)}/{len(verified)} segments with text")

    # Save results
    results_path = os.path.join(OUT_DIR, "manifest.jsonl")
    with open(results_path, "w") as f:
        for seg in with_text:
            f.write(json.dumps({
                "audio_filepath": seg["filepath"],
                "text": seg["text"],
                "duration": round(seg["duration"], 3),
                "sim": round(seg["sim"], 3),
            }, ensure_ascii=False) + "\n")

    final_dur = sum(s["duration"] for s in with_text)
    print(f"\n=== Result ===")
    print(f"  {len(with_text)} usable segments, {final_dur:.1f}s ({final_dur/60:.1f}min)")
    print(f"  Manifest: {results_path}")
    print("Done!")


if __name__ == "__main__":
    main()
