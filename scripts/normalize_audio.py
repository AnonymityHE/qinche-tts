"""Normalize all training audio segments to consistent loudness (RMS-based).

Reads train_manifest.jsonl and ref audio, normalizes all WAV files to
a target RMS level, and writes train_raw.jsonl for the finetuning pipeline.
"""
import argparse
import json
import os

import librosa
import numpy as np
import soundfile as sf


TARGET_RMS = 0.10
MIN_RMS = 1e-5
TAIL_SILENCE_SEC = 1.0


def rms_normalize(audio: np.ndarray, target: float = TARGET_RMS) -> np.ndarray:
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms < MIN_RMS:
        return audio
    gain = target / current_rms
    normalized = audio * gain
    peak = np.max(np.abs(normalized))
    if peak > 0.99:
        normalized = normalized * (0.99 / peak)
    return normalized


def append_silence(audio: np.ndarray, sr: int,
                   duration_sec: float = TAIL_SILENCE_SEC) -> np.ndarray:
    """Append silence to the end of audio. Helps prevent rushed endings and
    noise artifacts at the end of generated speech (community-verified fix)."""
    silence = np.zeros(int(sr * duration_sec), dtype=audio.dtype)
    return np.concatenate([audio, silence])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--ref_audio", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for normalized WAV files")
    parser.add_argument("--output_jsonl", type=str, required=True,
                        help="Output train_raw.jsonl path")
    parser.add_argument("--target_rms", type=float, default=TARGET_RMS)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Normalize reference audio
    ref_y, ref_sr = librosa.load(args.ref_audio, sr=None)
    ref_rms = np.sqrt(np.mean(ref_y ** 2))
    ref_norm = rms_normalize(ref_y, args.target_rms)
    ref_norm = append_silence(ref_norm, ref_sr)
    ref_out = os.path.join(args.output_dir, "ref_normalized.wav")
    sf.write(ref_out, ref_norm, ref_sr)
    print(f"Reference: {args.ref_audio}")
    print(f"  original RMS={ref_rms:.4f} → normalized RMS={np.sqrt(np.mean(ref_norm**2)):.4f}")
    print(f"  saved to {ref_out}")

    # Process training samples
    entries = []
    with open(args.train_manifest) as f:
        for line in f:
            entries.append(json.loads(line.strip()))

    stats = {"total": len(entries), "normalized": 0, "skipped_silent": 0}
    output_lines = []

    for entry in entries:
        src_path = entry["audio_filepath"]
        y, sr = librosa.load(src_path, sr=None)
        current_rms = np.sqrt(np.mean(y ** 2))

        basename = os.path.basename(src_path)
        source_dir = os.path.basename(os.path.dirname(src_path))
        out_subdir = os.path.join(args.output_dir, source_dir)
        os.makedirs(out_subdir, exist_ok=True)
        out_path = os.path.join(out_subdir, basename)

        if current_rms < MIN_RMS:
            stats["skipped_silent"] += 1
            continue

        y_norm = rms_normalize(y, args.target_rms)
        y_norm = append_silence(y_norm, sr)
        sf.write(out_path, y_norm, sr)
        stats["normalized"] += 1

        output_lines.append(json.dumps({
            "audio": out_path,
            "text": entry["text"],
            "ref_audio": ref_out,
        }, ensure_ascii=False))

    with open(args.output_jsonl, "w") as f:
        for line in output_lines:
            f.write(line + "\n")

    print(f"\nNormalization complete:")
    print(f"  Total:   {stats['total']}")
    print(f"  Kept:    {stats['normalized']}")
    print(f"  Dropped: {stats['skipped_silent']} (near-silent, RMS < {MIN_RMS})")
    print(f"  Output:  {args.output_jsonl} ({len(output_lines)} entries)")


if __name__ == "__main__":
    main()
