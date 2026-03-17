"""Deep analysis of qinche_01 segments: per-segment speaker embedding + SNR check."""
import os
import torch
import torchaudio
import numpy as np
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from pyannote.audio import Model, Inference

SEGMENTS_DIR = "/home/ubuntu/yunlin/TTS/data/segments"
DEMUCS_DIR = "/home/ubuntu/yunlin/TTS/data/demucs_output/htdemucs"


def compute_snr(vocals_path, no_vocals_path, start_s, end_s):
    """Compute SNR between vocals and no_vocals (background) for a time range."""
    voc, sr_v = torchaudio.load(vocals_path, frame_offset=int(start_s * 24000), num_frames=int((end_s - start_s) * 24000))
    bg, sr_b = torchaudio.load(no_vocals_path, frame_offset=int(start_s * 24000), num_frames=int((end_s - start_s) * 24000))
    voc_power = (voc ** 2).mean().item()
    bg_power = (bg ** 2).mean().item()
    if bg_power < 1e-10:
        return 99.0
    return 10 * np.log10(voc_power / bg_power)


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

    # Build reference embedding from pure sources
    print("Building reference embedding from pure sources...")
    ref_embs = []
    for pure_dir in ["qinche_pure_p02", "qinche_pure_p07", "qinche_pure_p11"]:
        seg_dir = os.path.join(SEGMENTS_DIR, pure_dir)
        if not os.path.isdir(seg_dir):
            continue
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

    # Check vocals/no_vocals for background noise
    vocals_path = os.path.join(DEMUCS_DIR, "qinche_01", "vocals.wav")
    no_vocals_path = os.path.join(DEMUCS_DIR, "qinche_01", "no_vocals.wav")
    has_bg = os.path.exists(no_vocals_path)

    # Analyze each segment in qinche_01
    seg_dir = os.path.join(SEGMENTS_DIR, "qinche_01")
    wavs = sorted([f for f in os.listdir(seg_dir) if f.endswith(".wav")])

    print(f"{'Segment':30s} {'Dur(s)':>7s} {'SIM':>7s} {'SNR(dB)':>8s} {'Verdict':>10s}")
    print("-" * 70)

    keep = []
    drop = []
    for wav_name in wavs:
        wav_path = os.path.join(seg_dir, wav_name)
        info = torchaudio.info(wav_path)
        dur = info.num_frames / info.sample_rate

        try:
            emb = get_embedding(wav_path)
            sim = float(np.dot(emb, ref_mean))
        except Exception as e:
            sim = -1.0

        # Estimate SNR from demucs separation
        snr_str = "N/A"
        if has_bg:
            seg_idx = int(wav_name.split("_")[-1].replace(".wav", ""))
            # rough time estimate from segment ordering and duration
            # We need actual timestamps — approximate from cumulative duration
            pass

        verdict = "KEEP" if sim >= 0.60 else "DROP"
        if verdict == "KEEP":
            keep.append((wav_name, dur, sim))
        else:
            drop.append((wav_name, dur, sim))

        print(f"{wav_name:30s} {dur:7.2f} {sim:7.3f} {snr_str:>8s} {verdict:>10s}")

    print(f"\n=== Summary ===")
    print(f"  Total: {len(wavs)} segments")
    print(f"  KEEP:  {len(keep)} segments, {sum(d for _, d, _ in keep):.1f}s")
    print(f"  DROP:  {len(drop)} segments, {sum(d for _, d, _ in drop):.1f}s")
    print(f"  Keep avg SIM: {np.mean([s for _, _, s in keep]):.3f}" if keep else "")
    print(f"  Drop avg SIM: {np.mean([s for _, _, s in drop]):.3f}" if drop else "")

    # Also quick-check qinche_02
    print(f"\n=== qinche_02 per-segment analysis (sample) ===")
    seg_dir2 = os.path.join(SEGMENTS_DIR, "qinche_02")
    wavs2 = sorted([f for f in os.listdir(seg_dir2) if f.endswith(".wav")])
    indices2 = np.linspace(0, len(wavs2) - 1, min(10, len(wavs2)), dtype=int)
    sims2 = []
    for idx in indices2:
        wav_path = os.path.join(seg_dir2, wavs2[idx])
        info = torchaudio.info(wav_path)
        dur = info.num_frames / info.sample_rate
        try:
            emb = get_embedding(wav_path)
            sim = float(np.dot(emb, ref_mean))
        except:
            sim = -1.0
        sims2.append(sim)
        print(f"  {wavs2[idx]:35s} {dur:6.2f}s  SIM={sim:.3f}")
    print(f"  qinche_02 sample avg SIM: {np.mean(sims2):.3f}")


if __name__ == "__main__":
    main()
