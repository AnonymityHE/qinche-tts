"""Fix speaker embedding in finetuned checkpoints.

The original training accumulated speaker embeddings in bf16 which caused
massive precision loss (~83% norm reduction). This script recomputes the
correct speaker embedding using the base model's speaker_encoder in fp32,
then patches all checkpoints.
"""
import os
import json
import glob
import argparse

import torch
import librosa
import numpy as np
from safetensors.torch import load_file, save_file
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram


def compute_speaker_embedding(base_model_path, ref_audio_paths, device="cuda:0"):
    """Compute mean speaker embedding from reference audios using base model's speaker_encoder."""
    from qwen_tts import Qwen3TTSModel

    print(f"Loading base model from {base_model_path}...")
    base = Qwen3TTSModel.from_pretrained(
        base_model_path,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    embeddings = []
    for ref_path in ref_audio_paths:
        audio, sr = librosa.load(ref_path, sr=None)
        assert sr == 24000, f"Expected 24kHz, got {sr}"
        mel = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024, num_mels=128, sampling_rate=24000,
            hop_size=256, win_size=1024, fmin=0, fmax=12000,
        ).transpose(1, 2).to(device).to(torch.bfloat16)

        with torch.no_grad():
            emb = base.model.speaker_encoder(mel).float()
        embeddings.append(emb)
        print(f"  {os.path.basename(ref_path)}: norm={emb.norm():.4f}")

    mean_emb = torch.cat(embeddings, dim=0).mean(dim=0, keepdim=True)
    print(f"  Mean embedding: norm={mean_emb.norm():.4f}")

    del base
    torch.cuda.empty_cache()
    return mean_emb


def patch_checkpoint(ckpt_dir, speaker_embedding, spk_id=3000):
    """Replace speaker embedding in a checkpoint's model.safetensors."""
    safetensors_path = os.path.join(ckpt_dir, "model.safetensors")
    state = load_file(safetensors_path)

    old_emb = state["talker.model.codec_embedding.weight"][spk_id]
    old_norm = old_emb.float().norm().item()

    state["talker.model.codec_embedding.weight"][spk_id] = (
        speaker_embedding[0].to(old_emb.device).to(old_emb.dtype)
    )

    new_emb = state["talker.model.codec_embedding.weight"][spk_id]
    new_norm = new_emb.float().norm().item()

    save_file(state, safetensors_path)
    print(f"  {os.path.basename(ckpt_dir)}: norm {old_norm:.4f} -> {new_norm:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--ref_audios", type=str, nargs="+", required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    spk_emb = compute_speaker_embedding(args.base_model, args.ref_audios, args.device)

    checkpoints = sorted(glob.glob(os.path.join(args.ckpt_dir, "checkpoint-epoch-*")))
    print(f"\nPatching {len(checkpoints)} checkpoints in {args.ckpt_dir}...")
    for ckpt in checkpoints:
        patch_checkpoint(ckpt, spk_emb)

    print("\nDone!")


if __name__ == "__main__":
    main()
