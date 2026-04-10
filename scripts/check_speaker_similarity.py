"""Compare speaker embeddings across different audio sources to verify voice consistency."""
import os
import torch
import torchaudio
import numpy as np
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from pyannote.audio import Model, Inference

SEGMENTS_DIR = "/home/ubuntu/yunlin/TTS/data/segments"

SOURCES_TO_CHECK = {
    "qinche_01": "qinche_01",
    "qinche_02": "qinche_02",
    "pure_p01": "qinche_pure_p01",
    "pure_p02": "qinche_pure_p02",
    "pure_p05": "qinche_pure_p05",
    "pure_p07": "qinche_pure_p07",
    "pure_p11": "qinche_pure_p11",
}


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

    print("Computing speaker embeddings...")
    source_embeddings = {}
    for label, dirname in SOURCES_TO_CHECK.items():
        seg_dir = os.path.join(SEGMENTS_DIR, dirname)
        if not os.path.isdir(seg_dir):
            print(f"  {label}: directory not found, skip")
            continue
        wavs = sorted([f for f in os.listdir(seg_dir) if f.endswith(".wav")])
        indices = np.linspace(0, len(wavs) - 1, min(5, len(wavs)), dtype=int)
        embs = []
        for idx in indices:
            path = os.path.join(seg_dir, wavs[idx])
            try:
                embs.append(get_embedding(path))
            except Exception as e:
                print(f"  Skip {os.path.basename(path)}: {e}")
        if embs:
            mean_emb = np.mean(embs, axis=0)
            source_embeddings[label] = mean_emb / np.linalg.norm(mean_emb)
            print(f"  {label}: {len(embs)} samples")

    names = list(source_embeddings.keys())
    print(f"\n{'':14s}", end="")
    for n in names:
        print(f"{n:>14s}", end="")
    print()

    for n1 in names:
        print(f"{n1:14s}", end="")
        for n2 in names:
            sim = float(np.dot(source_embeddings[n1], source_embeddings[n2]))
            print(f"{sim:14.3f}", end="")
        print()

    print("\n> 0.75 = likely same speaker | 0.5-0.75 = uncertain | < 0.5 = different")


if __name__ == "__main__":
    main()
