# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Chinese TTS voice cloning project** targeting a specific speaker (秦彻/qinche). The goal is to clone the speaker's voice using SFT fine-tuning of Qwen3-TTS-12Hz-1.7B-Base, with evaluation against Fish Audio S2 Pro as a zero-shot baseline.

**Current best result:** Qwen3 SFT v5 epoch 3, FasterQwen3TTS inference (SIM_gt=0.6892, WER=0.0425, RTF=0.357)

## Environments

All scripts require activating the appropriate conda environment:

```bash
conda activate qwen3-tts   # For Qwen3-TTS training, inference, and evaluation
conda activate fish-speech  # For Fish Audio S2 Pro inference only
```

## Common Commands

### Data Preprocessing Pipeline

```bash
# Step 1: Download audio from Bilibili
conda activate qwen3-tts
python scripts/download_bilibili.py

# Step 2: Vocal separation (run once per raw audio file)
CUDA_VISIBLE_DEVICES=7 python -m demucs --two-stems vocals \
    -o data/demucs_output data/raw/*.wav

# Step 3-6: Diarization + VAD + ASR + dataset construction
CUDA_VISIBLE_DEVICES=7 python scripts/preprocess_audio.py

# Optional: Verify speaker consistency across sources
python scripts/check_speaker_similarity.py

# Optional: Speaker filtering + train/test split (if re-purifying)
python scripts/purify_dataset.py
```

### Fine-tuning

The finetune scripts run three sequential steps: (1) RMS audio normalization, (2) audio code extraction via `prepare_data.py`, (3) SFT training via `sft_12hz.py`.

```bash
# Run v5 fine-tune (current best configuration)
cd /home/ubuntu/yunlin/TTS
bash scripts/run_finetune_v5.sh 2>&1 | tee logs/finetune_v5.log
```

Key training parameters (v5):
- LR: `2e-6`, effective batch size: 32 (2×16 grad accum), 10 epochs
- Data: `Qwen3-TTS/finetuning/train_with_codes.jsonl` (664 samples, ~28 min)
- Output: `output/qinche_sft_v5/`

### Evaluation

```bash
# Native inference evaluation (slower, RTF ~2.5)
CUDA_VISIBLE_DEVICES=7 python scripts/eval_finetuned.py 2>&1 | tee logs/eval_native.log

# FasterQwen3TTS CUDA Graph evaluation (recommended, RTF ~0.35)
CUDA_VISIBLE_DEVICES=0 CKPT_DIR=output/qinche_sft_v5 EVAL_TAG=ft_v5_fast \
  python scripts/eval_fast.py 2>&1 | tee logs/eval_v5_fast.log

# Inference speed benchmark
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_inference.py \
  --ckpt output/qinche_sft_v5/checkpoint-epoch-3 --num-samples 5

# Qwen3 zero-shot (baseline)
CUDA_VISIBLE_DEVICES=7 python scripts/eval_qwen_zeroshot.py

# Fish S2 Pro zero-shot (with compile, RTF ~0.64)
conda activate fish-speech
CUDA_VISIBLE_DEVICES=7 python scripts/eval_fish_zeroshot.py --compile 2>&1 | tee logs/eval_fish_zeroshot_compile.log

# WER + SIM combined evaluation on existing generated audio
python scripts/eval_sim_wer.py
```

Eval scripts are controlled via environment variables:
- `CKPT_DIR`: parent directory containing `checkpoint-epoch-*` subdirs
- `EVAL_TAG`: prefix for output dirs/files in `eval/`
- `EVAL_DIR`: base directory for eval output (default: `/home/ubuntu/yunlin/TTS/eval`)
- `CHECKPOINTS`: comma-separated checkpoint names (default varies per script)

## Architecture

### Model Architecture: Qwen3-TTS

Qwen3-TTS uses a **dual-component LM** ("talker + predictor"):
- **Talker**: 28-layer Transformer decoder that generates semantic tokens
- **Code predictor**: predicts residual VQ codes for the codec
- **Codec**: 16-codebook RVQ at 12Hz frame rate
- **Speaker encoder**: extracts a speaker embedding injected into codec embedding at the `spk_id` position

SFT fine-tuning trains all parameters (not LoRA). The speaker embedding is computed by averaging encoder outputs over all training samples. **Critical**: the averaging must be done in fp32 (not bf16), then cast back — bf16 cumulative sum of high-norm vectors (~17) causes catastrophic precision loss. This was the root cause of SIM_gt=0.04 in v1/v2/v3.

### Fine-tuning Code: `Qwen3-TTS/finetuning/`

| File | Purpose |
|------|---------|
| `sft_12hz.py` | Main training script; handles speaker embedding accumulation in fp32, checkpoint saving, LR scheduling |
| `dataset.py` | Dataset class and collate_fn; reads `train_with_codes.jsonl` |
| `prepare_data.py` | Pre-tokenizes audio → `audio_codes` using the 12Hz tokenizer; output fed to `sft_12hz.py` |
| `train_raw.jsonl` | RMS-normalized audio paths + transcripts |
| `train_with_codes.jsonl` | Like `train_raw.jsonl` but with pre-extracted `audio_codes` arrays |

### Inference: FasterQwen3TTS (Recommended)

`faster-qwen3-tts` wraps Qwen3-TTS with CUDA Graph capture for both the talker and code predictor, achieving RTF=0.35 (7.1x over native). Load fine-tuned checkpoints with `from_pretrained("output/qinche_sft_v5/checkpoint-epoch-3")`. Streaming mode: RTF=0.395, TTFA=305ms.

### Evaluation Metrics

| Metric | Meaning | Good threshold |
|--------|---------|---------------|
| `SIM_gt` | Cosine similarity (pyannote/embedding) between generated and ground-truth audio | >0.6 |
| `SIM_ref` | Cosine similarity between generated and reference audio | >0.6 |
| `WER` | Word error rate via WhisperX large-v3 + jiwer | <0.1 |
| `RTF` | Real-time factor (generation time / audio duration) | <1.0 |

### Data Pipeline

```
Raw WAV → Demucs (vocal separation) → Pyannote (speaker diarization)
    → Silero-VAD (1–15s segments) → WhisperX large-v3 (ASR)
    → OpenCC (traditional→simplified) → purify_dataset.py (SIM filtering + split)
    → normalize_audio.py (RMS normalize + 1s trailing silence)
    → prepare_data.py (extract audio_codes) → sft_12hz.py (train)
```

VAD parameters: min speech 500ms, min silence 800ms, threshold 0.5, output 24kHz. Speaker filter: keep segments where target speaker overlap > all others combined. Pure-voice sources (`qinche_pure_*`) skip diarization and speaker filtering.

## Key File Paths

All scripts hardcode paths under `/home/ubuntu/yunlin/TTS/` (the original training machine). When running in a different environment, update paths or set env vars accordingly.

| Path | Contents |
|------|---------|
| `models/qwen3-tts/` | All Qwen3-TTS model weights (Base, CustomVoice, VoiceDesign, Tokenizer) |
| `models/fish-speech-s2-pro/` | Fish S2 Pro weights |
| `output/qinche_sft_v5/` | **Current best** fine-tuned checkpoints (epoch 0-9) |
| `data/dataset/train_manifest.jsonl` | 664-sample training manifest |
| `data/dataset/test_manifest.jsonl` | 18-sample test manifest |
| `data/ref_audio/ref_00~04.wav` | 5 reference audio files for zero-shot/SIM_ref |
| `.env` | `HF_TOKEN` for pyannote model access |
| `eval/*_comparison.json` | Aggregated evaluation metrics per experiment |

## Additional Scripts

| Script | Purpose |
|--------|---------|
| `scripts/fix_speaker_embedding.py` | Post-hoc patch for v1/v2/v3 checkpoints: re-computes speaker embedding in fp32 without retraining |
| `scripts/process_new_data.py` / `scripts/download_and_process_new.py` | Ingest new Bilibili sources into existing dataset (runs download → demucs → preprocess for new BV numbers) |
| `scripts/normalize_audio.py` | Standalone RMS normalization + 1s trailing silence; called by `run_finetune_v5.sh` step 1 |
| `upload_r2.py` | Uploads eval audio samples to Cloudflare R2 bucket (`tencent-tts`) for sharing/reporting |

Detailed preprocessing pipeline documentation: `docs/audio_preprocessing_pipeline.md`

## Known Non-Viable Optimizations

- **INT8 quantization (bitsandbytes)**: incompatible with Qwen3-TTS dual-LM architecture → SIM_gt=0.07, WER=1.0
- **Speculative decoding (0.6B draft)**: acceptance rate 0-5%, slower than baseline
- **Self-speculative (early exit)**: 28-layer talker has <7% inter-layer agreement, no redundant layers
- **Fish S2 Pro fine-tuning**: RL-trained model; official warning against fine-tuning; requires 10h+ data
