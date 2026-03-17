#!/usr/bin/env bash
set -e

SCRIPTS_DIR="/home/ubuntu/yunlin/TTS/scripts"
FINETUNE_DIR="/home/ubuntu/yunlin/TTS/Qwen3-TTS/finetuning"

DEVICE="cuda:0"
TOKENIZER="/home/ubuntu/yunlin/TTS/models/qwen3-tts/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL="/home/ubuntu/yunlin/TTS/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-Base"
TRAIN_MANIFEST="/home/ubuntu/yunlin/TTS/data/dataset/train_manifest.jsonl"
REF_AUDIO="/home/ubuntu/yunlin/TTS/data/ref_audio/ref_00.wav"
NORM_DIR="/home/ubuntu/yunlin/TTS/data/normalized"
RAW_JSONL="${FINETUNE_DIR}/train_raw.jsonl"
TRAIN_JSONL="${FINETUNE_DIR}/train_with_codes.jsonl"
OUTPUT_DIR="/home/ubuntu/yunlin/TTS/output/qinche_sft_v2"

BATCH_SIZE=2
GRAD_ACCUM=16
LR=2e-6
EPOCHS=10
SPEAKER="qinche"
LANGUAGE="Chinese"

echo "=========================================="
echo "Step 1: Normalizing audio loudness..."
echo "=========================================="
python -u "${SCRIPTS_DIR}/normalize_audio.py" \
  --train_manifest "${TRAIN_MANIFEST}" \
  --ref_audio "${REF_AUDIO}" \
  --output_dir "${NORM_DIR}" \
  --output_jsonl "${RAW_JSONL}" \
  --target_rms 0.10

echo ""
echo "=========================================="
echo "Step 2: Extracting audio_codes..."
echo "=========================================="
cd "${FINETUNE_DIR}"
python -u prepare_data.py \
  --device ${DEVICE} \
  --tokenizer_model_path ${TOKENIZER} \
  --input_jsonl ${RAW_JSONL} \
  --output_jsonl ${TRAIN_JSONL}

echo ""
echo "=========================================="
echo "Step 3: Running SFT (${EPOCHS} epochs, lr=${LR}, bs=${BATCH_SIZE}×${GRAD_ACCUM})..."
echo "=========================================="
python -u sft_12hz.py \
  --init_model_path ${INIT_MODEL} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --batch_size ${BATCH_SIZE} \
  --grad_accum_steps ${GRAD_ACCUM} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --speaker_name ${SPEAKER} \
  --language ${LANGUAGE}

echo ""
echo "=========================================="
echo "Fine-tuning complete!"
echo "Checkpoints at: ${OUTPUT_DIR}"
echo "=========================================="
ls -la ${OUTPUT_DIR}/
