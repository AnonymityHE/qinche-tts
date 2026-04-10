#!/usr/bin/env bash
set -e

SCRIPTS_DIR="/home/ubuntu/yunlin/TTS/scripts"
FINETUNE_DIR="/home/ubuntu/yunlin/TTS/Qwen3-TTS/finetuning"

DEVICE="cuda:0"
TOKENIZER="/home/ubuntu/yunlin/TTS/models/qwen3-tts/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL="/home/ubuntu/yunlin/TTS/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-Base"

# Reuse normalized data from v2
TRAIN_JSONL="${FINETUNE_DIR}/train_with_codes.jsonl"
OUTPUT_DIR="/home/ubuntu/yunlin/TTS/output/qinche_sft_v3"

BATCH_SIZE=2
GRAD_ACCUM=16
LR=1e-5
EPOCHS=10
SPEAKER="qinche"
LANGUAGE="Chinese"

echo "=========================================="
echo "Running SFT v3 (${EPOCHS} epochs, lr=${LR}, bs=${BATCH_SIZE}×${GRAD_ACCUM})..."
echo "=========================================="
cd "${FINETUNE_DIR}"
python -u sft_12hz.py \
  --init_model_path ${INIT_MODEL} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --batch_size ${BATCH_SIZE} \
  --grad_accum_steps ${GRAD_ACCUM} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --speaker_name ${SPEAKER} \
  --language ${LANGUAGE} \
  --warmup_ratio 0.05

echo ""
echo "=========================================="
echo "Fine-tuning v3 complete!"
echo "Checkpoints at: ${OUTPUT_DIR}"
echo "=========================================="
ls -la ${OUTPUT_DIR}/
