#!/usr/bin/env bash
set -e

ROOT="/mnt/afs_e/wangyifu/qinche-tts"
SCRIPTS_DIR="${ROOT}/scripts"
FINETUNE_DIR="${ROOT}/Qwen3-TTS/finetuning"

DEVICE="cuda:0"
TOKENIZER="${ROOT}/models/qwen3-tts/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL="${ROOT}/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-Base"
TRAIN_MANIFEST="${ROOT}/data/dataset/train_manifest.jsonl"
REF_AUDIO="${ROOT}/data/ref_audio/ref_00.wav"
NORM_DIR="${ROOT}/data/normalized"
RAW_JSONL="${FINETUNE_DIR}/train_raw.jsonl"
TRAIN_JSONL="${FINETUNE_DIR}/train_with_codes.jsonl"
OUTPUT_DIR="${ROOT}/output/qinche_sft_v5"

BATCH_SIZE=2
GRAD_ACCUM=16
LR=2e-6
EPOCHS=10
SPEAKER="qinche"
LANGUAGE="Chinese"

echo "=========================================="
echo "v5 Training: expanded dataset (664 samples)"
echo "=========================================="

echo ""
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
echo "Step 3: Running SFT (${EPOCHS} epochs, lr=${LR}, bs=${BATCH_SIZE}x${GRAD_ACCUM})..."
echo "=========================================="
python -u sft_12hz.py \
  --init_model_path ${INIT_MODEL} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --speaker_name ${SPEAKER}

echo ""
echo "=========================================="
echo "v5 Fine-tuning complete!"
echo "Checkpoints at: ${OUTPUT_DIR}"
echo "=========================================="
ls -la ${OUTPUT_DIR}/
