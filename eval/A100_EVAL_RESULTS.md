# A100 Evaluation Results

## Environment

- **Machine**: 8x NVIDIA A100-SXM4-80GB
- **CUDA Driver**: 12.4
- **PyTorch**: 2.6.0+cu124
- **Model**: Qwen3-TTS SFT v5 epoch 3 (CustomVoice) + Qwen3-TTS Base (clone)
- **Date**: 2026-04-16

## Pipeline Timing (test_scene_01, 4 秦彻 sentences)

| Component | Total | Mean/sentence |
|-----------|-------|---------------|
| Init (model load) | 496s | — |
| Emotion Analysis (GPT-4o via OpenRouter) | 28.5s | 7.1s |
| RAG Retrieval (BGE-large-zh) | 5.7s | 1.4s |
| `auto` TTS generation | 9.0s* | 9.0s |
| `clone_xvec` TTS generation | 8.7s* | 8.7s |
| `baseline` TTS generation | 8.7s* | 8.7s |

\* Mean per sentence excluding first-call model loading warmup.

## Inference Speed Benchmark

| Method | Mean RTF | Mean Gen Time | Speedup |
|--------|----------|---------------|---------|
| Native SFT (CustomVoice, SDPA) | 1.329 | 8.13s | 1x |
| **CUDA Graph SFT** (FasterQwen3TTS) | **0.341** | **2.01s** | **3.9x** |
| Clone x-vec (Base model, native) | 1.342 | 9.53s | ~1x |

CUDA Graph acceleration achieves **3.9x speedup** (RTF 1.33 → 0.34), making the model real-time capable.

## Evaluation Metrics (SIM_ref + WER)

| Condition | SIM_ref | WER | Notes |
|-----------|---------|-----|-------|
| auto | 0.7020 | 0.1645 | Smart routing → clone_blend |
| clone_xvec | 0.6665 | 0.1507 | Fell back to ICL clone |
| baseline | 0.7409 | 0.1599 | SFT CustomVoice, no emotion |

### Differences from original eval environment

| Item | Original (Ubuntu training machine) | This run (A100 cluster) | Impact |
|------|-------------------------------------|-------------------------|--------|
| WER ASR model | whisperx large-v3 | openai-whisper base | **WER values ~3x higher** — not directly comparable to CLAUDE.md numbers (0.0425) |
| torch.load | Normal | Monkey-patched `weights_only=False` | No functional impact; needed for pyannote on torch 2.6 |
| cuDNN | Enabled | Disabled (`torch.backends.cudnn.enabled = False`) | Slight inference slowdown; workaround for CUDNN_STATUS_NOT_INITIALIZED |
| SIM_gt | Available via test_manifest | N/A | Pipeline-generated lines don't index-match test_manifest ground truth |
| clone_xvec | Pre-computed avg x-vectors | Fell back to ICL clone | `compute_avg_xvec_prompts()` was not called before pipeline run |
| flash_attn | Available | Not installed (GLIBC incompatible) | Uses SDPA fallback; slightly slower |

### How to get comparable WER numbers

To match the original eval (WER ~0.04), use `whisperx` with `large-v3`:
```bash
pip install whisperx  # may require torch>=2.8, handle version conflict
```
Or use the standalone eval scripts from the original project branch:
```bash
CUDA_VISIBLE_DEVICES=0 CKPT_DIR=output/qinche_sft_v5 EVAL_TAG=ft_v5_fast \
  python scripts/eval_fast.py
```

## Files

| File | Description |
|------|-------------|
| `output/pipeline_results_timed.json` | Pipeline results with per-line timing |
| `output/benchmark_results.json` | Native vs CUDA Graph speed comparison |
| `output/eval_report.json` | SIM_ref + WER evaluation report |
| `logs/pipeline_timed.log` | Full pipeline run log |
| `logs/benchmark_backends.log` | Benchmark run log |
| `logs/eval_patched.log` | Evaluation run log |
| `scripts/run_pipeline_timed.py` | Pipeline with timing instrumentation |
| `scripts/benchmark_backends.py` | Native vs CUDA Graph benchmark script |
| `scripts/run_eval_patched.py` | Eval wrapper with torch.load patch |
