# Final Report — Internal Working Notes

> **Audience:** the four-person team. Anyone should be able to pick up any section
> and turn it into LaTeX. Every claim cites a primary source (eval JSON, code path,
> log file, GitHub issue, or arXiv id).
>
> **Date:** May 9, 2026
> **Status:** draft / reorg before LaTeX writing

---

## 0. Project at a glance

> **One-sentence pitch.** We turn a flat 28-min-trained Qwen3-TTS voice cloner into
> an "AI emotional director" by stacking an LLM context engine, a character RAG,
> an emotion-arc tracker, and an 8-backend dispatcher in front of it — all served
> by a FastAPI + React single-page front end.

Two phases:

| Phase | Window | Goal | Outcome |
|---|---|---|---|
| **Phase 1 — Voice Cloning** | 2026-02 → 03 | "音色像不像" | Qwen3-TTS 1.7B SFT v5 ep3 → **SIM_gt = 0.689 / WER = 4.25% / RTF = 0.357** |
| **Phase 2 — Emotional Context** | 2026-04 → 05 | "情感平不平" | LLM + RAG + 8 backends + FastAPI + React + **A100 end-to-end pipeline** |

Live demo: <https://qinche.darkdark.me>
Repo: <https://github.com/HKUST-Group/qinche-tts> (branch `eval-a100`)

---

## 1. Data pipeline (used by both phases)

```
Bilibili 原始音频
  ├─ Demucs  (vocals separation)
  ├─ pyannote diarization (speaker filtering for mixed sources)
  ├─ silero-VAD (1–15 s segments)
  ├─ WhisperX large-v3 (Mandarin ASR)
  ├─ OpenCC (繁→简)
  ├─ purify_dataset.py (speaker-similarity filter + train/test split)
  └─ normalize_audio.py (RMS normalisation + 1 s tail-silence)
       └─→ data/normalized/*.wav + data/train_manifest.jsonl
```

| Set | Samples | Total | Per clip |
|---|---|---|---|
| Train (v5) | **664** | ~28 min | 1.0–13.3 s, mean 2.5 s |
| Test | 18 | 1.7 min | 4.7–11.7 s, mean 5.8 s |
| Reference audio | 5 | — | 3–10 s |
| Emotion buckets (curated) | **6 × 5 = 30** | — | from `data/emotion_buckets.json` |

Raw scripts live under `data/scripts/` and `scripts/preprocess_audio.py`.

### Data versions

| Ver | Count | Min | Notes | Outcome |
|---|---|---|---|---|
| v1 | 387 | ~14.7 | first cut, no scheduler | failed |
| v2/v3/v4 | 461 | ~19.7 | + new sources, RMS norm, 1 s tail | baseline |
| **v5** | **664** | **~28** | + `qinche_pure_p09~p11` | **best** |
| v6 | 738 | ~36 | + 74 `qinche_nobgm_01` | **failed → reverted** |

---

## 2. Phase 1 — Voice Cloning Foundation

### 2.1 Model & training

- **Base:** Qwen3-TTS-12Hz-1.7B-Base (Apache-2.0)
- **Method:** SFT, full parameters (not LoRA)
- **Hardware:** 8× NVIDIA A100-SXM4-80GB
- **Env:** Python 3.12, PyTorch 2.6, FlashAttention 2

### 2.2 Five fine-tuning runs (training log)

| Ver | LR | Sched | Eff. BS | Epochs | Data | Loss | SIM_gt / WER / RTF (best ep) | Verdict |
|---|---|---|---|---|---|---|---|---|
| v1 | 2e-5 | none | 2 | 5 | 387 | not converged | 0.04–0.10 / 2.9–4.4 / — | ❌ failed (LR too high, no scheduler, spk-emb taken from first batch only) |
| v2 (buggy) | 2e-6 | warm10% + cosine | 32 | 10 | 461 | 14→7 | 0.21–0.29 / 0.02–0.03 / — | bf16 spk-emb bug undetected |
| v2 (fixed) | 2e-6 | warm10% + cosine | 32 | 10 | 461 | — | **0.696 / 0.0365 / 2.58** (ep6) | bf16 bug fixed post-hoc |
| v3 | **1e-5** | warm5% + cosine | 32 | 10 | 461 | 15→4.6 | 0.66–0.68 / 0.029–0.038 / 2.43 | ❌ higher LR ≠ better |
| v4 | 2e-6 | warm10% + cosine | 32 | 10 | 461 | 17→7.5 | **0.678 / 0.053 / 0.351** (ep7) | bug fixed natively |
| **v5** | 2e-6 | warm10% + cosine | 32 | 10 | **664** | similar | **0.6892 / 0.0425 / 0.357** (ep3) | ✅ **best** |

(Numbers from `docs/eval_summary.json` + Phase-1 README §5.)

### 2.3 The bf16 speaker-embedding bug — single biggest lever

```python
# Buggy (Phase-1 v2 default)
spk_embed_sum += spk_encoder(wav)  # bf16 accumulation

# Fixed (Phase-1 v2-fixed onwards)
spk_embed_sum += spk_encoder(wav).float()  # fp32 running sum
spk_embed = (spk_embed_sum / N).bfloat16()  # cast back at save
```

- **Symptom:** `speaker_encoder` outputs norm ~17, but accumulated embedding norm
  collapses to ~2.98 after a few hundred batches.
- **Effect:** SIM_gt stuck at ~0.29; everyone thought it was the model ceiling.
- **Lift:** **+141 % SIM_gt (0.29 → 0.70)** after one-line fix.

> 📌 **Lesson:** any "accumulate-then-normalise" metric in `mixed_precision="bf16"`
> needs an fp32 detour. Standard mixed-precision training tutorials never warn
> about this because token-loss accumulation is not norm-sensitive — speaker
> embedding cosine is.

### 2.4 Inference acceleration (Phase 1)

| Method | RTF (orig) | RTF (opt) | Speed-up | Quality | Status |
|---|---|---|---|---|---|
| **FasterQwen3TTS (CUDA Graph)** | 2.542 | **0.357** | **7.1×** | lossless | ✅ shipped |
| Fish Speech `--compile` | 4.315 | 0.639 | 6.75× | lossless | ✅ verified |
| Flash Attention 2 | — | — | — | — | ✅ default-on |
| **INT8 (bitsandbytes)** | — | 7.0 (worse) | — | SIM_gt 0.07, WER 1.0 | ❌ failed |
| **Speculative dec.** (0.6B Base draft) | — | slower than baseline | — | accept rate 0–5% | ❌ failed |
| **Self-spec / early-exit** | — | — | — | mid-layer agreement < 7% | ❌ failed |

(See `scripts/benchmark_inference.py`, `output/benchmark_results.json`.)

### 2.5 Phase-1 final number table (cite as Table I)

| Model / setting | SIM_gt | SIM_ref | WER | RTF | Inference |
|---|---|---|---|---|---|
| Qwen3 1.7B Zero-shot | **0.7104** | 0.7563 | 0.1203 | 2.401 | native |
| **Qwen3 SFT v5 ep3** | **0.6892** | 0.7161 | 0.0425 | **0.357** | FasterQwen3TTS |
| Qwen3 SFT v5 ep3 | 0.6885 | **0.7281** | 0.0451 | 2.542 | native |
| Qwen3 SFT v5 ep5 | 0.6791 | 0.6998 | **0.0279** | 0.356 | FasterQwen3TTS |
| Qwen3 SFT v5 ep9 | 0.6795 | 0.7180 | **0.0269** | 0.354 | FasterQwen3TTS |
| Fish S2 Pro ZS (compile) | 0.6627 | 0.6824 | 0.0209 | 0.639 | torch.compile |

(All numbers from `docs/eval_summary.json` + Phase-1 README §6.2.)

---

## 3. Phase 2 — Emotional-Aware Context Pipeline

### 3.1 Module map (everything new lives under `src/`)

| Module | Path | Purpose |
|---|---|---|
| EmotionAnnotation contract | `src/context_engine/models.py` | Strict Pydantic v2: `{emotion, intensity ∈ [0,1], pace ∈ {slow,normal,fast}, style, ref_emotion_category ∈ {tender,calm,playful,intense,cold,intimate}, fish_audio_tags}` |
| **LLM Context Engine** | `src/context_engine/analyzer.py` | system prompt + 4 few-shot exemplars; GPT-4o via OpenAI/OpenRouter; `response_format=json_object`; 3-retry; T=0.3 |
| Emotion-bucket classifier | `src/context_engine/classify_samples.py` | Build `data/emotion_buckets.json` (top-5 per of 6 categories) |
| **Character RAG** | `src/rag/{knowledge_base,retriever}.py` | 4 markdown files in `data/character_kb/` → bge-large-zh-v1.5 → ChromaDB → top-k injection |
| **Emotion Arc Tracker** | `src/emotion_tracker/tracker.py` | 10-turn sliding window + transition-distance table; flags abrupt jumps; feeds last N states back into LLM prompt |
| **Multi-backend TTS dispatcher** (8 backends) | `src/tts/*` | see §3.2 |
| End-to-end pipeline | `src/pipeline.py` | `analyzer → tracker → RAG → dispatcher → wav` |
| FastAPI service | `src/api/server.py` | `/api/{analyze, pipeline, audio/<sess>/<backend>/<file>, health}` |
| React front end | `frontend/src/pages/EmotionalTTSPage.tsx` | scene input, presets, live emotion-arc chart, per-line detail panel, **Demo Mode** for offline iteration |
| Eval harness | `src/eval/{run_eval, sim, wer, emotion_acc}.py` | per-backend × per-condition |

### 3.2 The 8 backends and why

| Backend | Idea | When to use |
|---|---|---|
| `baseline` | SFT CustomVoice direct, no emotion | upper-bound timbre |
| `qwen` | Qwen3-TTS zero-shot | reference baseline |
| `clone` | SFT + single reference | naive cloning |
| `clone_xvec` | Per-emotion averaged x-vector picked by LLM label | emotion-conditioned |
| `clone_blend` | calm-xvec ↔ emotion-xvec linear interpolation by LLM intensity | trades emotion vs timbre |
| `auto` | Rule-based router on top of LLM output | recommended default |
| `auto_quality` | Speaker-SIM gate using a canonical x-vector; falls back to baseline if gate fails | quality insurance |
| `fish` | Fish S2 Pro + LLM-generated `[tag]` inline annotations | cross-architecture A/B |

### 3.3 The "auto_quality" gate — engineered fix for emotion vs timbre conflict

Code path: `src/tts/auto_backend.py`. The dispatcher first picks an emotion-conditioned
backend. It then compares the generated audio's speaker x-vector against a
canonical reference; if cosine < 0.6 it falls back to `baseline` (timbre-only).
This is the cleanest engineering compromise we found between "more emotion =
more timbre drift".

### 3.4 Front-end / Demo Mode

`EmotionalTTSPage.tsx` carries a hard-coded preset annotations bundle so anyone
without an A100 or an OpenRouter key can see the full UI render. Vital for the
4-person / 1-A100 dev cycle.

---

## 4. End-to-end evaluation on A100 (`eval-a100` branch, commit `c6ab413`)

> Environment differs from Phase-1 training machine; numbers are intra-condition
> comparable but **not directly comparable to Phase-1 WER** (see §4.4).

### 4.1 Pipeline timing — `output/pipeline_results_timed.json`

(test_scene_01, 4 秦彻 sentences)

| Component | Total | Mean / sentence |
|---|---|---|
| Init (model load) | 496 s | — |
| **Emotion Analysis** (GPT-4o, OpenRouter) | 28.5 s | **7.1 s** |
| **RAG Retrieval** (bge-large-zh, ChromaDB) | 5.7 s | **1.4 s** |
| TTS `auto` | 9.0 s* | 9.0 s |
| TTS `clone_xvec` | 8.7 s* | 8.7 s |
| TTS `baseline` | 8.7 s* | 8.7 s |

\* Excluding first-call warm-up (the very first `auto` call took 238 s, all in
warm-up + model load).

### 4.2 Inference benchmark — `output/benchmark_results.json`

| Method | Mean RTF | Mean gen | Speed-up |
|---|---|---|---|
| Native SFT (CustomVoice, SDPA) | 1.329 | 8.13 s | 1× |
| **CUDA Graph SFT (FasterQwen3TTS)** | **0.341** | **2.01 s** | **3.9×** |
| `clone_xvec` (Base + native) | 1.342 | 9.53 s | ~1× |

### 4.3 Three-condition SIM_ref + WER — `output/eval_report.json`

| Condition | SIM_ref | WER | Note |
|---|---|---|---|
| `auto` | **0.7020** | 0.1645 | router → `clone_blend` |
| `clone_xvec` | 0.6665 | **0.1507** | fell back to ICL (no pre-computed avg-xvec) |
| `baseline` | **0.7409** | 0.1599 | SFT CustomVoice, no emotion |

### 4.4 Cross-environment caveats — `eval/A100_EVAL_RESULTS.md`

| Item | Original (training machine) | A100 cluster | Impact |
|---|---|---|---|
| WER ASR model | whisperx large-v3 | openai-whisper base | **WER ~3× higher**, not directly comparable to Phase-1 numbers |
| flash_attn | available | not installed (GLIBC) | SDPA fallback, slightly slower |
| cuDNN | enabled | **disabled** (CUDNN_STATUS_NOT_INITIALIZED workaround in `src/tts/_common.py`) | small slowdown |
| SIM_gt | available | N/A | pipeline lines don't index-match test_manifest |
| `clone_xvec` | pre-computed avg | fell back to ICL | `compute_avg_xvec_prompts()` not called pre-pipeline |

---

## 5. Successes (the high-light reel for the report)

1. **Production-grade timbre cloning with 28 min of data.** SIM_gt 0.689,
   WER 4.25 %, RTF 0.357 — beats Fish S2 Pro zero-shot on SIM and matches
   Qwen3 ZS on SIM with **a third of the WER**.

2. **Discovered + fixed a +141 % SIM bug** (bf16 spk-emb accumulation). Single
   biggest quality lever in the entire project.

3. **Real-time inference at single-A100 scale** via FasterQwen3TTS CUDA Graph
   (RTF 0.357), independently re-verified on the A100 cluster (RTF 0.341 with
   SDPA fallback).

4. **End-to-end emotional pipeline running.** From plain scene description to
   emotion-tagged WAV, all dispatched through one FastAPI endpoint, visualised
   in real time on a deployed React page.

5. **Eight-backend dispatcher with a quality gate.** Engineered solution to
   the "more emotion = more timbre drift" problem, rather than betting on
   prompt magic.

6. **Demo Mode = 4 devs, 1 GPU, no blocking.** The React page renders the full
   UI from offline preset annotations, so UX iteration never queues for the A100.

---

## 6. Failures and root causes (be honest in §5 of the final report)

Ranked by what is most instructive to write up.

### 6.1 INT8 quantisation (bitsandbytes) — Qwen3-TTS architecture mismatch

- **Symptom:** SIM_gt 0.07, WER 1.0, RTF 7.0 (worse than baseline).
- **Root cause:** Qwen3-TTS is **dual-track LM** (28-layer talker + code
  predictor) with RVQ codec embedding. `bitsandbytes` LLM.int8 assumes a single
  transformer stack; quantising the codec embedding (`spk_id` slot) and the
  small predictor head destroys the numerical distribution.
- **Lesson:** TTS ≠ LLM. Generic LLM quantisation toolchains do **not** carry
  over.
- **Fix path (see §8.1 below):** FP8 via vLLM-Omni (post-training, no retrain).

### 6.2 Speculative decoding with Qwen3-TTS-0.6B-Base as draft

- **Symptom:** acceptance rate 0–5 %, slower than baseline.
- **Root cause:** speculative decoding requires draft and target token
  distributions to be close. The 0.6B-Base is **not fine-tuned** to the same
  speaker; the SFT-1.7B has a very different codec-token distribution.
- **Lesson:** without an SFT'd draft model, spec-dec is dead.

### 6.3 Self-speculative / early-exit on the talker

- **Symptom:** middle-layer vs final-layer token agreement < 7 %.
- **Root cause:** the 28-layer talker does substantial feature transformation
  at every layer (phoneme → prosody → codec). Unlike LLMs that have "redundant
  middle layers", Qwen3-TTS does not.
- **Lesson:** architecturally infeasible. We can write this off cleanly.

### 6.4 v3 LR = 1e-5 (5× v2/v4)

- **Symptom:** loss converged faster (15 → 4.6) but SIM_gt slightly worse.
- **Root cause:** the speaker-embedding head is very LR-sensitive; high LR
  causes batch-to-batch spk-emb thrash, and the running average ends up not
  pointing at the real speaker.
- **External validation:** the [community LoRA fine-tuning guide][instavar]
  also reports the official 2e-5 default produces "pure noise"; 2e-6 is the
  correct value. Our v2/v4/v5 = 2e-6 was right; v3 confirmed the warning.
- **Lesson:** training loss ≠ TTS evaluation. SIM_gt is the only ground truth.

[instavar]: https://instavar.com/blog/ai-production-stack/LoRA_Finetuning_Qwen3_TTS_Custom_Voices

### 6.5 v6 data expansion (74 nobgm clips → SIM dropped)

- **Symptom:** +16 % data → −0.01 SIM_gt.
- **Root cause:** the new `qinche_nobgm_01` source has BGM-removal residue;
  speaker encoder learnt the artefact distribution, not the timbre.
- **Lesson:** **data quality > quantity**. 28 min clean > 36 min noisy.

### 6.6 bf16 spk-embed precision (already covered §2.3)

### 6.7 A100 cluster environment drift

- **Symptom:** WER 0.04 (training machine, whisperx large-v3) ↔ WER 0.16 (A100,
  openai-whisper base). Same audio.
- **Root cause:** the eval harness was tightly coupled to the original env;
  the A100 cluster needed an ASR swap (env conflict), flash_attn off (GLIBC),
  and cuDNN off (CUDNN_STATUS_NOT_INITIALIZED).
- **Lesson:** evaluation scripts should pin (a) the ASR model hash into the
  output JSON, (b) a frozen conda env, (c) cache all model weights offline.
  Otherwise cross-environment numbers are not comparable.

### 6.8 Possibly hit but not yet verified — official Qwen3-TTS upstream bugs

> **Action item for the team:** check our forked `Qwen3-TTS/finetuning/sft_12hz.py`
> against upstream PR #278.

Two confirmed upstream bugs in the official Qwen3-TTS finetuning code that we
may also have hit silently:

1. **Double label-shift** ([Issue #179][q3-179] / [PR #278][q3-278]):
   `sft_12hz.py` and `modeling_qwen3_tts.py` both manually shift, plus
   `ForCausalLMLoss` shifts internally → **double shift → speech progressively
   accelerates each epoch**.
2. **Missing `text_projection`** ([Issue #39][q3-39] / [Issue #120][q3-120]):
   training calls `text_embedding()` directly, inference calls `text_projection`
   → **noise during inference even though training loss decreases**.

[q3-179]: https://github.com/QwenLM/Qwen3-TTS/issues/179
[q3-278]: https://github.com/QwenLM/Qwen3-TTS/pull/278
[q3-39]: https://github.com/QwenLM/Qwen3-TTS/issues/39
[q3-120]: https://github.com/QwenLM/Qwen3-TTS/issues/120

If we have these bugs in our v5, then 0.689 SIM_gt is **not** our ceiling and a
clean re-train could give us a free upgrade. Even if we don't have them, this is
a good "future work" hook.

---

## 7. Bottleneck analysis (section that drives Future Work)

### 7.1 SIM_gt ceiling at 0.689

| Source | Number |
|---|---|
| ZS upper bound | 0.7104 |
| Our best (v5 ep3) | **0.6892** |
| Headroom | **0.021** |

Possible gaps: (a) latent upstream bugs (§6.8), (b) data quantity, (c) data diversity.

### 7.2 Emotion realism — currently no objective number

We have **0** objective emotion-accuracy or naturalness numbers. Only SIM and
WER. This is the report's most obvious gap.

### 7.3 End-to-end latency

Per-line breakdown from §4.1:

```
LLM (GPT-4o, network) : 7.1 s    ← 67 %
RAG (bge + Chroma)    : 1.4 s    ← 13 %
TTS (CUDA Graph)      : 2.0 s    ← 19 %
TOTAL                 :~10.5 s   per sentence
```

### 7.4 Single-GPU TTS throughput

CUDA Graph already at single-card limit; further gains need batched / disaggregated
serving.

### 7.5 6-class emotion granularity

Engineering-stable but coarse. Needs validation against finer-grained models.

### 7.6 Cross-environment evaluation reproducibility

Already covered in §6.7. Affects how we present numbers in §4 of the report.

---

## 8. Solutions (the section the team needs for "Future Work")

For each failure / bottleneck, the most concrete option ordered by effort.

### 8.1 Replace failed INT8 → FP8 via vLLM-Omni (post-training, no retrain)

**Source:** [vLLM-Omni v0.16.0 release][vllm-016] (2026-02), [FP8 W8A8 docs][fp8-docs].

- vLLM-Omni v0.16.0 ships **production-ready Qwen3-TTS** support: RTF 0.22–0.45,
  TTFP −90 %, CUDA Graph for the speech tokenizer (Code2Wav), batched code2wav
  decoding, disaggregated 2-stage inference (Talker on big GPU, Code2Wav on
  small GPU/CPU).
- FP8 dynamic + AWQ combination tested in [vLLM compressor PR #2330][awq-fp8].
- Existing FP8 model on HF: [`drbaph/s2-pro-fp8`][s2pro-fp8] (Fish S2 Pro
  quantised, 2026-03-11).

[vllm-016]: https://github.com/vllm-project/vllm-omni/releases/tag/v0.16.0
[fp8-docs]: https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/quantization/fp8/
[awq-fp8]: https://github.com/vllm-project/llm-compressor/pull/2330
[s2pro-fp8]: https://hf.co/drbaph/s2-pro-fp8

### 8.2 Improve SIM by checking upstream bugs + switching to ICL Mode

- **Check / fix upstream bugs** (§6.8) — a few lines, free upgrade.
- **Use Qwen3-TTS Voice Clone Prompt API in ICL Mode** (5 audios per call) —
  community confirms ICL > X-vector-only ([Voice Clone Prompt docs][vcp-docs]).
  We already have 5 clips per emotion in `data/emotion_buckets.json` → perfect
  fit.

[vcp-docs]: https://mintlify.com/QwenLM/Qwen3-TTS/api/model/voice-clone-prompt

### 8.3 Improve SIM via VC-augmented data

- [arXiv 2011.05707][vc-aug]: 15 min target + voice-conversion-augmented data
  achieves expressive TTS.
- **Tool:** [CosyVoice 3][cv3] (Apache-2.0) zero-shot voice cloning, 9 langs +
  18 Chinese dialects, MOS ~5.5, 150 ms first-packet.
- Plan: take a small set of high-quality target clips → use CosyVoice 3 to
  borrow other speakers' diverse content → generate "the target voice saying
  more diverse lines" → fold back into v5 training set after emotion2vec
  filtering.

[vc-aug]: https://arxiv.org/abs/2011.05707
[cv3]: https://github.com/FunAudioLLM/CosyVoice

### 8.4 Try LoRA + scale 0.3 inference (community-validated recipe)

- [Instavar 2026 LoRA guide][instavar]: LoRA fine-tuning of Qwen3-TTS 1.7B at
  LR 2e-6, inference scale 0.3–0.35 (not the default 1.0), stop at epoch 10.
  Cleaner spk-emb learning than full SFT for low-data speakers.
- [Public repo][lora-repo].

[lora-repo]: https://github.com/instavar/qwen3-tts-lora-finetuning

### 8.5 Per-emotion residual adapters (~0.1 % params each)

- [arXiv 2210.15868][adapter]: residual adapters give few-shot speaker
  adaptation with ~0.1 % params, backbone frozen.
- We could train **6 adapters, one per emotion bucket**, instead of averaging
  x-vectors. Plug-and-play via the dispatcher.

[adapter]: https://arxiv.org/pdf/2210.15868

### 8.6 DPO post-training for emotional expressiveness

Reachable extension on top of v5:

| Paper | Highlight |
|---|---|
| [Emo-DPO][emo-dpo] | Differentiate emotional nuances via preference pairs |
| [arXiv 2409.12403][dpo-tts] | DPO on 1.15B LM-TTS improved SIM, WER, MOS — some metrics > human |
| [EASPO][easpo] | Stepwise DPO at intermediate denoising steps for diffusion TTS |
| [TKTO][tkto] | Token-level DPO for data-efficient alignment |

[emo-dpo]: https://xiaoxue1117.github.io/Emo-tts-dpo/
[dpo-tts]: https://arxiv.org/abs/2409.12403
[easpo]: https://arxiv.org/html/2509.25416v2
[tkto]: https://arxiv.org/html/2510.05799v1

### 8.7 Add objective emotion + naturalness numbers (close §7.2 gap)

| Tool | What it gives | Cost |
|---|---|---|
| [emotion2vec+ large][e2v] (9-class) | Per-clip emotion logits → emotion-classification accuracy on our 6 buckets | minutes |
| [UTMOSv2][utmos] | Auto-MOS naturalness, VoiceMOS-2024 winner in 7/16 metrics | minutes |
| [NISQA-TTS][nisqa] | 5 dimensions: Quality / Noisiness / Coloration / Discontinuity / Loudness → radar chart for the report | minutes |
| [Audio Turing Test (ATT)][att] | ICLR-2026 protocol: "is this human or AI?" instead of MOS | needs human raters but easier than 5-point MOS |
| [Emo-Emilia test set][c2ser] | 1400-clip Mandarin/English 7-class benchmark | one-shot |
| [EmoBox][emobox] | 32 datasets / 14 langs SER toolkit, emotion2vec baseline | reference |

[e2v]: https://github.com/ddlbojack/emotion2vec
[utmos]: https://github.com/sarulab-speech/UTMOSv2
[nisqa]: https://github.com/gabrielmittag/NISQA
[att]: https://openreview.net/forum?id=MzpSOMTt3Z
[c2ser]: https://github.com/zxzhao0/C2SER
[emobox]: https://arxiv.org/html/2406.07162

### 8.8 Cut LLM-engine latency from 7.1 s → < 1 s

- **Local Qwen2.5-14B-Instruct + AWQ 4-bit + structured-output JSON** —
  no network round-trip, fits on one A100, 128k context, [official docs][q25].
- **Prompt caching** (OpenAI auto / vLLM manual): up to 80 % latency, 90 % cost
  reduction; TTFT improvement 13–31 %.
- **FusionRAG** ([arXiv 2601.12904][fusionrag]): 2.66× – 9.39× TTFT reduction
  for RAG-augmented LLMs.
- **Inline static KB** in system prompt — our 4 KB markdown files total < 5 k
  tokens; we don't actually need ChromaDB at inference if we just bake them in.

[q25]: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M
[fusionrag]: https://arxiv.org/abs/2601.12904

### 8.9 Streaming TTS for sub-second perceived latency

- FasterQwen3TTS streaming mode already has TTFA = 305 ms.
- VoXtream2 (74 ms TTFP) and Qwen3-TTS 0.6B (97 ms) are the SOTA references for
  the report.
- vLLM-Omni v0.16.0 disaggregated inference for production deployment.

### 8.10 Pin evaluation environment

- Add ASR model SHA + library versions to `eval_report.json`.
- Provide a single `environment.yml` for the eval harness.
- Cache ASR/SER weights in `models/` next to TTS weights.

---

## 9. Future-work, ROI-ranked

### 9A. Immediate (1–2 weeks, all post-training, no retrain)

| # | Task | Source | Expected lift |
|---|---|---|---|
| 1 | Run **emotion2vec+ / UTMOSv2 / NISQA** on existing 18 × 8 generations | §8.7 | **+1 figure, +1 table** for §4 of the final report |
| 2 | Switch dispatcher's `clone` family to **Qwen3-TTS ICL Mode** with 5 refs | §8.2 | likely +0.01–0.02 SIM_ref |
| 3 | Replace GPT-4o with **local Qwen2.5-14B + AWQ** + prompt caching | §8.8 | LLM stage 7.1 s → ~1 s |
| 4 | Inline the 4-file character KB into the LLM system prompt; drop runtime ChromaDB | §8.8 | RAG stage 1.4 s → < 0.1 s |

### 9B. Mid-term (1 month, may require retrain / re-deploy)

| # | Task | Source | Expected lift |
|---|---|---|---|
| 5 | Verify + fix upstream **Qwen3-TTS label-shift / text_projection** bugs; retrain v5 | §6.8, PR #278 | possible +0.02–0.05 SIM_gt |
| 6 | Migrate inference to **vLLM-Omni v0.16.0** + FP8 + disaggregated serve | §8.1 | RTF 0.34 → ~0.22; TTFP −90 % |
| 7 | **CosyVoice 3 VC** to expand 28 min → 60–100 min effective data | §8.3 | possible +0.02 SIM_gt |
| 8 | Pin eval environment (ASR hash, conda env, weights cache) | §8.10 | cross-env comparability |

### 9C. Exploratory (open-ended)

| # | Task | Source | Why |
|---|---|---|---|
| 9 | **Emo-DPO** preference alignment on top of v5 | §8.6 | preference-pair training to push emotional expressiveness |
| 10 | **Per-emotion residual adapters** (6 adapters × 0.1 % params) | §8.5 | replace `clone_xvec` / `clone_blend` with a cleaner mechanism |
| 11 | **EAGLE-3 feature-level speculative** on TTS talker | (no prior work) | high-risk research direction; nobody has done it |

---

## 10. Mapping notes → final-report sections

This is the section-by-section content drop for `final.tex`.

| Final-report section | Pulls from notes section |
|---|---|
| 1. Introduction & Problem | §0 |
| 2. Phase 1 — Voice Cloning Foundation | §1, §2 |
| 3. Phase 2 — Emotional-Aware Context Pipeline | §3 |
| 4. End-to-End Evaluation on A100 | §4 |
| 5. Lessons Learned: What Worked, What Failed | §5, §6 |
| 6. Bottleneck Analysis | §7 |
| 7. Future Work (Solutions) | §8, §9 |
| 8. Conclusion | one-liner from §0 |

---

## 11. Figures inventory (what we already have, what we need to make)

### Already in repo (re-usable)

| File | Source | Use in final report |
|---|---|---|
| `docs/report/figs/bf16_bug_impact.pdf` | Phase-1 | §2.3 bf16 fix |
| `docs/report/figs/sim_gt_across_versions.pdf` | Phase-1 | §2.5 final table |
| `docs/report/figs/rtf_comparison_bar.pdf` | Phase-1 | §2.4 acceleration |
| `docs/report/figs/native_vs_fast_v4.pdf` | Phase-1 | §2.4 |
| `docs/report/figs/dataset_duration_distribution.pdf` | Phase-1 | §1 data |
| `docs/report/figs/preprocessing_pipeline.pdf` | Phase-1 | §1 pipeline |
| `docs/milestone/figs/emotion_and_phase1.pdf` | Milestone | §3 emotion buckets + §2.5 Phase-1 numbers |
| `docs/milestone/figs/ui_arc.png` | Milestone | §3 frontend |
| `docs/milestone/figs/ui_detail.png` | Milestone | §3 frontend |

### Need to make new for the final report

| Figure | Data source | Tool |
|---|---|---|
| Updated system architecture (TikZ, possibly extended to show eval harness) | hand-drawn | TikZ in LaTeX |
| **Pipeline timing pie / stacked bar** (LLM/RAG/TTS shares) | `output/pipeline_results_timed.json` | matplotlib |
| **Native vs CUDA Graph bar (A100)** | `output/benchmark_results.json` | matplotlib |
| **Three-condition SIM_ref + WER bar** (auto / clone_xvec / baseline) | `output/eval_report.json` | matplotlib |
| **Failure timeline** (INT8 → spec-dec → self-spec → v3 LR → v6 data) | this notes file | TikZ timeline or a table |
| Future-work roadmap (Immediate / Mid / Exploratory) | §9 above | simple table |

---

## 12. Open questions for the team

1. Did anyone diff our forked `sft_12hz.py` against upstream PR #278? If not,
   that's the cheapest possible SIM upgrade.
2. Is there a recorded MOS / A-B test session with humans, or do we only have
   objective numbers? (Determines whether we need §8.7 immediately.)
3. Repo URL on the cover page — keep `HKUST-Group/qinche-tts` (the eval-a100
   branch has the freshest pipeline) or set up a clean read-only mirror?
4. Group/class number to print on the cover page.

---

*End of notes. Ready to lift into `final.tex`.*
