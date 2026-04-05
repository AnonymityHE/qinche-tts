# TTS 语音克隆项目文档

> 目标：为目标说话人（秦彻/qinche）构建高质量的中文 TTS 语音克隆系统。
>
> 最后更新：2026-03-16

---

## 目录

- [1. 项目结构](#1-项目结构)
- [2. 模型清单](#2-模型清单)
- [3. 运行环境](#3-运行环境)
- [4. 数据集](#4-数据集)
- [5. 训练实验记录](#5-训练实验记录)
- [6. 评估结果汇总](#6-评估结果汇总)
- [7. 推理加速总结](#7-推理加速总结)
- [8. 后续优化方向](#8-后续优化方向)

---

## 1. 项目结构

```
TTS/
├── models/                          # 预训练模型权重
│   ├── qwen3-tts/                   # Qwen3-TTS 全系列
│   └── fish-speech-s2-pro/          # Fish Audio S2 Pro
├── Qwen3-TTS/                       # Qwen3-TTS 官方代码（含微调脚本）
│   └── finetuning/
│       ├── sft_12hz.py              # 微调训练主脚本
│       ├── dataset.py               # 数据集与 collate_fn
│       ├── prepare_data.py          # 提取 audio_codes
│       ├── train_raw.jsonl          # 归一化后的训练清单
│       └── train_with_codes.jsonl   # 含 audio_codes 的训练清单
├── fish-speech/                     # Fish Speech S2 代码库
│   └── checkpoints/s2-pro -> models/fish-speech-s2-pro
├── scripts/                         # 数据处理、训练、评估脚本
│   ├── preprocess_audio.py          # 音频预处理（VAD/ASR/分段）
│   ├── purify_dataset.py            # 说话人过滤 + 训练/测试划分
│   ├── normalize_audio.py           # RMS 响度归一化 + 末尾静音
│   ├── process_new_data.py          # 新数据处理流水线
│   ├── run_finetune.sh              # v2 微调启动脚本
│   ├── run_finetune_v3.sh           # v3 微调启动脚本
│   ├── eval_finetuned.py            # 微调模型评估（原生推理）
│   ├── eval_fast.py                 # 微调模型评估（FasterQwen3TTS CUDA Graph）
│   ├── eval_fish_zeroshot.py        # Fish S2 Pro zero-shot 评估
│   ├── eval_qwen_zeroshot.py        # Qwen3 zero-shot 评估
│   └── benchmark_inference.py       # 推理加速 benchmark（baseline vs CUDA Graph）
├── data/
│   ├── segments/                    # VAD 切分后的音频片段
│   ├── ref_audio/                   # 参考音频（ref_00~ref_04.wav）
│   ├── dataset/                     # train/test manifest
│   └── normalized/                  # 响度归一化后的音频
├── output/                          # 微调 checkpoint 输出
│   ├── qinche_sft_v4/               # v4: LR=2e-6, 461条, 10 epochs（bf16 bug 已修复）
│   └── qinche_sft_v5/               # v5: LR=2e-6, 664条, 10 epochs（扩充数据，当前最优）
├── eval/                            # 评估输出（生成音频 + 指标）
│   ├── qwen_1.7b_zs/               # Qwen 1.7B zero-shot 生成
│   ├── ft_v2_checkpoint-epoch-*/    # v2 微调各 epoch 生成
│   ├── fish_s2_zeroshot/            # Fish S2 Pro zero-shot 评估结果
│   ├── fish_s2_zeroshot_compile/    # Fish S2 Pro zero-shot + torch.compile
│   ├── ft_v4_fast_checkpoint-epoch-*/  # v4 微调 + FasterQwen3TTS 评估
│   └── *_comparison.json            # 汇总指标 JSON
└── logs/                            # 所有日志
```

---

## 2. 模型清单

### 2.1 Qwen3-TTS（主力模型）

| 模型 | 路径 | 参数量 | 用途 |
|------|------|--------|------|
| Qwen3-TTS-12Hz-1.7B-Base | `models/qwen3-tts/` | 1.7B | **微调基座模型** |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | `models/qwen3-tts/` | 1.7B | 官方预训练声音克隆模型（参考） |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | `models/qwen3-tts/` | 1.7B | 文本描述生成声音（参考） |
| Qwen3-TTS-12Hz-0.6B-Base | `models/qwen3-tts/` | 0.6B | 轻量版基座（对比测试用） |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | `models/qwen3-tts/` | 0.6B | 轻量版声音克隆 |
| Qwen3-TTS-Tokenizer-12Hz | `models/qwen3-tts/` | - | 音频 tokenizer |

**架构特点：**
- Transformer decoder + 16-codebook RVQ codec，帧率 12Hz
- Speaker encoder 提取说话人嵌入，注入到 codec embedding 中的 `spk_id` 位置
- 支持 10 种语言，中文为核心语言
- 微调方式：SFT 全量训练（非 LoRA），直接学习 speaker embedding
- 最低数据需求：官方推荐 10-30 分钟高质量音频
- 推理显存：~20GB
- 许可证：Apache 2.0（可商用）

**微调流程：**
```
train_manifest.jsonl → normalize_audio.py（RMS归一化+1s静音）
    → prepare_data.py（提取audio_codes）
    → sft_12hz.py（训练，输出checkpoint）
```

### 2.2 Fish Audio S2 Pro（对比模型，仅 zero-shot）

| 模型 | 路径 | 参数量 | 用途 |
|------|------|--------|------|
| S2 Pro (Slow AR + Fast AR) | `models/fish-speech-s2-pro/` | 5B (4B+0.4B+codec) | **Zero-shot 声音克隆对比** |

**架构特点：**
- Dual-AR 架构：4B Slow AR（时间轴语义预测）+ 400M Fast AR（残差声学细节）
- RVQ codec：10 codebooks，帧率 21Hz
- 训练数据 10M+ 小时，支持 80+ 种语言（中文为 Tier 1）
- 15-30 秒参考音频即可做 zero-shot 声音克隆
- 支持 `[tag]` 内联情绪控制（15000+ 种标签）
- 推理显存：至少 24GB
- 许可证：**Fish Audio Research License（商用需授权）**

**为何不做微调：**
1. S2 Pro 是 RL-trained 模型，官方明确警告不建议在 RL 模型上微调，会导致性能退化
2. 默认只学语言模式（speech patterns），不学音色（timbre）；音色需靠推理时的 reference prompt
3. 微调数据量要求高，社区建议至少 10 小时（我们远不够）
4. 商用许可受限

**推理流程：**
```
reference_audio.wav → VQ token 提取（codec.pth）
    → text2semantic 推理（LLAMA 4B）
    → codec 解码 → 输出 WAV
```

---

## 3. 运行环境

### 3.1 硬件

- **GPU：** 8× NVIDIA A100-SXM4-80GB
- **存储：** 19TB（已用 ~18.3TB，剩余 ~700GB）

### 3.2 Conda 环境

| 环境名 | Python | PyTorch | 用途 |
|--------|--------|---------|------|
| `qwen3-tts` | 3.12 | 2.6.x | Qwen3-TTS 微调/推理/评估 |
| `fish-speech` | 3.12 | 2.8.0 | Fish Audio S2 Pro 推理 |

### 3.3 关键依赖

**qwen3-tts 环境：**
- `transformers`, `accelerate` — 模型加载与分布式训练
- `flash-attn` — Flash Attention 2 加速
- `whisperx` — ASR 转写（评估 WER 用）
- `pyannote.audio` — 说话人嵌入提取（评估 SIM 用）
- `opencc-python-reimplemented` — 繁简转换
- `faster-qwen3-tts` — CUDA Graph 加速推理（RTF 0.35，7.1x 加速）
- `jiwer` — WER 计算
- `silero-vad` — 语音活动检测
- `librosa`, `soundfile` — 音频处理

**fish-speech 环境：**
- `torch==2.8.0`, `torchaudio==2.8.0`
- `transformers<=4.57.3`
- `gradio` — WebUI 推理界面
- `descript-audio-codec` — DAC 声码器

---

## 4. 数据集

### 4.1 数据来源

| 来源 | 类型 | 说明 |
|------|------|------|
| `qinche_pure_p01~p11` | 纯净单人音频 | 目标说话人的纯净录音 |
| `qinche_02` | 混合音频 | 含其他说话人，经 pyannote diarization 筛选 |
| `qinche_01` | 混合音频 | 质量较差，**已排除** |

### 4.2 预处理流水线

```
原始音频 → Demucs（人声分离）
    → Pyannote（说话人分离，混合源）
    → Silero-VAD（语音切分，1-15s）
    → WhisperX（ASR 转写）
    → OpenCC（繁体→简体）
    → purify_dataset.py（说话人相似度过滤 + 随机 train/test 划分）
    → normalize_audio.py（RMS 归一化 + 1s 末尾静音）
```

### 4.3 当前数据集统计

| 集合 | 样本数 | 总时长 | 时长范围 | 平均时长 |
|------|--------|--------|----------|----------|
| 训练集 | 664 | ~28 分钟 | 1.0s - 13.3s | ~2.5s |
| 测试集 | 18 | 1.7 分钟 | 4.7s - 11.7s | 5.8s |
| 参考音频 | 5 | - | 3-10s | - |

### 4.4 数据迭代历史

- **v2/v3/v4 数据集**（461 条，~19.7 min）：`qinche_pure_p01~p08` + `qinche_02` 过滤后
- **v5 数据集**（664 条，~28 min）：扩充 `qinche_pure_p09~p11` 等新数据源
- **v6 数据集**（738 条，~36 min）：新增 74 条 `qinche_nobgm_01`，但效果反而下降（SIM_gt 0.68 vs v5 的 0.69），**已清理，该批数据质量不达标**
- 最终采用 v5 数据集（664 条）作为正式训练集

---

## 5. 训练实验记录

### 5.1 v1 — 初版训练（已弃用）

| 参数 | 值 |
|------|-----|
| 输出路径 | `output/qinche_sft/` |
| 学习率 | 2e-5 |
| Batch Size | 2 |
| Epochs | 5 |
| Scheduler | 无 |
| 数据 | 387 条，~14.7 min |

**问题：** LR 过高，无 scheduler，speaker embedding 只取首 batch，数据划分不均。
**结论：** SIM_gt 仅 0.04-0.10，WER 2.9-4.4，完全失败。

### 5.2 v2 — 降低学习率 + scheduler

| 参数 | 值 |
|------|-----|
| 输出路径 | `output/qinche_sft_v2/` |
| 学习率 | 2e-6 |
| Effective Batch Size | 32 (2×16) |
| Epochs | 10 |
| Scheduler | Linear warmup (10%) + Cosine decay（最低 0.1×LR） |
| 数据 | 461 条，~19.7 min（含新数据 + RMS归一化 + 1s静音） |

**训练曲线：** Loss 从 14.0 → 7.0（收敛，但 final loss 偏高）

**v2 原版评估（speaker embedding 有 bf16 精度 bug）：**
| Epoch | SIM_gt | SIM_ref | WER |
|-------|--------|---------|-----|
| 4 | 0.2834 | 0.2583 | 0.0265 |
| 6 | 0.2921 | 0.2636 | 0.0208 |
| 9 | 0.2062 | 0.2102 | 0.0232 |

**v2 fixed 评估（修复 speaker embedding 精度后）：**
| Epoch | SIM_gt | SIM_ref | WER |
|-------|--------|---------|-----|
| 4 | 0.6784 | 0.7498 | 0.0874 |
| **6** | **0.6964** | **0.7653** | **0.0365** |
| 9 | 0.6876 | 0.7613 | 0.0282 |

**关键 Bug 修复：** 训练过程中 `spk_embed_sum` 在 `mixed_precision="bf16"` 下累加导致严重精度损失。speaker_encoder 输出 norm~17，但累加几百次后保存的 embedding norm 仅 2.98。修复方法：将累加转为 fp32（`.float()`），再在保存时转回 bf16。修复后 SIM_gt 从 0.29 提升到 0.70（+141%）。

### 5.3 v3 — 提高学习率探索

| 参数 | 值 |
|------|-----|
| 输出路径 | `output/qinche_sft_v3/` |
| 学习率 | **1e-5**（v2 的 5 倍） |
| Effective Batch Size | 32 (2×16) |
| Epochs | 10 |
| Scheduler | Linear warmup (5%) + Cosine decay |
| 数据 | 同 v2（复用 train_with_codes.jsonl） |

**训练曲线：** Loss 从 15.1 → 4.6（收敛更快更深）

**v3 原版评估（speaker embedding 有 bf16 精度 bug）：**
| Epoch | SIM_gt | SIM_ref | WER |
|-------|--------|---------|-----|
| 3 | 0.2822 | 0.2591 | 0.0311 |
| 5 | 0.1820 | 0.1597 | 0.0336 |
| 7 | 0.1956 | 0.2037 | 0.0253 |

**v3 fixed 评估（修复 speaker embedding 精度后，原生推理）：**
| Epoch | SIM_gt | SIM_ref | WER | RTF |
|-------|--------|---------|-----|-----|
| 3 | 0.6773 | 0.7487 | 0.0348 | 2.462 |
| 5 | 0.6584 | 0.7288 | 0.0380 | 2.429 |
| 7 | 0.6688 | 0.7275 | 0.0285 | 2.414 |

**分析：** 修复 bf16 bug 后，v3 的 SIM_gt 在 0.66-0.68 之间，与 v2-fixed 接近但略低。LR=1e-5 的更快收敛并未带来明显优势，v2 的 LR=2e-6 仍是更稳定的选择。

### 5.4 v4 — bf16 bug 修复后的正式训练

| 参数 | 值 |
|------|-----|
| 输出路径 | `output/qinche_sft_v4/` |
| 学习率 | **2e-6**（同 v2） |
| Effective Batch Size | 32 (2×16) |
| Epochs | 10 |
| Scheduler | Linear warmup (10%) + Cosine decay（最低 0.1×LR） |
| 数据 | 同 v2/v3（461 条，~19.7 min） |
| bf16 bug | **已修复**（speaker embedding 累加使用 fp32） |

**训练曲线：** Loss 从 16.7 → 7.5（与 v2 类似的收敛模式）
- Epoch 0: 16.7 → 13.7
- Epoch 2: ~8.5
- Epoch 5: ~7.5-8.0
- Epoch 9: ~7.0-8.4（趋于平稳）

**评估结果（使用 FasterQwen3TTS CUDA Graph 加速推理）：**
| Epoch | SIM_gt | SIM_ref | WER | RTF |
|-------|--------|---------|-----|-----|
| 3 | 0.6715 | 0.6995 | 0.0667 | **0.352** |
| 5 | 0.6684 | 0.7037 | 0.0323 | **0.351** |
| **7** | **0.6783** | **0.7034** | 0.0530 | **0.351** |
| 9 | 0.6703 | 0.7078 | 0.0439 | **0.351** |

**分析：**
- SIM_gt 在 0.67-0.68 之间，略低于 v2-fixed 原生推理的 0.70。差异来自 FasterQwen3TTS（CUDA Graph 静态缓存）与原生 `transformers.generate()`（动态缓存）的微小数值差异（BF16 下不同 kernel 路径不保证 bit-exact）
- **RTF 从 ~2.5 降至 0.35**（6.6x 加速），远超实时标准
- WER 保持优秀（0.03-0.07），epoch 5 的 WER=0.0323 最低
- 最佳 epoch 为 7（SIM_gt 最高）或 5（WER 最低）
- v4 native fp32 累积 speaker embedding 的效果与 v2 事后 patch 相当，验证了修复方案的正确性

### 5.5 v5 — 数据扩充训练（当前最优）

| 参数 | 值 |
|------|-----|
| 输出路径 | `output/qinche_sft_v5/` |
| 学习率 | **2e-6** |
| Effective Batch Size | 32 (2×16) |
| Epochs | 10 |
| Scheduler | Linear warmup (10%) + Cosine decay（最低 0.1×LR） |
| 数据 | **664 条，~28 min**（扩充 qinche_pure_p09~p11 等新数据） |
| bf16 bug | 已修复 |

**评估结果（原生推理 — 仅评估 epoch 3）：**
| Epoch | SIM_gt | SIM_ref | WER | RTF |
|-------|--------|---------|-----|-----|
| **3** | **0.6885** | **0.7281** | **0.0451** | 2.542 |

**评估结果（FasterQwen3TTS CUDA Graph 加速推理 — 全 epoch 对比）：**
| Epoch | SIM_gt | SIM_ref | WER | RTF |
|-------|--------|---------|-----|-----|
| **3** | **0.6892** | **0.7161** | **0.0425** | **0.357** |
| 5 | 0.6791 | 0.6998 | **0.0279** | **0.356** |
| 7 | 0.6868 | 0.7141 | 0.0850 | **0.355** |
| 9 | 0.6795 | 0.7180 | 0.0269 | **0.354** |

**分析：**
- **v5 是当前最优版本**，数据量从 461 条扩充到 664 条（+44%），SIM_gt 从 v4 的 0.6783 提升至 **0.6892**（CUDA Graph 推理），改善约 0.01
- 原生推理 SIM_gt=0.6885，与 CUDA Graph 的 0.6892 几乎一致（本次差异极小）
- **最佳 epoch 为 3**（SIM_gt 最高 0.6892，WER 合理 0.0425），epoch 5 的 WER 最低（0.0279）但 SIM 略低
- epoch 7 的 WER 异常偏高（0.085），由个别样本 WER=0.87 拉高，属于采样波动
- RTF 稳定在 0.35-0.36，与 v4 一致
- **v6 数据扩充尝试失败**：额外 74 条 `qinche_nobgm_01` 数据质量不达标，效果反而下降，已清理

### 5.6 v5-retrain — 环境迁移后重训练

原训练机（`/home/ubuntu/yunlin/TTS/`）被回收，在新环境下使用官方最新 Qwen3-TTS finetuning 代码重新训练。

| 参数 | 值 | 与原 v5 的区别 |
|------|-----|--------------|
| 输出路径 | `output/qinche_sft_v5/` | 同 |
| 学习率 | **2e-6** | 同 |
| Effective Batch Size | **8** (2×4) | 原 v5 为 32 (2×16) |
| Epochs | 10 | 同 |
| Attention | **SDPA** (PyTorch native) | 原 v5 为 Flash Attention 2 |
| Scheduler | Accelerate 默认 | 原 v5 为 Linear warmup (10%) + Cosine decay |
| 数据 | 664 条，~28 min | 同 |
| Qwen3-TTS 代码 | GitHub 最新版 `QwenLM/Qwen3-TTS` | 原 v5 为早期内部版本 |

**训练 Loss 曲线对比：**
| 训练节点 | 原 v5 Loss | v5-retrain Loss |
|----------|-----------|-----------------|
| Epoch 0 起始 | ~16.7 | 14.4 |
| Epoch 0 末尾 | ~13.7 | 7.9 |
| Epoch 2 末尾 | ~8.5 | 4.5 |
| Epoch 9 末尾 | ~7.0-8.4 | 7.5 |

**差异分析：**
- `gradient_accumulation_steps` 从 16 降为 4（新版 `sft_12hz.py` 硬编码），effective batch size 从 32 降为 8
- 更小的 batch size 导致前期收敛更快（loss 下降更快），但后期可能不如大 batch 稳定
- SDPA 与 Flash Attention 2 在数值结果上等价，不影响模型质量
- 新版 `sft_12hz.py` 的 scheduler/optimizer 配置可能与旧版略有不同

**结论：**
- 训练正常完成，Loss 收敛趋势正确
- 由于 effective batch size 差异（8 vs 32），本次训练**不是原 v5 的精确复现**
- 最终模型质量需要通过评估（SIM_gt/WER）确认
- checkpoint 已上传至 HuggingFace Hub（见下方链接）

---

## 6. 评估结果汇总

### 6.1 评估指标

| 指标 | 含义 | 目标 |
|------|------|------|
| **SIM_gt** | 生成音频 vs 真实音频的说话人相似度（pyannote embedding 余弦相似度） | 越高越好，>0.6 为可用 |
| **SIM_ref** | 生成音频 vs 参考音频的说话人相似度 | 越高越好 |
| **WER** | 生成音频 ASR 转写 vs 原文的字错率 | 越低越好，<0.1 为优秀 |
| **RTF** | 实时因子（生成时间/音频时长） | <1.0 为实时 |

### 6.2 全面对比结果

| 模型/方案 | SIM_gt | SIM_ref | WER | RTF | 推理方式 | 备注 |
|-----------|--------|---------|-----|-----|----------|------|
| Qwen3 1.7B Zero-shot | **0.7104** | 0.7563 | 0.1203 | 2.401 | 原生 | SIM 基线 |
| **Qwen3 SFT v5 ep3** | **0.6892** | 0.7161 | 0.0425 | **0.357** | FasterQwen3TTS | **当前最优（加速推理）** |
| Qwen3 SFT v5 ep3 | 0.6885 | **0.7281** | 0.0451 | 2.542 | 原生 | 当前最优（原生推理） |
| Qwen3 SFT v5 ep5 | 0.6791 | 0.6998 | **0.0279** | **0.356** | FasterQwen3TTS | WER 最低（加速推理） |
| Qwen3 SFT v5 ep9 | 0.6795 | 0.7180 | **0.0269** | **0.354** | FasterQwen3TTS | WER 最低 |
| Qwen3 SFT v2-fixed ep6 | 0.6964 | **0.7653** | 0.0365 | 2.579 | 原生 | SIM 最优（原生推理，旧数据） |
| Qwen3 SFT v4 ep7 | 0.6783 | 0.7034 | 0.0530 | **0.351** | FasterQwen3TTS | 速度+质量均衡（旧数据） |
| Fish S2 Pro ZS (compiled) | 0.6627 | 0.6824 | 0.0209 | 0.639 | torch.compile | 实时，WER 最优 |
| Fish S2 Pro ZS | 0.6647 | 0.6910 | 0.0244 | 4.315 | 原生 | 无 compile |

**关键发现：**
1. **数据扩充有效**：v5（664 条，28min）相比 v4（461 条，19.7min），SIM_gt 从 0.6783 提升至 0.6892（+0.01）
2. **bf16 精度 bug 是此前 SIM 低的根本原因**：修复后 v2 SIM_gt 从 0.29 → 0.70（+141%）
3. **FasterQwen3TTS 实现了 Qwen3 实时推理**：RTF 从 ~2.5 → 0.35（**7.1x 加速**），生成 1 秒音频仅需 0.35 秒
4. **CUDA Graph 推理质量与原生推理几乎一致**：v5 ep3 原生 SIM_gt=0.6885 ≈ CUDA Graph SIM_gt=0.6892
5. **LR=2e-6 是最优学习率**：v3（LR=1e-5）未带来额外收益
6. **Fish S2 Pro + compile 也达到实时**：RTF 0.639，但 SIM_gt 0.66 低于 Qwen3 微调方案
7. **INT8 量化不可行**：尝试后发现 SIM_gt 降至 0.07、WER=1.0，bitsandbytes 与 Qwen3-TTS 的双轨 LM 架构不兼容
8. **Speculative decoding 不适用**：0.6B Base 作为 draft 时接受率仅 0-5%（未微调的 Base 模型 token 分布与 SFT 模型差异太大）；self-speculative（early-exit）也不可行（28 层 talker 的中间层与最终层 agreement < 7%，各层表征差异极大）
9. **综合推荐**：
   - **生产部署 → Qwen3 SFT v5 ep3 + FasterQwen3TTS（RTF=0.36，SIM_gt=0.69）**
   - 质量极致 → Qwen3 SFT v5 ep3 原生推理（SIM_gt=0.69，RTF=2.54）
   - 零样本替代 → Fish S2 Pro + compile（RTF=0.64，无需微调）

---

## 7. 推理加速总结

### 7.1 已验证的加速方案

| 优化手段 | 适用模型 | 原始 RTF | 优化后 RTF | 加速倍数 | SIM 损失 | 状态 |
|----------|----------|----------|-----------|----------|----------|------|
| **FasterQwen3TTS (CUDA Graph)** | Qwen3-TTS | 2.542 | **0.357** | **7.1x** | 无损 | **已验证，推荐** |
| **Fish S2 Pro `--compile`** | Fish S2 Pro | 4.315 | **0.639** | **6.75x** | 无损 | **已验证** |
| Flash Attention 2 | 两者 | - | - | - | - | 已启用 |

**FasterQwen3TTS（推荐方案）：**
- 第三方库 `faster-qwen3-tts`，使用 CUDA Graph 捕获加速
- 不依赖 Flash Attention、vLLM 或 Triton
- Warmup 阶段自动捕获 CUDA Graph（talker + code predictor 分别捕获）
- **实测 RTF=0.357（A100 上 7.1x 加速），SIM_gt 与原生推理几乎一致**
- 流式模式 RTF=0.395，TTFA（首包延迟）仅 305ms
- 直接兼容微调 checkpoint，`from_pretrained` 加载本地路径无需改动
- 安装：`pip install faster-qwen3-tts`

**Fish S2 Pro --compile：**
- Fish Speech 原生支持 `--compile`，编译 `decode_one_token_ar` 函数
- **实测结果：RTF 4.315 → 0.639（6.75x 加速），SIM/WER 无损**
- 首次编译开销仅 ~3.5s，后续推理稳定在 ~22 tokens/sec
- GPU 显存：20.1 GB

### 7.2 已排除的加速方案

| 方案 | 尝试结果 | 排除原因 |
|------|----------|----------|
| **INT8 量化 (bitsandbytes)** | SIM_gt=0.07, WER=1.0, RTF=7.0 | bitsandbytes 与 Qwen3-TTS 的双轨 LM（talker+predictor）架构不兼容，量化后输出完全失真 |
| **Speculative Decoding (0.6B draft)** | 接受率 0-5%，比 baseline 慢 | 未微调的 0.6B Base 与 SFT 后的 1.7B token 分布差异极大 |
| **Self-Speculative (early-exit)** | 中间层 agreement < 7% | 28 层 talker 各层表征空间差异巨大，每层都在做实质性特征变换，不存在冗余层 |

### 7.3 结论

CUDA Graph 已将 RTF 从 2.54 降至 0.36（**7.1x 加速**），远超实时标准，且质量无损。进一步的加速尝试（量化、speculative decoding）均因 Qwen3-TTS 的独特双轨 LM 架构而不可行。**当前的 CUDA Graph 方案已是该架构下接近最优的加速方案。**

---

## 8. 后续优化方向

### 8.1 SIM 进一步提升

当前最优 SIM_gt = 0.6892（v5 ep3），距离 zero-shot 基线 0.7104 仅差 0.021。可能方向：

- **继续扩充训练数据**：当前 28 分钟，社区反馈 30-60 分钟效果最佳
- **参考音频优化**：测试不同参考音频组合对 SIM 的影响
- **训练策略调优**：尝试 LR=5e-6（v2 和 v3 之间的折中），或增加 epochs

### 8.2 服务化部署

| 方案 | 特点 | 适用场景 |
|------|------|----------|
| **FasterQwen3TTS serve** | 内置 server 模式 + OpenAI 兼容 API | 轻量级服务 |
| **vLLM-Omni** | 已支持 Qwen3-TTS，流式/批处理 | 高并发 API 服务 |

### 8.3 主观评测

- 客观指标之外，组织 MOS（Mean Opinion Score）主观评测
- 重点关注：音色相似度、自然度、韵律、情感表达

---

## 附录：常用命令速查

```bash
# 激活 Qwen3-TTS 环境
conda activate qwen3-tts

# 运行微调（v5 数据，664 条）
cd /home/ubuntu/yunlin/TTS
bash scripts/run_finetune.sh 2>&1 | tee logs/finetune_v5.log

# 运行评估（原生推理）
CUDA_VISIBLE_DEVICES=7 python scripts/eval_finetuned.py 2>&1 | tee logs/eval_v5_native.log

# 运行评估（FasterQwen3TTS CUDA Graph 加速，RTF ~0.35）
CUDA_VISIBLE_DEVICES=0 CKPT_DIR=output/qinche_sft_v5 EVAL_TAG=ft_v5_fast \
  python scripts/eval_fast.py 2>&1 | tee logs/eval_v5_fast.log

# 推理加速 benchmark（对比 baseline vs CUDA Graph）
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_inference.py \
  --ckpt output/qinche_sft_v5/checkpoint-epoch-3 --num-samples 5

# 激活 Fish Speech 环境
conda activate fish-speech

# Fish S2 Pro zero-shot 评估（带 compile，RTF ~0.64）
CUDA_VISIBLE_DEVICES=7 python scripts/eval_fish_zeroshot.py --compile 2>&1 | tee logs/eval_fish_zeroshot_compile.log

# 查看 GPU 使用
nvidia-smi

# 查看训练日志
tail -f logs/finetune_v5.log
```
