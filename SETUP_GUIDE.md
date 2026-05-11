# Context-Aware Emotional TTS — 配置与运行指南

> 适用分支：`project-v2`
> 最后更新：2026-04-10

---

## 目录

- [1. 环境搭建](#1-环境搭建)
- [2. API 配置（OpenRouter / OpenAI）](#2-api-配置openrouter--openai)
- [3. 模型权重放置](#3-模型权重放置)
- [4. 初始化 RAG 知识库](#4-初始化-rag-知识库)
- [5. 运行端到端 Pipeline](#5-运行端到端-pipeline)
- [6. 运行评估（新 Eval）](#6-运行评估新-eval)
- [7. 启动 Web 服务（FastAPI + 前端）](#7-启动-web-服务fastapi--前端)
- [8. 常见问题](#8-常见问题)

---

## 1. 环境搭建

### 1.1 Python 环境

```bash
conda create -n emotional-tts python=3.12 -y
conda activate emotional-tts
pip install -r requirements.txt
```

**额外依赖（GPU 推理时需要）：**

```bash
# Qwen3-TTS 推理
pip install qwen-tts

# CUDA Graph 加速（可选，推荐）
pip install faster-qwen3-tts

# Flash Attention 2（可选，推荐）
pip install flash-attn --no-build-isolation
```

### 1.2 前端环境（可选，仅需运行 Web UI 时）

```bash
cd frontend
npm install
```

---

## 2. API 配置（OpenRouter / OpenAI）

项目使用 **OpenAI Python SDK** 调用 LLM（GPT-4o），但支持通过 `LLM_BASE_URL` 切换到 OpenRouter 等兼容代理。

### 2.1 创建 `.env` 文件

在项目根目录创建 `.env`（参考 `.env.example`）：

```bash
cp .env.example .env
```

### 2.2 方案 A：通过 OpenRouter（推荐，便宜且稳定）

OpenRouter 是一个 LLM API 代理，兼容 OpenAI SDK，只需改 `base_url` 和 API key。

```env
OPENAI_API_KEY=sk-or-v1-你的openrouter密钥
LLM_BASE_URL=https://openrouter.ai/api/v1
```

**获取 OpenRouter API Key：**
1. 访问 https://openrouter.ai/keys
2. 注册/登录后创建 API Key
3. 充值 $5-10 即可跑完整个项目

**Provider 选择：** 代码中 `model="gpt-4o"`，OpenRouter 会自动路由到 OpenAI 的 GPT-4o。OpenRouter 控制台可以在 https://openrouter.ai/settings/preferences 设置偏好的 provider 路由策略（默认自动选最优，不需要手动改）。

**切换模型：** 在 `.env` 中设置 `LLM_MODEL` 即可（代码已支持环境变量覆盖，见 2.4 节）：

```env
# 在 .env 中添加，不设则默认 gpt-4o
LLM_MODEL=minimax/minimax-m2.7
```

### 2.2.1 OpenRouter 可用模型对比

项目的 LLM 用途是中文情感分析 + 结构化 JSON 输出，以下模型均可胜任：

| 模型 | OpenRouter model ID | Input $/M | Output $/M | 中文能力 | 推荐场景 |
|------|-------------------|-----------|------------|---------|---------|
| **GPT-4o** | `openai/gpt-4o` | $2.50 | $10.00 | 优秀 | 追求最优质量 |
| **GPT-4o-mini** | `openai/gpt-4o-mini` | $0.15 | $0.60 | 良好 | 性价比首选 |
| **MiniMax M2.7** | `minimax/minimax-m2.7` | $0.30 | $1.20 | 良好 | 新一代 agentic 模型 |
| **MiniMax M2.5** | `minimax/minimax-m2.5` | $0.12 | $0.99 | 良好 | 便宜够用 |
| **MiniMax M2.5 免费** | `minimax/minimax-m2.5:free` | **$0** | **$0** | 良好 | 免费测试 / 调试 |
| **DeepSeek Chat** | `deepseek/deepseek-chat` | $0.14 | $0.28 | 优秀 | 极致省钱 |
| **Qwen3-235B** | `qwen/qwen3-235b` | $0.40 | $2.20 | 最强 | 中文理解最优 |

**费用估算：** 整个项目（情感分析 + 样本分类 ~664 条）大约消耗 0.5-1M tokens。用 GPT-4o 约 $5-10，用 GPT-4o-mini / MiniMax M2.7 不到 $1，用 MiniMax M2.5 免费版 $0。

> **建议：** 开发调试阶段用 `minimax/minimax-m2.5:free`（免费），正式跑评估换 `openai/gpt-4o-mini` 或 `minimax/minimax-m2.7`（便宜且稳定），报告中标注用的模型即可。

### 2.3 方案 B：直接用 OpenAI 官方 API

```env
OPENAI_API_KEY=sk-你的openai密钥
# LLM_BASE_URL 留空或不设置，SDK 默认连 api.openai.com
```

### 2.4 工作原理

```
.env
 ├── OPENAI_API_KEY  → 传给 OpenAI(api_key=...)
 └── LLM_BASE_URL    → 传给 OpenAI(base_url=...)，为空则用官方地址
```

代码位置：`src/context_engine/analyzer.py`

```python
self._client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("LLM_BASE_URL"),  # None → 默认 OpenAI
)
```

---

## 3. 模型权重放置

模型权重不在 Git 中（`.gitignore` 排除），需手动下载/拷贝。

```
models/
├── qwen3-tts/           # Qwen3-TTS SFT 微调 checkpoint（v5 epoch-3）
│                        # 来源：朋友跑完训练后传给你的权重
├── qwen3-tts-base/      # Qwen3-TTS Base 模型（用于 clone 路径）
│                        # 来源：huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base
├── fish-speech/         # Fish Audio S2 Pro（用于 fish 路径）
│                        # 来源：huggingface-cli download fishaudio/s2-pro
├── bge-large-zh/        # BGE embedding（RAG 用，首次运行会自动下载缓存）
└── emotion2vec/         # emotion2vec（评估 Emotion Acc 用，可选）
```

**最低要求：** 只放 `qwen3-tts/` 就能跑 `qwen` 和 `baseline` 两条路径。其他模型缺失时会自动降级为 mock（生成静音 wav）。

### 下载命令

```bash
# Qwen3-TTS Base（clone 路径需要）
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir models/qwen3-tts-base

# Fish Audio S2 Pro（fish 路径需要）
huggingface-cli download fishaudio/s2-pro --local-dir models/fish-speech

# BGE（首次运行 RAG 时会自动下载到 models/bge-large-zh，也可手动）
huggingface-cli download BAAI/bge-large-zh-v1.5 --local-dir models/bge-large-zh
```

---

## 4. 初始化 RAG 知识库

首次运行前需构建 ChromaDB 向量库：

```bash
python -c "from src.rag.knowledge_base import build_knowledge_base; build_knowledge_base()"
```

这会读取 `data/character_kb/*.md`（秦彻角色知识），分块后写入 `chroma_db/`。只需跑一次，后续复用。

---

## 5. 运行端到端 Pipeline

### 5.1 用测试剧本跑 Pipeline

```bash
# 使用默认后端 (auto + clone_xvec + baseline)
python -m src.pipeline data/scripts/test_scene_01.json

# 指定后端
python -m src.pipeline data/scripts/test_scene_01.json --backends qwen,clone_xvec,fish,baseline

# 批量跑所有测试剧本
for i in 01 02 03 04 05; do
  python -m src.pipeline data/scripts/test_scene_$i.json
done
```

### 5.2 Pipeline 做了什么

```
输入: data/scripts/test_scene_XX.json（游戏剧本）
  → RAG 检索角色知识（ChromaDB + BGE）
  → GPT-4o 情感分析（通过 OpenRouter/OpenAI）
  → Emotion Arc 追踪
  → 选择情感参考音频（emotion_buckets.json）
  → TTS 生成（多后端并行）
输出: output/<backend>/line_XXX.wav + output/pipeline_results.json
```

### 5.3 可用的 TTS 后端

| 后端 | 模型 | 需要的权重 | 说明 |
|------|------|-----------|------|
| `qwen` | Qwen3-TTS SFT | `models/qwen3-tts/` | 微调模型 + instruct 情感控制 |
| `baseline` | Qwen3-TTS SFT | `models/qwen3-tts/` | 同 qwen 但不加情感指令，作对照组 |
| `clone` | Qwen3-TTS Base | `models/qwen3-tts-base/` | 单条参考音频 ICL |
| `clone_xvec` | Qwen3-TTS Base | `models/qwen3-tts-base/` | 每类情感平均 x-vector |
| `clone_blend` | Qwen3-TTS Base | `models/qwen3-tts-base/` | calm↔emotion x-vector 插值 |
| `auto` | SFT + Base | 两个都要 | 智能路由：calm→baseline，其他→clone_blend |
| `auto_quality` | SFT + Base | 两个都要 | 双路生成 + speaker similarity 质量门控 |
| `fish` | Fish S2 Pro | `models/fish-speech/` | inline emotion tag 控制 |

---

## 6. 运行评估（新 Eval）

### 6.1 前提：先跑完 Pipeline 生成音频

确保 `output/` 下已有各后端的 wav 文件：

```
output/
├── auto/           line_000.wav, line_001.wav, ...
├── clone_xvec/     line_000.wav, line_001.wav, ...
├── baseline/       line_000.wav, line_001.wav, ...
└── pipeline_results.json
```

### 6.2 运行多条件对比评估

```bash
# 评估所有条件（默认: qwen, clone, fish, baseline）
python -m src.eval.run_eval

# 只评估特定条件
python -m src.eval.run_eval --conditions auto,clone_xvec,baseline

# 跳过 emotion2vec（如果没装 funasr 或没有 emotion2vec 权重）
python -m src.eval.run_eval --conditions auto,clone_xvec,baseline --skip-emotion

# 指定输出目录和报告路径
python -m src.eval.run_eval \
  --output-dir output \
  --ref-audio-dir data/ref_audio \
  --test-manifest data/test_manifest.jsonl \
  --conditions auto,clone_xvec,baseline \
  --report-path output/eval_report.json
```

### 6.3 评估指标

| 指标 | 说明 | 依赖 |
|------|------|------|
| **SIM_ref** | 生成音频 vs 参考音频集的说话人相似度 | pyannote.audio |
| **SIM_gt** | 生成音频 vs 真实音频的说话人相似度 | pyannote.audio |
| **WER** | ASR 转写 vs 原文的字错率 | whisperx + jiwer |
| **Emotion Acc** | 音频情感分类 vs LLM 预测标签的匹配率 | funasr + emotion2vec（可选） |

### 6.4 查看历史评估结果（不需要 GPU）

```bash
# 汇总 archive 中的所有历史评估结果
python -m src.eval.analyze_existing

# 导出为 JSON
python -m src.eval.analyze_existing --output docs/eval_summary.json
```

---

## 7. 启动 Web 服务（FastAPI + 前端）

### 7.1 启动后端

```bash
# 开发模式（热重载）
python -m src.api.server
# 或
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

后端 API：
- `POST /api/analyze` — 情感分析（RAG + LLM + 情感弧线，不生成音频）
- `POST /api/pipeline` — 完整 Pipeline（含 TTS 生成）
- `GET /api/audio/{session_id}/{backend}/{filename}` — 提供生成的音频
- `GET /api/health` — 健康检查

### 7.2 启动前端

```bash
cd frontend
npm run dev
```

前端运行在 http://localhost:5173，开发模式下 `/api` 请求会自动代理到后端 `http://127.0.0.1:8000`。

### 7.3 仅看前端 Demo（不需要后端和 GPU）

前端 `EmotionalTTSPage` 内置了 Demo Mode，使用预设情感数据，不需要后端 API。直接访问 http://localhost:5173 即可体验。

---

## 8. 常见问题

### Q: 没有 GPU 能跑吗？

- **情感分析（`/api/analyze`）**：可以，只需要 OpenAI/OpenRouter API key
- **TTS 生成**：不行，Qwen3-TTS 需要 ~20GB 显存，Fish S2 Pro 需要 ~24GB
- **评估**：`analyze_existing` 不需要 GPU；`run_eval` 的 SIM/WER 需要 GPU（pyannote + whisperx）
- **前端 Demo Mode**：可以，不依赖后端

### Q: OpenRouter 的费用大概多少？

GPT-4o 通过 OpenRouter 的价格约 $2.50/M input tokens + $10/M output tokens。整个项目（情感分析 + 样本分类）预计 $10-15，充 $20 够用。

### Q: 模型权重放在哪里？

所有权重放在 `models/` 目录下，该目录被 `.gitignore` 排除。如果是从朋友那里拿的微调 checkpoint，解压到 `models/qwen3-tts/` 即可。路径对应关系：

| 代码中的路径 | 环境变量覆盖 | 说明 |
|-------------|-------------|------|
| `models/qwen3-tts/` | `QWEN3_TTS_MODEL_PATH` | SFT 微调模型 |
| `models/qwen3-tts-base/` | — | Base 模型（clone 用） |
| `models/fish-speech/` | `FISH_SPEECH_MODEL_PATH` | Fish S2 Pro |
| `models/bge-large-zh/` | `BGE_MODEL_PATH` | BGE embedding |

### Q: Pipeline 跑到一半报 OpenAI API 错误？

检查 `.env` 中的 `OPENAI_API_KEY` 是否正确。如果用 OpenRouter，key 格式为 `sk-or-v1-...`，同时确保 `LLM_BASE_URL=https://openrouter.ai/api/v1`。

### Q: 如何只跑评估不跑 Pipeline？

如果已有之前生成的音频文件，直接跑 `run_eval.py` 指定 `--output-dir` 到对应目录即可。

---

## 快速开始（最小配置）

```bash
# 1. 克隆并切到 project-v2
git clone https://github.com/HKUST-Group/qinche-tts.git
cd qinche-tts
git checkout project-v2

# 2. 安装依赖
conda create -n emotional-tts python=3.12 -y
conda activate emotional-tts
pip install -r requirements.txt
pip install qwen-tts faster-qwen3-tts

# 3. 配置 API key
cp .env.example .env
# 编辑 .env，填入 OpenRouter 或 OpenAI 的 key

# 4. 放置模型权重
# 把微调 checkpoint 放到 models/qwen3-tts/

# 5. 初始化 RAG
python -c "from src.rag.knowledge_base import build_knowledge_base; build_knowledge_base()"

# 6. 跑 Pipeline（生成音频）
python -m src.pipeline data/scripts/test_scene_01.json

# 7. 跑评估
python -m src.eval.run_eval --conditions auto,clone_xvec,baseline --skip-emotion
```
