# Context-Aware Emotional TTS — 开发清单

> 基于 proposal + Fish Audio S2 借鉴分析  
> 创建日期：2026-04-05 | 最后更新：2026-04-05  
> 今日进度：Week 1-2（Mar 28 – Apr 10）

---

## 费用 & 资源总览


| 资源                    | 方案                                             | 费用            |
| --------------------- | ---------------------------------------------- | ------------- |
| **GPT-4o API**        | 用于 LLM Context Engine、情感分类、RAG 推理              | ~$10-15（整个项目） |
| **Fish Audio S2 Pro** | 本地部署（开源），从 GitHub/HuggingFace 下载权重             | 免费            |
| **Qwen3-TTS**         | 朋友跑微调后给 checkpoint                             | 免费（GPU 由朋友提供） |
| **GPU（推理阶段）**         | 1× A100-80GB，Qwen3-TTS 和 Fish Audio S2 Pro 轮流跑 | 取决于已有资源       |


> **唯一的硬性花费：OpenAI API key，充 $20 绰绰有余。**  
> Fish Audio S2 Pro 是完全开源的（[GitHub](https://github.com/fishaudio/fish-speech) / [HuggingFace](https://huggingface.co/fishaudio/s2-pro)），本地部署不需要 API 费用。

---

## 0. 前置准备

### 0.1 模型权重获取（由朋友负责）

- 从 `project` 分支获取微调代码和配置
- 在 A100 上跑 Qwen3-TTS v5 的 SFT 微调训练
- 导出 v5 epoch-3 最佳 checkpoint（SIM_gt=0.69, WER=0.04）
- 将模型权重传给你（网盘 / R2 / HuggingFace）
- 验证 checkpoint 能正常加载推理

### 0.2 Fish Audio S2 Pro 本地部署

- Clone 开源仓库：`git clone https://github.com/fishaudio/fish-speech`
- 从 HuggingFace 下载 S2 Pro 权重：`huggingface-cli download fishaudio/s2-pro`
- 安装依赖，配置 SGLang 推理引擎
- 用秦彻参考音频测试 zero-shot 语音克隆效果
- 测试 inline emotion tag（如 `[gentle]你还好吗？`）验证情感控制有效性

### 0.3 开发环境搭建

- 本地 Python 环境：`torch`, `transformers`, `chromadb`, `openai`, `pyannote.audio`, `jiwer` 等
- 确认 GPU 推理环境（至少 1× A100-80GB，Qwen3-TTS 和 Fish Audio S2 Pro 轮流跑推理）
- 配置 OpenAI GPT-4o API key（用于 LLM Context Engine，~$10-15 整个项目够用）
- Clone `project` 分支到本地（含训练数据、评估脚本等）
- 创建 `.env` 文件存放 API key（不要提交到 git）

---

## 1. 角色知识库 + RAG Pipeline

**目标：** 构建秦彻角色知识库，为 LLM 提供角色人设上下文

### 1.1 知识库内容收集

- 收集秦彻角色资料（wiki、性格分析、剧情总结等，目标 5,000-10,000 字）
- 按四个维度整理结构化文档：
  - `personality` — 性格特征（例：表面冷漠、内心温柔、极度自律等）
  - `relationships` — 人际关系（与女主、与其他角色的关系）
  - `emotional_patterns` — 情感反应模式（什么场景下会温柔、什么场景下会冷淡等）
  - `catchphrases` — 口头禅、说话风格特征

### 1.2 RAG 实现

- 将结构化文档分块（chunk），每块 200-500 字
- 用 `bge-large-zh` 做 embedding
- 存入 ChromaDB（或 FAISS）
- 实现检索函数：输入场景描述 → 返回 top-3 角色上下文片段
- 编写单元测试：给几个典型场景，验证检索结果合理

**产出：** `rag/` 模块，可被 LLM Context Engine 调用

---

## 2. 训练样本情感分类

**目标：** 将 664 条训练音频按情感分到 6 个桶中

### 2.1 情感分类体系（借鉴 Fish Audio 思路）

- 定义 6 个情感类别：
  - `tender`（温柔） / `calm`（沉稳） / `playful`（俏皮）
  - `intense`（激烈） / `cold`（冷淡） / `intimate`（亲密）
- （可选扩展）参考 Fish Audio 的自然语言标签，给每条样本加更细粒度的描述标签

### 2.2 用 LLM 自动分类

- 从 `project` 分支获取 664 条样本的文本转录（`train_manifest.jsonl`）
- 编写 prompt：输入台词文本 → 输出情感类别 + 置信度 + 自然语言描述
- 用 GPT-4o 批量标注所有 664 条样本
- 人工抽查 50 条，验证标注准确率 > 85%
- 每个桶选出 3-5 条高质量候选作为参考音频

### 2.3 （可选）用 emotion2vec 做交叉验证

- 用 emotion2vec 模型对 664 条**音频**做情感识别
- 与 LLM 文本标注结果对比，找出不一致的样本人工复核

**产出：** `emotion_buckets.json` — 每条样本的情感标签映射；`ref_audio/` — 每个桶的精选参考音频

---

## 3. LLM Context Engine

**目标：** 给定游戏台词 + 场景 → 输出结构化情感标注 JSON

### 3.1 Prompt 设计

- 设计 system prompt，包含：
  - 角色身份定义
  - 情感分类体系说明（6 类 + 自然语言描述）
  - 输出格式规范
- 设计 few-shot 示例（每个情感类别至少 1-2 个示例）
- 整合 RAG 检索结果到 prompt 中

### 3.2 核心函数实现

- `analyze_emotion(line, scene_description, dialogue_history, character_context)` → JSON
- 输出格式包含 emotion, intensity, pace, style, ref_emotion_category, fish_audio_tags
- 支持 GPT-4o API（via OpenRouter）
- 添加重试机制和 JSON 格式校验（含 APITimeoutError 指数退避）

### 3.3 （借鉴 Fish Audio）为 Fish Audio S2 Pro 路径生成 inline tag

- 将 LLM 输出的 emotion/style 映射为 Fish Audio inline tag 格式
- 例如：`emotion: "gentle_reassuring"` → `[温柔][轻松]你还好吗？`

**产出：** `context_engine/` 模块

---

## 4. 情感弧线追踪

**目标：** 维护多轮对话的情感状态，确保情感过渡自然

### 4.1 基础版（prompt-based，用于 Milestone）

- 维护滑动窗口：最近 5-10 轮的 `{text, emotion, intensity}`
- 将历史情感状态序列化后注入 LLM prompt

### 4.2 进阶版（emotion state machine，用于 Final）

- 定义 6 类情感之间的合理转移矩阵（哪些转换是自然的、哪些需要渐变）
- 添加软约束：如果 LLM 输出的情感与前一轮"距离"过大，触发 abrupt transition 警告
- 支持在分支剧情点手动重置情感状态（`tracker.reset()`）

**产出：** `emotion_tracker/` 模块

---

## 5. TTS 生成层

**目标：** 双路径生成 — Qwen3-TTS（主）+ Fish Audio S2 Pro（对比）

### 5.1 Qwen3-TTS 路径（主路径 — SFT CustomVoice + instruct）

- 加载 v5 epoch-3 checkpoint（`qwen_backend.py`）
- 实现情感参考音频选择：`ref_emotion_category` → 从对应桶中选参考音频（intensity 加权）
- 实现 instruct 格式化：LLM style → `"用…的语气说"` 格式
- 集成 FasterQwen3TTS + CUDA Graph 加速（RTF ≤ 0.36）
- 实现 `generate_speech(text, emotion, output_path)` → wav

### 5.1b Voice Clone 路径（Audio-as-Emotion-Prompt）

- 下载 Qwen3-TTS Base 模型用于 voice clone（`clone_backend.py`）
- 方案 A：ICL 模式 — emotion-matched 参考音频 + ref_text
- 方案 B：平均 x-vector — 每情感类别多 clip 平均 speaker embedding
- VoiceClonePromptItem 缓存，避免重复编码

### 5.2 Fish Audio S2 Pro 路径（对比路径）

- 实现 inline tag 生成：LLM emotion JSON → `[tag]文本` 格式（`fish_backend.py`）
- 实现 `generate_speech(text, emotion, ref_audio_path, output_path)` → wav
- 在 A100 上实际部署测试

### 5.3 Baseline（无情感控制）

- 用 Qwen3-TTS CustomVoice 无 instruct 生成（`baseline_backend.py`）
- baseline 复用 qwen_backend 已加载的模型，避免双份内存

**产出：** `tts/` 模块，统一接口 `synthesize(text, emotion_config, backend="qwen"|"fish"|"baseline")`

---

## 6. 端到端 Pipeline 集成

**目标：** 游戏剧本 → 情感语音，一键跑通

### 6.1 Pipeline 编排

- 实现主函数 `run_pipeline()`，完整链路：
  ```
  游戏剧本（JSON）
    → RAG 检索角色上下文（带场景级缓存）
    → LLM Context Engine 分析情感
    → Emotion Arc Tracker 更新状态
    → Intensity 加权参考音频选择
    → TTS 生成（qwen / clone / fish / baseline 四条路径）
    → 输出音频 + 元数据 JSON
  ```
- 准备 5 套测试剧本（`data/scripts/test_scene_01..05.json`）
- 端到端跑通验证

### 6.2 剧本格式定义

- 定义输入 JSON 格式（scene + dialogue list）

### 6.3 FastAPI 后端

- `POST /api/analyze` — 场景分析（RAG + LLM + 情感弧线）
- `POST /api/pipeline` — 完整 pipeline（async，线程池隔离）
- `GET /api/audio/{session_id}/{backend}/{filename}` — 音频服务
- 全局异常处理 + 结构化错误返回

**产出：** `pipeline.py` — 可命令行运行的完整 pipeline

---

## 7. 评估框架

**目标：** 三组对比实验 + 客观 + 主观评估

### 7.1 客观评估（自动化）

- **SIM_gt / SIM_ref**：pyannote embedding cosine similarity（`src/eval/sim.py`）
- **WER**：WhisperX ASR + jiwer（`src/eval/wer.py`）
- **Emotion Accuracy**：emotion2vec 支持（`src/eval/emotion_acc.py`）
- 统一评测框架 `run_eval.py`，支持 `--conditions qwen,clone,fish,baseline`
- 汇总成表格：多条件对比

### 7.2 主观评估

- 准备 20 句对比音频对（Baseline vs LLM-Context）
- 招募 10+ 评估者
- A/B preference test：哪个更好？
- MOS 5 分制打分：自然度、情感恰当性、角色一致性
- 设计评估问卷（Google Form / 问卷星）

### 7.3 消融实验（加分项）

- RAG 有 vs 无
- Emotion Arc Tracking 有 vs 无
- Instruction prompting vs Reference audio selection vs 两者结合

**产出：** `eval/` 目录，包含评估脚本和结果

---

## 8. 前端 Dashboard

**目标：** 扩展 master 分支现有前端，添加 demo 交互功能

### 8.1 新增页面/组件

- **Scene Input Panel**：输入场景描述 + 对话台词（`EmotionalTTSPage`）
- **LLM Analysis Card**：实时显示 LLM 输出的情感标注 JSON
- **RAG Results Panel**：显示检索到的角色上下文片段
- **Emotion Arc Visualization**：Recharts 折线图展示多轮对话的情感变化轨迹
- **Reference Audio Display**：显示选中的参考音频波形
- **A/B Playback**：并排对比不同后端的音频

### 8.2 后端 API

- FastAPI 本地后端（Vite 开发代理到 :8000）
- Cloudflare Workers / 边缘部署方案

**产出：** 更新 `master` 分支前端代码

---

## 9. 报告 & 交付物

### 9.1 Milestone Report（Week 3，Apr 11-17）

- 2-4 张标注截图（pipeline 各步骤的输出可视化）
- 10 句端到端测试结果
- 简要进度说明

### 9.2 Final Report（Week 6，May 2-10）

- 4-6 页正式报告
- 包含：问题定义、方法、实验结果、对比分析、Fish Audio 借鉴讨论
- 三组对比实验的完整数据表格 + 图表

### 9.3 Demo Video（3 分钟）

- 展示：输入剧本 → 系统分析情感 → 生成语音 的完整流程
- 对比播放：无情感 vs 有情感的语音

### 9.4 代码提交

- 整理代码仓库，确保可复现
- 更新 README（环境配置、运行方式、目录结构说明）

---

## 时间线对照


| 周次           | 日期              | 核心任务                                    | 对应清单             |
| ------------ | --------------- | --------------------------------------- | ---------------- |
| **Week 1-2** | Mar 28 - Apr 10 | RAG + 情感分类 + LLM Engine + 基础 Pipeline   | §1, §2, §3, §6   |
| **Week 3**   | Apr 11 - Apr 17 | Emotion Arc + 10 句测试 + Milestone Report | §4, §6.1, §9.1   |
| **Week 4-5** | Apr 18 - May 1  | 前端 + 三组评估 + 主观测试                        | §7, §8           |
| **Week 6**   | May 2 - May 10  | Final Report + Demo Video + 代码整理        | §9.2, §9.3, §9.4 |


> **注意：** §0（模型权重）和 §5（TTS 生成层）是贯穿始终的——等朋友跑完训练给权重后，随时可以接入。在等待期间，可以先用 Fish Audio S2 Pro 本地开源版做开发和测试（它是 zero-shot 的，不需要微调权重就能跑）。

---

## 分工建议


| 成员           | 建议负责模块                                   |
| ------------ | ---------------------------------------- |
| 你（HE Yunlin） | 整体架构 + TTS 集成 + Pipeline 编排（§5, §6）      |
| 跑训练的朋友       | 模型微调 + 权重导出（§0.1）                        |
| 成员 2         | RAG + 角色知识库（§1）                          |
| 成员 3         | LLM Context Engine + Emotion Arc（§3, §4） |
| 成员 4         | 前端 Dashboard + 评估框架（§7, §8）              |
| 全员           | 情感分类标注审核（§2）、主观评估、报告撰写（§9）               |


