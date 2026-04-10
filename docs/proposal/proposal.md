# Project Proposal: Context-Aware Emotional TTS for Game AI Dubbing

**Course:** MAIE5531 GenAI and LLM  
**Date:** March 27, 2026

## Team Members

| Full Name | Student ID | HKUST Email |
|-----------|-----------|-------------|
| HE Yunlin | 21270701 | yheec@connect.ust.hk |
| WANG Yifu | 20978112 | ywangqy@connect.ust.hk |
| SU Ziyao | 21272577 | zsubc@connect.ust.hk |
| HAN Shipeng | 21270775 | shanaq@connect.ust.hk |

---

## English Version

**Problem Statement:**
Voice acting is a critical but expensive component of game production. In our previous work, we built a TTS voice cloning system for a game character by fine-tuning Qwen3-TTS on 664 audio samples (28 minutes), achieving a speaker similarity of SIM_gt = 0.69 and real-time inference at RTF = 0.36. While the cloned voice faithfully reproduces the character's timbre, it generates speech in a "flat" manner — the synthesized voice sounds like the character, but not like the character *in a specific dramatic scene*. For narrative-driven games such as otome games, emotional expressiveness aligned with scene context is essential for player immersion.

**Objective:**
We will build a **Context-Aware Emotional TTS** system for game AI dubbing. The system takes game scripts (dialogue + scene descriptions) as input and generates emotionally appropriate character voice, automatically adapting tone, pace, and emotional intensity to match the dramatic context — acting as an "AI Emotional Director."

**Competitors:**
Existing TTS solutions (ElevenLabs, Azure TTS, CosyVoice) focus on voice quality and timbre cloning but lack scene-level emotional understanding. Fish Audio supports emotion tags but requires manual annotation per line. Our solution is unique in using LLMs as an automated emotional director that reads the game script, retrieves character personality via RAG, tracks emotional arcs across dialogue turns, and generates context-aware control signals for TTS — eliminating manual emotional annotation entirely. A live demo of our previous voice cloning results is available at: [https://qinche.darkdark.me](https://qinche.darkdark.me)

**Proposed Solution:**
1. **LLM Context Engine:** Use GPT-4o/Qwen to analyze game dialogue context (scene, character relationships, emotional arc) and output structured emotional annotations (emotion label, intensity, pace, style).
2. **Character RAG:** Build a character knowledge base (personality, relationships, speech patterns) and retrieve relevant traits to guide LLM emotion analysis.
3. **Emotion Arc Tracking:** Maintain multi-turn dialogue emotion state to ensure natural emotional transitions across consecutive lines.
4. **Emotion-Aware Reference Audio Selection:** Classify training audio by emotion category, dynamically select matching reference clips during inference.
5. **Fine-tuned TTS Generation:** Leverage our pre-trained Qwen3-TTS model (SFT on 664 samples, 28 min of target speaker data, SIM_gt=0.69, RTF=0.36) with context-driven control signals.
6. **Three-way Comparative Evaluation:** No-context baseline vs. manual emotion annotation vs. LLM-automated context, evaluated with both objective metrics (SIM, WER, Emotion Accuracy) and subjective A/B MOS tests.

**Business Value:**
The Chinese gaming market exceeds 300 billion RMB annually, with voice acting costs accounting for 10–15% of production budgets. AI dubbing with context awareness can reduce voice acting costs by over 98%, compress production timelines from months to days, enable rapid script iteration without re-recording, and support dynamic NPC dialogue that cannot be pre-recorded — unlocking new interactive storytelling possibilities.

---

## 中文版本

**问题陈述：**
配音是游戏制作中关键但昂贵的环节。在我们此前的工作中，我们为一个游戏角色构建了 TTS 语音克隆系统，基于 Qwen3-TTS 对 664 条音频样本（28 分钟）进行微调，实现了说话人相似度 SIM_gt = 0.69 和实时推理 RTF = 0.36。尽管克隆的声音高保真地还原了角色音色，但生成的语音缺乏情感变化——合成的声音"听起来像这个角色"，却不像"这个角色在特定剧情场景中的说话方式"。对于以剧情为核心的游戏（如乙女游戏），与场景上下文匹配的情感表达对玩家沉浸感至关重要。

**目标：**
我们将构建一个**上下文感知的情感 TTS 系统**，用于游戏 AI 配音。系统接收游戏剧本（对白 + 场景描述）作为输入，自动调整语气、语速和情感强度以匹配剧情上下文，生成情感恰当的角色语音——充当"AI 情感导演"。

**竞品分析：**
现有 TTS 方案（ElevenLabs、Azure TTS、CosyVoice）聚焦于音质和音色克隆，缺乏场景级情感理解能力。Fish Audio 支持情感标签但需要逐句人工标注。我们的方案独特之处在于：利用 LLM 作为自动化的"情感导演"，阅读游戏剧本、通过 RAG 检索角色人设、追踪对话中的情感弧线，并生成上下文感知的 TTS 控制信号——完全消除人工情感标注。我们此前语音克隆工作的在线演示：[https://qinche.darkdark.me](https://qinche.darkdark.me)

**技术方案：**
1. **LLM 上下文引擎：** 使用 GPT-4o/Qwen 分析游戏对白上下文（场景、角色关系、情感弧线），输出结构化情感标注（情感标签、强度、语速、风格）。
2. **角色知识库 RAG：** 构建角色知识库（性格、人际关系、说话模式），通过检索增强引导 LLM 的情感分析。
3. **情感弧线追踪：** 维护多轮对话的情感状态，确保连续台词间的情感过渡自然。
4. **情感感知的参考音频选择：** 将训练音频按情感类别分类，推理时动态选择情感匹配的参考音频。
5. **微调 TTS 生成：** 基于我们预训练的 Qwen3-TTS 模型（664 条样本 SFT 微调，28 分钟目标说话人数据，SIM_gt=0.69，RTF=0.36），结合上下文驱动的控制信号。
6. **三组对比评估：** 无上下文基线 vs 人工情感标注 vs LLM 自动上下文，使用客观指标（SIM、WER、情感准确率）和主观 A/B MOS 测试综合评估。

**商业价值：**
中国游戏市场年规模超过 3000 亿元，配音成本占制作预算的 10-15%。具备上下文感知的 AI 配音可将配音成本降低 98% 以上，将制作周期从数月压缩至数天，支持快速剧本迭代而无需重新录制，并支持无法预录的动态 NPC 对话——开启全新的互动叙事可能性。

---

## Appendix A: System Architecture

### English

The system comprises five layers:

1. **Input Layer.** Accepts game script (dialogue + scene description) and maintains a dialogue history buffer.

2. **Character RAG Layer.** A vector database (ChromaDB/FAISS) stores a structured character knowledge base covering personality, relationships, catchphrases, and emotional patterns. Before each LLM call, the scene description queries top-k relevant character fragments, injected into the LLM prompt.

3. **LLM Context Engine.** The core innovation. An LLM (GPT-4o / Qwen2.5) receives the current line, surrounding dialogue, scene description, retrieved character context, and recent emotion states. It outputs structured JSON:
   ```json
   {"emotion": "gentle_reassuring", "intensity": 0.6, "pace": "slow",
    "style": "tender with slight fatigue", "ref_emotion_category": "tender"}
   ```
   An emotion arc tracker maintains a sliding window of recent states, ensuring natural transitions.

4. **TTS Generation Layer.** Two parallel pathways:
   - **Qwen3-TTS (primary):** `ref_emotion_category` selects reference audio from the matching emotion bucket, directly influencing speaking style.
   - **Fish Audio S2 Pro (comparison):** Emotion maps to inline `[tag]` annotations, validating the engine's portability.

5. **Output & Evaluation Layer.** Three-way comparison (no-context / manual / LLM-context) with objective metrics + subjective listening tests. A visualization dashboard shows the full pipeline state.

### 中文

系统由五层组成：

1. **输入层：** 接收游戏剧本（对白 + 场景描述），维护对话历史缓冲区。

2. **角色 RAG 层：** 向量数据库（ChromaDB/FAISS）存储结构化角色知识库，涵盖性格、人际关系、口头禅和情感模式。每次 LLM 调用前，用场景描述检索 top-k 相关角色片段注入 prompt。

3. **LLM 上下文引擎：** 核心创新层。LLM 接收当前台词、上下文、场景描述、检索到的角色特征及近 5-10 轮情感状态，输出结构化 JSON 标注。情感弧线追踪器通过滑动窗口维护情感连续性。

4. **TTS 生成层：** 双路径——Qwen3-TTS 通过情感参考音频选择影响说话风格；Fish Audio S2 Pro 通过内联情感标签对比验证。

5. **输出与评估层：** 三组对比实验 + 客观指标 + 主观听感测试，可视化 Dashboard 实时展示全流程。

---

## Appendix B: Technical Feasibility Analysis / 技术可行性分析

### B.1 LLM Context Engine — Feasibility: High / 可行性：高

- GPT-4o / Qwen2.5 Chinese game script comprehension is mature; structured JSON output via function calling is production-ready; few-shot prompts achieve >85% accuracy on emotion classification.
- Cost: ~$0.005/line (GPT-4o) or near-zero (local Qwen2.5-72B on our A100 cluster).
- **Risk:** Insufficient nuance for specific characters. **Mitigation:** RAG injects character-specific knowledge.

### B.2 Character RAG — Feasibility: High / 可行性：高

- RAG is the most mature LLM augmentation technique. Target character has extensive community wikis (~5,000-10,000 words).
- Implementation: Collect material → structure into personality/relationships/emotional_patterns/catchphrases → embed with bge-large-zh → store in ChromaDB → retrieve top-3 per query.

### B.3 Emotion Arc Tracking — Feasibility: Medium-High / 可行性：中高

- Sliding window of last 5-10 turns, each recording `{text, emotion, intensity}`.
- Milestone: prompt-based tracking. Final: emotion state machine with soft transition constraints.
- **Risk:** Non-sequential dialogue (branching). **Mitigation:** Manual emotion state reset at branch points.

### B.4 Emotion-Aware Reference Audio Selection — Feasibility: High / 可行性：高

- Classify 664 training samples into 6 emotion buckets (tender/calm/playful/intense/cold/intimate) via LLM text analysis.
- Each bucket: 3-5 high-quality candidates. Inference selects by `ref_emotion_category`.
- **Key assumption validated:** Different reference clips produce measurably different speaking styles in Qwen3-TTS.

### B.5 Visualization Dashboard — Feasibility: High / 可行性：高

- Existing React + TypeScript + Vite frontend on `master` branch, deployed to Cloudflare.
- Extensions: scene input panel, real-time LLM analysis visualization, reference audio waveform, A/B playback.

### B.6 Evaluation Framework — Feasibility: High / 可行性：高

- **Objective:** SIM (pyannote), WER (WhisperX + jiwer), Emotion Accuracy (emotion2vec) — all automated, code reused from previous work.
- **Subjective:** A/B preference test (10+ evaluators, 20 pairs), MOS 5-point scale on naturalness, emotional appropriateness, character consistency.

---

## Appendix C: Previous Work Summary / 前期工作总结

Our prior voice cloning effort serves as the foundation:

- **Data:** 664 training samples (~28 min), processed through Demucs → Pyannote → Silero-VAD → WhisperX → RMS normalization.
- **Model:** Qwen3-TTS-12Hz-1.7B-Base, full SFT (LR=2e-6, batch=32, cosine decay + 10% warmup).
- **Best result:** v5 epoch-3 — SIM_gt=0.69, WER=0.04, RTF=0.36 (FasterQwen3TTS CUDA Graph, 7.1x speedup).
- **Comparison:** Fish Audio S2 Pro zero-shot — SIM_gt=0.66, WER=0.02, RTF=0.64.
- **Key finding:** Timbre is well-cloned; emotional expressiveness remains flat — motivating this project.

---

## Appendix D: Technology Stack / 技术栈

- **LLM:** OpenAI GPT-4o / Qwen2.5-72B (local, 8×A100-80GB)
- **RAG:** ChromaDB + bge-large-zh embeddings
- **TTS (primary):** Qwen3-TTS 1.7B fine-tuned v5 + FasterQwen3TTS (CUDA Graph, RTF=0.36)
- **TTS (comparison):** Fish Audio S2 Pro (inline [tag] emotion control)
- **Frontend:** React + TypeScript + Vite + Recharts; Cloudflare Workers
- **Backend:** A100 GPU server (TTS inference); Cloudflare Workers (API proxy)
- **Evaluation:** pyannote.audio (SIM), WhisperX (ASR), jiwer (WER), emotion2vec (SER)
- **Languages:** Python (backend/ML), TypeScript (frontend)

---

## Appendix E: Project Timeline / 项目时间线

- **Week 1-2 (Mar 28 - Apr 10):** Build character KB + RAG; design few-shot prompts (6 emotion categories); classify 664 samples; implement basic pipeline.
- **Week 3 (Apr 11 - Apr 17):** Integrate emotion arc tracking; 10-sentence end-to-end test; milestone report with screenshots.
- **Week 4-5 (Apr 18 - May 1):** Extend frontend dashboard; run three-way evaluation; subjective A/B test (10+ evaluators).
- **Week 6 (May 2 - May 10):** Final report (4-6 pages); 3-minute demo video; source code submission.
