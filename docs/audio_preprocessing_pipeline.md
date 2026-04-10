# 音频预处理技术链文档

## 完整流水线概览

```
B站视频 → 下载原始音频 → 人声/伴奏分离 → 说话人分离 → VAD语音切割 → ASR转写 → 数据集构建
```

整个流程分 5 个阶段，使用 6 个核心工具。

---

## 阶段 1：数据下载

**脚本**：`TTS/scripts/download_bilibili.py`

**工具**：`bilibili-api-python` + `httpx` + `ffmpeg`

**流程**：
1. 通过 `bilibili-api-python` 调用 B 站 API 获取视频的音频流 URL
2. 使用 `httpx` 异步流式下载音频（m4a 格式），支持断点重试（最多 3 次）
3. 使用 `ffmpeg` 将 m4a 转换为 WAV（24kHz 单声道）

**数据源**：
| BV号 | 名称 | 说明 |
|---|---|---|
| BV1NjzAB2EPY | qinche_01 | 秦彻相关混剪，~11min |
| BV168UQBeEKp | qinche_02 | 秦彻相关，~35min |
| BV1Z7HTecEBD | qinche_pure_p01~p11 | 纯秦彻语音，11个分P，~27min |

**执行命令**：
```bash
conda activate qwen3-tts
python scripts/download_bilibili.py
```

**产出**：`TTS/data/raw/*.wav`（24kHz 单声道 WAV）

---

## 阶段 2：人声/伴奏分离 (Music Source Separation)

**工具**：`demucs` (htdemucs 模型, by Meta/Facebook Research)

**原理**：深度学习模型将音频分离为 vocals（人声）和 no_vocals（伴奏/BGM/音效），有效去除背景音乐和环境音。

**执行命令**：
```bash
conda activate qwen3-tts
CUDA_VISIBLE_DEVICES=7 python -m demucs --two-stems vocals \
    -o data/demucs_output \
    data/raw/qinche_01.wav data/raw/qinche_02.wav

# 对 pure 系列也执行同样操作
CUDA_VISIBLE_DEVICES=7 python -m demucs --two-stems vocals \
    -o data/demucs_output \
    data/raw/qinche_pure_p01.wav ... data/raw/qinche_pure_p11.wav
```

**关键参数**：
- `--two-stems vocals`：只分离人声和非人声两路
- 模型：`htdemucs`（Hybrid Transformer Demucs）

**产出**：`TTS/data/demucs_output/htdemucs/<name>/vocals.wav`

---

## 阶段 3：说话人分离 (Speaker Diarization)

**工具**：`pyannote-audio` (通过 `whisperx.diarize.DiarizationPipeline`)

**原理**：
1. 使用 `pyannote/segmentation-3.0` 模型检测说话人变化点
2. 使用 `pyannote/embedding` 模型提取每段的说话人向量 (speaker embedding)
3. 聚类算法将各段聚为不同说话人 (SPEAKER_00, SPEAKER_01, ...)
4. 选择占比最大的说话人作为"目标说话人"（dominant speaker）

**适用条件**：
- 对混合说话人音频（`qinche_01`, `qinche_02`）执行 diarization
- 对纯秦彻音频（`qinche_pure_*`）跳过此步骤（代码中 `PURE_VOICE_PREFIXES = ["qinche_pure_"]`）

**代码逻辑**（`preprocess_audio.py` 中的 `diarize_vocals` 函数）：
```python
pipeline = DiarizationPipeline(token=hf_token, device="cuda")
diarize_df = pipeline(audio)  # 返回 DataFrame: [start, end, speaker]
# 统计每个 speaker 的总时长，选 dominant
dominant = max(speaker_durations, key=speaker_durations.get)
```

**依赖**：需要 HuggingFace Token（配置在 `TTS/.env`），并在 HF 上接受 `pyannote/segmentation-3.0` 和 `pyannote/embedding` 的 license。

---

## 阶段 4：VAD 语音活动检测 + 切割

**工具**：`silero-vad`（Silero 团队出品，轻量 VAD 模型）

**原理**：逐帧预测"当前是否有人在说话"，输出语音段的起止时间戳。

**代码逻辑**（`preprocess_audio.py` 中的 `vad_segment` 函数）：
```python
model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad")
get_speech_timestamps = utils[0]

timestamps = get_speech_timestamps(audio_16k, model,
    sampling_rate=16000,
    min_speech_duration_ms=500,    # 最短语音段 0.5s
    min_silence_duration_ms=800,   # 最短静音间隔 0.8s（防止过度切碎）
    threshold=0.5                  # VAD 置信度阈值
)
```

**说话人过滤**（`filter_segments_by_speaker` 函数）：
- 对每个 VAD 切出的段，计算其与各说话人 diarization 标注的重叠
- 只保留与目标说话人（dominant speaker）重叠大于其他说话人的段
- 纯秦彻音频跳过此过滤

**切割参数**：
- 最短段：1.0s（`MIN_DURATION`）
- 最长段：15.0s（`MAX_DURATION`，超长段会被等分切割）
- 输出采样率：24kHz

**产出**：`TTS/data/segments/<source_name>/<source_name>_XXXX.wav`

---

## 阶段 5：ASR 自动语音识别（转写）

**工具**：`whisperx`（基于 `faster-whisper` / `ctranslate2`，使用 `large-v3` 模型）

**原理**：对每个切割好的音频段进行语音识别，输出对应的文本。

**代码逻辑**（`preprocess_audio.py` 中的 `transcribe_segments` 函数）：
```python
model = whisperx.load_model("large-v3", "cuda", compute_type="float16")
for seg in segments:
    audio = whisperx.load_audio(seg["filepath"])
    result = model.transcribe(audio, batch_size=16, language="zh")
    seg["text"] = "".join([s["text"].strip() for s in result["segments"]])
```

**关键参数**：
- 模型：`large-v3`（OpenAI Whisper 的 CTranslate2 优化版）
- 计算类型：`float16`
- 语言：`zh`（中文）
- 重试机制：最多 3 次（应对模型加载偶发失败）

---

## 阶段 6：数据集构建

**代码逻辑**（`preprocess_audio.py` 中的 `select_ref_and_split` + `save_dataset`）：

1. **选择参考音频**：从所有段中选出 5 条时长在 2~10s 的高质量片段作为 voice clone 的参考音频
2. **划分数据集**：
   - Train: ~90%
   - Test: ~10%（最多 20 条）
   - Ref: 5 条参考音频
3. **输出 manifest**：JSONL 格式

**产出**：
```
TTS/data/dataset/train_manifest.jsonl   # {"audio_filepath": ..., "text": ..., "duration": ...}
TTS/data/dataset/test_manifest.jsonl
TTS/data/dataset/ref_manifest.json
TTS/data/ref_audio/ref_00.wav ~ ref_04.wav
```

---

## 统一执行命令

```bash
# 环境
conda activate qwen3-tts

# Step 1: 下载
python scripts/download_bilibili.py

# Step 2: Demucs 人声分离
CUDA_VISIBLE_DEVICES=7 python -m demucs --two-stems vocals \
    -o data/demucs_output data/raw/*.wav

# Step 3-6: 预处理流水线（含 diarization + VAD + ASR + 数据集构建）
CUDA_VISIBLE_DEVICES=7 python scripts/preprocess_audio.py
```

---

## 数据质量保障

1. **Demucs 分离**：去除 BGM、音效等非人声干扰
2. **Speaker Diarization**：识别多说话人场景中的目标说话人
3. **Speaker Filter**：只保留目标说话人占主导的语音段
4. **VAD 切割**：精确到语音边界，避免包含静音
5. **Speaker Embedding 验证**（`check_speaker_similarity.py`）：使用 `pyannote/embedding` 模型计算各源与参考声纹的余弦相似度，确认声音一致性

---

## 依赖清单

| 工具 | 版本 | 用途 |
|---|---|---|
| bilibili-api-python | latest | B站API下载 |
| httpx | latest | 异步HTTP下载 |
| ffmpeg | system | 音频格式转换 |
| demucs (htdemucs) | latest | 人声/伴奏分离 |
| pyannote-audio | latest | 说话人分离 + 声纹embedding |
| silero-vad | latest | 语音活动检测 |
| whisperx (faster-whisper + large-v3) | latest | ASR转写 |
| soundfile | latest | WAV 读写 |
| torchaudio | 2.8+ | 音频加载/重采样 |

---

## 目录结构

```
TTS/
├── scripts/
│   ├── download_bilibili.py          # 阶段1: 下载
│   ├── preprocess_audio.py           # 阶段2-6: 核心流水线
│   ├── check_speaker_similarity.py   # 声纹一致性验证
│   └── purify_dataset.py             # 数据提纯（可选）
├── data/
│   ├── raw/                          # 原始下载的WAV
│   ├── demucs_output/htdemucs/       # demucs分离结果
│   │   └── <source>/vocals.wav
│   ├── segments/                     # VAD切割的语音段
│   │   └── <source>/<source>_XXXX.wav
│   ├── ref_audio/                    # 参考音频
│   ├── test/                         # 测试集音频
│   └── dataset/                      # 最终manifest
│       ├── train_manifest.jsonl
│       ├── test_manifest.jsonl
│       └── ref_manifest.json
├── logs/                             # 各阶段运行日志
├── models/                           # 下载的模型权重
└── .env                              # HF_TOKEN 配置
```
