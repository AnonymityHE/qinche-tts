// ============================================================
// TTS evaluation data extracted from /home/ubuntu/yunlin/TTS/eval/
// ============================================================

export interface ModelResult {
  name: string
  label: string
  sim_gt: number
  sim_ref: number
  wer: number
  rtf: number
  color: string
}

export interface EpochResult {
  epoch: number
  sim_gt: number
  wer: number
}

export interface AudioSample {
  id: number
  text: string
  // R2 URLs will be filled in after upload; relative paths for local dev
  urls: Record<string, string>
}

// Main model comparison (best checkpoints)
export const modelResults: ModelResult[] = [
  {
    name: 'fish_s2_zeroshot',
    label: 'Fish Audio S2 Pro (Zero-shot)',
    sim_gt: 0.6647,
    sim_ref: 0.6910,
    wer: 0.0244,
    rtf: 4.315,
    color: '#f59e0b',
  },
  {
    name: 'fish_s2_compile',
    label: 'Fish Audio S2 Pro (Compiled)',
    sim_gt: 0.6635,
    sim_ref: 0.6887,
    wer: 0.0245,
    rtf: 1.823,
    color: '#f97316',
  },
  {
    name: 'qwen3_v5_native',
    label: 'Qwen3-TTS v5 (Native)',
    sim_gt: 0.6892,
    sim_ref: 0.7161,
    wer: 0.0425,
    rtf: 1.12,
    color: '#8b5cf6',
  },
  {
    name: 'qwen3_v5_fast',
    label: 'Qwen3-TTS v5 (CUDA Graph)',
    sim_gt: 0.6892,
    sim_ref: 0.7161,
    wer: 0.0425,
    rtf: 0.357,
    color: '#3b82f6',
  },
]

// v5 CUDA Graph epoch progression
export const v5EpochData: EpochResult[] = [
  { epoch: 3, sim_gt: 0.6892, wer: 0.0425 },
  { epoch: 5, sim_gt: 0.6791, wer: 0.0073 },
  { epoch: 7, sim_gt: 0.6868, wer: 0.0677 },
  { epoch: 9, sim_gt: 0.6795, wer: 0.0260 },
]

// v4 vs v5 epoch comparison (CUDA Graph)
export const v4v5EpochData = [
  { epoch: 3, v4: 0.6315, v5: 0.6892 },
  { epoch: 5, v4: 0.6600, v5: 0.6791 },
  { epoch: 7, v4: 0.6895, v5: 0.6868 },
  { epoch: 9, v4: 0.6600, v5: 0.6795 },
]

// Pipeline steps
export const pipelineSteps = [
  { id: 1, icon: '🎵', label: 'Audio Collection', sub: 'Bilibili scraping' },
  { id: 2, icon: '🎛️', label: 'Source Separation', sub: 'Demucs' },
  { id: 3, icon: '👤', label: 'Speaker Diarization', sub: 'Pyannote' },
  { id: 4, icon: '✂️', label: 'VAD Segmentation', sub: 'Silero-VAD' },
  { id: 5, icon: '📝', label: 'ASR Transcription', sub: 'WhisperX' },
  { id: 6, icon: '🔊', label: 'Normalization', sub: 'RMS loudness' },
  { id: 7, icon: '🧹', label: 'Speaker Purification', sub: 'Cosine sim filter' },
  { id: 8, icon: '🤖', label: 'SFT Training', sub: 'Qwen3-TTS / Fish S2' },
]

// Audio sample texts (first 6 from eval set)
export const sampleTexts = [
  '但还有名词的解释,只带一种爪子锋利又养不熟的猫,也可以特指一个十分可恶又让人不忍心下手的人。',
  '你今天一直在搜N109区的乌鸦,如何关闭机械鸟的电源,尝试了喂鸭喝水敲鸭脑袋宝石溜鸭,这些我都知道。',
  '慌什么肩上的伤早就愈合了听我的声音这不是好得很',
  '我不喜欢这种感觉,就像有人把整个世界打包卖掉,而我是附赠的赠品。',
  '算了,我就知道你会这么说。',
  '秦彻的声音在夜里特别好听,像是从很远的地方传来的,带着一点点沙。',
]

// Data sources
export const dataSources = [
  { name: 'qinche_pure_p01~p11', samples: 296, type: '纯净单说话人', quality: 'high' },
  { name: 'qinche_02', samples: 77, type: '多说话人混合', quality: 'medium' },
  { name: 'qinche_call_p01~p06', samples: 64, type: '通话场景录音', quality: 'medium' },
  { name: 'qinche_01', samples: 38, type: '混合音频', quality: 'low' },
]
