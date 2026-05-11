import { useState, useCallback, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import {
  ArrowLeft, Send, Loader2, BookOpen, Brain, AudioLines,
  ChevronDown, ChevronUp, Sparkles, Waves,
} from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, ReferenceLine,
} from 'recharts'

const EMOTION_COLORS: Record<string, string> = {
  tender: '#f472b6',
  calm: '#60a5fa',
  playful: '#34d399',
  intense: '#f87171',
  cold: '#94a3b8',
  intimate: '#c084fc',
}

const EMOTION_LABELS: Record<string, string> = {
  tender: '温柔',
  calm: '沉稳',
  playful: '俏皮',
  intense: '激烈',
  cold: '冷淡',
  intimate: '亲密',
}

interface DialogueLine {
  speaker: string
  text: string
}

interface LineResult {
  index: number
  speaker: string
  text: string
  emotion: Record<string, unknown> | null
  rag_context: string[]
  emotion_history: Record<string, unknown>[]
  ref_audio: string | null
  audio: Record<string, string>
  skipped: boolean
}

interface ArcPoint {
  index: number
  speaker: string
  category: string | null
  intensity: number
  emotion?: string
  style?: string
}

interface AnalyzeResponse {
  session_id: string
  results: LineResult[]
  emotion_arc: ArcPoint[]
}

const PRESET_SCENES = [
  {
    label: '战后安慰',
    scene: '战斗结束后的废墟走廊，烟尘还没散尽，女主角跑过来查看秦彻的伤势。',
    dialogue: [
      { speaker: '女主', text: '你没事吧？！让我看看你的伤！' },
      { speaker: '秦彻', text: '慌什么，肩上的伤早就愈合了，听我的声音，这不是好得很。' },
      { speaker: '女主', text: '你每次都这样说……我真的很担心。' },
      { speaker: '秦彻', text: '别妄自菲薄，你比你想象中有用得多，这几天他都会跟着你出门。' },
      { speaker: '女主', text: '那你呢？你能好好休息吗？' },
      { speaker: '秦彻', text: '好吧，林空试试，你的地盘你做主，有事给我打电话。' },
    ],
  },
  {
    label: '深夜通话',
    scene: '深夜，女主角独自在公寓里，窗外下着雨，手机突然响起。',
    dialogue: [
      { speaker: '秦彻', text: '还没睡？窗外的雨声很大，我在电话里都能听到。' },
      { speaker: '女主', text: '嗯……睡不着，有点想事情。' },
      { speaker: '秦彻', text: '什么事让你这么费神？说给我听听。' },
      { speaker: '女主', text: '就是觉得最近什么都做不好。' },
      { speaker: '秦彻', text: '别人怎么说不重要，你心里那把尺才是准绳，相信自己的直觉。' },
    ],
  },
  {
    label: '日常调侃',
    scene: 'N109区的咖啡厅，阳光透过落地窗照进来，秦彻坐在女主对面翻看文件。',
    dialogue: [
      { speaker: '女主', text: '你今天怎么有空来这种地方？' },
      { speaker: '秦彻', text: '有个很久没见的人想在N109约会，除了这个，还能是因为什么？' },
      { speaker: '女主', text: '谁跟你约会了！我就是路过！' },
      { speaker: '秦彻', text: '路过还点了两杯咖啡？你的演技需要提升。' },
    ],
  },
]

function SceneInputPanel({
  scene, setScene, dialogue, setDialogue, onSubmit, loading
}: {
  scene: string
  setScene: (s: string) => void
  dialogue: DialogueLine[]
  setDialogue: (d: DialogueLine[]) => void
  onSubmit: () => void
  loading: boolean
}) {
  const [showPresets, setShowPresets] = useState(false)

  const addLine = () => setDialogue([...dialogue, { speaker: '秦彻', text: '' }])
  const removeLine = (i: number) => setDialogue(dialogue.filter((_, idx) => idx !== i))
  const updateLine = (i: number, field: 'speaker' | 'text', val: string) => {
    const copy = [...dialogue]
    copy[i] = { ...copy[i], [field]: val }
    setDialogue(copy)
  }

  const loadPreset = (idx: number) => {
    const p = PRESET_SCENES[idx]
    setScene(p.scene)
    setDialogue([...p.dialogue])
    setShowPresets(false)
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold text-purple-300 uppercase tracking-wider flex items-center gap-2">
          <BookOpen size={14} /> Scene Input
        </h3>
        <button
          onClick={() => setShowPresets(!showPresets)}
          className="text-xs text-gray-400 hover:text-white flex items-center gap-1 transition-colors"
        >
          Presets {showPresets ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        </button>
      </div>

      <AnimatePresence>
        {showPresets && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
            <div className="flex gap-2 pb-2">
              {PRESET_SCENES.map((p, i) => (
                <button key={i} onClick={() => loadPreset(i)}
                  className="px-3 py-1.5 rounded-full text-xs bg-white/10 hover:bg-purple-600/40 text-gray-300 hover:text-white transition-all border border-white/10 hover:border-purple-500/50"
                >
                  {p.label}
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <textarea
        value={scene}
        onChange={(e) => setScene(e.target.value)}
        placeholder="Describe the scene…  e.g. 战斗结束后的废墟走廊，烟尘还没散尽"
        rows={2}
        className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-purple-500/50 resize-none"
      />

      <div className="space-y-2 max-h-[280px] overflow-y-auto pr-1">
        {dialogue.map((line, i) => (
          <motion.div key={i} layout initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
            className="flex gap-2 items-start"
          >
            <select
              value={line.speaker}
              onChange={(e) => updateLine(i, 'speaker', e.target.value)}
              className="bg-white/5 border border-white/10 rounded-lg px-2 py-2 text-xs text-gray-300 w-20 shrink-0 focus:outline-none focus:ring-1 focus:ring-purple-500/50"
            >
              <option value="秦彻">秦彻</option>
              <option value="女主">女主</option>
              <option value="旁白">旁白</option>
            </select>
            <input
              value={line.text}
              onChange={(e) => updateLine(i, 'text', e.target.value)}
              placeholder="Enter dialogue…"
              className="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-purple-500/50"
            />
            <button onClick={() => removeLine(i)}
              className="text-gray-500 hover:text-red-400 p-2 transition-colors text-xs shrink-0"
            >×</button>
          </motion.div>
        ))}
      </div>

      <div className="flex gap-2">
        <button onClick={addLine}
          className="flex-1 py-2 border border-dashed border-white/15 rounded-lg text-gray-400 hover:text-white hover:border-purple-500/30 text-xs transition-all"
        >+ Add Line</button>
        <button onClick={onSubmit} disabled={loading || !scene.trim() || dialogue.length === 0}
          className="flex-1 py-2 bg-purple-600 hover:bg-purple-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-all"
        >
          {loading ? <><Loader2 size={14} className="animate-spin" /> Analyzing…</> : <><Send size={14} /> Analyze</>}
        </button>
      </div>
    </div>
  )
}

function EmotionArcChart({ arc }: { arc: ArcPoint[] }) {
  const categories = ['tender', 'calm', 'playful', 'intense', 'cold', 'intimate']
  const catToY: Record<string, number> = {}
  categories.forEach((c, i) => { catToY[c] = i })

  const data = arc.map((p) => ({
    index: p.index + 1,
    category: p.category,
    catY: p.category ? catToY[p.category] ?? 3 : null,
    intensity: p.intensity,
    emotion: p.emotion,
    style: p.style,
    speaker: p.speaker,
    label: p.category ? `${EMOTION_LABELS[p.category] || p.category}` : '—',
  }))

  return (
    <div className="bg-white/5 border border-white/10 rounded-xl p-4">
      <h3 className="text-sm font-bold text-purple-300 uppercase tracking-wider flex items-center gap-2 mb-3">
        <Waves size={14} /> Emotion Arc
      </h3>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
          <XAxis
            dataKey="index"
            stroke="#6b7280"
            tick={{ fill: '#9ca3af', fontSize: 10 }}
            label={{ value: 'Turn', position: 'insideBottom', offset: -2, fill: '#9ca3af', fontSize: 10 }}
          />
          <YAxis
            domain={[-0.5, 5.5]}
            ticks={[0, 1, 2, 3, 4, 5]}
            tickFormatter={(v) => EMOTION_LABELS[categories[v]] || ''}
            stroke="#6b7280"
            tick={{ fill: '#9ca3af', fontSize: 9 }}
            width={40}
          />
          <Tooltip
            contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8, color: '#f9fafb', fontSize: 12 }}
            formatter={(_: unknown, __: unknown, props: { payload?: { label?: string; style?: string; intensity?: number } }) => {
              const d = props?.payload
              if (!d) return [String(_), '']
              return [`${d.label || ''} · ${d.style || ''}`, `Intensity: ${d.intensity ?? 0}`]
            }}
            labelFormatter={(v) => `Turn ${v}`}
          />
          <Legend wrapperStyle={{ fontSize: 10 }} />
          {categories.map((c, i) => (
            <ReferenceLine key={c} y={i} stroke={EMOTION_COLORS[c] + '30'} strokeDasharray="3 3" />
          ))}
          <Line
            type="monotone"
            dataKey="catY"
            name="Emotion Category"
            stroke="#a855f7"
            strokeWidth={2}
            dot={(props: { cx?: number; cy?: number; index?: number }) => {
              const idx = props.index ?? 0
              const d = data[idx]
              if (!d || d.catY === null || props.cx == null || props.cy == null) return <></>
              const color = EMOTION_COLORS[d.category || 'calm'] || '#a855f7'
              const r = 4 + (d.intensity || 0) * 6
              return (
                <circle cx={props.cx} cy={props.cy} r={r} fill={color} stroke={color} strokeWidth={2} fillOpacity={0.7} />
              )
            }}
            connectNulls
          />
        </LineChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap gap-2 mt-2 justify-center">
        {categories.map((c) => (
          <span key={c} className="text-xs flex items-center gap-1">
            <span className="w-2 h-2 rounded-full" style={{ background: EMOTION_COLORS[c] }} />
            {EMOTION_LABELS[c]}
          </span>
        ))}
      </div>
    </div>
  )
}

function ResultCard({ result, isActive, onClick }: { result: LineResult; isActive: boolean; onClick: () => void }) {
  const em = result.emotion
  const category = em?.ref_emotion_category as string || ''
  const color = EMOTION_COLORS[category] || '#6b7280'

  return (
    <motion.div
      layout
      onClick={onClick}
      className={`rounded-xl p-4 cursor-pointer transition-all border ${
        isActive ? 'bg-white/10 border-purple-500/50 shadow-lg shadow-purple-500/10' : 'bg-white/5 border-white/10 hover:bg-white/[0.07]'
      } ${result.skipped ? 'opacity-50' : ''}`}
    >
      <div className="flex items-start gap-3">
        <div className={`text-xs font-semibold px-2 py-0.5 rounded-full shrink-0 mt-0.5 ${
          result.speaker === '秦彻' ? 'bg-purple-500/20 text-purple-300' : 'bg-blue-500/20 text-blue-300'
        }`}>
          {result.speaker}
        </div>
        <p className="text-sm text-gray-200 leading-relaxed flex-1">{result.text}</p>
      </div>

      {em && !result.skipped && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          <span className="text-xs px-2 py-0.5 rounded-full font-medium"
            style={{ background: color + '25', color, border: `1px solid ${color}40` }}>
            {EMOTION_LABELS[category] || category}
          </span>
          <span className="text-xs px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-gray-400">
            {em.emotion as string}
          </span>
          <span className="text-xs px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-gray-400">
            intensity: {(em.intensity as number)?.toFixed(1)}
          </span>
          <span className="text-xs px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-gray-400">
            {em.pace as string}
          </span>
        </div>
      )}
    </motion.div>
  )
}

function DetailPanel({ result }: { result: LineResult | null }) {
  if (!result || result.skipped) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500 text-sm">
        Select a 秦彻 dialogue line to view details
      </div>
    )
  }

  const em = result.emotion!
  const category = em.ref_emotion_category as string
  const color = EMOTION_COLORS[category] || '#a855f7'

  return (
    <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} key={result.index} className="space-y-4 overflow-y-auto max-h-[calc(100vh-220px)] pr-1">
      <div className="bg-white/5 border border-white/10 rounded-xl p-4">
        <h4 className="text-xs font-bold text-purple-300 uppercase tracking-wider mb-3 flex items-center gap-2">
          <Brain size={13} /> LLM Emotion Analysis
        </h4>
        <div className="space-y-2.5">
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500 w-20 shrink-0">Category</span>
            <span className="text-sm font-semibold px-3 py-1 rounded-full"
              style={{ background: color + '25', color, border: `1px solid ${color}40` }}>
              {EMOTION_LABELS[category]} ({category})
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500 w-20 shrink-0">Emotion</span>
            <span className="text-sm text-white">{em.emotion as string}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500 w-20 shrink-0">Intensity</span>
            <div className="flex-1 bg-white/10 rounded-full h-2 overflow-hidden">
              <div className="h-full rounded-full transition-all" style={{ width: `${(em.intensity as number) * 100}%`, background: color }} />
            </div>
            <span className="text-xs text-gray-300 w-8 text-right">{(em.intensity as number)?.toFixed(1)}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500 w-20 shrink-0">Pace</span>
            <span className="text-sm text-white">{em.pace as string}</span>
          </div>
          <div>
            <span className="text-xs text-gray-500 block mb-1">Style Description</span>
            <p className="text-sm text-gray-200 bg-white/5 rounded-lg p-3 border border-white/5 leading-relaxed italic">
              "{em.style as string}"
            </p>
          </div>
          {!!em.fish_audio_tags && (
            <div>
              <span className="text-xs text-gray-500 block mb-1">Fish Audio Tags</span>
              <code className="text-xs text-green-300 bg-green-500/10 border border-green-500/20 rounded-lg p-2 block break-all">
                {String(em.fish_audio_tags)}
              </code>
            </div>
          )}
        </div>
      </div>

      {result.rag_context.length > 0 && (
        <div className="bg-white/5 border border-white/10 rounded-xl p-4">
          <h4 className="text-xs font-bold text-blue-300 uppercase tracking-wider mb-3 flex items-center gap-2">
            <BookOpen size={13} /> RAG Character Context
          </h4>
          <div className="space-y-2">
            {result.rag_context.map((ctx, i) => (
              <div key={i} className="text-xs text-gray-300 bg-white/5 rounded-lg p-3 border border-white/5 leading-relaxed">
                {ctx.length > 200 ? ctx.slice(0, 200) + '…' : ctx}
              </div>
            ))}
          </div>
        </div>
      )}

      {result.ref_audio && (
        <div className="bg-white/5 border border-white/10 rounded-xl p-4">
          <h4 className="text-xs font-bold text-green-300 uppercase tracking-wider mb-3 flex items-center gap-2">
            <AudioLines size={13} /> Reference Audio
          </h4>
          <p className="text-xs text-gray-400 break-all">{result.ref_audio}</p>
        </div>
      )}

      {result.emotion_history.length > 0 && (
        <div className="bg-white/5 border border-white/10 rounded-xl p-4">
          <h4 className="text-xs font-bold text-orange-300 uppercase tracking-wider mb-3 flex items-center gap-2">
            <Sparkles size={13} /> Emotion History Window
          </h4>
          <div className="flex flex-wrap gap-1.5">
            {result.emotion_history.map((s, i) => {
              const c = s.ref_emotion_category as string
              const hcolor = EMOTION_COLORS[c] || '#6b7280'
              return (
                <span key={i} className="text-xs px-2 py-0.5 rounded-full"
                  style={{ background: hcolor + '20', color: hcolor, border: `1px solid ${hcolor}30` }}>
                  {s.emotion as string}({(s.intensity as number)?.toFixed(1)})
                </span>
              )
            })}
          </div>
        </div>
      )}
    </motion.div>
  )
}

const DEMO_EMOTIONS: Record<string, { emotion: string; category: string; intensity: number; pace: string; style: string; fish_audio_tags: string }> = {
  '慌什么，肩上的伤早就愈合了，听我的声音，这不是好得很。': {
    emotion: 'reassuring warmth', category: 'tender', intensity: 0.65, pace: 'slow',
    style: '低沉温柔，带有安抚性的笑意，声线平稳但尾音微微上扬',
    fish_audio_tags: 'calm male voice, warm tone, gentle reassurance',
  },
  '别妄自菲薄，你比你想象中有用得多，这几天他都会跟着你出门。': {
    emotion: 'protective gentleness', category: 'calm', intensity: 0.45, pace: 'moderate',
    style: '沉稳而郑重，声线比上句低了半度，每个字都带着不容置疑的力量',
    fish_audio_tags: 'deep male voice, steady pace, authoritative warmth',
  },
  '好吧，林空试试，你的地盘你做主，有事给我打电话。': {
    emotion: 'relaxed affection', category: 'playful', intensity: 0.4, pace: 'moderate',
    style: '轻松随意的语气，尾音略带调侃味，像在故意让步',
    fish_audio_tags: 'casual male voice, slightly playful, light-hearted',
  },
  '还没睡？窗外的雨声很大，我在电话里都能听到。': {
    emotion: 'quiet concern', category: 'intimate', intensity: 0.55, pace: 'slow',
    style: '压低的声线，像怕吵醒谁，每个音节都裹着夜晚的柔软',
    fish_audio_tags: 'whisper-like male voice, intimate, night atmosphere',
  },
  '什么事让你这么费神？说给我听听。': {
    emotion: 'gentle inquiry', category: 'tender', intensity: 0.5, pace: 'slow',
    style: '温和的询问，语速放慢，声音里有引导对方倾诉的耐心',
    fish_audio_tags: 'soft male voice, patient, caring tone',
  },
  '别人怎么说不重要，你心里那把尺才是准绳，相信自己的直觉。': {
    emotion: 'firm tenderness', category: 'intense', intensity: 0.7, pace: 'moderate',
    style: '语气忽然认真起来，每个字掷地有声，但底色仍然温暖',
    fish_audio_tags: 'firm male voice, earnest conviction, warm undertone',
  },
  '有个很久没见的人想在N109约会，除了这个，还能是因为什么？': {
    emotion: 'teasing', category: 'playful', intensity: 0.6, pace: 'moderate',
    style: '带着笑意的反问，声线上扬，明显在逗对方',
    fish_audio_tags: 'amused male voice, teasing, confident charm',
  },
  '路过还点了两杯咖啡？你的演技需要提升。': {
    emotion: 'playful mockery', category: 'playful', intensity: 0.7, pace: 'moderate',
    style: '语速适中但咬字清晰，尾音带着压抑的笑声，满是调侃',
    fish_audio_tags: 'witty male voice, sarcastic amusement, charming',
  },
}

function generateDemoResponse(dialogue: DialogueLine[]): AnalyzeResponse {
  const results: LineResult[] = dialogue.map((line, i) => {
    const isQinche = line.speaker === '秦彻'
    const demoEmo = DEMO_EMOTIONS[line.text]
    return {
      index: i,
      speaker: line.speaker,
      text: line.text,
      emotion: isQinche && demoEmo ? {
        emotion: demoEmo.emotion,
        ref_emotion_category: demoEmo.category,
        intensity: demoEmo.intensity,
        pace: demoEmo.pace,
        style: demoEmo.style,
        fish_audio_tags: demoEmo.fish_audio_tags,
      } : null,
      rag_context: isQinche ? [
        `[性格] 秦彻外表冷峻强势，但对亲近的人有着不易察觉的温柔。`,
        `[说话风格] 语言简洁有力，偶尔带有调侃或反问，很少直接表达关心但行动上总是第一时间出现。`,
      ] : [],
      emotion_history: [],
      ref_audio: isQinche ? `data/ref_audio/${demoEmo?.category || 'calm'}_sample.wav` : null,
      audio: isQinche
        ? { auto: `output/auto/line_${i}.wav`, baseline: `output/baseline/line_${i}.wav` }
        : ({} as Record<string, string>),
      skipped: !isQinche,
    }
  })

  const emotion_arc: ArcPoint[] = results
    .filter(r => !r.skipped && r.emotion)
    .map(r => ({
      index: r.index,
      speaker: r.speaker,
      category: (r.emotion?.ref_emotion_category as string) || null,
      intensity: (r.emotion?.intensity as number) || 0,
      emotion: r.emotion?.emotion as string,
      style: r.emotion?.style as string,
    }))

  return { session_id: `demo-${Date.now()}`, results, emotion_arc }
}

export default function EmotionalTTSPage() {
  const navigate = useNavigate()
  const [scene, setScene] = useState(PRESET_SCENES[0].scene)
  const [dialogue, setDialogue] = useState<DialogueLine[]>([...PRESET_SCENES[0].dialogue])
  const [loading, setLoading] = useState(false)
  const [response, setResponse] = useState<AnalyzeResponse | null>(null)
  const [activeIdx, setActiveIdx] = useState<number | null>(null)
  const [demoMode, setDemoMode] = useState(false)
  const resultsRef = useRef<HTMLDivElement>(null)

  const handleAnalyze = useCallback(async () => {
    setLoading(true)
    setResponse(null)
    setActiveIdx(null)
    setDemoMode(true)

    await new Promise(r => setTimeout(r, 800))
    const data = generateDemoResponse(dialogue)
    setResponse(data)
    const firstQinche = data.results.findIndex(r => !r.skipped)
    if (firstQinche >= 0) setActiveIdx(firstQinche)
    setLoading(false)
  }, [scene, dialogue])

  useEffect(() => {
    if (response && resultsRef.current) {
      resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [response])

  const activeResult = response && activeIdx !== null ? response.results[activeIdx] : null

  return (
    <div className="min-h-screen bg-black text-white">
      <div className="fixed inset-0 z-0 bg-gradient-to-br from-purple-950/30 via-black to-blue-950/20" />

      <div className="relative z-10 max-w-7xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <motion.button initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
            onClick={() => navigate('/')}
            className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
          >
            <ArrowLeft size={18} /> Back
          </motion.button>
          <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="text-center">
            <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
              Context-Aware Emotional TTS
            </h1>
            <p className="text-gray-500 text-xs mt-1">LLM Scene Understanding · RAG Character Context · Emotion Arc Tracking</p>
          </motion.div>
          <div className="w-20" />
        </div>

        {/* Input Panel */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          className="bg-white/[0.03] border border-white/10 rounded-2xl p-6 mb-8 backdrop-blur-sm"
        >
          <SceneInputPanel
            scene={scene} setScene={setScene}
            dialogue={dialogue} setDialogue={setDialogue}
            onSubmit={handleAnalyze} loading={loading}
          />
        </motion.div>

        {demoMode && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 mb-6 text-amber-300 text-sm flex items-center gap-3"
          >
            <Sparkles size={16} className="shrink-0" />
            <div>
              <span className="font-semibold">Demo Mode</span> — 当前页面为演示版，展示预设的情感分析与情绪弧线效果。
              更多能力（实时分析与在线合成）敬请期待。
            </div>
          </motion.div>
        )}

        {/* Results */}
        {response && (
          <motion.div ref={resultsRef} initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
            {/* Emotion Arc Chart */}
            <EmotionArcChart arc={response.emotion_arc} />

            {/* Two-column: Results list + Detail panel */}
            <div className="grid grid-cols-5 gap-6 min-h-[400px]">
              {/* Left: result cards */}
              <div className="col-span-2 space-y-2 overflow-y-auto max-h-[600px] pr-1">
                {response.results.map((r) => (
                  <ResultCard key={r.index} result={r} isActive={activeIdx === r.index} onClick={() => setActiveIdx(r.index)} />
                ))}
              </div>

              {/* Right: detail panel */}
              <div className="col-span-3">
                <DetailPanel result={activeResult} />
              </div>
            </div>

            {/* Session info */}
            <div className="text-center text-gray-600 text-xs pb-8">
              Session: {response.session_id} · {response.results.filter(r => !r.skipped).length} lines analyzed by GPT-4o via RAG + Emotion Arc
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}
