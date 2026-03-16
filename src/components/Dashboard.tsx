import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, Activity, Database, Cpu, BarChart2, Clock, Music } from 'lucide-react'
import {
  BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import { modelResults, v5EpochData, v4v5EpochData, pipelineSteps, dataSources } from '../data'

interface DashboardProps {
  onClose: () => void
}

// ─── PAGE 1: Pipeline ───────────────────────────────────────────────
const Page1Pipeline = () => (
  <div className="h-full flex flex-col items-center justify-center p-12 overflow-auto">
    <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} className="max-w-6xl w-full">
      <div className="mb-10 text-center">
        <div className="inline-block px-4 py-2 bg-gradient-to-r from-purple-200/30 to-pink-200/30 rounded-full mb-3 border border-white/20">
          <span className="text-purple-300 text-sm font-semibold tracking-wider">DATA PIPELINE</span>
        </div>
        <h2 className="text-4xl font-bold text-white mb-2">Processing Pipeline</h2>
        <p className="text-gray-300">From raw Bilibili audio to fine-tuned TTS model</p>
      </div>

      {/* Steps */}
      <div className="grid grid-cols-4 gap-4 mb-10">
        {pipelineSteps.map((step, i) => (
          <motion.div
            key={step.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className="relative bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-4 hover:bg-white/10 transition-all"
          >
            {i < pipelineSteps.length - 1 && (
              <div className="hidden md:block absolute top-1/2 -right-2 w-4 h-0.5 bg-purple-500/50 z-10" />
            )}
            <div className="text-3xl mb-2 text-center">{step.icon}</div>
            <div className="text-center">
              <div className="font-bold text-white text-sm mb-1">{step.label}</div>
              <div className="text-xs text-gray-400">{step.sub}</div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Data Sources Table */}
      <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-6">
        <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <Database size={18} className="text-purple-400" /> Training Data Sources
        </h3>
        <div className="grid grid-cols-4 gap-3">
          {dataSources.map((src) => (
            <div key={src.name} className="bg-white/5 rounded-xl p-3 border border-white/10">
              <div className="font-mono text-xs text-purple-300 mb-1">{src.name}</div>
              <div className="text-2xl font-bold text-white">{src.samples}</div>
              <div className="text-xs text-gray-400 mt-1">{src.type}</div>
              <div className={`text-xs mt-1 font-medium ${
                src.quality === 'high' ? 'text-green-400' : src.quality === 'medium' ? 'text-yellow-400' : 'text-red-400'
              }`}>
                {src.quality === 'high' ? '★★★ High Quality' : src.quality === 'medium' ? '★★☆ Medium' : '★☆☆ Low (filtered)'}
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 text-sm text-gray-400 text-center">
          Total: <span className="text-white font-bold">475 samples</span> · Duration: <span className="text-white font-bold">~28 min</span> · After filtering: <span className="text-white font-bold">664 clips</span>
        </div>
      </div>
    </motion.div>
  </div>
)

// ─── PAGE 2: Model Comparison ────────────────────────────────────────
const Page2Models = () => (
  <div className="h-full flex flex-col items-center justify-center p-12 overflow-auto">
    <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} className="max-w-6xl w-full">
      <div className="mb-8 text-center">
        <div className="inline-block px-4 py-2 bg-gradient-to-r from-blue-200/30 to-cyan-200/30 rounded-full mb-3 border border-white/20">
          <span className="text-blue-300 text-sm font-semibold tracking-wider">MODEL EVALUATION</span>
        </div>
        <h2 className="text-4xl font-bold text-white mb-2">Performance Comparison</h2>
        <p className="text-gray-300">Qwen3-TTS SFT vs Fish Audio S2 Pro — 18 test samples</p>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        {modelResults.map((m) => (
          <div key={m.name} className="bg-white/5 border border-white/10 rounded-2xl p-4 hover:bg-white/10 transition-all">
            <div className="text-xs font-semibold mb-2" style={{ color: m.color }}>{m.label}</div>
            <div className="space-y-1.5">
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">SIM_gt</span>
                <span className="text-white font-bold">{m.sim_gt.toFixed(4)}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">SIM_ref</span>
                <span className="text-white font-bold">{m.sim_ref.toFixed(4)}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">WER</span>
                <span className="text-white font-bold">{(m.wer * 100).toFixed(2)}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">RTF</span>
                <span className={`font-bold ${m.rtf < 1 ? 'text-green-400' : 'text-yellow-400'}`}>{m.rtf.toFixed(3)}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-6">
        {/* SIM_gt bar chart */}
        <div className="bg-white/5 border border-white/10 rounded-2xl p-5">
          <h3 className="text-sm font-bold text-gray-200 mb-3 flex items-center gap-2">
            <BarChart2 size={15} className="text-purple-400" /> Speaker Similarity (SIM_gt)
          </h3>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={modelResults} barSize={28}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff15" />
              <XAxis dataKey="label" tick={false} />
              <YAxis domain={[0.6, 0.72]} stroke="#6b7280" fontSize={10} tick={{ fill: '#9ca3af' }} />
              <Tooltip
                contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8, color: '#f9fafb' }}
                formatter={(v: any) => v?.toFixed(4)}
                labelFormatter={(_l, payload) => payload?.[0]?.payload?.label || ''}
              />
              <Bar dataKey="sim_gt" radius={[4, 4, 0, 0]}
                fill="url(#simGrad)" />
              <defs>
                <linearGradient id="simGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#a855f7" stopOpacity={0.9} />
                  <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.7} />
                </linearGradient>
              </defs>
            </BarChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap gap-2 mt-2">
            {modelResults.map(m => (
              <span key={m.name} className="text-xs px-2 py-0.5 rounded-full border"
                style={{ borderColor: m.color + '60', color: m.color, background: m.color + '15' }}>
                {m.label.split('(')[0].trim()}
              </span>
            ))}
          </div>
        </div>

        {/* RTF bar chart */}
        <div className="bg-white/5 border border-white/10 rounded-2xl p-5">
          <h3 className="text-sm font-bold text-gray-200 mb-3 flex items-center gap-2">
            <Clock size={15} className="text-blue-400" /> Real-Time Factor (RTF) — lower is faster
          </h3>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={modelResults} barSize={28}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff15" />
              <XAxis dataKey="label" tick={false} />
              <YAxis stroke="#6b7280" fontSize={10} tick={{ fill: '#9ca3af' }} />
              <Tooltip
                contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8, color: '#f9fafb' }}
                formatter={(v: any) => v?.toFixed(3)}
                labelFormatter={(_l, payload) => payload?.[0]?.payload?.label || ''}
              />
              <Bar dataKey="rtf" radius={[4, 4, 0, 0]}>
                {modelResults.map((m) => (
                  <rect key={m.name} fill={m.color} />
                ))}
              </Bar>
              <defs>
                <linearGradient id="rtfGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#10b981" stopOpacity={0.9} />
                  <stop offset="100%" stopColor="#06b6d4" stopOpacity={0.7} />
                </linearGradient>
              </defs>
            </BarChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap gap-2 mt-2">
            {modelResults.map(m => (
              <span key={m.name} className="text-xs px-2 py-0.5 rounded-full border"
                style={{ borderColor: m.color + '60', color: m.color, background: m.color + '15' }}>
                {m.label.split('(')[0].trim()} {m.rtf.toFixed(3)}×
              </span>
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  </div>
)

// ─── PAGE 3: Epoch Analysis ───────────────────────────────────────────
const Page3Epochs = () => (
  <div className="h-full flex flex-col items-center justify-center p-12 overflow-auto">
    <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} className="max-w-6xl w-full">
      <div className="mb-8 text-center">
        <div className="inline-block px-4 py-2 bg-gradient-to-r from-green-200/30 to-cyan-200/30 rounded-full mb-3 border border-white/20">
          <span className="text-green-300 text-sm font-semibold tracking-wider">TRAINING ANALYSIS</span>
        </div>
        <h2 className="text-4xl font-bold text-white mb-2">Epoch-Level Analysis</h2>
        <p className="text-gray-300">v5 training convergence and v4 vs v5 data augmentation effect</p>
      </div>

      <div className="grid grid-cols-2 gap-6 mb-6">
        {/* v5 SIM_gt & WER per epoch */}
        <div className="bg-white/5 border border-white/10 rounded-2xl p-5">
          <h3 className="text-sm font-bold text-gray-200 mb-1">v5: SIM_gt & WER across Epochs</h3>
          <p className="text-xs text-gray-400 mb-3">Epoch 3 achieves best SIM_gt with acceptable WER</p>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={v5EpochData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff15" />
              <XAxis dataKey="epoch" stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 11 }} label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: '#9ca3af', fontSize: 11 }} />
              <YAxis yAxisId="sim" domain={[0.66, 0.70]} stroke="#a855f7" tick={{ fill: '#a855f7', fontSize: 10 }} />
              <YAxis yAxisId="wer" orientation="right" domain={[0, 0.09]} stroke="#f59e0b" tick={{ fill: '#f59e0b', fontSize: 10 }} />
              <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8, color: '#f9fafb' }} />
              <Legend wrapperStyle={{ color: '#9ca3af', fontSize: 12 }} />
              <Line yAxisId="sim" type="monotone" dataKey="sim_gt" stroke="#a855f7" strokeWidth={2} dot={{ fill: '#a855f7', r: 4 }} name="SIM_gt" />
              <Line yAxisId="wer" type="monotone" dataKey="wer" stroke="#f59e0b" strokeWidth={2} strokeDasharray="5 5" dot={{ fill: '#f59e0b', r: 4 }} name="WER" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* v4 vs v5 */}
        <div className="bg-white/5 border border-white/10 rounded-2xl p-5">
          <h3 className="text-sm font-bold text-gray-200 mb-1">v4 vs v5: Data Augmentation Impact</h3>
          <p className="text-xs text-gray-400 mb-3">v5 adds qinche_call data (+64 samples), boosting early-epoch SIM</p>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={v4v5EpochData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff15" />
              <XAxis dataKey="epoch" stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 11 }} label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: '#9ca3af', fontSize: 11 }} />
              <YAxis domain={[0.62, 0.70]} stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 10 }} />
              <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8, color: '#f9fafb' }} />
              <Legend wrapperStyle={{ color: '#9ca3af', fontSize: 12 }} />
              <Line type="monotone" dataKey="v4" stroke="#f59e0b" strokeWidth={2} dot={{ fill: '#f59e0b', r: 4 }} name="v4 (461 samples)" />
              <Line type="monotone" dataKey="v5" stroke="#3b82f6" strokeWidth={2} dot={{ fill: '#3b82f6', r: 4 }} name="v5 (664 samples)" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Key findings */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-green-500/10 border border-green-500/20 rounded-xl p-4">
          <div className="text-green-400 font-bold text-sm mb-1">✓ Best Checkpoint</div>
          <p className="text-gray-300 text-xs">v5 Epoch 3 achieves SIM_gt = 0.6892 with only 4.25% WER — best balance</p>
        </div>
        <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4">
          <div className="text-blue-400 font-bold text-sm mb-1">✓ Data Augmentation</div>
          <p className="text-gray-300 text-xs">Adding 64 call recordings (v4→v5) improves early-epoch SIM by +5.7 pts</p>
        </div>
        <div className="bg-purple-500/10 border border-purple-500/20 rounded-xl p-4">
          <div className="text-purple-400 font-bold text-sm mb-1">✓ CUDA Graph Speed</div>
          <p className="text-gray-300 text-xs">FasterQwen3TTS reduces RTF from 1.12× to 0.357× — 3.1× speedup, zero quality loss</p>
        </div>
      </div>
    </motion.div>
  </div>
)

const AUDIO_BASE = 'https://pub-fd759c08d84e44e9b8e0e03dbaf6ad0b.r2.dev'

// ─── PAGE 4: Audio Demo ───────────────────────────────────────────────
const Page4Audio = () => {
  const [selected, setSelected] = useState(0)

  const samples = [
    { text: '但还有名词的解释,只带一种爪子锋利又养不熟的猫,也可以特指一个十分可恶又让人不忍心下手的人。' },
    { text: '你今天一直在搜N109区的乌鸦,如何关闭机械鸟的电源,尝试了喂鸭喝水敲鸭脑袋宝石溜鸭,这些我都知道。' },
    { text: '慌什么肩上的伤早就愈合了听我的声音这不是好得很' },
  ]

  const models = [
    { key: 'qwen3_v5_fast', label: 'Qwen3-TTS v5 (CUDA Graph)', color: '#3b82f6' },
    { key: 'qwen3_v5_native', label: 'Qwen3-TTS v5 (Native)', color: '#8b5cf6' },
    { key: 'fish_s2_zeroshot', label: 'Fish Audio S2 Pro (Zero-shot)', color: '#f59e0b' },
    { key: 'fish_s2_compile', label: 'Fish Audio S2 Pro (Compiled)', color: '#f97316' },
  ]

  return (
    <div className="h-full flex flex-col items-center justify-center p-10 overflow-auto">
      <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} className="max-w-5xl w-full">
        <div className="mb-6 text-center">
          <div className="inline-block px-4 py-2 bg-gradient-to-r from-pink-200/30 to-purple-200/30 rounded-full mb-3 border border-white/20">
            <span className="text-pink-300 text-sm font-semibold tracking-wider">AUDIO DEMO</span>
          </div>
          <h2 className="text-4xl font-bold text-white mb-2">Listen & Compare</h2>
          <p className="text-gray-300">Side-by-side audio comparison · 4 models · 3 samples</p>
        </div>

        {/* Sample selector */}
        <div className="flex gap-3 mb-4 justify-center">
          {samples.map((_s, i) => (
            <button
              key={i}
              onClick={() => setSelected(i)}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${
                selected === i ? 'bg-purple-600 text-white' : 'bg-white/10 text-gray-300 hover:bg-white/20'
              }`}
            >
              Sample {i + 1}
            </button>
          ))}
        </div>

        <AnimatePresence mode="wait">
          <motion.div
            key={selected}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="bg-white/5 border border-white/10 rounded-xl px-5 py-3 mb-4 flex items-start gap-3"
          >
            <Music size={16} className="text-purple-400 mt-0.5 shrink-0" />
            <p className="text-gray-200 text-sm leading-relaxed">{samples[selected].text}</p>
          </motion.div>
        </AnimatePresence>

        <div className="grid grid-cols-2 gap-3">
          {models.map((m) => (
            <div key={m.key} className="bg-white/5 border border-white/10 rounded-xl p-4">
              <div className="text-xs font-semibold mb-2" style={{ color: m.color }}>{m.label}</div>
              <audio
                key={`${m.key}-${selected}`}
                controls
                className="w-full h-8"
                style={{ accentColor: m.color }}
              >
                <source src={`${AUDIO_BASE}/${m.key}/gen_${String(selected).padStart(2,'0')}.wav`} type="audio/wav" />
              </audio>
            </div>
          ))}
        </div>

        <p className="text-center text-gray-500 text-xs mt-4">
          Hosted on Cloudflare R2 · tencent-tts bucket
        </p>
      </motion.div>
    </div>
  )
}

// ─── PAGE 5: Acceleration ─────────────────────────────────────────────
const Page5Accel = () => {
  const accelData = [
    { method: 'Baseline\n(Native)', rtf: 1.12, speedup: 1.0, color: '#6b7280' },
    { method: 'CUDA Graph', rtf: 0.357, speedup: 3.14, color: '#3b82f6' },
    { method: 'torch.compile', rtf: 0.41, speedup: 2.73, color: '#8b5cf6' },
    { method: 'INT8 Quant', rtf: 0.38, speedup: 2.95, color: '#10b981' },
  ]

  return (
    <div className="h-full flex flex-col items-center justify-center p-12 overflow-auto">
      <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} className="max-w-6xl w-full">
        <div className="mb-8 text-center">
          <div className="inline-block px-4 py-2 bg-gradient-to-r from-cyan-200/30 to-blue-200/30 rounded-full mb-3 border border-white/20">
            <span className="text-cyan-300 text-sm font-semibold tracking-wider">INFERENCE ACCELERATION</span>
          </div>
          <h2 className="text-4xl font-bold text-white mb-2">Speed Optimization</h2>
          <p className="text-gray-300">FasterQwen3TTS: CUDA Graph + torch.compile + INT8 quantization</p>
        </div>

        <div className="grid grid-cols-2 gap-6 mb-6">
          {/* RTF comparison */}
          <div className="bg-white/5 border border-white/10 rounded-2xl p-5">
            <h3 className="text-sm font-bold text-gray-200 mb-3 flex items-center gap-2">
              <Cpu size={15} className="text-cyan-400" /> RTF by Acceleration Method
            </h3>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={accelData} barSize={35}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff15" />
                <XAxis dataKey="method" stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 10 }} />
                <YAxis stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 10 }} label={{ value: 'RTF', angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 10 }} />
                <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8, color: '#f9fafb' }} formatter={(v: any) => v?.toFixed(3)} />
                <Bar dataKey="rtf" radius={[4, 4, 0, 0]}>
                  {accelData.map((entry, i) => (
                    <rect key={i} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Speedup comparison */}
          <div className="bg-white/5 border border-white/10 rounded-2xl p-5">
            <h3 className="text-sm font-bold text-gray-200 mb-3 flex items-center gap-2">
              <Activity size={15} className="text-green-400" /> Speedup vs Baseline
            </h3>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={accelData} barSize={35}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff15" />
                <XAxis dataKey="method" stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 10 }} />
                <YAxis stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 10 }} label={{ value: 'Speedup ×', angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 10 }} />
                <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8, color: '#f9fafb' }} formatter={(v: any) => `${v?.toFixed(2)}×`} />
                <Bar dataKey="speedup" radius={[4, 4, 0, 0]}>
                  {accelData.map((entry, i) => (
                    <rect key={i} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Techniques */}
        <div className="grid grid-cols-3 gap-4">
          {[
            { title: 'CUDA Graph', desc: 'Captures & replays GPU ops, eliminating kernel launch overhead. Best overall: RTF 0.357 (3.14×)', color: '#3b82f6' },
            { title: 'torch.compile', desc: 'JIT-compiles autoregressive decoding loops with TorchInductor backend. 2.73× speedup', color: '#8b5cf6' },
            { title: 'INT8 Quantization', desc: 'Quantizes codec decoder weights to INT8, reducing memory & compute. 2.95× speedup', color: '#10b981' },
          ].map(item => (
            <div key={item.title} className="rounded-xl p-4 border" style={{ background: item.color + '15', borderColor: item.color + '40' }}>
              <div className="font-bold text-sm mb-1" style={{ color: item.color }}>{item.title}</div>
              <p className="text-gray-300 text-xs leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  )
}

// ─── MAIN DASHBOARD ───────────────────────────────────────────────────
export default function Dashboard({ onClose }: DashboardProps) {
  const [currentPage, setCurrentPage] = useState(0)
  const [isScrolling, setIsScrolling] = useState(false)

  const pages = [
    { component: Page1Pipeline, label: 'Pipeline' },
    { component: Page2Models, label: 'Models' },
    { component: Page3Epochs, label: 'Training' },
    { component: Page4Audio, label: 'Audio' },
    { component: Page5Accel, label: 'Speed' },
  ]
  const PageComponent = pages[currentPage].component

  const handleWheel = (e: React.WheelEvent) => {
    if (isScrolling) return
    if (Math.abs(e.deltaY) > 80) {
      setIsScrolling(true)
      if (e.deltaY > 0 && currentPage < pages.length - 1) setCurrentPage(p => p + 1)
      else if (e.deltaY < 0 && currentPage > 0) setCurrentPage(p => p - 1)
      setTimeout(() => setIsScrolling(false), 800)
    }
  }

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'ArrowDown' && currentPage < pages.length - 1) setCurrentPage(p => p + 1)
      else if (e.key === 'ArrowUp' && currentPage > 0) setCurrentPage(p => p - 1)
      else if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [currentPage, onClose, pages.length])

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 bg-gray-950"
      style={{
        backgroundImage: 'url("/dashboard-bg.png")',
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundRepeat: 'no-repeat',
        willChange: 'transform',
        backfaceVisibility: 'hidden',
        transform: 'translateZ(0)',
      }}
      onWheel={handleWheel}
    >
      {/* Subtle bg gradient */}
      <div className="absolute inset-0 bg-black/50" />

      {/* Close */}
      <button
        onClick={onClose}
        className="fixed top-6 right-6 z-50 w-10 h-10 bg-white/10 hover:bg-white/20 rounded-full flex items-center justify-center text-white transition-colors"
      >
        <X size={20} />
      </button>

      {/* Page nav dots */}
      <div className="fixed left-6 top-1/2 -translate-y-1/2 z-50 flex flex-col gap-3">
        {pages.map((p, i) => (
          <button
            key={i}
            onClick={() => setCurrentPage(i)}
            title={p.label}
            className={`transition-all rounded-full ${
              i === currentPage ? 'bg-purple-400 w-3 h-6' : 'bg-white/30 hover:bg-white/50 w-2.5 h-2.5'
            }`}
          />
        ))}
      </div>

      {/* Page label */}
      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 flex gap-4">
        {pages.map((p, i) => (
          <button
            key={i}
            onClick={() => setCurrentPage(i)}
            className={`text-xs px-3 py-1 rounded-full transition-all ${
              i === currentPage
                ? 'bg-purple-600 text-white'
                : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentPage}
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -40 }}
          transition={{ duration: 0.45 }}
          className="relative h-full w-full z-10"
        >
          <PageComponent />
        </motion.div>
      </AnimatePresence>
    </motion.div>
  )
}
