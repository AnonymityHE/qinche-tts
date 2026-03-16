import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Music, Play, ChevronLeft, ChevronRight } from 'lucide-react'
import { modelResults, sampleTexts } from '../data'

const AUDIO_BASE = 'https://pub-fd759c08d84e44e9b8e0e03dbaf6ad0b.r2.dev'

interface AudioSampleCardProps {
  modelName: string
  label: string
  color: string
  audioUrl: string | null
  text: string
}

function AudioCard({ label, color, audioUrl, text }: AudioSampleCardProps) {
  return (
    <div className="bg-white/5 border border-white/10 rounded-xl p-4 flex flex-col gap-3">
      <div className="text-sm font-semibold" style={{ color }}>{label}</div>
      {audioUrl ? (
        <audio controls className="w-full h-8" style={{ accentColor: color }}>
          <source src={audioUrl} type="audio/wav" />
        </audio>
      ) : (
        <div className="bg-white/5 rounded-lg p-3 text-center border border-dashed border-white/10">
          <Play size={20} className="mx-auto mb-1 text-gray-600" />
          <span className="text-gray-500 text-xs">Audio coming soon · Upload to R2 first</span>
        </div>
      )}
      <p className="text-gray-400 text-xs leading-relaxed line-clamp-2">{text}</p>
    </div>
  )
}

export default function DemoPage() {
  const navigate = useNavigate()
  const [sampleIdx, setSampleIdx] = useState(0)

  // Map model results to audio URLs — fill in after R2 upload
  const modelAudioMap: Record<string, string[]> = {
    qwen3_v5_fast: sampleTexts.map((_, i) =>
      AUDIO_BASE ? `${AUDIO_BASE}/qwen3_v5_fast/gen_${String(i).padStart(2, '0')}.wav` : ''
    ),
    fish_s2_zeroshot: sampleTexts.map((_, i) =>
      AUDIO_BASE ? `${AUDIO_BASE}/fish_s2_zeroshot/gen_${String(i).padStart(2, '0')}.wav` : ''
    ),
    qwen3_v5_native: sampleTexts.map((_, i) =>
      AUDIO_BASE ? `${AUDIO_BASE}/qwen3_v5_native/gen_${String(i).padStart(2, '0')}.wav` : ''
    ),
    fish_s2_compile: sampleTexts.map((_, i) =>
      AUDIO_BASE ? `${AUDIO_BASE}/fish_s2_compile/gen_${String(i).padStart(2, '0')}.wav` : ''
    ),
  }

  const displayModels = modelResults.slice(0, 4)

  return (
    <div className="min-h-screen bg-black text-white">
      <div className="fixed inset-0 z-0" style={{
        backgroundImage: 'url("/demo-bg.png")',
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        willChange: 'transform',
        backfaceVisibility: 'hidden',
        transform: 'translateZ(0)',
      }} />
      <div className="fixed inset-0 z-0 bg-black/50" />

      <div className="relative z-10 max-w-5xl mx-auto px-6 py-10">
        {/* Back */}
        <motion.button
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          onClick={() => navigate('/')}
          className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors mb-10"
        >
          <ArrowLeft size={18} /> Back to Overview
        </motion.button>

        {/* Header */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-10">
          <h1 className="text-4xl font-bold mb-2">Audio Demo</h1>
          <p className="text-gray-400">Compare synthesized Qinche voice across models — 18 test prompts</p>
        </motion.div>

        {/* Sample navigation */}
        <div className="bg-white/5 border border-white/10 rounded-2xl p-6 mb-8">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Music size={18} className="text-purple-400" />
              <span className="font-semibold">Sample {sampleIdx + 1} / {sampleTexts.length}</span>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setSampleIdx(i => Math.max(0, i - 1))}
                disabled={sampleIdx === 0}
                className="w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 disabled:opacity-30 flex items-center justify-center transition-all"
              >
                <ChevronLeft size={16} />
              </button>
              <button
                onClick={() => setSampleIdx(i => Math.min(sampleTexts.length - 1, i + 1))}
                disabled={sampleIdx === sampleTexts.length - 1}
                className="w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 disabled:opacity-30 flex items-center justify-center transition-all"
              >
                <ChevronRight size={16} />
              </button>
            </div>
          </div>

          {/* Sample dots */}
          <div className="flex gap-1.5 flex-wrap mb-4">
            {sampleTexts.map((_, i) => (
              <button
                key={i}
                onClick={() => setSampleIdx(i)}
                className={`w-2 h-2 rounded-full transition-all ${i === sampleIdx ? 'bg-purple-400 w-4' : 'bg-white/20 hover:bg-white/40'}`}
              />
            ))}
          </div>

          <AnimatePresence mode="wait">
            <motion.p
              key={sampleIdx}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="text-gray-200 text-base leading-relaxed"
            >
              {sampleTexts[sampleIdx]}
            </motion.p>
          </AnimatePresence>
        </div>

        {/* Audio cards grid */}
        <AnimatePresence mode="wait">
          <motion.div
            key={sampleIdx}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="grid grid-cols-2 gap-4 mb-10"
          >
            {displayModels.map((m) => (
              <AudioCard
                key={m.name}
                modelName={m.name}
                label={m.label}
                color={m.color}
                audioUrl={modelAudioMap[m.name]?.[sampleIdx] || null}
                text={sampleTexts[sampleIdx]}
              />
            ))}
          </motion.div>
        </AnimatePresence>

        {/* Model metrics summary */}
        <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
          <h3 className="font-bold text-white mb-4">Model Comparison Summary</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-400 border-b border-white/10">
                  <th className="text-left pb-3 font-medium">Model</th>
                  <th className="text-right pb-3 font-medium">SIM_gt ↑</th>
                  <th className="text-right pb-3 font-medium">SIM_ref ↑</th>
                  <th className="text-right pb-3 font-medium">WER ↓</th>
                  <th className="text-right pb-3 font-medium">RTF ↓</th>
                </tr>
              </thead>
              <tbody>
                {modelResults.map((m, i) => (
                  <tr key={m.name} className={`border-b border-white/5 ${i === 0 ? '' : ''}`}>
                    <td className="py-2.5">
                      <span className="text-xs font-semibold px-2 py-0.5 rounded-full" style={{ color: m.color, background: m.color + '20' }}>
                        {m.label}
                      </span>
                    </td>
                    <td className="text-right py-2.5 text-white font-mono">{m.sim_gt.toFixed(4)}</td>
                    <td className="text-right py-2.5 text-white font-mono">{m.sim_ref.toFixed(4)}</td>
                    <td className="text-right py-2.5 text-white font-mono">{(m.wer * 100).toFixed(2)}%</td>
                    <td className={`text-right py-2.5 font-mono font-bold ${m.rtf < 1 ? 'text-green-400' : 'text-yellow-400'}`}>{m.rtf.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}
