import { useState, useRef } from 'react'
import { AnimatePresence, motion, useScroll, useTransform } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { ChevronDown, Activity, Mic2, Brain } from 'lucide-react'
import GradientText from '../components/GradientText'
import Dashboard from '../components/Dashboard'

// ─── HERO ─────────────────────────────────────────────────────────────
const HeroSection = ({ onDemo, onEmotionalTTS }: { onDemo: () => void; onEmotionalTTS: () => void }) => {
  const ref = useRef<HTMLDivElement>(null)
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start start', 'end start'] })
  const titleY = useTransform(scrollYProgress, [0, 1], [0, -180])
  const titleOpacity = useTransform(scrollYProgress, [0, 0.5], [1, 0])
  const descOpacity = useTransform(scrollYProgress, [0, 0.3], [1, 0])
  const descY = useTransform(scrollYProgress, [0, 1], [0, 140])

  return (
    <div ref={ref} className="min-h-screen flex flex-col items-center justify-center relative px-8">
      {/* Blobs */}
      <div className="absolute top-1/3 left-1/4 w-80 h-80 bg-purple-600/20 rounded-full blur-3xl" style={{ animation: 'float 8s ease-in-out infinite', willChange: 'transform' }} />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-600/15 rounded-full blur-3xl" style={{ animation: 'float 8s ease-in-out infinite', animationDelay: '4s', willChange: 'transform' }} />

      <div className="text-center z-10">
        <motion.div style={{ opacity: descOpacity, y: descY }} className="mb-6">
          <span className="text-purple-400 text-sm font-medium tracking-widest uppercase">Chinese Voice Cloning · 秦彻</span>
        </motion.div>

        <motion.div style={{ y: titleY, opacity: titleOpacity }} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }} className="mb-10">
          <GradientText
            colors={['#ffffff', '#a855f7', '#3b82f6', '#06b6d4', '#a855f7', '#ffffff']}
            animationSpeed={10}
            className="text-8xl md:text-9xl font-bold"
          >
            QINCHE
          </GradientText>
        </motion.div>

        <motion.p style={{ y: descY, opacity: descOpacity }} className="text-lg text-gray-300 max-w-xl mx-auto mb-10 leading-relaxed">
          Full-pipeline Chinese TTS voice cloning · Qwen3-TTS SFT vs Fish Audio S2 Pro<br />
          <span className="text-gray-500 text-sm">CUDA Graph acceleration · 3.1× real-time</span>
        </motion.p>

        <motion.div style={{ opacity: descOpacity }} className="flex gap-4 justify-center flex-wrap">
          <motion.button
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            onClick={onDemo}
            className="px-8 py-3.5 bg-white text-black rounded-lg font-semibold flex items-center gap-2 hover:bg-gray-100 transition-all"
          >
            <Mic2 size={17} /> Listen to Samples
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            onClick={onEmotionalTTS}
            className="px-8 py-3.5 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg font-semibold flex items-center gap-2 hover:from-purple-500 hover:to-blue-500 transition-all"
          >
            <Brain size={17} /> Emotional TTS Demo
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.03 }}
            onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })}
            className="px-8 py-3.5 border border-white/20 rounded-lg font-semibold hover:bg-white/5 transition-all text-white"
          >
            View Results
          </motion.button>
        </motion.div>
      </div>

      <motion.div
        animate={{ y: [0, 10, 0] }}
        transition={{ repeat: Infinity, duration: 2 }}
        style={{ opacity: descOpacity }}
        className="absolute bottom-10 text-gray-400 cursor-pointer"
        onClick={() => window.scrollBy({ top: window.innerHeight, behavior: 'smooth' })}
      >
        <ChevronDown size={30} />
      </motion.div>
    </div>
  )
}

// ─── METRICS SECTION ─────────────────────────────────────────────────
const MetricsSection = () => {
  const metrics = [
    { value: '0.6892', label: 'SIM_gt (Best)', sub: 'Qwen3-TTS v5 Ep.3', color: 'text-purple-400' },
    { value: '0.357×', label: 'Real-Time Factor', sub: 'FasterQwen3TTS (CUDA Graph)', color: 'text-blue-400' },
    { value: '3.1×', label: 'Speedup', sub: 'vs Native inference', color: 'text-cyan-400' },
    { value: '664', label: 'Training Clips', sub: '~28 min total audio', color: 'text-green-400' },
  ]

  return (
    <div id="features" className="py-24 px-8">
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-14">
          <h2 className="text-5xl font-bold mb-4">Key Results</h2>
          <p className="text-gray-400">Evaluation on 18 held-out test samples</p>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {metrics.map((m, i) => (
            <motion.div
              key={m.label}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
              viewport={{ once: true }}
              className="bg-white/5 border border-white/10 rounded-2xl p-6 text-center hover:bg-white/10 transition-all"
            >
              <div className={`text-4xl font-bold mb-2 ${m.color}`}>{m.value}</div>
              <div className="text-white font-semibold mb-1">{m.label}</div>
              <div className="text-gray-400 text-xs">{m.sub}</div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ─── FEATURES ────────────────────────────────────────────────────────
const FeaturesSection = () => {
  const [active, setActive] = useState(0)

  const features = [
    {
      title: 'Data Pipeline',
      subtitle: 'Demucs · Pyannote · WhisperX',
      desc: 'Raw Bilibili audio → source separation (Demucs) → speaker diarization (Pyannote) → VAD segmentation (Silero-VAD) → ASR transcription (WhisperX) → RMS normalization → 664 clean clips',
      tags: ['Demucs', 'Pyannote', 'Silero-VAD', 'WhisperX', 'OpenCC'],
      stat: '475 raw → 664 filtered clips'
    },
    {
      title: 'Qwen3-TTS SFT',
      subtitle: '1.7B · Transformer Decoder · 16-codebook RVQ',
      desc: 'Supervised fine-tuning of Qwen3-TTS (1.7B) on target speaker data. bf16 mixed precision, Flash Attention 2, speaker embedding accumulation over training batches.',
      tags: ['1.7B params', 'RVQ 16-codebook', 'Flash Attention 2', 'bf16', 'SFT'],
      stat: 'SIM_gt 0.6892 @ Epoch 3'
    },
    {
      title: 'Fish Audio S2 Pro',
      subtitle: '5B · Dual-AR · 10-codebook RVQ',
      desc: 'Zero-shot voice cloning with Fish Audio S2 Pro (5B Dual-AR). Serves as baseline comparison. Also tested with torch.compile acceleration.',
      tags: ['5B params', 'RVQ 10-codebook', 'Zero-shot', 'Dual-AR'],
      stat: 'SIM_gt 0.6647, RTF 4.32×'
    },
    {
      title: 'FasterQwen3TTS',
      subtitle: 'CUDA Graph · torch.compile · INT8',
      desc: 'Custom inference acceleration: CUDA Graph captures GPU ops for repeated execution, eliminating kernel launch overhead. Combined with torch.compile and INT8 quantization.',
      tags: ['CUDA Graph', 'torch.compile', 'INT8 Quant', 'Speculative Decoding'],
      stat: 'RTF 0.357 — 3.1× faster'
    },
  ]

  return (
    <div className="py-16 px-8">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-5xl font-bold mb-4">System Overview</h2>
          <p className="text-gray-400">Technical pipeline from data to deployment</p>
        </div>
        <div className="grid grid-cols-5 gap-8">
          {/* Left list */}
          <div className="col-span-2 space-y-2">
            {features.map((f, i) => (
              <motion.div
                key={f.title}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.1 }}
                viewport={{ once: true }}
                onClick={() => setActive(i)}
                className={`p-4 cursor-pointer transition-all border-l-4 ${
                  active === i ? 'bg-white/10 border-purple-500' : 'hover:bg-white/5 border-transparent'
                }`}
              >
                <div className={`font-bold text-lg ${active === i ? 'text-white' : 'text-gray-300'}`}>{f.title}</div>
                <div className="text-purple-400 text-xs font-medium mt-0.5">{f.subtitle}</div>
              </motion.div>
            ))}
          </div>

          {/* Right panel */}
          <div className="col-span-3">
            <AnimatePresence mode="wait">
              <motion.div
                key={active}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.25 }}
                className="bg-white/5 border border-white/10 rounded-2xl p-6 space-y-5"
              >
                <div>
                  <div className="text-xs text-purple-400 font-semibold uppercase tracking-wider mb-2">Description</div>
                  <p className="text-gray-200 leading-relaxed">{features[active].desc}</p>
                </div>
                <div>
                  <div className="text-xs text-blue-400 font-semibold uppercase tracking-wider mb-2">Technologies</div>
                  <div className="flex flex-wrap gap-2">
                    {features[active].tags.map(t => (
                      <span key={t} className="text-xs bg-blue-500/20 border border-blue-500/30 px-2.5 py-1 rounded-full text-blue-200">{t}</span>
                    ))}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-green-400 font-semibold uppercase tracking-wider mb-2">Key Metric</div>
                  <div className="text-white font-bold">{features[active].stat}</div>
                </div>
              </motion.div>
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  )
}

// ─── INNOVATIONS ───────────────────────────────────────────────────────
const InnovationsSection = ({ onDashboard }: { onDashboard: () => void }) => {
  const items = [
    {
      num: '01',
      title: 'End-to-End Pipeline',
      sub: 'Collection → Processing → Training → Eval',
      desc: 'Complete voice cloning pipeline from raw web audio to deployed TTS model, with automated quality filtering at each stage.'
    },
    {
      num: '02',
      title: 'Model Comparison',
      sub: 'Qwen3-TTS SFT vs Fish S2 Pro',
      desc: 'Head-to-head evaluation of fine-tuned Qwen3-TTS against zero-shot Fish Audio S2 Pro across SIM, WER, and RTF metrics.'
    },
    {
      num: '03',
      title: 'Real-Time Acceleration',
      sub: 'CUDA Graph · 3.1× speedup',
      desc: 'FasterQwen3TTS achieves sub-realtime synthesis (RTF 0.357) with zero quality degradation, enabling production deployment.'
    },
  ]

  return (
    <div className="py-16 px-8">
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-5xl font-bold mb-4">Core Innovations</h2>
          <p className="text-gray-400">What makes this pipeline unique</p>
        </div>

        <div className="space-y-10 mb-12">
          {items.map((item, i) => (
            <motion.div
              key={item.num}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.15 }}
              viewport={{ once: true }}
              className="group flex items-start gap-6"
            >
              <div className="text-6xl font-bold text-purple-600/60 group-hover:text-purple-400 transition-colors">{item.num}</div>
              <div className="pt-1">
                <h3 className="text-2xl font-bold mb-1 group-hover:text-purple-300 transition-colors">{item.title}</h3>
                <p className="text-purple-400 text-sm font-medium mb-2">{item.sub}</p>
                <p className="text-gray-300 leading-relaxed">{item.desc}</p>
              </div>
            </motion.div>
          ))}
        </div>

        <div className="flex justify-center">
          <motion.button
            whileHover={{ scale: 1.03 }}
            onClick={onDashboard}
            className="px-8 py-3.5 border border-white/20 rounded-lg font-semibold hover:bg-white/5 transition-all flex items-center gap-2"
          >
            <Activity size={17} /> Open Dashboard
          </motion.button>
        </div>
        <p className="text-center text-gray-500 text-sm mt-4">Pipeline · Metrics · Epoch Analysis · Audio · Speed</p>
      </div>
    </div>
  )
}

// ─── CALL TO ACTION ───────────────────────────────────────────────────
const CTASection = ({ onDemo }: { onDemo: () => void }) => (
  <div className="min-h-screen flex flex-col justify-center items-center px-8">
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className="text-center max-w-3xl"
    >
      <h2 className="text-6xl font-bold mb-8">Hear the Difference</h2>
      <p className="text-xl text-gray-300 mb-3 leading-relaxed">
        Compare cloned Qinche voice across training epochs, models, and acceleration methods.
      </p>
      <p className="text-sm text-gray-500 mb-10">Metrics below are from the best model: Qwen3-TTS v5 Epoch 3 (CUDA Graph)</p>
        <div className="grid grid-cols-3 gap-5 mb-12">
          {[
            { val: '0.357×', label: 'Best RTF', color: 'text-blue-400' },
            { val: '0.6892', label: 'Best SIM_gt', color: 'text-purple-400' },
            { val: '4.25%', label: 'WER', color: 'text-green-400' },
          ].map(s => (
          <div key={s.label} className="bg-white/5 border border-white/10 rounded-xl p-5">
            <div className={`text-3xl font-bold mb-1 ${s.color}`}>{s.val}</div>
            <div className="text-gray-400 text-sm">{s.label}</div>
          </div>
        ))}
      </div>
      <motion.button
        whileHover={{ scale: 1.03 }}
        whileTap={{ scale: 0.97 }}
        onClick={onDemo}
        className="px-10 py-4 bg-white text-black rounded-lg font-bold text-base flex items-center gap-3 mx-auto hover:bg-gray-100 transition-all"
      >
        <Mic2 size={18} /> Listen to Samples
      </motion.button>
    </motion.div>
  </div>
)

// ─── MAIN ─────────────────────────────────────────────────────────────
export default function LandingPage() {
  const navigate = useNavigate()
  const [showDashboard, setShowDashboard] = useState(false)

  return (
    <div className="relative bg-black text-white">
      {/* Background */}
      <div className="fixed inset-0 z-0" style={{
        backgroundImage: 'url("/landing-page.png")',
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        willChange: 'transform',
        backfaceVisibility: 'hidden',
        transform: 'translateZ(0)',
      }} />
      <div className="fixed inset-0 z-0 bg-black/30" />

      <div className="relative z-10">
        <HeroSection onDemo={() => navigate('/demo')} onEmotionalTTS={() => navigate('/emotional-tts')} />
        <MetricsSection />
        <FeaturesSection />
        <InnovationsSection onDashboard={() => setShowDashboard(true)} />
        <CTASection onDemo={() => navigate('/demo')} />
      </div>

      <AnimatePresence>
        {showDashboard && <Dashboard onClose={() => setShowDashboard(false)} />}
      </AnimatePresence>
    </div>
  )
}
