import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LabelList,
} from 'recharts'
import {
  ExternalLink, Presentation, AlertTriangle, HelpCircle, Workflow, Microscope, Sliders,
  TrendingDown, Gauge, GitBranch, AlertOctagon, Box, Sparkles, Heart,
  ChevronLeft, ChevronRight,
} from 'lucide-react'

/* ─── Shared palette (matches the rest of the site) ─────────────────────── */

const SCALE_COLORS: Record<string, string> = {
  binary: '#2196F3',
  ternary: '#4CAF50',
  quaternary: '#FF9800',
  continuous: '#E91E63',
}

/* ─── Data from data/json_consistency_results.json (5-repeat run) ───────── */

const IDENTICAL_JSON_DATA = [
  { scale: 'Binary',     pct: 60 },
  { scale: 'Ternary',    pct: 50 },
  { scale: 'Quaternary', pct: 40 },
  { scale: 'Continuous', pct: 0  },
]

const LOGPROB_TABLE = [
  { scale: 'Binary',     meanProb: 0.9821, meanEntropy: 0.0615, minProb: 0.500, color: SCALE_COLORS.binary },
  { scale: 'Ternary',    meanProb: 0.9809, meanEntropy: 0.0637, minProb: 0.500, color: SCALE_COLORS.ternary },
  { scale: 'Quaternary', meanProb: 0.9718, meanEntropy: 0.0978, minProb: 0.500, color: SCALE_COLORS.quaternary },
  { scale: 'Continuous', meanProb: 0.9432, meanEntropy: 0.1964, minProb: 0.243, color: SCALE_COLORS.continuous },
]

/* ─── Data from data/json_consistency_analysis.json (per-family) ────────── */

const FAMILY_INSTABILITY = [
  { family: 'concreteness',   pct: 20.0 },
  { family: 'negation',       pct: 15.0 },
  { family: 'spatial',        pct: 12.5 },
  { family: 'intent',         pct: 12.5 },
  { family: 'actions_events', pct: 10.0 },
  { family: 'causality',      pct: 10.0 },
  { family: 'numeric',        pct: 5.0  },
  { family: 'emotion',        pct: 5.0  },
  { family: 'social',         pct: 5.0  },
  { family: 'dialogue',       pct: 5.0  },
  { family: 'identity',       pct: 5.0  },
  { family: 'uncertainty',    pct: 2.5  },
  { family: 'sentiment',      pct: 2.5  },
  { family: 'first_person',   pct: 2.5  },
  { family: 'comparison',     pct: 2.5  },
  { family: 'named_entities', pct: 0.0  },
  { family: 'temporal',       pct: 0.0  },
  { family: 'modality',       pct: 0.0  },
  { family: 'imperative',     pct: 0.0  },
  { family: 'normative',      pct: 0.0  },
]

/* ─── Top "Research" table (Scale Degradation Summary) ──────────────────── */
/* Mirrors the headline table on ResearchTab + LogprobsTab so the audience  */
/* only needs this single page. Values from the logprobs run (run 2).      */

const SCALE_SUMMARY = [
  { scale: 'binary',     meanVar: 0.002788, zeroVarPct: 98.5, highVarPct: 1.5,  meanConsistency: 0.9957 },
  { scale: 'ternary',    meanVar: 0.003463, zeroVarPct: 94.5, highVarPct: 5.5,  meanConsistency: 0.9878 },
  { scale: 'quaternary', meanVar: 0.005605, zeroVarPct: 87.5, highVarPct: 11.0, meanConsistency: 0.9745 },
  { scale: 'continuous', meanVar: 0.001610, zeroVarPct: 79.5, highVarPct: 4.0,  meanConsistency: 0.9782 },
]

/* Reliability coefficients from logprobs run */
const RELIABILITY = [
  { scale: 'binary',     icc: 0.984, kappa: 0.985, kAlpha: 0.984 },
  { scale: 'ternary',    icc: 0.976, kappa: 0.994, kAlpha: 0.977 },
  { scale: 'quaternary', icc: 0.954, kappa: 1.000, kAlpha: 0.952 },
  { scale: 'continuous', icc: 0.962, kappa: null,  kAlpha: 0.961 },
]

/* ─── Slide registry — used for the sticky nav at the top ───────────────── */

const SLIDES = [
  { id: 'cover',    label: '1 · Cover' },
  { id: 'problem',  label: '2 · Problem' },
  { id: 'question', label: '3 · Research Q' },
  { id: 'method',   label: '4 · Method' },
  { id: 'probes',   label: '5 · Probes' },
  { id: 'scales',   label: '6 · Scales' },
  { id: 'summary',  label: '7 · Headline table' },
  { id: 'finding1', label: '8 · Finding 1' },
  { id: 'finding2', label: '9 · Finding 2' },
  { id: 'finding3', label: '10 · Finding 3' },
  { id: 'finding4', label: '11 · Finding 4' },
  { id: 'pca',      label: '12 · PCA' },
  { id: 'meaning',  label: '13 · So what?' },
  { id: 'thanks',   label: '14 · Thanks' },
]

const DEMO_URL = 'https://grading-llm.vercel.app/'

/* ─── Tiny helpers ──────────────────────────────────────────────────────── */

function SlideCard({
  id, num, total, kicker, title, icon: Icon, accent = 'indigo', children,
}: {
  id: string; num: number; total: number; kicker?: string; title: string
  icon: any; accent?: string; children: React.ReactNode
}) {
  return (
    <section id={id} className="scroll-mt-32">
      <div className="card relative overflow-hidden">
        {/* Slide number ribbon */}
        <div className="absolute top-0 right-0 px-3 py-1 text-[11px] font-mono text-gray-400 bg-gray-50 rounded-bl-lg border-l border-b border-gray-100">
          {num} / {total}
        </div>
        <div className="flex items-start gap-3 mb-4">
          <div className={`p-2 rounded-lg bg-${accent}-50`}>
            <Icon className={`w-5 h-5 text-${accent}-600`} />
          </div>
          <div className="flex-1 pr-16">
            {kicker && (
              <div className={`text-[11px] font-semibold uppercase tracking-wider text-${accent}-600 mb-1`}>
                {kicker}
              </div>
            )}
            <h2 className="text-xl font-bold text-gray-900 leading-tight">{title}</h2>
          </div>
        </div>
        <div className="text-gray-700 leading-relaxed">{children}</div>
      </div>
    </section>
  )
}

function Stat({ value, label, accent = 'indigo' }: { value: string; label: string; accent?: string }) {
  return (
    <div className={`bg-${accent}-50 border border-${accent}-200 rounded-lg px-4 py-3 text-center`}>
      <div className={`text-3xl font-bold text-${accent}-700 leading-none`}>{value}</div>
      <div className="text-xs text-gray-600 mt-1.5">{label}</div>
    </div>
  )
}

/* ─── Component ─────────────────────────────────────────────────────────── */

export default function PresentationTab() {
  const [active, setActive] = useState('cover')

  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) setActive(e.target.id)
        })
      },
      { rootMargin: '-30% 0px -60% 0px', threshold: 0 }
    )
    SLIDES.forEach((s) => {
      const el = document.getElementById(s.id)
      if (el) obs.observe(el)
    })
    return () => obs.disconnect()
  }, [])

  const goTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  const idx = SLIDES.findIndex((s) => s.id === active)
  const prev = idx > 0 ? SLIDES[idx - 1] : null
  const next = idx < SLIDES.length - 1 ? SLIDES[idx + 1] : null

  return (
    <div className="space-y-6">
      {/* ── Hero / link banner ──────────────────────────────────────────── */}
      <div className="card bg-gradient-to-br from-indigo-600 via-blue-600 to-cyan-600 text-white border-none">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-indigo-100 mb-2">
              <Presentation className="w-4 h-4" />
              Symposium walkthrough · 14 sections
            </div>
            <h1 className="text-3xl font-bold leading-tight">Grading-LLM</h1>
            <p className="text-indigo-100 mt-1">
              Measuring an LLM&apos;s consistency across answer granularities
            </p>
            <p className="text-sm text-indigo-200 mt-3 max-w-xl">
              Follow along live with the talk. Every chart on this page comes from one of the
              16,000-evaluation batch runs over <strong>10 statement archetypes × 20 mechanistic
              questions × 4 scales × 20 repeats</strong>. You can rerun any of it on your own
              statement from the Analyze tab.
            </p>
          </div>
          <a
            href={DEMO_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-5 py-3 bg-white text-indigo-700 font-semibold rounded-lg hover:bg-indigo-50 transition-colors shadow-md"
          >
            <ExternalLink className="w-4 h-4" />
            grading-llm.vercel.app
          </a>
        </div>
      </div>

      {/* ── Sticky slide nav ─────────────────────────────────────────────── */}
      <div className="sticky top-0 z-20 bg-white/95 backdrop-blur border-b border-gray-200 -mx-4 px-4 py-2">
        <div className="flex items-center gap-2">
          <button
            disabled={!prev}
            onClick={() => prev && goTo(prev.id)}
            className="p-1.5 rounded hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed"
            title={prev ? `Previous: ${prev.label}` : ''}
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <div className="flex-1 overflow-x-auto">
            <div className="flex gap-1.5 min-w-max">
              {SLIDES.map((s) => (
                <button
                  key={s.id}
                  onClick={() => goTo(s.id)}
                  className={`px-2.5 py-1 text-[11px] font-medium rounded transition-colors whitespace-nowrap ${
                    active === s.id
                      ? 'bg-indigo-600 text-white'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>
          <button
            disabled={!next}
            onClick={() => next && goTo(next.id)}
            className="p-1.5 rounded hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed"
            title={next ? `Next: ${next.label}` : ''}
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* ── 1. Cover ─────────────────────────────────────────────────────── */}
      <SlideCard id="cover" num={1} total={14} kicker="STAT 298 · Interpretability research"
        title="If an LLM grades the same statement twice, does it agree with itself?"
        icon={Presentation} accent="indigo"
      >
        <p>
          We probe <strong>1,600 evaluations per statement</strong> to find out where, and why, the
          model changes its mind, even at temperature 0.
        </p>
        <p className="mt-3 text-sm text-gray-500">Atharva Patel · Spring 2026</p>
      </SlideCard>

      {/* ── 2. Problem ───────────────────────────────────────────────────── */}
      <SlideCard id="problem" num={2} total={14} kicker="The problem"
        title="LLMs are unreliable judges, even at temperature 0"
        icon={AlertTriangle} accent="amber"
      >
        <p>
          Researchers increasingly use LLMs to grade other models, rate alignment, score rubrics,
          and interpret behavior. If the same prompt produces a different score on re-run,{' '}
          <strong>every downstream metric inherits that noise.</strong>
        </p>
        <div className="mt-4 grid md:grid-cols-3 gap-3">
          <div className="border border-gray-200 rounded-lg p-3">
            <div className="text-xs font-mono text-gray-400 mb-1">01</div>
            <div className="font-semibold text-gray-900 text-sm">Where does it change its mind?</div>
          </div>
          <div className="border border-gray-200 rounded-lg p-3">
            <div className="text-xs font-mono text-gray-400 mb-1">02</div>
            <div className="font-semibold text-gray-900 text-sm">Does finer granularity make it worse?</div>
          </div>
          <div className="border border-gray-200 rounded-lg p-3">
            <div className="text-xs font-mono text-gray-400 mb-1">03</div>
            <div className="font-semibold text-gray-900 text-sm">Can token-level confidence catch it?</div>
          </div>
        </div>
        <div className="mt-4 grid grid-cols-2 gap-3">
          <Stat value="60%" label="of binary-scale runs reproduce the same JSON byte-for-byte" accent="green" />
          <Stat value="0%" label="for the continuous scale" accent="red" />
        </div>
      </SlideCard>

      {/* ── 3. Research Question ─────────────────────────────────────────── */}
      <SlideCard id="question" num={3} total={14} kicker="Research question"
        title="Does answer granularity predict where the judge becomes unstable?"
        icon={HelpCircle} accent="purple"
      >
        <p>
          And can PCA recover those unstable directions as interpretable semantic axes?
        </p>
        <div className="mt-4 bg-purple-50 border border-purple-200 rounded-lg p-4">
          <div className="text-xs font-semibold uppercase tracking-wider text-purple-700 mb-1.5">
            Hypothesis
          </div>
          <p className="text-purple-900">
            Finer scales offer more numerical degrees of freedom, so the model spends those degrees
            on noise rather than signal, and PCA on grading vectors reveals which factor families
            absorb the wobble.
          </p>
        </div>
      </SlideCard>

      {/* ── 4. Method ────────────────────────────────────────────────────── */}
      <SlideCard id="method" num={4} total={14} kicker="Method"
        title="1,600 graded responses per statement"
        icon={Workflow} accent="blue"
      >
        <div className="grid md:grid-cols-4 gap-3">
          {[
            { n: '1', title: 'Statement', desc: 'One text input (claim or excerpt)' },
            { n: '2', title: '20 Questions', desc: 'Probe orthogonal semantic features' },
            { n: '3', title: '4 Scales × 20 reps', desc: 'Binary / Ternary / Quat. / Continuous' },
            { n: '4', title: 'Analyze', desc: 'Variance, entropy, mode, 3-D PCA' },
          ].map((s) => (
            <div key={s.n} className="border border-gray-200 rounded-lg p-3">
              <div className="text-2xl font-bold text-blue-600 mb-1">{s.n}</div>
              <div className="font-semibold text-gray-900 text-sm">{s.title}</div>
              <div className="text-xs text-gray-500 mt-1">{s.desc}</div>
            </div>
          ))}
        </div>
        <div className="mt-4 bg-blue-50 border border-blue-200 rounded-lg p-4 text-center font-mono text-sm">
          20 questions × 4 scales × 20 repeats ={' '}
          <strong className="text-blue-700">1,600 API calls</strong> / statement
        </div>
        <p className="mt-3 text-xs text-gray-500 text-center">
          All saved batch results on this page use the <strong className="text-blue-700">mechanistic</strong> question
          bank (<code>data/questions_mech.json</code>). The interpretability bank is wired into the live
          Analyze tab so you can compare yourself, but is not part of the 16,000-grade dataset shown here.
        </p>
      </SlideCard>

      {/* ── 5. Probes (mech vs interp) ──────────────────────────────────── */}
      <SlideCard id="probes" num={5} total={14} kicker="Probe design"
        title="Two question modes: surface form vs inferred meaning"
        icon={Microscope} accent="purple"
      >
        <div className="grid md:grid-cols-2 gap-4">
          <div className="border border-blue-200 rounded-lg p-4 bg-blue-50/40">
            <div className="text-xs font-semibold uppercase tracking-wider text-blue-700 mb-1">
              Mechanistic mode
            </div>
            <div className="font-semibold text-gray-900 mb-2">Explicit linguistic features</div>
            <ul className="text-sm space-y-1.5 text-gray-700 list-disc pl-5">
              <li>Does it mention a specific person by name?</li>
              <li>Does it contain explicit negation?</li>
              <li>Does it use first-person pronouns?</li>
              <li>Does it describe an action being performed?</li>
              <li>Does it contain explicit numbers?</li>
            </ul>
          </div>
          <div className="border border-pink-200 rounded-lg p-4 bg-pink-50/40">
            <div className="text-xs font-semibold uppercase tracking-wider text-pink-700 mb-1">
              Interpretability mode
            </div>
            <div className="font-semibold text-gray-900 mb-2">Implicit meaning &amp; inference</div>
            <ul className="text-sm space-y-1.5 text-gray-700 list-disc pl-5">
              <li>Is the speaker&apos;s tone ambiguous?</li>
              <li>What unstated reasons are implied?</li>
              <li>Is there an implicit power dynamic?</li>
              <li>What does the speaker leave out?</li>
              <li>How does context change the meaning?</li>
            </ul>
          </div>
        </div>
      </SlideCard>

      {/* ── 6. Scales ────────────────────────────────────────────────────── */}
      <SlideCard id="scales" num={6} total={14} kicker="Answer granularity"
        title="Four scales, one prompt, more numerical freedom"
        icon={Sliders} accent="orange"
      >
        <div className="grid md:grid-cols-4 gap-3">
          {[
            { name: 'Binary',     opts: '2 options', vals: '{0, 1}',              c: SCALE_COLORS.binary },
            { name: 'Ternary',    opts: '3 options', vals: '{0, 0.5, 1}',         c: SCALE_COLORS.ternary },
            { name: 'Quaternary', opts: '4 options', vals: '{0, 0.33, 0.66, 1}',  c: SCALE_COLORS.quaternary },
            { name: 'Continuous', opts: '∞ options', vals: '[0, 1]',              c: SCALE_COLORS.continuous },
          ].map((s) => (
            <div key={s.name} className="border-2 rounded-lg p-3" style={{ borderColor: s.c }}>
              <div className="font-bold text-lg" style={{ color: s.c }}>{s.name}</div>
              <div className="text-xs text-gray-500 mb-2">{s.opts}</div>
              <div className="text-xs font-mono text-gray-700">{s.vals}</div>
            </div>
          ))}
        </div>
        <p className="mt-4 text-sm text-gray-500 italic">
          Same prompt, same temperature: only the allowed numeric vocabulary changes.
        </p>
      </SlideCard>

      {/* ── 7. Headline table (Research-tab top table) ──────────────────── */}
      <SlideCard id="summary" num={7} total={14} kicker="Headline table"
        title="Scale degradation summary (16,000 evaluations)"
        icon={TrendingDown} accent="blue"
      >
        <p className="text-sm text-gray-600 mb-3">
          The top-of-page table from the Research and Logprobs tabs. Each row aggregates 4,000
          grades (10 texts × 20 mechanistic questions × 20 samples). Total = 16,000.
        </p>

        {/* The 10 archetypes that get graded */}
        <details className="mb-4 border border-gray-200 rounded-lg overflow-hidden">
          <summary className="cursor-pointer px-4 py-2.5 bg-gray-50 hover:bg-gray-100 text-sm font-medium text-gray-700 flex items-center justify-between">
            <span>The 10 statement archetypes (click to expand)</span>
            <span className="text-xs text-gray-400 font-mono">data/batch_input_10.jsonl</span>
          </summary>
          <div className="divide-y divide-gray-100">
            {[
              { id: 'factual_simple',         tag: 'Factual',     text: 'The Eiffel Tower was completed in 1889 and stands 330 meters tall in Paris, France.' },
              { id: 'sentiment_positive',     tag: 'Evaluative+', text: 'The community garden transformed our neighborhood — strangers became friends, children learned patience, and even the most skeptical residents admitted the fresh tomatoes were worth the effort.' },
              { id: 'sentiment_negative',     tag: 'Evaluative−', text: 'The project was a catastrophic failure that wasted millions in taxpayer money, displaced vulnerable families, and left behind nothing but empty promises and crumbling infrastructure.' },
              { id: 'ambiguous',              tag: 'Ambiguous',   text: 'She said she was fine with the decision, but everyone in the room knew that wasn’t the whole story.' },
              { id: 'medical_clinical',       tag: 'Clinical',    text: 'Patient presents with acute onset chest pain radiating to the left arm, diaphoresis, and shortness of breath. ECG shows ST-elevation in leads II, III, and aVF. Troponin levels pending.' },
              { id: 'negation_heavy',         tag: 'Negation',    text: 'The study found no significant evidence that the treatment was neither ineffective nor harmful, leaving researchers unable to draw any definitive conclusions.' },
              { id: 'imperative_action',      tag: 'Imperative',  text: 'Immediately evacuate the building through the nearest emergency exit. Do not use the elevators. Proceed to the designated assembly point and wait for further instructions from emergency personnel.' },
              { id: 'abstract_philosophical', tag: 'Abstract',    text: 'Consciousness may be less like a light switch and more like a dimmer — not something that is simply present or absent, but something that exists in degrees across a spectrum we barely understand.' },
              { id: 'narrative_paragraph',    tag: 'Narrative',   text: 'Maria had been working at the clinic for twelve years when the new director arrived. Within weeks, he had restructured the scheduling system, fired two senior nurses, and implemented a policy requiring all staff to log their breaks. Morale plummeted. Patients noticed the tension. By March, three more staff members had quietly submitted their resignations.' },
              { id: 'technical_ml',           tag: 'Technical',   text: 'The transformer architecture uses multi-head self-attention to compute weighted representations of input tokens, where attention weights are derived from scaled dot-product similarity between query and key projections, enabling the model to capture long-range dependencies without recurrence.' },
            ].map((t, i) => (
              <div key={t.id} className="px-4 py-2.5 text-sm">
                <div className="flex items-center gap-2 mb-1">
                  <span className="w-5 h-5 rounded-full bg-gray-100 text-gray-600 flex items-center justify-center text-[10px] font-bold font-mono">
                    {i + 1}
                  </span>
                  <code className="text-[11px] font-mono text-gray-700 bg-gray-100 px-1.5 py-0.5 rounded">{t.id}</code>
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-gray-500">{t.tag}</span>
                </div>
                <p className="text-gray-700 leading-relaxed pl-7">{t.text}</p>
              </div>
            ))}
          </div>
        </details>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-2 pr-4 font-medium text-gray-500">Scale</th>
                <th className="text-right py-2 px-3 font-medium text-gray-500">Mean Var</th>
                <th className="text-right py-2 px-3 font-medium text-gray-500">% Zero-Var</th>
                <th className="text-right py-2 px-3 font-medium text-gray-500">% High-Var</th>
                <th className="text-right py-2 pl-3 font-medium text-gray-500">Mean Mode Consistency</th>
              </tr>
            </thead>
            <tbody>
              {SCALE_SUMMARY.map((row) => (
                <tr key={row.scale} className="border-b border-gray-100">
                  <td className="py-2 pr-4 capitalize font-medium" style={{ color: SCALE_COLORS[row.scale] }}>
                    {row.scale}
                  </td>
                  <td className="text-right py-2 px-3 font-mono">{row.meanVar.toFixed(4)}</td>
                  <td className="text-right py-2 px-3 font-mono">{row.zeroVarPct.toFixed(1)}%</td>
                  <td className="text-right py-2 px-3 font-mono">{row.highVarPct.toFixed(1)}%</td>
                  <td className="text-right py-2 pl-3 font-mono">{(row.meanConsistency * 100).toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-5">
          <div className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-2">
            Reliability coefficients (logprobs run)
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-2 pr-4 font-medium text-gray-500">Scale</th>
                <th className="text-right py-2 px-3 font-medium text-gray-500">ICC</th>
                <th className="text-right py-2 px-3 font-medium text-gray-500">Cohen&apos;s κ</th>
                <th className="text-right py-2 pl-3 font-medium text-gray-500">Krippendorff α</th>
              </tr>
            </thead>
            <tbody>
              {RELIABILITY.map((row) => (
                <tr key={row.scale} className="border-b border-gray-100">
                  <td className="py-2 pr-4 capitalize font-medium" style={{ color: SCALE_COLORS[row.scale] }}>
                    {row.scale}
                  </td>
                  <td className="text-right py-2 px-3 font-mono">{row.icc.toFixed(3)}</td>
                  <td className="text-right py-2 px-3 font-mono">
                    {row.kappa === null ? <span className="text-gray-400">n/a</span> : row.kappa.toFixed(3)}
                  </td>
                  <td className="text-right py-2 pl-3 font-mono">{row.kAlpha.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-gray-500">
            Cohen&apos;s κ is undefined for continuous (no discrete categories). Cronbach&apos;s α
            is omitted here because it divides by zero on saturated cells; see the Logprobs tab for
            the full Tier 1-3 panel.
          </p>
        </div>

        <div className="mt-4 bg-blue-50 border border-blue-200 rounded-lg p-4 text-sm text-blue-800">
          <p className="font-semibold mb-1">Read this row by row.</p>
          <p>
            Variance rises binary → ternary → quaternary, then{' '}
            <em>drops</em> on continuous, but continuous has the lowest zero-variance rate
            (79.5%). Quiet, distributed noise instead of loud bin-flipping.
          </p>
        </div>
      </SlideCard>

      {/* ── 8. Finding 1 ─────────────────────────────────────────────────── */}
      <SlideCard id="finding1" num={8} total={14} kicker="Finding 01"
        title="More numerical choices → fewer identical re-runs"
        icon={TrendingDown} accent="orange"
      >
        <p className="text-sm text-gray-600 mb-3">
          Percent of (text × scale) pairs where 5 repeats produced byte-identical JSON.
          <span className="ml-2 text-gray-400">
            Source: <code>json_consistency_results.json</code>
          </span>
        </p>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={IDENTICAL_JSON_DATA} margin={{ top: 24, right: 12, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="scale" />
            <YAxis tickFormatter={(v) => `${v}%`} domain={[0, 100]} />
            <Tooltip formatter={(v: number) => `${v}%`} />
            <Bar dataKey="pct">
              {IDENTICAL_JSON_DATA.map((entry) => (
                <Cell key={entry.scale} fill={SCALE_COLORS[entry.scale.toLowerCase()]} />
              ))}
              <LabelList dataKey="pct" position="top" formatter={(v: number) => `${v}%`} fill="#374151" fontSize={12} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="mt-4 bg-orange-50 border border-orange-200 rounded-lg p-4 text-sm text-orange-900">
          <p className="font-semibold mb-1">Reproducibility collapses</p>
          <p>
            Even with <code>temperature=0</code> and <code>response_format=json_object</code>, the
            model fails every single time on the continuous scale (0 / 10). Granularity isn&apos;t
            ornamental; it leaks into the sampler.
          </p>
        </div>
      </SlideCard>

      {/* ── 9. Finding 2 ─────────────────────────────────────────────────── */}
      <SlideCard id="finding2" num={9} total={14} kicker="Finding 02"
        title="Token-level confidence tracks the same gradient"
        icon={Gauge} accent="indigo"
      >
        <p className="text-sm text-gray-600 mb-3">
          Mean chosen-token probability and entropy over the numeric score tokens in the JSON
          response.
        </p>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200">
              <th className="text-left py-2 pr-4 font-medium text-gray-500">Scale</th>
              <th className="text-right py-2 px-3 font-medium text-gray-500">Mean prob ↑</th>
              <th className="text-right py-2 px-3 font-medium text-gray-500">Mean entropy ↓</th>
              <th className="text-right py-2 pl-3 font-medium text-gray-500">Min prob</th>
            </tr>
          </thead>
          <tbody>
            {LOGPROB_TABLE.map((row) => (
              <tr key={row.scale} className="border-b border-gray-100">
                <td className="py-2 pr-4 font-medium" style={{ color: row.color }}>{row.scale}</td>
                <td className="text-right py-2 px-3 font-mono">{row.meanProb.toFixed(4)}</td>
                <td className="text-right py-2 px-3 font-mono">{row.meanEntropy.toFixed(4)}</td>
                <td className="text-right py-2 pl-3 font-mono">{row.minProb.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="mt-4 bg-indigo-50 border border-indigo-200 rounded-lg p-4 text-sm text-indigo-900">
          <p className="font-semibold mb-1">What it means</p>
          <p>
            Mean token entropy <strong>triples</strong> from binary to continuous. Min chosen-token
            probability collapses from <strong>0.50 → 0.24</strong>: the model genuinely doesn&apos;t
            know which decimal to emit next. Logprobs are early-warning signals, not safety
            guarantees.
          </p>
        </div>
      </SlideCard>

      {/* ── 10. Finding 3 ────────────────────────────────────────────────── */}
      <SlideCard id="finding3" num={10} total={14} kicker="Finding 03"
        title="Some semantic factors are systematically less stable"
        icon={GitBranch} accent="green"
      >
        <p className="text-sm text-gray-600 mb-3">
          Percent of (text × scale) cells where 5 repeats disagreed, by factor family.
        </p>
        <ResponsiveContainer width="100%" height={420}>
          <BarChart data={FAMILY_INSTABILITY} layout="vertical" margin={{ left: 80, right: 24 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" tickFormatter={(v) => `${v}%`} domain={[0, 25]} />
            <YAxis type="category" dataKey="family" width={120} tick={{ fontSize: 11 }} />
            <Tooltip formatter={(v: number) => `${v}%`} />
            <Bar dataKey="pct" fill="#22c55e">
              {FAMILY_INSTABILITY.map((entry, i) => {
                const ratio = entry.pct / 25
                const c = ratio > 0.6 ? '#ef4444' : ratio > 0.3 ? '#f59e0b' : ratio > 0 ? '#84cc16' : '#22c55e'
                return <Cell key={i} fill={c} />
              })}
              <LabelList dataKey="pct" position="right" formatter={(v: number) => `${v}%`} fill="#374151" fontSize={11} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p className="mt-3 text-sm text-gray-600 italic">
          Concrete vs abstract, polarity flips, spatial language, inferred intent: the same
          categories humans disagree on.
        </p>
      </SlideCard>

      {/* ── 11. Finding 4 ────────────────────────────────────────────────── */}
      <SlideCard id="finding4" num={11} total={14} kicker="Finding 04"
        title="High confidence ≠ stable answer"
        icon={AlertOctagon} accent="red"
      >
        <p className="text-sm text-gray-600 mb-4">
          Cases where the model emitted near-certain logprobs yet still flipped its score:
        </p>
        <div className="grid md:grid-cols-2 gap-3">
          <div className="border border-gray-200 rounded-lg p-4">
            <div className="text-xs font-mono text-gray-500 mb-1">
              narrative_paragraph / quaternary / Q19 (identity)
            </div>
            <div className="font-mono text-sm bg-gray-50 p-2 rounded mb-2">
              5 repeats → [0, 1, 1, 1, 1]
            </div>
            <div className="text-xs text-gray-600">
              Variance <strong>0.16</strong> · Mean token prob <strong>0.949</strong> · Min{' '}
              <strong>0.82</strong>
            </div>
            <div className="mt-2 text-sm text-red-700">
              → Model is 95% sure each time, and still produces a different answer once.
            </div>
          </div>
          <div className="border border-gray-200 rounded-lg p-4">
            <div className="text-xs font-mono text-gray-500 mb-1">
              sentiment_negative / binary / Q7 (negation)
            </div>
            <div className="font-mono text-sm bg-gray-50 p-2 rounded mb-2">
              5 repeats → [0, 1, 0, 1, 0]
            </div>
            <div className="text-xs text-gray-600">
              Variance <strong>0.24</strong> · Mean token prob <strong>0.665</strong> · Min{' '}
              <strong>0.56</strong>
            </div>
            <div className="mt-2 text-sm text-gray-600">
              → Here logprobs do correlate; the exception, not the rule.
            </div>
          </div>
        </div>
        <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="text-xs font-semibold uppercase tracking-wider text-red-700 mb-1">
            Mann-Whitney U · stable vs unstable cells (mean token prob)
          </div>
          <div className="flex items-baseline gap-4">
            <div className="text-3xl font-bold text-red-700 font-mono">p = 0.059</div>
            <div className="text-sm text-red-800">
              Near-significant, but the magnitude is small: stable mean prob{' '}
              <strong>0.968</strong> vs unstable <strong>0.926</strong>.
            </div>
          </div>
          <p className="mt-2 text-sm text-red-800">
            <strong>Don&apos;t trust confidence alone.</strong>
          </p>
        </div>
      </SlideCard>

      {/* ── 12. PCA ──────────────────────────────────────────────────────── */}
      <SlideCard id="pca" num={12} total={14} kicker="Analytical core"
        title="PCA reveals which semantic axes absorb the wobble"
        icon={Box} accent="indigo"
      >
        <p className="text-sm text-gray-600 mb-4">
          An <strong>80 × 20 matrix per statement</strong> (20 repeats × 4 scales = 80 grading
          vectors, one row per sample, one column per question) goes through an eigendecomposition
          into 3 PCs that typically explain 60-80% of the spread. Star markers are scale centroids.
          Tight clusters mean the model is consistent. Spread clusters mean the model is changing
          its mind on those questions.
        </p>

        <div className="grid md:grid-cols-3 gap-4 mb-4">
          {[
            {
              id: 'factual_simple',
              label: 'factual_simple',
              desc: 'Stable. All four scales collapse near the origin.',
            },
            {
              id: 'ambiguous',
              label: 'ambiguous',
              desc: 'Quaternary and continuous (orange / pink) drift far from the binary cluster.',
            },
            {
              id: 'narrative_paragraph',
              label: 'narrative_paragraph',
              desc: 'Quaternary spreads dramatically; the wobble lives in mid-resolution scales.',
            },
          ].map((t) => (
            <figure key={t.id} className="border border-gray-200 rounded-lg overflow-hidden bg-white">
              <img
                src={`/presentation/pca_${t.id}.png`}
                alt={`PCA of 80 grading vectors for ${t.label}`}
                className="w-full"
                loading="lazy"
              />
              <figcaption className="p-3 text-xs text-gray-600 border-t border-gray-200">
                <div className="font-mono font-semibold text-gray-800 mb-1">{t.label}</div>
                <div>{t.desc}</div>
              </figcaption>
            </figure>
          ))}
        </div>

        <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4 text-sm">
          <div className="grid md:grid-cols-3 gap-4">
            <div>
              <div className="text-xs font-semibold uppercase tracking-wider text-indigo-700 mb-1">
                Input matrix
              </div>
              <div className="font-mono text-lg text-gray-900">80 × 20</div>
              <div className="text-xs text-gray-600">samples × questions; each cell in [0, 1]</div>
            </div>
            <div>
              <div className="text-xs font-semibold uppercase tracking-wider text-indigo-700 mb-1">
                Decompose
              </div>
              <div className="font-mono text-lg text-gray-900">3 PCs</div>
              <div className="text-xs text-gray-600">center → cov → eigendecomposition</div>
            </div>
            <div>
              <div className="text-xs font-semibold uppercase tracking-wider text-indigo-700 mb-1">
                Interpret
              </div>
              <div className="font-mono text-lg text-gray-900">3-D scatter</div>
              <div className="text-xs text-gray-600">colored by scale; loadings = wobble drivers</div>
            </div>
          </div>
          <p className="mt-3 text-indigo-900">
            <strong>Want a live one?</strong> Switch to the <em>Analyze Statement</em> tab, paste
            any sentence, and you get the interactive 3-D PCA on your own text.
          </p>
        </div>
      </SlideCard>

      {/* ── 13. So what ──────────────────────────────────────────────────── */}
      <SlideCard id="meaning" num={13} total={14} kicker="What it means"
        title="Implications for LLM-as-judge evaluation"
        icon={Sparkles} accent="purple"
      >
        <div className="space-y-3">
          {[
            {
              n: '01',
              h: 'Pick the coarsest scale that still answers your question.',
              b: 'If binary captures your decision boundary, the variance you save matters more than the precision you lose.',
            },
            {
              n: '02',
              h: 'Probe per factor family, not per metric average.',
              b: 'Five families absorb most of the instability. An averaged score hides which axes wobble.',
            },
            {
              n: '03',
              h: 'Don’t trust logprobs as a stability proxy.',
              b: 'p = 0.059 in our data. Confident tokens still produce different scores. Repeat-and-check is the only honest measure.',
            },
          ].map((row) => (
            <div key={row.n} className="border border-gray-200 rounded-lg p-4 flex gap-4">
              <div className="text-2xl font-bold text-purple-600 font-mono">{row.n}</div>
              <div>
                <div className="font-semibold text-gray-900">{row.h}</div>
                <div className="text-sm text-gray-600 mt-1">{row.b}</div>
              </div>
            </div>
          ))}
        </div>
        <p className="mt-4 text-sm text-gray-600 italic">
          Next: cross-model comparison (gpt-4o ↔ Claude ↔ Llama), longer contexts, and reweighting
          eval rubrics by per-family reliability.
        </p>
      </SlideCard>

      {/* ── 14. Thanks ───────────────────────────────────────────────────── */}
      <SlideCard id="thanks" num={14} total={14} kicker="Thanks for stopping by"
        title="Try it live. Pick a statement. Watch the model wobble."
        icon={Heart} accent="pink"
      >
        <div className="flex flex-col md:flex-row md:items-center gap-4 mt-2">
          <a
            href={DEMO_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-6 py-3 bg-pink-600 text-white font-semibold rounded-lg hover:bg-pink-700 transition-colors shadow-md"
          >
            <ExternalLink className="w-4 h-4" />
            grading-llm.vercel.app
          </a>
          <p className="text-sm text-gray-600">
            Bring your own OpenAI key. Run on a statement of your choice. Inspect the PCA in 3-D.
          </p>
        </div>
        <div className="mt-4 text-sm text-gray-500">
          Atharva Patel &middot;{' '}
          <a className="text-indigo-600 hover:underline" href="mailto:atharvajpatel@berkeley.edu">
            atharvajpatel@berkeley.edu
          </a>{' '}
          &middot;{' '}
          <a className="text-indigo-600 hover:underline" href="https://github.com/atharvajpatel/Grading-LLM" target="_blank" rel="noopener noreferrer">
            github.com/atharvajpatel/Grading-LLM
          </a>
        </div>
      </SlideCard>
    </div>
  )
}
