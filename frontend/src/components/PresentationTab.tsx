import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LabelList,
} from 'recharts'

/* ─── Shared palette (single source of truth) ───────────────────────────── */
/* SCALE_COLORS now comes from the api client (gold luminance ramp), so this  */
/* page's legend stays in sync with every other tab. magnitudeColor gives a   */
/* monotonic-luminance ramp for the family instability chart; PALETTE + CHART  */
/* supply the data-mark and structural chart colors.                          */
import { SCALE_COLORS } from '../api/client'
import { PALETTE, magnitudeColor, CHART } from '../theme/palette'

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
  id, num, total, kicker, title, icon: _icon, accent: _accent = 'indigo', children,
}: {
  // `icon` and `accent` are retained for call-site compatibility only; they no
  // longer affect styling (dynamic accent classes + icons were removed so every
  // slide gets the same cream/ink/gold treatment).
  id: string; num: number; total: number; kicker?: string; title: string
  icon?: any; accent?: string; children: React.ReactNode
}) {
  void _icon; void _accent
  return (
    <section id={id} className="scroll-mt-32">
      <div className="card relative overflow-hidden">
        {/* Slide number ribbon */}
        <div className="absolute top-0 right-0 px-3 py-1 text-[11px] font-mono text-mute bg-cream rounded-bl-lg border-l border-b border-hair">
          {num} / {total}
        </div>
        <div className="mb-4 pr-16">
          {kicker && <div className="panel-title mb-1.5">{kicker}</div>}
          <h2 className="section-title text-xl">{title}</h2>
          <div className="metric-rule" />
        </div>
        <div className="text-ink leading-relaxed">{children}</div>
      </div>
    </section>
  )
}

function Stat({ value, label, accent: _accent = 'indigo' }: { value: string; label: string; accent?: string }) {
  // `accent` retained for call-site compatibility only; valence now lives in the label text.
  void _accent
  return (
    <div className="border border-hair rounded-ctl px-4 py-3 text-center bg-cream">
      <div className="stat-num text-3xl leading-none">{value}</div>
      <div className="text-xs muted mt-1.5">{label}</div>
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
      <div className="card border-none" style={{ background: PALETTE.ink }}>
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <div className="panel-title mb-2">
              Symposium walkthrough · 14 sections
            </div>
            <h1 className="text-3xl font-bold leading-tight" style={{ color: PALETTE.cream }}>Grading-LLM</h1>
            <p className="mt-1" style={{ color: PALETTE.cream }}>
              Measuring an LLM&apos;s consistency across answer granularities
            </p>
            <p className="text-sm mt-3 max-w-xl" style={{ color: PALETTE.hair }}>
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
            className="btn-primary"
          >
            grading-llm.vercel.app
          </a>
        </div>
      </div>

      {/* ── Sticky slide nav ─────────────────────────────────────────────── */}
      <div className="sticky top-0 z-20 bg-cream/95 backdrop-blur border-b border-hair -mx-4 px-4 py-2">
        <div className="flex items-center gap-2">
          <button
            disabled={!prev}
            onClick={() => prev && goTo(prev.id)}
            className="px-2 py-1 text-[11px] font-medium rounded-ctl text-mute hover:text-ink disabled:opacity-30 disabled:cursor-not-allowed"
            title={prev ? `Previous: ${prev.label}` : ''}
          >
            Prev
          </button>
          <div className="flex-1 overflow-x-auto">
            <div className="flex gap-1.5 min-w-max">
              {SLIDES.map((s) => (
                <button
                  key={s.id}
                  onClick={() => goTo(s.id)}
                  className={`px-2.5 py-1 text-[11px] font-medium rounded-ctl transition-colors whitespace-nowrap ${
                    active === s.id
                      ? 'bg-gold text-white'
                      : 'text-mute hover:text-ink'
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
            className="px-2 py-1 text-[11px] font-medium rounded-ctl text-mute hover:text-ink disabled:opacity-30 disabled:cursor-not-allowed"
            title={next ? `Next: ${next.label}` : ''}
          >
            Next
          </button>
        </div>
      </div>

      {/* ── 1. Cover ─────────────────────────────────────────────────────── */}
      <SlideCard id="cover" num={1} total={14} kicker="STAT 298 · Interpretability research"
        title="If an LLM grades the same statement twice, does it agree with itself?"
      >
        <p>
          We probe <strong>1,600 evaluations per statement</strong> to find out where, and why, the
          model changes its mind, even at temperature 0.
        </p>
        <p className="mt-3 text-sm muted">Atharva Patel · Spring 2026</p>
      </SlideCard>

      {/* ── 2. Problem ───────────────────────────────────────────────────── */}
      <SlideCard id="problem" num={2} total={14} kicker="The problem"
        title="LLMs are unreliable judges, even at temperature 0"
      >
        <p>
          Researchers increasingly use LLMs to grade other models, rate alignment, score rubrics,
          and interpret behavior. If the same prompt produces a different score on re-run,{' '}
          <strong>every downstream metric inherits that noise.</strong>
        </p>
        <div className="mt-4 grid md:grid-cols-3 gap-3">
          <div className="border border-hair rounded-ctl p-3">
            <div className="text-xs font-mono text-mute mb-1">01</div>
            <div className="font-semibold text-ink text-sm">Where does it change its mind?</div>
          </div>
          <div className="border border-hair rounded-ctl p-3">
            <div className="text-xs font-mono text-mute mb-1">02</div>
            <div className="font-semibold text-ink text-sm">Does finer granularity make it worse?</div>
          </div>
          <div className="border border-hair rounded-ctl p-3">
            <div className="text-xs font-mono text-mute mb-1">03</div>
            <div className="font-semibold text-ink text-sm">Can token-level confidence catch it?</div>
          </div>
        </div>
        <div className="mt-4 grid grid-cols-2 gap-3">
          <Stat value="60%" label="reproduce identical JSON — binary scale" />
          <Stat value="0%" label="reproduce identical JSON — continuous scale" />
        </div>
      </SlideCard>

      {/* ── 3. Research Question ─────────────────────────────────────────── */}
      <SlideCard id="question" num={3} total={14} kicker="Research question"
        title="Does answer granularity predict where the judge becomes unstable?"
      >
        <p>
          And can PCA recover those unstable directions as interpretable semantic axes?
        </p>
        <div className="mt-4 callout">
          <div className="panel-title mb-1.5">
            Hypothesis
          </div>
          <p className="text-ink">
            Finer scales offer more numerical degrees of freedom, so the model spends those degrees
            on noise rather than signal, and PCA on grading vectors reveals which factor families
            absorb the wobble.
          </p>
        </div>
      </SlideCard>

      {/* ── 4. Method ────────────────────────────────────────────────────── */}
      <SlideCard id="method" num={4} total={14} kicker="Method"
        title="1,600 graded responses per statement"
      >
        <div className="grid md:grid-cols-4 gap-3">
          {[
            { n: '1', title: 'Statement', desc: 'One text input (claim or excerpt)' },
            { n: '2', title: '20 Questions', desc: 'Probe orthogonal semantic features' },
            { n: '3', title: '4 Scales × 20 reps', desc: 'Binary / Ternary / Quat. / Continuous' },
            { n: '4', title: 'Analyze', desc: 'Variance, entropy, mode, 3-D PCA' },
          ].map((s) => (
            <div key={s.n} className="border border-hair rounded-ctl p-3">
              <div className="text-2xl font-bold text-gold mb-1">{s.n}</div>
              <div className="font-semibold text-ink text-sm">{s.title}</div>
              <div className="text-xs muted mt-1">{s.desc}</div>
            </div>
          ))}
        </div>
        <div className="mt-4 callout text-center font-mono text-sm">
          20 questions × 4 scales × 20 repeats ={' '}
          <strong className="text-ink">1,600 API calls</strong> / statement
        </div>
        <p className="mt-3 text-xs muted text-center">
          All saved batch results on this page use the <strong className="text-ink">mechanistic</strong> question
          bank (<code>data/questions_mech.json</code>). The interpretability bank is wired into the live
          Analyze tab so you can compare yourself, but is not part of the 16,000-grade dataset shown here.
        </p>
      </SlideCard>

      {/* ── 5. Probes (mech vs interp) ──────────────────────────────────── */}
      <SlideCard id="probes" num={5} total={14} kicker="Probe design"
        title="Two question modes: surface form vs inferred meaning"
      >
        <div className="grid md:grid-cols-2 gap-4">
          <div className="callout p-4">
            <div className="panel-title mb-1">
              Mechanistic mode
            </div>
            <div className="font-semibold text-ink mb-2">Explicit linguistic features</div>
            <ul className="text-sm space-y-1.5 text-ink list-disc pl-5">
              <li>Does it mention a specific person by name?</li>
              <li>Does it contain explicit negation?</li>
              <li>Does it use first-person pronouns?</li>
              <li>Does it describe an action being performed?</li>
              <li>Does it contain explicit numbers?</li>
            </ul>
          </div>
          <div className="callout-ink callout p-4">
            <div className="panel-title mb-1" style={{ color: PALETTE.ink }}>
              Interpretability mode
            </div>
            <div className="font-semibold text-ink mb-2">Implicit meaning &amp; inference</div>
            <ul className="text-sm space-y-1.5 text-ink list-disc pl-5">
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
      >
        <div className="grid md:grid-cols-4 gap-3">
          {[
            { name: 'Binary',     opts: '2 options', vals: '{0, 1}',              c: SCALE_COLORS.binary },
            { name: 'Ternary',    opts: '3 options', vals: '{0, 0.5, 1}',         c: SCALE_COLORS.ternary },
            { name: 'Quaternary', opts: '4 options', vals: '{0, 0.33, 0.66, 1}',  c: SCALE_COLORS.quaternary },
            { name: 'Continuous', opts: '∞ options', vals: '[0, 1]',              c: SCALE_COLORS.continuous },
          ].map((s) => (
            <div
              key={s.name}
              className="rounded-ctl p-3"
              style={{ border: `2px solid ${s.c}`, boxShadow: `inset 0 0 0 1px ${PALETTE.hair}` }}
            >
              <div className="font-bold text-lg flex items-center gap-2" style={{ color: PALETTE.ink }}>
                <span
                  className="inline-block w-3 h-3 rounded-full"
                  style={{ background: s.c, border: `1px solid ${PALETTE.hair}` }}
                />
                {s.name}
              </div>
              <div className="text-xs muted mb-2">{s.opts}</div>
              <div className="text-xs font-mono text-ink">{s.vals}</div>
            </div>
          ))}
        </div>
        <p className="mt-4 text-sm muted italic">
          Same prompt, same temperature: only the allowed numeric vocabulary changes.
        </p>
      </SlideCard>

      {/* ── 7. Headline table (Research-tab top table) ──────────────────── */}
      <SlideCard id="summary" num={7} total={14} kicker="Headline table"
        title="Scale degradation summary (16,000 evaluations)"
      >
        <p className="text-sm muted mb-3">
          The top-of-page table from the Research and Logprobs tabs. Each row aggregates 4,000
          grades (10 texts × 20 mechanistic questions × 20 samples). Total = 16,000.
        </p>

        {/* The 10 archetypes that get graded */}
        <details className="mb-4 border border-hair rounded-ctl overflow-hidden">
          <summary className="cursor-pointer px-4 py-2.5 bg-cream hover:bg-hair text-sm font-medium text-ink flex items-center justify-between">
            <span>The 10 statement archetypes (click to expand)</span>
            <span className="text-xs muted font-mono">data/batch_input_10.jsonl</span>
          </summary>
          <div className="divide-y divide-hair">
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
                  <span className="w-5 h-5 rounded-full bg-cream text-ink flex items-center justify-center text-[10px] font-bold font-mono border border-hair">
                    {i + 1}
                  </span>
                  <code className="text-[11px] font-mono text-ink bg-cream px-1.5 py-0.5 rounded">{t.id}</code>
                  <span className="text-[10px] font-semibold uppercase tracking-wider muted">{t.tag}</span>
                </div>
                <p className="text-ink leading-relaxed pl-7">{t.text}</p>
              </div>
            ))}
          </div>
        </details>

        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Scale</th>
                <th className="num">Mean Var</th>
                <th className="num">% Zero-Var</th>
                <th className="num">% High-Var</th>
                <th className="num">Mean Mode Consistency</th>
              </tr>
            </thead>
            <tbody>
              {SCALE_SUMMARY.map((row) => (
                <tr key={row.scale}>
                  <td className="capitalize font-medium">
                    <span
                      className="inline-block w-2.5 h-2.5 rounded-full mr-2 align-middle"
                      style={{ background: SCALE_COLORS[row.scale], border: `1px solid ${PALETTE.hair}` }}
                    />
                    {row.scale}
                  </td>
                  <td className="num">{row.meanVar.toFixed(4)}</td>
                  <td className="num">{row.zeroVarPct.toFixed(1)}%</td>
                  <td className="num">{row.highVarPct.toFixed(1)}%</td>
                  <td className="num">{(row.meanConsistency * 100).toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-5">
          <div className="panel-title mb-2">
            Reliability coefficients (logprobs run)
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Scale</th>
                <th className="num">ICC</th>
                <th className="num">Cohen&apos;s κ</th>
                <th className="num">Krippendorff α</th>
              </tr>
            </thead>
            <tbody>
              {RELIABILITY.map((row) => (
                <tr key={row.scale}>
                  <td className="capitalize font-medium">
                    <span
                      className="inline-block w-2.5 h-2.5 rounded-full mr-2 align-middle"
                      style={{ background: SCALE_COLORS[row.scale], border: `1px solid ${PALETTE.hair}` }}
                    />
                    {row.scale}
                  </td>
                  <td className="num">{row.icc.toFixed(3)}</td>
                  <td className="num">
                    {row.kappa === null ? <span className="muted">n/a</span> : row.kappa.toFixed(3)}
                  </td>
                  <td className="num">{row.kAlpha.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs muted">
            Cohen&apos;s κ is undefined for continuous (no discrete categories). Cronbach&apos;s α
            is omitted here because it divides by zero on saturated cells; see the Logprobs tab for
            the full Tier 1-3 panel.
          </p>
        </div>

        <div className="mt-4 callout text-sm text-ink">
          <p className="font-semibold mb-1">Read this row by row.</p>
          <p>
            Variance rises binary → ternary → quaternary, then{' '}
            <em>drops</em> on continuous, but continuous has the lowest zero-variance rate
            (79.5%). Quiet, distributed noise instead of loud bin-flipping.
          </p>
        </div>

        {/* Metric glossary (always-on, plain English) */}
        <details className="mt-4 border border-hair rounded-ctl overflow-hidden group" open>
          <summary className="cursor-pointer px-4 py-2.5 bg-cream hover:bg-hair text-sm font-semibold text-ink flex items-center justify-between">
            <span>What every column means (presenter glossary)</span>
            <span className="text-xs muted group-open:hidden">click to expand</span>
          </summary>
          <div className="p-4 space-y-3 text-sm text-ink bg-white">
            <div>
              <div className="font-semibold text-ink">Mean Var</div>
              <p className="muted">
                Variance of the 20 repeats per cell, then averaged across all 200 cells in the
                scale. Lower means the model gave the same answer every time. Theoretical max on
                [0, 1] data is 0.25 (a 50/50 flip).
              </p>
            </div>
            <div>
              <div className="font-semibold text-ink">% Zero-Var</div>
              <p className="muted">
                Fraction of cells where all 20 repeats returned the exact same number. The cleanest
                &ldquo;100% consistent&rdquo; indicator. Continuous looks calm on{' '}
                <em>Mean Var</em> but is the <strong>worst</strong> here (79.5%): it wobbles a
                tiny bit on almost every cell instead of flipping bins on a few.
              </p>
            </div>
            <div>
              <div className="font-semibold text-ink">% High-Var</div>
              <p className="muted">
                Fraction of cells with variance &gt; 0.1, i.e. genuinely loud disagreement
                (a sample like [0, 1, 0, 1, 0]). Quaternary&apos;s 11% is three times worse than
                binary &mdash; this is the bin-flipping failure mode.
              </p>
            </div>
            <div>
              <div className="font-semibold text-ink">Mean Mode Consistency</div>
              <p className="muted">
                For each cell, count how many of the 20 repeats matched the most-common answer.
                Equivalent to &ldquo;if I had to bet on one number, how often would I be right?&rdquo;
              </p>
            </div>
            <div>
              <div className="font-semibold text-ink">ICC (Intraclass Correlation)</div>
              <p className="muted">
                Psychometric reliability score. How much of the total variance is real
                between-cell signal vs noise across repeats. Range 0&ndash;1; &gt; 0.9 is
                &ldquo;excellent.&rdquo; All four scales pass; the ordering still tracks
                instability.
              </p>
            </div>
            <div>
              <div className="font-semibold text-ink">Cohen&apos;s κ (kappa)</div>
              <p className="muted">
                Inter-rater agreement corrected for chance, treating two random repeats as two
                raters. Range &minus;1 to +1; &gt; 0.8 is &ldquo;almost perfect.&rdquo; Quaternary
                wins here because when it does flip, it flips between adjacent values &mdash;
                kappa undercounts that. <em>n/a</em> for continuous (no discrete categories).
              </p>
            </div>
            <div>
              <div className="font-semibold text-ink">Krippendorff α</div>
              <p className="muted">
                The honest one. Modern standard for inter-annotator reliability across binary,
                ordinal, and continuous data. Range &minus;1 to +1; &gt; 0.8 is
                &ldquo;publishable.&rdquo; The binary &rarr; quaternary dip (0.984 &rarr; 0.952)
                mirrors the variance story without kappa&apos;s quirks. Use this when you want
                one defensible reliability number per scale.
              </p>
            </div>
          </div>
        </details>
      </SlideCard>

      {/* ── 8. Finding 1 ─────────────────────────────────────────────────── */}
      <SlideCard id="finding1" num={8} total={14} kicker="Finding 01"
        title="More numerical choices → fewer identical re-runs"
      >
        <p className="text-sm muted mb-3">
          Percent of (text × scale) pairs where 5 repeats produced byte-identical JSON.
          <span className="ml-2 muted">
            Source: <code>json_consistency_results.json</code>
          </span>
        </p>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={IDENTICAL_JSON_DATA} margin={{ top: 24, right: 12, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
            <XAxis dataKey="scale" />
            <YAxis tickFormatter={(v) => `${v}%`} domain={[0, 100]} />
            <Tooltip formatter={(v: number) => `${v}%`} />
            <Bar dataKey="pct">
              {IDENTICAL_JSON_DATA.map((entry) => (
                <Cell key={entry.scale} fill={SCALE_COLORS[entry.scale.toLowerCase()]} />
              ))}
              <LabelList dataKey="pct" position="top" formatter={(v: number) => `${v}%`} fill={CHART.label} fontSize={12} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="mt-4 callout text-sm text-ink">
          <p className="font-semibold mb-1">Reproducibility collapses</p>
          <p>
            Even with <code>temperature=0</code> and <code>response_format=json_object</code>, the
            model fails every single time on the continuous scale (0 / 10). Granularity isn&apos;t
            ornamental; it leaks into the sampler.
          </p>
        </div>

        <details className="mt-3 text-sm muted border border-hair rounded-ctl p-3 bg-cream">
          <summary className="cursor-pointer font-semibold text-ink">
            Why &ldquo;byte-identical&rdquo; is the strictest possible test
          </summary>
          <p className="mt-2">
            Variance smooths over small wobbles. Mode consistency lets one repeat differ. This bar
            asks: did all 5 of the same prompts return the exact same JSON string &mdash; same
            scores, same brackets, same whitespace? It bypasses every smoothing trick the math
            offers. The continuous-scale 0% is the most viscerally honest evidence that the
            sampler is non-deterministic regardless of <code>temperature=0</code>.
          </p>
        </details>
      </SlideCard>

      {/* ── 9. Finding 2 ─────────────────────────────────────────────────── */}
      <SlideCard id="finding2" num={9} total={14} kicker="Finding 02"
        title="Token-level confidence tracks the same gradient"
      >
        <p className="text-sm muted mb-3">
          Mean chosen-token probability and entropy over the numeric score tokens in the JSON
          response.
        </p>
        <table className="data-table">
          <thead>
            <tr>
              <th>Scale</th>
              <th className="num">Mean prob ↑</th>
              <th className="num">Mean entropy ↓</th>
              <th className="num">Min prob</th>
            </tr>
          </thead>
          <tbody>
            {LOGPROB_TABLE.map((row) => (
              <tr key={row.scale}>
                <td className="font-medium">
                  <span
                    className="inline-block w-2.5 h-2.5 rounded-full mr-2 align-middle"
                    style={{ background: row.color, border: `1px solid ${PALETTE.hair}` }}
                  />
                  {row.scale}
                </td>
                <td className="num">{row.meanProb.toFixed(4)}</td>
                <td className="num">{row.meanEntropy.toFixed(4)}</td>
                <td className="num">{row.minProb.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="mt-4 callout text-sm text-ink">
          <p className="font-semibold mb-1">What it means</p>
          <p>
            Mean token entropy <strong>triples</strong> from binary to continuous. Min chosen-token
            probability collapses from <strong>0.50 → 0.24</strong>: the model genuinely doesn&apos;t
            know which decimal to emit next. Logprobs are early-warning signals, not safety
            guarantees.
          </p>
        </div>

        <details className="mt-3 text-sm text-ink border border-hair rounded-ctl p-4 bg-white space-y-2">
          <summary className="cursor-pointer font-semibold text-ink">
            How to read each column
          </summary>
          <div className="mt-2 space-y-3">
            <div>
              <div className="font-semibold text-ink">Mean prob ↑</div>
              <p className="muted">
                The probability the model assigned to the token it actually emitted, averaged across
                every numeric score token in the response. Higher = the model was confident in its
                pick. Binary 0.98 vs continuous 0.94 means the model is genuinely less certain
                which decimal to choose.
              </p>
            </div>
            <div>
              <div className="font-semibold text-ink">Mean entropy ↓</div>
              <p className="muted">
                Shannon entropy: <code>&minus;Σ p log₂ p</code> over the top-5 alternative tokens.
                Zero entropy = all probability mass on one option (totally sure). Higher entropy =
                the distribution is spread across multiple plausible tokens. The 3&times; jump
                from binary (0.06) to continuous (0.20) is the model spreading its bets.
              </p>
            </div>
            <div>
              <div className="font-semibold text-ink">Min prob</div>
              <p className="muted">
                Worst case across the whole run. Binary&apos;s 0.50 is a literal coin flip;
                continuous&apos;s 0.24 means the model&apos;s favorite token was only its 1-in-4
                pick. The worst case bounds how much you can trust any single grade.
              </p>
            </div>
          </div>
        </details>
      </SlideCard>

      {/* ── 10. Finding 3 ────────────────────────────────────────────────── */}
      <SlideCard id="finding3" num={10} total={14} kicker="Finding 03"
        title="Some semantic factors are systematically less stable"
      >
        <p className="text-sm muted mb-3">
          Percent of (text × scale) cells where 5 repeats disagreed, by factor family.
        </p>
        <ResponsiveContainer width="100%" height={420}>
          <BarChart data={FAMILY_INSTABILITY} layout="vertical" margin={{ left: 80, right: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
            <XAxis type="number" tickFormatter={(v) => `${v}%`} domain={[0, 25]} />
            <YAxis type="category" dataKey="family" width={120} tick={{ fontSize: 11 }} />
            <Tooltip formatter={(v: number) => `${v}%`} />
            <Bar dataKey="pct" fill={PALETTE.gold}>
              {FAMILY_INSTABILITY.map((entry, i) => {
                const ratio = entry.pct / 25
                return <Cell key={i} fill={magnitudeColor(ratio)} />
              })}
              <LabelList dataKey="pct" position="right" formatter={(v: number) => `${v}%`} fill={CHART.label} fontSize={11} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p className="mt-3 text-sm muted italic">
          Concrete vs abstract, polarity flips, spatial language, inferred intent: the same
          categories humans disagree on.
        </p>
      </SlideCard>

      {/* ── 11. Finding 4 ────────────────────────────────────────────────── */}
      <SlideCard id="finding4" num={11} total={14} kicker="Finding 04"
        title="High confidence ≠ stable answer"
      >
        <p className="text-sm muted mb-4">
          Cases where the model emitted near-certain logprobs yet still flipped its score:
        </p>
        <div className="grid md:grid-cols-2 gap-3">
          <div className="border border-hair rounded-ctl p-4">
            <div className="text-xs font-mono muted mb-1">
              narrative_paragraph / quaternary / Q19 (identity)
            </div>
            <div className="font-mono text-sm bg-cream p-2 rounded mb-2">
              5 repeats → [0, 1, 1, 1, 1]
            </div>
            <div className="text-xs muted">
              Variance <strong>0.16</strong> · Mean token prob <strong>0.949</strong> · Min{' '}
              <strong>0.82</strong>
            </div>
            <div className="mt-2 text-sm text-ink font-semibold">
              → Model is 95% sure each time, and still produces a different answer once.
            </div>
          </div>
          <div className="border border-hair rounded-ctl p-4">
            <div className="text-xs font-mono muted mb-1">
              sentiment_negative / binary / Q7 (negation)
            </div>
            <div className="font-mono text-sm bg-cream p-2 rounded mb-2">
              5 repeats → [0, 1, 0, 1, 0]
            </div>
            <div className="text-xs muted">
              Variance <strong>0.24</strong> · Mean token prob <strong>0.665</strong> · Min{' '}
              <strong>0.56</strong>
            </div>
            <div className="mt-2 text-sm muted">
              → Here logprobs do correlate; the exception, not the rule.
            </div>
          </div>
        </div>
        <div className="mt-4 callout-ink callout p-4">
          <div className="panel-title mb-1" style={{ color: PALETTE.ink }}>
            Mann-Whitney U · stable vs unstable cells (mean token prob)
          </div>
          <div className="flex items-baseline gap-4">
            <div className="stat-num text-3xl font-mono">p = 0.059</div>
            <div className="text-sm text-ink">
              Near-significant, but the magnitude is small: stable mean prob{' '}
              <strong>0.968</strong> vs unstable <strong>0.926</strong>.
            </div>
          </div>
          <p className="mt-2 text-sm text-ink">
            <strong>Don&apos;t trust confidence alone.</strong>
          </p>
        </div>

        <details className="mt-3 text-sm text-ink border border-hair rounded-ctl p-4 bg-white">
          <summary className="cursor-pointer font-semibold text-ink">
            What Mann-Whitney is doing here &middot; and what p = 0.059 actually means
          </summary>
          <div className="mt-2 space-y-2">
            <p>
              <strong>Mann-Whitney U</strong> is a non-parametric test that asks: do two
              distributions have different medians? It doesn&apos;t assume normality, which matters
              because logprob distributions are heavily skewed (most tokens are near probability 1).
            </p>
            <p>
              We split every cell into two buckets &mdash; <em>stable</em> (all repeats agreed) and{' '}
              <em>unstable</em> (at least one flip) &mdash; then ran the test on their mean
              chosen-token probabilities.
            </p>
            <ul className="list-disc pl-5 space-y-1">
              <li>Stable cells mean prob: <strong>0.968</strong></li>
              <li>Unstable cells mean prob: <strong>0.926</strong></li>
              <li>Mann-Whitney U test: <strong>p = 0.059</strong></li>
            </ul>
            <p>
              The conventional &ldquo;statistically significant&rdquo; threshold is p &lt; 0.05.
              We&apos;re at 0.059 &mdash; barely above it. The plain-English read:
            </p>
            <ul className="list-disc pl-5 space-y-1">
              <li>Yes, on average, unstable cells have lower confidence than stable ones.</li>
              <li>
                No, the difference is not reliably distinguishable from chance at the standard
                threshold.
              </li>
              <li>The effect is tiny: a 4-point gap on a 0&ndash;1 scale.</li>
            </ul>
            <p className="font-semibold text-ink">
              Engineering takeaway: logprobs correlate with stability in the right direction, but
              the correlation is too weak to filter on. You cannot say &ldquo;only trust grades
              where token prob &gt; X.&rdquo; The only honest measure of stability is repeat-and-check
              &mdash; which is exactly what this whole experiment does.
            </p>
          </div>
        </details>
      </SlideCard>

      {/* ── 12. PCA ──────────────────────────────────────────────────────── */}
      <SlideCard id="pca" num={12} total={14} kicker="Analytical core"
        title="PCA reveals which semantic axes absorb the wobble"
      >
        <p className="text-sm muted mb-4">
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
            <figure key={t.id} className="border border-hair rounded-ctl overflow-hidden bg-white">
              <img
                src={`/presentation/pca_${t.id}.png`}
                alt={`PCA of 80 grading vectors for ${t.label}`}
                className="w-full"
                loading="lazy"
              />
              <figcaption className="p-3 text-xs muted border-t border-hair">
                <div className="font-mono font-semibold text-ink mb-1">{t.label}</div>
                <div>{t.desc}</div>
              </figcaption>
            </figure>
          ))}
        </div>

        <div className="callout p-4 text-sm">
          <div className="grid md:grid-cols-3 gap-4">
            <div>
              <div className="panel-title mb-1">
                Input matrix
              </div>
              <div className="font-mono text-lg text-ink">80 × 20</div>
              <div className="text-xs muted">samples × questions; each cell in [0, 1]</div>
            </div>
            <div>
              <div className="panel-title mb-1">
                Decompose
              </div>
              <div className="font-mono text-lg text-ink">3 PCs</div>
              <div className="text-xs muted">center → cov → eigendecomposition</div>
            </div>
            <div>
              <div className="panel-title mb-1">
                Interpret
              </div>
              <div className="font-mono text-lg text-ink">3-D scatter</div>
              <div className="text-xs muted">colored by scale; loadings = wobble drivers</div>
            </div>
          </div>
          <p className="mt-3 text-ink">
            <strong>Want a live one?</strong> Switch to the <em>Analyze Statement</em> tab, paste
            any sentence, and you get the interactive 3-D PCA on your own text.
          </p>
        </div>

        <details className="mt-3 text-sm text-ink border border-hair rounded-ctl p-4 bg-white">
          <summary className="cursor-pointer font-semibold text-ink">
            How to read the three PCA plots above
          </summary>
          <div className="mt-2 space-y-2">
            <p>
              Each plot shows 80 dots for one statement: 20 repeats of each scale. Color = scale
              (blue binary, green ternary, orange quaternary, pink continuous). The big{' '}
              <strong>star</strong> is that scale&apos;s centroid (the mean position of its 20
              repeats).
            </p>
            <ul className="list-disc pl-5 space-y-1">
              <li>
                <strong>Tight color cluster</strong> = the model gave the same 20-question answer
                vector every time on that scale. Consistent.
              </li>
              <li>
                <strong>Spread color cluster</strong> = the model produced different score
                vectors across repeats. It&apos;s changing its mind on at least some of the 20
                questions.
              </li>
              <li>
                <strong>Stars close together</strong> = the four scales agree on what the underlying
                answer is, even if their precision differs.
              </li>
              <li>
                <strong>Stars far apart</strong> = the scale itself is shifting what the model
                returns. The choice of scale is changing the verdict, not just the resolution.
              </li>
            </ul>
            <p>
              The three statements were picked deliberately:
              <strong> factual_simple</strong> collapses near the origin (easy case),{' '}
              <strong>ambiguous</strong> sends orange and pink drifting hard from the blue
              cluster, and <strong>narrative_paragraph</strong> shows the dramatic quaternary
              spread &mdash; the wobble lives in mid-resolution scales for that text.
            </p>
          </div>
        </details>
      </SlideCard>

      {/* ── 13. So what ──────────────────────────────────────────────────── */}
      <SlideCard id="meaning" num={13} total={14} kicker="What it means"
        title="Implications for LLM-as-judge evaluation"
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
            <div key={row.n} className="border border-hair rounded-ctl p-4 flex gap-4">
              <div className="text-2xl font-bold text-gold font-mono">{row.n}</div>
              <div>
                <div className="font-semibold text-ink">{row.h}</div>
                <div className="text-sm muted mt-1">{row.b}</div>
              </div>
            </div>
          ))}
        </div>
        <p className="mt-4 text-sm muted italic">
          Next: cross-model comparison (gpt-4o ↔ Claude ↔ Llama), longer contexts, and reweighting
          eval rubrics by per-family reliability.
        </p>
      </SlideCard>

      {/* ── 14. Thanks ───────────────────────────────────────────────────── */}
      <SlideCard id="thanks" num={14} total={14} kicker="Thanks for stopping by"
        title="Try it live. Pick a statement. Watch the model wobble."
      >
        {/* Presenter cheat sheet — the one-paragraph version */}
        <div className="mb-4 bg-cream border border-hair rounded-ctl p-4 text-sm text-ink leading-relaxed">
          <div className="panel-title mb-2">
            One-paragraph version (for the stranger who just walked up)
          </div>
          <p>
            We grade <strong>10 test statements</strong> with <strong>20 mechanistic semantic
            probes</strong>, repeating each grade <strong>20 times at temperature 0</strong>, on{' '}
            <strong>4 answer scales</strong> (binary through continuous). The headline finding is
            that finer numerical scales let the model spend its degrees of freedom on noise rather
            than signal &mdash; quaternary is the worst scale by every honest reliability metric
            (Krippendorff α 0.95 vs 0.98 for binary), and on continuous the model literally never
            reproduces the same JSON output twice. Token-level confidence correlates with stability
            in the right direction but is too weak to filter on (Mann-Whitney p = 0.059, effect
            size 4 points). The instability isn&apos;t uniform &mdash; 5 of 20 question families
            show zero disagreement, while abstract-integration families like concreteness and
            intent absorb almost all of it. The honest engineering recipe: pick the coarsest scale
            your decision actually needs, probe per factor family rather than averaging, and never
            trust a single grade &mdash; replicate.
          </p>
        </div>

        <div className="flex flex-col md:flex-row md:items-center gap-4 mt-2">
          <a
            href={DEMO_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="btn-primary"
          >
            grading-llm.vercel.app
          </a>
          <p className="text-sm muted">
            Bring your own OpenAI key. Run on a statement of your choice. Inspect the PCA in 3-D.
          </p>
        </div>
        <div className="mt-4 text-sm muted">
          Atharva Patel &middot;{' '}
          <a className="text-gold hover:underline" href="mailto:atharvajpatel@berkeley.edu">
            atharvajpatel@berkeley.edu
          </a>{' '}
          &middot;{' '}
          <a className="text-gold hover:underline" href="https://github.com/atharvajpatel/Grading-LLM" target="_blank" rel="noopener noreferrer">
            github.com/atharvajpatel/Grading-LLM
          </a>
        </div>
      </SlideCard>
    </div>
  )
}
