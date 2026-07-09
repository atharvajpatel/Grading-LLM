import { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell, LineChart, Line,
} from 'recharts'
import {
  PALETTE, magnitudeColor, onMagnitude, divergingColor, onDiverging, dashFor, CHART,
} from '../theme/palette'
import { SCALE_COLORS } from '../api/client'

// ─── Shared Constants ────────────────────────────────────────────────────────

const SCALE_ORDER = ['binary', 'ternary', 'quaternary', 'continuous']

// ─── Section 1: Run-to-Run Reproducibility ──────────────────────────────────

const SCALE_DELTAS = [
  { scale: 'binary', run1MeanVar: 0.002413, run2MeanVar: 0.002788, deltaPct: 15.5, run1Consistency: 0.9972, run2Consistency: 0.9957, run1ZeroVarPct: 97.0, run2ZeroVarPct: 98.5 },
  { scale: 'ternary', run1MeanVar: 0.003241, run2MeanVar: 0.003463, deltaPct: 6.8, run1Consistency: 0.9918, run2Consistency: 0.9878, run1ZeroVarPct: 95.5, run2ZeroVarPct: 94.5 },
  { scale: 'quaternary', run1MeanVar: 0.004974, run2MeanVar: 0.005605, deltaPct: 12.7, run1Consistency: 0.9685, run2Consistency: 0.9745, run1ZeroVarPct: 88.5, run2ZeroVarPct: 87.5 },
  { scale: 'continuous', run1MeanVar: 0.002014, run2MeanVar: 0.001610, deltaPct: -20.0, run1Consistency: 0.9748, run2Consistency: 0.9782, run1ZeroVarPct: 80.5, run2ZeroVarPct: 79.5 },
]

const HEATMAP_DIFF = [
  { question: 'Named Entities', binary: 0.0, ternary: 0.0, quaternary: 0.0, continuous: -0.000047 },
  { question: 'Actions/Events', binary: 0.0, ternary: 0.009437, quaternary: -0.002776, continuous: -0.000743 },
  { question: 'Causality', binary: 0.00325, ternary: 0.00225, quaternary: -0.003917, continuous: 0.00031 },
  { question: 'Temporal', binary: 0.0, ternary: 0.0, quaternary: 0.0, continuous: -0.00009 },
  { question: 'Spatial', binary: -0.00475, ternary: -0.003188, quaternary: 0.000246, continuous: 0.00002 },
  { question: 'Numeric', binary: 0.0, ternary: 0.0, quaternary: 0.000433, continuous: -0.000142 },
  { question: 'Negation', binary: 0.009, ternary: -0.0135, quaternary: 0.001908, continuous: -0.00128 },
  { question: 'Uncertainty', binary: -0.00475, ternary: 0.0, quaternary: 0.0, continuous: 0.000123 },
  { question: 'Modality', binary: 0.0, ternary: 0.0, quaternary: 0.0, continuous: 0.0 },
  { question: 'Sentiment', binary: 0.0, ternary: 0.0, quaternary: 0.0, continuous: -0.000813 },
  { question: 'Emotion', binary: -0.009, ternary: -0.001, quaternary: 0.001198, continuous: 0.0 },
  { question: 'Social', binary: 0.0, ternary: 0.0, quaternary: 0.001329, continuous: 0.000017 },
  { question: 'Dialogue', binary: 0.02275, ternary: 0.007, quaternary: -0.00375, continuous: 0.0 },
  { question: 'First Person', binary: -0.009, ternary: 0.0, quaternary: 0.0, continuous: 0.0 },
  { question: 'Imperative', binary: 0.0, ternary: 0.0, quaternary: 0.0, continuous: 0.0 },
  { question: 'Comparison', binary: 0.0, ternary: 0.0, quaternary: 0.0, continuous: 0.000048 },
  { question: 'Normative', binary: 0.0, ternary: 0.0, quaternary: 0.0, continuous: 0.0 },
  { question: 'Intent', binary: 0.0, ternary: 0.00125, quaternary: -0.005853, continuous: -0.00282 },
  { question: 'Concreteness', binary: 0.0, ternary: 0.001188, quaternary: -0.000885, continuous: -0.004368 },
  { question: 'Identity', binary: 0.0, ternary: 0.001, quaternary: 0.024677, continuous: 0.00171 },
]

const SCATTER_CORRELATION = { pearsonR: 0.6279, n: 800 }

// Advanced stats comparison (from both runs)
const TIER1_COMPARISON = [
  { scale: 'binary', run1ICC: 0.984923, run2ICC: 0.984448, run1Kripp: 0.984211, run2Kripp: 0.983723 },
  { scale: 'ternary', run1ICC: 0.975761, run2ICC: 0.971374, run1Kripp: 0.977118, run2Kripp: 0.97325 },
  { scale: 'quaternary', run1ICC: 0.954494, run2ICC: 0.954509, run1Kripp: 0.951957, run2Kripp: 0.958933 },
  { scale: 'continuous', run1ICC: 0.962245, run2ICC: 0.967779, run1Kripp: 0.960552, run2Kripp: 0.966306 },
]

const ETA_COMPARISON = [
  { factor: 'Question type', run1: 0.038963, run2: 0.050296 },
  { factor: 'Text type', run1: 0.020604, run2: 0.022173 },
  { factor: 'Scale type', run1: 0.004691, run2: 0.006311 },
]

// ─── Section 3: Logprob Analysis ────────────────────────────────────────────

const CONFIDENCE_HEATMAP = [
  { question: 'Named Entities', binary: 1.0, ternary: 1.0, quaternary: 1.0, continuous: 0.978 },
  { question: 'Actions/Events', binary: 0.9967, ternary: 0.9393, quaternary: 0.9877, continuous: 1.0 },
  { question: 'Causality', binary: 0.9111, ternary: 0.9994, quaternary: 0.9208, continuous: 0.9579 },
  { question: 'Temporal', binary: 1.0, ternary: 0.9773, quaternary: 0.9405, continuous: 0.6741 },
  { question: 'Spatial', binary: 0.9678, ternary: 0.9966, quaternary: 0.9665, continuous: 1.0 },
  { question: 'Numeric', binary: 0.9956, ternary: 0.9591, quaternary: 0.9953, continuous: 0.691 },
  { question: 'Negation', binary: 0.9778, ternary: 0.9295, quaternary: 0.9549, continuous: 0.9988 },
  { question: 'Uncertainty', binary: 0.9913, ternary: 0.9865, quaternary: 0.9594, continuous: 0.9337 },
  { question: 'Modality', binary: 1.0, ternary: 0.997, quaternary: 0.9891, continuous: 1.0 },
  { question: 'Sentiment', binary: 0.987, ternary: 0.9979, quaternary: 0.9569, continuous: 0.718 },
  { question: 'Emotion', binary: 0.9836, ternary: 0.9974, quaternary: 0.9987, continuous: 0.9906 },
  { question: 'Social', binary: 1.0, ternary: 0.9997, quaternary: 0.9638, continuous: 0.9078 },
  { question: 'Dialogue', binary: 0.9561, ternary: 1.0, quaternary: 0.9971, continuous: 1.0 },
  { question: 'First Person', binary: 0.9955, ternary: 0.9663, quaternary: 0.9704, continuous: 0.7858 },
  { question: 'Imperative', binary: 1.0, ternary: 1.0, quaternary: 0.9877, continuous: 1.0 },
  { question: 'Comparison', binary: 0.9996, ternary: 1.0, quaternary: 0.9952, continuous: 0.8636 },
  { question: 'Normative', binary: 0.9955, ternary: 0.9997, quaternary: 0.9893, continuous: 0.9877 },
  { question: 'Intent', binary: 0.907, ternary: 0.9962, quaternary: 0.9988, continuous: 0.9982 },
  { question: 'Concreteness', binary: 0.9913, ternary: 0.9788, quaternary: 0.9974, continuous: 1.0 },
  { question: 'Identity', binary: 0.982, ternary: 0.9752, quaternary: 0.9409, continuous: 0.9028 },
]

const CONFIDENCE_BY_SCALE = [
  { scale: 'binary', meanConf: 0.9819, medianConf: 1.0, minConf: 0.5, pctBelow90: 5.6, pctBelow50: 0.1 },
  { scale: 'ternary', meanConf: 0.9848, medianConf: 1.0, minConf: 0.5, pctBelow90: 4.25, pctBelow50: 0.0 },
  { scale: 'quaternary', meanConf: 0.9755, medianConf: 1.0, minConf: 0.5, pctBelow90: 8.5, pctBelow50: 0.1 },
  { scale: 'continuous', meanConf: 0.9194, medianConf: 1.0, minConf: 0.2506, pctBelow90: 16.73, pctBelow50: 8.4 },
]

const QUESTION_CONFIDENCE_RANKING = [
  { question: 'Temporal', family: 'temporal', meanConf: 0.6741, pctBelow90: 73.5 },
  { question: 'Numeric', family: 'numeric', meanConf: 0.691, pctBelow90: 60.0 },
  { question: 'Sentiment', family: 'sentiment', meanConf: 0.718, pctBelow90: 51.0 },
  { question: 'First Person', family: 'first_person', meanConf: 0.7858, pctBelow90: 40.0 },
  { question: 'Comparison', family: 'comparison', meanConf: 0.8636, pctBelow90: 20.0 },
  { question: 'Identity', family: 'identity', meanConf: 0.9028, pctBelow90: 19.5 },
  { question: 'Social', family: 'social', meanConf: 0.9078, pctBelow90: 20.0 },
  { question: 'Uncertainty', family: 'uncertainty', meanConf: 0.9337, pctBelow90: 20.0 },
  { question: 'Causality', family: 'causality', meanConf: 0.9579, pctBelow90: 10.0 },
  { question: 'Named Entities', family: 'named_entities', meanConf: 0.978, pctBelow90: 10.0 },
  { question: 'Normative', family: 'normative', meanConf: 0.9877, pctBelow90: 6.5 },
  { question: 'Emotion', family: 'emotion', meanConf: 0.9906, pctBelow90: 4.0 },
  { question: 'Intent', family: 'intent', meanConf: 0.9982, pctBelow90: 0.0 },
  { question: 'Negation', family: 'negation', meanConf: 0.9988, pctBelow90: 0.0 },
  { question: 'Actions/Events', family: 'actions_events', meanConf: 1.0, pctBelow90: 0.0 },
  { question: 'Spatial', family: 'spatial', meanConf: 1.0, pctBelow90: 0.0 },
  { question: 'Modality', family: 'modality', meanConf: 1.0, pctBelow90: 0.0 },
  { question: 'Dialogue', family: 'dialogue', meanConf: 1.0, pctBelow90: 0.0 },
  { question: 'Imperative', family: 'imperative', meanConf: 1.0, pctBelow90: 0.0 },
  { question: 'Concreteness', family: 'concreteness', meanConf: 1.0, pctBelow90: 0.0 },
]

const CONFIDENCE_DISTRIBUTION: Record<string, Array<{bin: string, count: number, pct: number}>> = {
  binary: [
    { bin: '<50%', count: 4, pct: 0.1 },
    { bin: '50-70%', count: 94, pct: 2.4 },
    { bin: '70-80%', count: 55, pct: 1.4 },
    { bin: '80-90%', count: 71, pct: 1.8 },
    { bin: '90-95%', count: 53, pct: 1.3 },
    { bin: '95-99%', count: 210, pct: 5.2 },
    { bin: '99-100%', count: 3513, pct: 87.8 },
  ],
  ternary: [
    { bin: '<50%', count: 0, pct: 0.0 },
    { bin: '50-70%', count: 94, pct: 2.4 },
    { bin: '70-80%', count: 30, pct: 0.8 },
    { bin: '80-90%', count: 46, pct: 1.1 },
    { bin: '90-95%', count: 43, pct: 1.1 },
    { bin: '95-99%', count: 174, pct: 4.3 },
    { bin: '99-100%', count: 3613, pct: 90.3 },
  ],
  quaternary: [
    { bin: '<50%', count: 4, pct: 0.1 },
    { bin: '50-70%', count: 112, pct: 2.8 },
    { bin: '70-80%', count: 90, pct: 2.2 },
    { bin: '80-90%', count: 134, pct: 3.4 },
    { bin: '90-95%', count: 77, pct: 1.9 },
    { bin: '95-99%', count: 197, pct: 4.9 },
    { bin: '99-100%', count: 3386, pct: 84.7 },
  ],
  continuous: [
    { bin: '<50%', count: 336, pct: 8.4 },
    { bin: '50-70%', count: 199, pct: 5.0 },
    { bin: '70-80%', count: 55, pct: 1.4 },
    { bin: '80-90%', count: 79, pct: 2.0 },
    { bin: '90-95%', count: 54, pct: 1.4 },
    { bin: '95-99%', count: 88, pct: 2.2 },
    { bin: '99-100%', count: 3189, pct: 79.7 },
  ],
}

const CONF_VS_VAR_CORRELATION = { pearsonR: -0.1513, spearmanR: -0.0871 }

// ─── Section 4: Predictive Power ────────────────────────────────────────────

const PREDICTIVE = { baseRate: 10.0, nUnstable: 80, nStable: 720, nTotal: 800, auc: 0.1093, bestThreshold: 0.99 }

const ROC_POINTS = [
  { threshold: 0.5, precision: 0.1429, recall: 0.0375, f1: 0.0594, accuracy: 0.8812, tpr: 0.0375, fpr: 0.025 },
  { threshold: 0.6, precision: 0.1875, recall: 0.075, f1: 0.1071, accuracy: 0.875, tpr: 0.075, fpr: 0.0361 },
  { threshold: 0.7, precision: 0.2444, recall: 0.1375, f1: 0.176, accuracy: 0.8712, tpr: 0.1375, fpr: 0.0472 },
  { threshold: 0.8, precision: 0.2, recall: 0.1375, f1: 0.163, accuracy: 0.8588, tpr: 0.1375, fpr: 0.0611 },
  { threshold: 0.85, precision: 0.2167, recall: 0.1625, f1: 0.1857, accuracy: 0.8575, tpr: 0.1625, fpr: 0.0653 },
  { threshold: 0.9, precision: 0.2, recall: 0.175, f1: 0.1867, accuracy: 0.8475, tpr: 0.175, fpr: 0.0778 },
  { threshold: 0.95, precision: 0.1928, recall: 0.2, f1: 0.1963, accuracy: 0.8363, tpr: 0.2, fpr: 0.0931 },
  { threshold: 0.99, precision: 0.177, recall: 0.25, f1: 0.2073, accuracy: 0.8087, tpr: 0.25, fpr: 0.1292 },
  { threshold: 0.999, precision: 0.1419, recall: 0.275, f1: 0.1872, accuracy: 0.7612, tpr: 0.275, fpr: 0.1847 },
  { threshold: 1.0, precision: 0.1226, recall: 0.4875, f1: 0.196, accuracy: 0.6, tpr: 0.4875, fpr: 0.3875 },
]

// ─── Section 5: Score Agreement ─────────────────────────────────────────────

const AGREEMENT = { overallAgreePct: 98.9, totalPairs: 800 }
const AGREEMENT_PER_SCALE = [
  { scale: 'binary', agreePct: 99.0, disagreeCount: 2 },
  { scale: 'ternary', agreePct: 99.0, disagreeCount: 2 },
  { scale: 'quaternary', agreePct: 98.5, disagreeCount: 3 },
  { scale: 'continuous', agreePct: 99.0, disagreeCount: 2 },
]

// ─── Utility Functions ──────────────────────────────────────────────────────

// Diverging scale for signed variance deltas (Run2 − Run1):
// negative → navy, zero → cream, positive → gold. Normalization unchanged
// (magnitude saturates at 0.025); only the endpoint colors changed.
function diffToColor(value: number): string {
  const t = Math.max(-1, Math.min(value / 0.025, 1))
  return divergingColor(t)
}

// Single-hue magnitude ramp for confidence (near 1.0 = strong).
function confidenceToColor(value: number): string {
  return magnitudeColor(value)
}

function confidenceBarColor(value: number): string {
  return magnitudeColor(value)
}

// ─── Component ──────────────────────────────────────────────────────────────

export default function ComparisonTab() {
  const [showSection1, setShowSection1] = useState(true)
  const [showSection2, setShowSection2] = useState(true)
  const [showSection3, setShowSection3] = useState(true)
  const [showSection4, setShowSection4] = useState(true)
  const [showSection5, setShowSection5] = useState(true)
  const [confDistScale, setConfDistScale] = useState<string>('binary')

  // Prepare eta comparison bar chart data
  const etaBarData = ETA_COMPARISON.map((e) => ({
    factor: e.factor,
    'Run 1': e.run1,
    'Run 2': e.run2,
  }))

  // Prepare ROC curve data with diagonal reference
  const rocCurveData = ROC_POINTS.map((p) => ({
    fpr: p.fpr,
    tpr: p.tpr,
    random: p.fpr, // diagonal reference line
    threshold: p.threshold,
  }))

  // Best F1 row
  const bestF1 = ROC_POINTS.reduce((best, curr) => curr.f1 > best.f1 ? curr : best, ROC_POINTS[0])

  return (
    <div className="space-y-8">
      {/* ═══════════════════════════════════════════════════════════════════════
          HEADER
          ═══════════════════════════════════════════════════════════════════════ */}
      <div className="card">
        <h2 className="page-title">
          Run-to-Run Comparison: 32,000 Evaluations
        </h2>
        <p className="muted leading-relaxed mt-2">
          Side-by-side comparison of two independent batch runs on identical prompts and texts.
          Run 2 additionally captures logprob confidence scores for every evaluation, enabling
          a first test of whether token-level confidence predicts multi-sample instability.
        </p>
        <div className="mt-3 flex flex-wrap gap-3 text-sm">
          <span className="chip">Run 1: Feb 2026</span>
          <span className="chip">Run 2: Mar 2026 (Logprobs)</span>
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════════
          SECTION 1: Run-to-Run Reproducibility
          ═══════════════════════════════════════════════════════════════════════ */}
      <div className="card">
        <button
          className="flex items-center justify-between w-full text-left"
          onClick={() => setShowSection1(!showSection1)}
        >
          <div>
            <div className="panel-title">Section 1</div>
            <h3 className="section-title">Run-to-Run Reproducibility</h3>
          </div>
          <span className="btn-secondary">{showSection1 ? 'Hide' : 'Show'}</span>
        </button>

        {showSection1 && (
          <div className="mt-4 space-y-6">
            {/* 1b. Side-by-side delta table */}
            <div>
              <h4 className="text-sm font-semibold text-ink mb-3">Variance &amp; Consistency Deltas by Scale</h4>
              <div className="overflow-x-auto">
                <table className="data-table w-full text-sm">
                  <thead>
                    <tr className="border-b border-hair">
                      <th className="text-left py-2 pr-4 font-medium text-mute">Scale</th>
                      <th className="text-right py-2 px-3 font-medium text-mute">Run 1 Var</th>
                      <th className="text-right py-2 px-3 font-medium text-mute">Run 2 Var</th>
                      <th className="text-right py-2 px-3 font-medium text-mute">Delta %</th>
                      <th className="text-right py-2 px-3 font-medium text-mute">Run 1 Consistency</th>
                      <th className="text-right py-2 pl-3 font-medium text-mute">Run 2 Consistency</th>
                    </tr>
                  </thead>
                  <tbody>
                    {SCALE_DELTAS.map((row) => (
                      <tr key={row.scale} className="border-b border-hair">
                        <td className="py-2 pr-4 capitalize font-medium" style={{ color: SCALE_COLORS[row.scale] }}>
                          {row.scale}
                        </td>
                        <td className="text-right py-2 px-3 num">{row.run1MeanVar.toFixed(4)}</td>
                        <td className="text-right py-2 px-3 num">{row.run2MeanVar.toFixed(4)}</td>
                        <td
                          className="text-right py-2 px-3 num font-semibold"
                          style={{ color: row.deltaPct < 0 ? PALETTE.ink : PALETTE.gold }}
                        >
                          {row.deltaPct > 0 ? '+' : row.deltaPct < 0 ? '−' : ''}{Math.abs(row.deltaPct).toFixed(1)}%
                        </td>
                        <td className="text-right py-2 px-3 num">{(row.run1Consistency * 100).toFixed(2)}%</td>
                        <td className="text-right py-2 pl-3 num">{(row.run2Consistency * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* 1c. Heatmap diff */}
            <div>
              <h4 className="text-sm font-semibold text-ink mb-1">Variance Diff Heatmap (Run 2 &minus; Run 1)</h4>
              <p className="text-xs muted mb-3">
                Navy = Run 2 improved (lower variance). Gold = Run 2 worse. Cream = no change. The signed number is shown in each cell.
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr>
                      <th className="text-left py-1 pr-2 font-medium text-mute w-36">Question</th>
                      {SCALE_ORDER.map((s) => (
                        <th key={s} className="py-1 px-2 font-medium text-mute capitalize text-center w-28">{s}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {HEATMAP_DIFF.map((row) => (
                      <tr key={row.question}>
                        <td className="py-1 pr-2 font-medium text-ink text-xs">{row.question}</td>
                        {SCALE_ORDER.map((s) => {
                          const val = (row as any)[s] as number
                          const t = Math.max(-1, Math.min(val / 0.025, 1))
                          const bg = diffToColor(val)
                          return (
                            <td key={s} className="py-1 px-2">
                              <div
                                className="rounded px-2 py-1 text-center font-mono"
                                style={{
                                  backgroundColor: bg,
                                  color: onDiverging(t),
                                }}
                                title={`${row.question} / ${s}: ${val > 0 ? '+' : val < 0 ? '−' : ''}${Math.abs(val).toFixed(6)}`}
                              >
                                {val > 0 ? '+' : val < 0 ? '−' : ''}{Math.abs(val).toFixed(4)}
                              </div>
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* 1d. Callout */}
            <div className="callout text-sm">
              <p className="font-semibold mb-1">Variance Scatter: Pearson r = {SCATTER_CORRELATION.pearsonR}</p>
              <p>
                Across {SCATTER_CORRELATION.n} points, instability patterns are moderately reproducible &mdash; the same
                questions tend to be unstable across runs, but the magnitude varies. The correlation is strong enough to
                confirm structural instability but not strong enough to predict exact variance magnitudes from one run to another.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* ═══════════════════════════════════════════════════════════════════════
          SECTION 2: Advanced Stats Comparison
          ═══════════════════════════════════════════════════════════════════════ */}
      <div className="card">
        <button
          className="flex items-center justify-between w-full text-left"
          onClick={() => setShowSection2(!showSection2)}
        >
          <div>
            <div className="panel-title">Section 2</div>
            <h3 className="section-title">Advanced Stats Comparison</h3>
          </div>
          <span className="btn-secondary">{showSection2 ? 'Hide' : 'Show'}</span>
        </button>

        {showSection2 && (
          <div className="mt-4 space-y-6">
            {/* 2a. Reliability comparison table */}
            <div>
              <h4 className="text-sm font-semibold text-ink mb-3">Reliability Metrics: ICC &amp; Krippendorff&apos;s Alpha</h4>
              <div className="overflow-x-auto">
                <table className="data-table w-full text-sm">
                  <thead>
                    <tr className="border-b border-hair">
                      <th className="text-left py-2 pr-4 font-medium text-mute">Scale</th>
                      <th className="text-right py-2 px-3 font-medium text-mute">Run 1 ICC</th>
                      <th className="text-right py-2 px-3 font-medium text-mute">Run 2 ICC</th>
                      <th className="text-right py-2 px-3 font-medium text-mute">Run 1 Kripp &alpha;</th>
                      <th className="text-right py-2 pl-3 font-medium text-mute">Run 2 Kripp &alpha;</th>
                    </tr>
                  </thead>
                  <tbody>
                    {TIER1_COMPARISON.map((row) => (
                      <tr key={row.scale} className="border-b border-hair">
                        <td className="py-2 pr-4 capitalize font-medium" style={{ color: SCALE_COLORS[row.scale] }}>
                          {row.scale}
                        </td>
                        <td className="text-right py-2 px-3 num">{row.run1ICC.toFixed(4)}</td>
                        <td className="text-right py-2 px-3 num">{row.run2ICC.toFixed(4)}</td>
                        <td className="text-right py-2 px-3 num">{row.run1Kripp.toFixed(4)}</td>
                        <td className="text-right py-2 pl-3 num">{row.run2Kripp.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* 2b. Eta-squared comparison */}
            <div>
              <h4 className="text-sm font-semibold text-ink mb-3">Eta-Squared (&eta;&sup2;) by Factor</h4>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={etaBarData} layout="vertical" margin={{ left: 110 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
                  <XAxis type="number" tickFormatter={(v: number) => v.toFixed(3)} tick={{ fill: CHART.tick }} />
                  <YAxis type="category" dataKey="factor" width={100} tick={{ fontSize: 12, fill: CHART.tick }} />
                  <Tooltip
                    formatter={(v: number) => v.toFixed(6)}
                    contentStyle={{ backgroundColor: CHART.tooltipBg, border: `1px solid ${CHART.tooltipBorder}`, borderRadius: 8, color: CHART.label }}
                  />
                  <Legend />
                  <Bar dataKey="Run 1" fill={PALETTE.gold} barSize={14} />
                  <Bar dataKey="Run 2" fill={PALETTE.ink} barSize={14} stroke={PALETTE.ink} strokeDasharray={dashFor(1)} />
                </BarChart>
              </ResponsiveContainer>

              <div className="mt-4 callout text-sm">
                <p className="font-semibold mb-1">Both runs confirm: question type explains 4-5x more variance than scale type.</p>
                <p>
                  Question type &eta;&sup2; = {ETA_COMPARISON[0].run1.toFixed(4)} (Run 1) vs {ETA_COMPARISON[0].run2.toFixed(4)} (Run 2).
                  Scale type &eta;&sup2; = {ETA_COMPARISON[2].run1.toFixed(4)} (Run 1) vs {ETA_COMPARISON[2].run2.toFixed(4)} (Run 2).
                  The hierarchy is stable across runs: <strong>question &gt; text &gt; scale</strong>.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ═══════════════════════════════════════════════════════════════════════
          SECTION 3: Logprob Analysis
          ═══════════════════════════════════════════════════════════════════════ */}
      <div className="card">
        <button
          className="flex items-center justify-between w-full text-left"
          onClick={() => setShowSection3(!showSection3)}
        >
          <div>
            <div className="panel-title">Section 3</div>
            <h3 className="section-title">Logprob Confidence Analysis</h3>
          </div>
          <span className="btn-secondary">{showSection3 ? 'Hide' : 'Show'}</span>
        </button>

        {showSection3 && (
          <div className="mt-4 space-y-6">
            {/* 3b. Confidence by scale - table */}
            <div>
              <h4 className="text-sm font-semibold text-ink mb-3">Confidence by Scale</h4>
              <div className="overflow-x-auto">
                <table className="data-table w-full text-sm">
                  <thead>
                    <tr className="border-b border-hair">
                      <th className="text-left py-2 pr-4 font-medium text-mute">Scale</th>
                      <th className="text-right py-2 px-3 font-medium text-mute">Mean Confidence</th>
                      <th className="text-right py-2 px-3 font-medium text-mute">% Below 90%</th>
                      <th className="text-right py-2 pl-3 font-medium text-mute">% Below 50%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {CONFIDENCE_BY_SCALE.map((row) => (
                      <tr key={row.scale} className="border-b border-hair">
                        <td className="py-2 pr-4 capitalize font-medium" style={{ color: SCALE_COLORS[row.scale] }}>
                          {row.scale}
                        </td>
                        <td className="text-right py-2 px-3 num">{(row.meanConf * 100).toFixed(2)}%</td>
                        <td className="text-right py-2 px-3 num">{row.pctBelow90.toFixed(2)}%</td>
                        <td className="text-right py-2 pl-3 num">{row.pctBelow50.toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Confidence by scale bar chart */}
              <div className="mt-4">
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={CONFIDENCE_BY_SCALE}>
                    <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
                    <XAxis dataKey="scale" tick={{ fill: CHART.tick }} />
                    <YAxis domain={[0.9, 1.0]} tickFormatter={(v: number) => (v * 100).toFixed(0) + '%'} tick={{ fill: CHART.tick }} />
                    <Tooltip
                      formatter={(v: number) => (v * 100).toFixed(2) + '%'}
                      contentStyle={{ backgroundColor: CHART.tooltipBg, border: `1px solid ${CHART.tooltipBorder}`, borderRadius: 8, color: CHART.label }}
                    />
                    <Bar dataKey="meanConf" name="Mean Confidence">
                      {CONFIDENCE_BY_SCALE.map((entry) => (
                        <Cell key={entry.scale} fill={SCALE_COLORS[entry.scale]} stroke={CHART.grid} strokeWidth={1} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* 3c. Confidence heatmap */}
            <div>
              <h4 className="text-sm font-semibold text-ink mb-1">Confidence Heatmap (continuous scale shown)</h4>
              <p className="text-xs muted mb-3">
                Mean logprob confidence per question &times; scale. Darker gold = high confidence (near 1.0). Pale = low confidence.
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr>
                      <th className="text-left py-1 pr-2 font-medium text-mute w-36">Question</th>
                      {SCALE_ORDER.map((s) => (
                        <th key={s} className="py-1 px-2 font-medium text-mute capitalize text-center w-28">{s}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {CONFIDENCE_HEATMAP.map((row) => (
                      <tr key={row.question}>
                        <td className="py-1 pr-2 font-medium text-ink text-xs">{row.question}</td>
                        {SCALE_ORDER.map((s) => {
                          const val = (row as any)[s] as number
                          const bg = confidenceToColor(val)
                          return (
                            <td key={s} className="py-1 px-2">
                              <div
                                className="rounded px-2 py-1 text-center font-mono"
                                style={{
                                  backgroundColor: bg,
                                  color: onMagnitude(val),
                                }}
                                title={`${row.question} / ${s}: ${(val * 100).toFixed(2)}%`}
                              >
                                {(val * 100).toFixed(1)}%
                              </div>
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* 3d. Question confidence ranking */}
            <div>
              <h4 className="text-sm font-semibold text-ink mb-3">Question Confidence Ranking (Continuous Scale, Lowest First)</h4>
              <ResponsiveContainer width="100%" height={520}>
                <BarChart data={QUESTION_CONFIDENCE_RANKING} layout="vertical" margin={{ left: 120 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
                  <XAxis type="number" domain={[0.6, 1.0]} tickFormatter={(v: number) => (v * 100).toFixed(0) + '%'} tick={{ fill: CHART.tick }} />
                  <YAxis type="category" dataKey="question" width={110} tick={{ fontSize: 11, fill: CHART.tick }} />
                  <Tooltip
                    formatter={(v: number) => (v * 100).toFixed(2) + '%'}
                    contentStyle={{ backgroundColor: CHART.tooltipBg, border: `1px solid ${CHART.tooltipBorder}`, borderRadius: 8, color: CHART.label }}
                  />
                  <Bar dataKey="meanConf" name="Mean Confidence">
                    {QUESTION_CONFIDENCE_RANKING.map((entry) => (
                      <Cell key={entry.question} fill={confidenceBarColor(entry.meanConf)} stroke={CHART.grid} strokeWidth={1} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* 3e. Confidence distribution */}
            <div>
              <h4 className="text-sm font-semibold text-ink mb-3">Confidence Distribution by Scale</h4>
              <div className="flex flex-wrap gap-2 mb-4">
                {SCALE_ORDER.map((s) => (
                  <button
                    key={s}
                    onClick={() => setConfDistScale(s)}
                    className={`px-3 py-1 rounded-full text-sm font-medium capitalize transition-colors border ${
                      confDistScale === s
                        ? 'bg-gold text-white border-transparent'
                        : 'bg-transparent text-mute border-hair'
                    }`}
                    style={confDistScale === s ? { backgroundColor: SCALE_COLORS[s] } : undefined}
                  >
                    {s}
                  </button>
                ))}
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={CONFIDENCE_DISTRIBUTION[confDistScale]}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
                  <XAxis dataKey="bin" tick={{ fill: CHART.tick }} />
                  <YAxis tickFormatter={(v: number) => v.toFixed(0) + '%'} tick={{ fill: CHART.tick }} />
                  <Tooltip
                    formatter={(v: number, name: string) => {
                      if (name === 'pct') return v.toFixed(1) + '%'
                      return v
                    }}
                    contentStyle={{ backgroundColor: CHART.tooltipBg, border: `1px solid ${CHART.tooltipBorder}`, borderRadius: 8, color: CHART.label }}
                  />
                  <Bar dataKey="pct" name="% of Evaluations" fill={SCALE_COLORS[confDistScale]}>
                    {CONFIDENCE_DISTRIBUTION[confDistScale].map((_, idx) => (
                      <Cell key={idx} fill={SCALE_COLORS[confDistScale]} opacity={0.6 + idx * 0.06} stroke={CHART.grid} strokeWidth={1} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* 3f. Callout */}
            <div className="callout text-sm">
              <p className="font-semibold mb-1">Confidence vs Variance Correlation</p>
              <p>
                Pearson r = {CONF_VS_VAR_CORRELATION.pearsonR}, Spearman &rho; = {CONF_VS_VAR_CORRELATION.spearmanR}.
                Weak negative correlation &mdash; logprobs alone are a noisy predictor of instability at the per-question level.
                A question can have high model confidence and still show variance across 20 repeated samples.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* ═══════════════════════════════════════════════════════════════════════
          SECTION 4: Predictive Power Test
          ═══════════════════════════════════════════════════════════════════════ */}
      <div className="card">
        <button
          className="flex items-center justify-between w-full text-left"
          onClick={() => setShowSection4(!showSection4)}
        >
          <div>
            <div className="panel-title">Section 4</div>
            <h3 className="section-title">Can Logprobs Replace Multi-Sample Testing?</h3>
          </div>
          <span className="btn-secondary">{showSection4 ? 'Hide' : 'Show'}</span>
        </button>

        {showSection4 && (
          <div className="mt-4 space-y-6">
            {/* 4b. Summary card */}
            <div className="callout text-sm">
              <p>
                <strong>Base rate:</strong> {PREDICTIVE.baseRate}% of (text, question, scale) triples show variance &gt; 0
                across 20 samples ({PREDICTIVE.nUnstable} unstable out of {PREDICTIVE.nTotal} total).
                Can we detect these from a single sample&apos;s logprob?
              </p>
            </div>

            {/* 4c. ROC-style table */}
            <div>
              <h4 className="text-sm font-semibold text-ink mb-3">Threshold Sweep: Precision / Recall / F1</h4>
              <div className="overflow-x-auto">
                <table className="data-table w-full text-sm">
                  <thead>
                    <tr className="border-b border-hair">
                      <th className="text-left py-2 pr-3 font-medium text-mute">Threshold</th>
                      <th className="text-right py-2 px-3 font-medium text-mute">Precision</th>
                      <th className="text-right py-2 px-3 font-medium text-mute">Recall</th>
                      <th className="text-right py-2 px-3 font-medium text-mute">F1</th>
                      <th className="text-right py-2 pl-3 font-medium text-mute">Accuracy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ROC_POINTS.map((row) => {
                      const isBest = row.threshold === bestF1.threshold
                      return (
                        <tr
                          key={row.threshold}
                          className={`border-b border-hair ${isBest ? 'bg-gold-badge font-semibold' : ''}`}
                        >
                          <td className={`py-2 pr-3 font-mono ${isBest ? 'font-semibold text-ink' : ''}`}>
                            {row.threshold}
                            {isBest && <span className="ml-2 text-xs text-gold">(best F1)</span>}
                          </td>
                          <td className="text-right py-2 px-3 num">{row.precision.toFixed(4)}</td>
                          <td className="text-right py-2 px-3 num">{row.recall.toFixed(4)}</td>
                          <td className={`text-right py-2 px-3 num ${isBest ? 'font-semibold text-ink' : ''}`}>
                            {row.f1.toFixed(4)}
                          </td>
                          <td className="text-right py-2 pl-3 num">{row.accuracy.toFixed(4)}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* 4d. ROC curve */}
            <div>
              <h4 className="text-sm font-semibold text-ink mb-3">ROC Curve</h4>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={rocCurveData} margin={{ left: 10, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
                  <XAxis
                    dataKey="fpr"
                    type="number"
                    domain={[0, 1]}
                    tickFormatter={(v: number) => v.toFixed(1)}
                    tick={{ fill: CHART.tick }}
                    label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5, fontSize: 12, fill: CHART.label }}
                  />
                  <YAxis
                    type="number"
                    domain={[0, 1]}
                    tickFormatter={(v: number) => v.toFixed(1)}
                    tick={{ fill: CHART.tick }}
                    label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', fontSize: 12, fill: CHART.label }}
                  />
                  <Tooltip
                    formatter={(v: number, name: string) => [v.toFixed(4), name]}
                    labelFormatter={(l: number) => `FPR: ${l.toFixed(4)}`}
                    contentStyle={{ backgroundColor: CHART.tooltipBg, border: `1px solid ${CHART.tooltipBorder}`, borderRadius: 8, color: CHART.label }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="tpr"
                    name="Logprob Classifier"
                    stroke={PALETTE.gold}
                    strokeWidth={2}
                    dot={{ r: 4, fill: PALETTE.gold }}
                  />
                  <Line
                    type="monotone"
                    dataKey="random"
                    name="Random (diagonal)"
                    stroke={CHART.reference}
                    strokeWidth={1}
                    strokeDasharray="5 5"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* 4e. Verdict callout */}
            <div className="callout-ink text-sm">
              <p className="font-semibold mb-2">Verdict: AUC = {PREDICTIVE.auc}</p>
              <p>
                Logprobs from a single sample are <strong>not</strong> a reliable predictor of multi-sample instability
                at the individual question level. The model can be confident in its answer AND still give a different
                answer next time. This is because instability at temperature=0 comes from sampling noise in the softmax,
                not from the model&apos;s token-level confidence.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* ═══════════════════════════════════════════════════════════════════════
          SECTION 5: Score Agreement
          ═══════════════════════════════════════════════════════════════════════ */}
      <div className="card">
        <button
          className="flex items-center justify-between w-full text-left"
          onClick={() => setShowSection5(!showSection5)}
        >
          <div>
            <div className="panel-title">Section 5</div>
            <h3 className="section-title">Cross-Run Score Agreement</h3>
          </div>
          <span className="btn-secondary">{showSection5 ? 'Hide' : 'Show'}</span>
        </button>

        {showSection5 && (
          <div className="mt-4 space-y-6">
            {/* 5b. Agreement summary cards */}
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              {/* Overall stat */}
              <div className="metric-card md:col-span-1 text-center">
                <p className="metric-label mb-1">Overall Agreement</p>
                <p className="stat-num">{AGREEMENT.overallAgreePct}%</p>
                <p className="text-xs muted mt-1">{AGREEMENT.totalPairs} triples</p>
              </div>

              {/* Per-scale cards */}
              {AGREEMENT_PER_SCALE.map((row) => (
                <div
                  key={row.scale}
                  className="metric-card text-center"
                >
                  <p
                    className="text-xs font-medium mb-1 capitalize"
                    style={{ color: SCALE_COLORS[row.scale] }}
                  >
                    {row.scale}
                  </p>
                  <p className="stat-num text-2xl text-ink">{row.agreePct}%</p>
                  <p className="text-xs muted mt-1">
                    {row.disagreeCount} / 200 disagree
                  </p>
                </div>
              ))}
            </div>

            {/* 5c. Callout */}
            <div className="callout text-sm">
              <p className="font-semibold mb-2">98.9% modal score agreement across runs</p>
              <p>
                Only 9 out of {AGREEMENT.totalPairs} (text, question, scale) triples have different modal scores between
                Run 1 and Run 2. The LLM&apos;s &quot;opinion&quot; is highly stable &mdash; it&apos;s the variance around
                that opinion that fluctuates between runs. This confirms that instability is about precision, not about the
                model changing its mind.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
