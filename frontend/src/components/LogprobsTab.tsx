import { useState } from 'react'
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell,
} from 'recharts'
import {
  BookOpen, BarChart3, Layers, TrendingUp, FlaskConical, ChevronDown, ChevronUp, Microscope, AlertTriangle, CheckCircle, Target,
} from 'lucide-react'

// ─── Hardcoded Data from Second Run with Logprobs ───────────────────────────

const SCALE_COLORS: Record<string, string> = {
  binary: '#2196F3',
  ternary: '#4CAF50',
  quaternary: '#FF9800',
  continuous: '#E91E63',
}

const SCALE_ORDER = ['binary', 'ternary', 'quaternary', 'continuous']

const HEATMAP_DATA = [
  { question: 'Named Entities', binary: 0.0, ternary: 0.0, quaternary: 0.0, continuous: 0.0 },
  { question: 'Actions/Events', binary: 0.0, ternary: 0.013437, quaternary: 0.002538, continuous: 0.003547 },
  { question: 'Causality', binary: 0.024, ternary: 0.00225, quaternary: 0.010188, continuous: 0.004088 },
  { question: 'Temporal', binary: 0.0, ternary: 0.0, quaternary: 0.0, continuous: 0.0 },
  { question: 'Spatial', binary: 0.0, ternary: 0.0, quaternary: 0.003082, continuous: 0.000477 },
  { question: 'Numeric', binary: 0.0, ternary: 0.0, quaternary: 0.002861, continuous: 0.000047 },
  { question: 'Negation', binary: 0.009, ternary: 0.010438, quaternary: 0.00404, continuous: 0.00329 },
  { question: 'Uncertainty', binary: 0.0, ternary: 0.0, quaternary: 0.0, continuous: 0.00065 },
  { question: 'Modality', binary: 0.0, ternary: 0.0, quaternary: 0.0, continuous: 0.0 },
  { question: 'Sentiment', binary: 0.0, ternary: 0.0, quaternary: 0.001388, continuous: 0.003188 },
  { question: 'Emotion', binary: 0.0, ternary: 0.006188, quaternary: 0.016988, continuous: 0.0 },
  { question: 'Social', binary: 0.0, ternary: 0.0, quaternary: 0.003179, continuous: 0.000418 },
  { question: 'Dialogue', binary: 0.02275, ternary: 0.016, quaternary: 0.009, continuous: 0.0 },
  { question: 'First Person', binary: 0.0, ternary: 0.0, quaternary: 0.0, continuous: 0.0 },
  { question: 'Imperative', binary: 0.0, ternary: 0.0, quaternary: 0.0, continuous: 0.0 },
  { question: 'Comparison', binary: 0.0, ternary: 0.0, quaternary: 0.0, continuous: 0.000048 },
  { question: 'Normative', binary: 0.0, ternary: 0.0, quaternary: 0.0, continuous: 0.0 },
  { question: 'Intent', binary: 0.0, ternary: 0.006, quaternary: 0.012959, continuous: 0.00459 },
  { question: 'Concreteness', binary: 0.0, ternary: 0.001188, quaternary: 0.01504, continuous: 0.010155 },
  { question: 'Identity', binary: 0.0, ternary: 0.01375, quaternary: 0.030834, continuous: 0.00171 },
]

const SCALE_SUMMARY = [
  { scale: 'binary', meanVar: 0.002788, medianVar: 0.0, zeroVarPct: 98.5, highVarPct: 1.5, medianEntropy: 0.0, meanConsistency: 0.9957 },
  { scale: 'ternary', meanVar: 0.003463, medianVar: 0.0, zeroVarPct: 94.5, highVarPct: 5.5, medianEntropy: 0.0, meanConsistency: 0.9878 },
  { scale: 'quaternary', meanVar: 0.005605, medianVar: 0.0, zeroVarPct: 87.5, highVarPct: 11.0, medianEntropy: 0.0, meanConsistency: 0.9745 },
  { scale: 'continuous', meanVar: 0.001610, medianVar: 0.0, zeroVarPct: 79.5, highVarPct: 4.0, medianEntropy: 0.0, meanConsistency: 0.9782 },
]

const DEGRADATION_DATA = [
  { scale: 'binary', factual_simple: 0.0, sentiment_positive: 0.0, sentiment_negative: 0.0045, ambiguous: 0.011375, medical_clinical: 0.0, negation_heavy: 0.012, imperative_action: 0.0, abstract_philosophical: 0.0, narrative_paragraph: 0.0, technical_ml: 0.0 },
  { scale: 'ternary', factual_simple: 0.004375, sentiment_positive: 0.0, sentiment_negative: 0.002375, ambiguous: 0.011, medical_clinical: 0.002375, negation_heavy: 0.003469, imperative_action: 0.0, abstract_philosophical: 0.002844, narrative_paragraph: 0.003688, technical_ml: 0.0045 },
  { scale: 'quaternary', factual_simple: 0.006634, sentiment_positive: 0.003954, sentiment_negative: 0.0, ambiguous: 0.014473, medical_clinical: 0.005562, negation_heavy: 0.0, imperative_action: 0.003581, abstract_philosophical: 0.0, narrative_paragraph: 0.021846, technical_ml: 0.0 },
  { scale: 'continuous', factual_simple: 0.000024, sentiment_positive: 0.001689, sentiment_negative: 0.000175, ambiguous: 0.002864, medical_clinical: 0.001851, negation_heavy: 0.000455, imperative_action: 0.001605, abstract_philosophical: 0.000604, narrative_paragraph: 0.001589, technical_ml: 0.005249 },
]

const TEXT_DIFFICULTY = [
  { id: 'ambiguous', meanVariance: 0.009928 },
  { id: 'narrative_paragraph', meanVariance: 0.006781 },
  { id: 'negation_heavy', meanVariance: 0.003981 },
  { id: 'factual_simple', meanVariance: 0.002758 },
  { id: 'medical_clinical', meanVariance: 0.002447 },
  { id: 'technical_ml', meanVariance: 0.002437 },
  { id: 'sentiment_negative', meanVariance: 0.001763 },
  { id: 'sentiment_positive', meanVariance: 0.001411 },
  { id: 'imperative_action', meanVariance: 0.001296 },
  { id: 'abstract_philosophical', meanVariance: 0.000862 },
]

const MODE_CONSISTENCY_DATA = [
  { id: 'factual_simple', binary: 1.0, ternary: 0.9775, quaternary: 0.99, continuous: 0.9975 },
  { id: 'sentiment_positive', binary: 1.0, ternary: 1.0, quaternary: 0.96, continuous: 0.99 },
  { id: 'sentiment_negative', binary: 0.995, ternary: 0.9975, quaternary: 1.0, continuous: 0.9875 },
  { id: 'ambiguous', binary: 0.9825, ternary: 0.97, quaternary: 0.95, continuous: 0.9575 },
  { id: 'medical_clinical', binary: 1.0, ternary: 0.9975, quaternary: 0.98, continuous: 0.9775 },
  { id: 'negation_heavy', binary: 0.98, ternary: 0.9825, quaternary: 1.0, continuous: 0.9825 },
  { id: 'imperative_action', binary: 1.0, ternary: 1.0, quaternary: 0.98, continuous: 0.975 },
  { id: 'abstract_philosophical', binary: 1.0, ternary: 0.9825, quaternary: 1.0, continuous: 0.975 },
  { id: 'narrative_paragraph', binary: 1.0, ternary: 0.975, quaternary: 0.885, continuous: 0.9875 },
  { id: 'technical_ml', binary: 1.0, ternary: 0.995, quaternary: 1.0, continuous: 0.9525 },
]

const PER_TEXT_TABLE = [
  { textId: 'factual_simple', scale: 'binary', avgVariance: 0.0, avgConsistency: 1.0, avgEntropy: 0.0, uniqueVectors: 1 },
  { textId: 'factual_simple', scale: 'ternary', avgVariance: 0.004375, avgConsistency: 0.9775, avgEntropy: 0.061, uniqueVectors: 3 },
  { textId: 'factual_simple', scale: 'quaternary', avgVariance: 0.006634, avgConsistency: 0.99, avgEntropy: 0.0448, uniqueVectors: 3 },
  { textId: 'factual_simple', scale: 'continuous', avgVariance: 0.000024, avgConsistency: 0.9975, avgEntropy: 0.0143, uniqueVectors: 2 },
  { textId: 'sentiment_positive', scale: 'binary', avgVariance: 0.0, avgConsistency: 1.0, avgEntropy: 0.0, uniqueVectors: 1 },
  { textId: 'sentiment_positive', scale: 'ternary', avgVariance: 0.0, avgConsistency: 1.0, avgEntropy: 0.0, uniqueVectors: 1 },
  { textId: 'sentiment_positive', scale: 'quaternary', avgVariance: 0.003954, avgConsistency: 0.96, avgEntropy: 0.1322, uniqueVectors: 6 },
  { textId: 'sentiment_positive', scale: 'continuous', avgVariance: 0.001689, avgConsistency: 0.99, avgEntropy: 0.0448, uniqueVectors: 3 },
  { textId: 'sentiment_negative', scale: 'binary', avgVariance: 0.0045, avgConsistency: 0.995, avgEntropy: 0.0234, uniqueVectors: 2 },
  { textId: 'sentiment_negative', scale: 'ternary', avgVariance: 0.002375, avgConsistency: 0.9975, avgEntropy: 0.0143, uniqueVectors: 2 },
  { textId: 'sentiment_negative', scale: 'quaternary', avgVariance: 0.0, avgConsistency: 1.0, avgEntropy: 0.0, uniqueVectors: 1 },
  { textId: 'sentiment_negative', scale: 'continuous', avgVariance: 0.000175, avgConsistency: 0.9875, avgEntropy: 0.0608, uniqueVectors: 4 },
  { textId: 'ambiguous', scale: 'binary', avgVariance: 0.011375, avgConsistency: 0.9825, avgEntropy: 0.0467, uniqueVectors: 2 },
  { textId: 'ambiguous', scale: 'ternary', avgVariance: 0.011, avgConsistency: 0.97, avgEntropy: 0.0846, uniqueVectors: 3 },
  { textId: 'ambiguous', scale: 'quaternary', avgVariance: 0.014473, avgConsistency: 0.95, avgEntropy: 0.1548, uniqueVectors: 7 },
  { textId: 'ambiguous', scale: 'continuous', avgVariance: 0.002864, avgConsistency: 0.9575, avgEntropy: 0.1239, uniqueVectors: 4 },
  { textId: 'medical_clinical', scale: 'binary', avgVariance: 0.0, avgConsistency: 1.0, avgEntropy: 0.0, uniqueVectors: 1 },
  { textId: 'medical_clinical', scale: 'ternary', avgVariance: 0.002375, avgConsistency: 0.9975, avgEntropy: 0.0143, uniqueVectors: 2 },
  { textId: 'medical_clinical', scale: 'quaternary', avgVariance: 0.005562, avgConsistency: 0.98, avgEntropy: 0.0917, uniqueVectors: 4 },
  { textId: 'medical_clinical', scale: 'continuous', avgVariance: 0.001851, avgConsistency: 0.9775, avgEntropy: 0.0984, uniqueVectors: 4 },
  { textId: 'negation_heavy', scale: 'binary', avgVariance: 0.012, avgConsistency: 0.98, avgEntropy: 0.0485, uniqueVectors: 2 },
  { textId: 'negation_heavy', scale: 'ternary', avgVariance: 0.003469, avgConsistency: 0.9825, avgEntropy: 0.064, uniqueVectors: 3 },
  { textId: 'negation_heavy', scale: 'quaternary', avgVariance: 0.0, avgConsistency: 1.0, avgEntropy: 0.0, uniqueVectors: 1 },
  { textId: 'negation_heavy', scale: 'continuous', avgVariance: 0.000455, avgConsistency: 0.9825, avgEntropy: 0.0467, uniqueVectors: 2 },
  { textId: 'imperative_action', scale: 'binary', avgVariance: 0.0, avgConsistency: 1.0, avgEntropy: 0.0, uniqueVectors: 1 },
  { textId: 'imperative_action', scale: 'ternary', avgVariance: 0.0, avgConsistency: 1.0, avgEntropy: 0.0, uniqueVectors: 1 },
  { textId: 'imperative_action', scale: 'quaternary', avgVariance: 0.003581, avgConsistency: 0.98, avgEntropy: 0.0938, uniqueVectors: 2 },
  { textId: 'imperative_action', scale: 'continuous', avgVariance: 0.001605, avgConsistency: 0.975, avgEntropy: 0.072, uniqueVectors: 4 },
  { textId: 'abstract_philosophical', scale: 'binary', avgVariance: 0.0, avgConsistency: 1.0, avgEntropy: 0.0, uniqueVectors: 1 },
  { textId: 'abstract_philosophical', scale: 'ternary', avgVariance: 0.002844, avgConsistency: 0.9825, avgEntropy: 0.0467, uniqueVectors: 2 },
  { textId: 'abstract_philosophical', scale: 'quaternary', avgVariance: 0.0, avgConsistency: 1.0, avgEntropy: 0.0, uniqueVectors: 1 },
  { textId: 'abstract_philosophical', scale: 'continuous', avgVariance: 0.000604, avgConsistency: 0.975, avgEntropy: 0.1027, uniqueVectors: 5 },
  { textId: 'narrative_paragraph', scale: 'binary', avgVariance: 0.0, avgConsistency: 1.0, avgEntropy: 0.0, uniqueVectors: 1 },
  { textId: 'narrative_paragraph', scale: 'ternary', avgVariance: 0.003688, avgConsistency: 0.975, avgEntropy: 0.064, uniqueVectors: 3 },
  { textId: 'narrative_paragraph', scale: 'quaternary', avgVariance: 0.021846, avgConsistency: 0.885, avgEntropy: 0.3266, uniqueVectors: 9 },
  { textId: 'narrative_paragraph', scale: 'continuous', avgVariance: 0.001589, avgConsistency: 0.9875, avgEntropy: 0.0664, uniqueVectors: 4 },
  { textId: 'technical_ml', scale: 'binary', avgVariance: 0.0, avgConsistency: 1.0, avgEntropy: 0.0, uniqueVectors: 1 },
  { textId: 'technical_ml', scale: 'ternary', avgVariance: 0.0045, avgConsistency: 0.995, avgEntropy: 0.0234, uniqueVectors: 2 },
  { textId: 'technical_ml', scale: 'quaternary', avgVariance: 0.0, avgConsistency: 1.0, avgEntropy: 0.0, uniqueVectors: 1 },
  { textId: 'technical_ml', scale: 'continuous', avgVariance: 0.005249, avgConsistency: 0.9525, avgEntropy: 0.1548, uniqueVectors: 8 },
]

const TEXT_PALETTE = [
  '#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
  '#ec4899', '#14b8a6', '#f97316', '#6366f1', '#84cc16',
]

const TEXT_IDS = [
  'factual_simple', 'sentiment_positive', 'sentiment_negative', 'ambiguous',
  'medical_clinical', 'negation_heavy', 'imperative_action', 'abstract_philosophical',
  'narrative_paragraph', 'technical_ml',
]

// ─── Advanced Stats (Tier 1-3) ──────────────────────────────────────────────

const TIER1_DATA = [
  { scale: 'binary', cronbachsAlpha: 0.7, icc: 0.984448, cohensKappa: 0.983693, krippendorffAlpha: 0.983723 },
  { scale: 'ternary', cronbachsAlpha: 0.092592, icc: 0.971374, cohensKappa: 0.997078, krippendorffAlpha: 0.97325 },
  { scale: 'quaternary', cronbachsAlpha: -0.214548, icc: 0.954509, cohensKappa: 0.99599, krippendorffAlpha: 0.958933 },
  { scale: 'continuous', cronbachsAlpha: 0.065423, icc: 0.967779, cohensKappa: null, krippendorffAlpha: 0.966306 },
]

const TIER2_TEST_RETEST = [
  { scale: 'binary', pearsonR: 0.996851, spearmanRho: 0.994987 },
  { scale: 'ternary', pearsonR: 0.999, spearmanRho: 0.986755 },
  { scale: 'quaternary', pearsonR: 0.995877, spearmanRho: 0.968377 },
  { scale: 'continuous', pearsonR: 0.99841, spearmanRho: 0.989931 },
]

const TIER2_BLAND_ALTMAN = [
  { scale: 'binary', meanBias: 0.0009, meanStdDiff: 0.03915119 },
  { scale: 'ternary', meanBias: 0.0005, meanStdDiff: 0.07349987 },
  { scale: 'quaternary', meanBias: 0.001053, meanStdDiff: 0.0769618 },
  { scale: 'continuous', meanBias: -0.00048, meanStdDiff: 0.05290967 },
]

const REGRESSION = { slope: -0.00013889, rSquared: 0.01143, pValue: 0.89309 }

const TIER3_BOOTSTRAP = [
  { scale: 'binary', varPoint: 0.002788, varLower: 0.0, varUpper: 0.006388, conPoint: 0.99575, conLower: 0.99, conUpper: 1.0 },
  { scale: 'ternary', varPoint: 0.003463, varLower: 0.001369, varUpper: 0.006169, conPoint: 0.98775, conLower: 0.978, conUpper: 0.995506 },
  { scale: 'quaternary', varPoint: 0.005605, varLower: 0.002825, varUpper: 0.00868, conPoint: 0.9745, conLower: 0.962244, conUpper: 0.986 },
  { scale: 'continuous', varPoint: 0.00161, varLower: 0.000706, varUpper: 0.002701, conPoint: 0.97825, conLower: 0.96775, conUpper: 0.987756 },
]

const FRIEDMAN = { chiSquared: 42.617866, pValue: 2.97e-09, significant: true }
const ETA_SQUARED = { byScale: 0.006311, byText: 0.022173, byQuestion: 0.050296 }

// ─── Utility ──────────────────────────────────────────────────────────────────

function varianceToColor(value: number, maxValue: number): string {
  if (maxValue === 0) return '#22c55e'
  const ratio = Math.min(value / maxValue, 1)
  if (ratio < 0.5) {
    const t = ratio * 2
    return `rgb(${Math.round(34 + t * 211)},${Math.round(197 + t * -39)},${Math.round(94 + t * -83)})`
  }
  const t = (ratio - 0.5) * 2
  return `rgb(${Math.round(245 + t * -6)},${Math.round(158 + t * -90)},${Math.round(11 + t * 57)})`
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function LogprobsTab() {
  const [showTier1, setShowTier1] = useState(true)
  const [showTier2, setShowTier2] = useState(true)
  const [showTier3, setShowTier3] = useState(true)

  const heatmapMaxVar = Math.max(
    ...HEATMAP_DATA.flatMap((row) => SCALE_ORDER.map((s) => (row as any)[s] as number)),
    0.001
  )

  return (
    <div className="space-y-8">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className="card bg-gradient-to-br from-indigo-50 to-blue-50 border-l-4 border-indigo-500">
        <div className="flex items-center space-x-3 mb-3">
          <BookOpen className="w-7 h-7 text-indigo-600" />
          <h2 className="text-2xl font-bold text-gray-900">
            SECOND Run with Logprobs - 16,000 Evaluations
          </h2>
        </div>
        <p className="text-gray-600 leading-relaxed">
          This page presents hardcoded results from the <strong>second batch run</strong> with logprobs enabled.
          10 texts &times; 20 questions &times; 4 scales &times; 20 samples = <strong>16,000 individual LLM evaluations</strong> on
          GPT-4o-mini at temperature=0. Logprobs were collected to enable advanced statistical reliability analysis
          (Cronbach&rsquo;s &alpha;, ICC, Cohen&rsquo;s &kappa;, Krippendorff&rsquo;s &alpha;, bootstrap CIs, Friedman test).
        </p>
        <div className="mt-3 flex flex-wrap gap-3 text-sm">
          <span className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full font-medium">GPT-4o-mini</span>
          <span className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full font-medium">Temperature = 0</span>
          <span className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full font-medium">20 repeated samples</span>
          <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full font-medium">Logprobs Enabled</span>
          <span className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full font-medium">Second Run</span>
        </div>
      </div>

      {/* ── Scale Degradation Summary ──────────────────────────────────────── */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <Layers className="w-5 h-5 text-blue-600" />
          <h3 className="text-lg font-semibold text-gray-900">Scale Degradation Summary</h3>
        </div>
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

        {/* Key finding callout */}
        <div className="mt-4 bg-blue-50 border border-blue-200 rounded-lg p-4 text-sm text-blue-800">
          <p className="font-semibold mb-1">Finding: Quaternary remains the least reliable scale</p>
          <p>
            Quaternary has the <strong>highest</strong> mean variance (0.0056) and highest high-variance rate (11.0%).
            Continuous shows the <strong>lowest</strong> mean variance (0.0016) but also the lowest zero-variance rate (79.5%),
            confirming the pattern of quiet, distributed noise.
          </p>
        </div>
      </div>

      {/* ── Mean Variance by Scale (bar chart) ─────────────────────────────── */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <BarChart3 className="w-5 h-5 text-blue-600" />
          <h3 className="text-lg font-semibold text-gray-900">Mean Variance by Scale</h3>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={SCALE_SUMMARY}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="scale" />
            <YAxis tickFormatter={(v: number) => v.toFixed(4)} />
            <Tooltip formatter={(v: number) => v.toFixed(6)} />
            <Bar dataKey="meanVar" name="Mean Variance">
              {SCALE_SUMMARY.map((entry) => (
                <Cell key={entry.scale} fill={SCALE_COLORS[entry.scale]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* ── Question Stability Heatmap ─────────────────────────────────────── */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-2">
          <Microscope className="w-5 h-5 text-purple-600" />
          <h3 className="text-lg font-semibold text-gray-900">Question Stability Across Scales</h3>
        </div>
        <p className="text-sm text-gray-500 mb-4">
          Mean variance per question across all 10 texts. Green = stable (zero variance). Red = unstable.
        </p>

        {/* Interactive heatmap */}
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr>
                <th className="text-left py-1 pr-2 font-medium text-gray-500 w-36">Question</th>
                {SCALE_ORDER.map((s) => (
                  <th key={s} className="py-1 px-2 font-medium text-gray-500 capitalize text-center w-28">{s}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {HEATMAP_DATA.map((row) => (
                <tr key={row.question}>
                  <td className="py-1 pr-2 font-medium text-gray-700 text-xs">{row.question}</td>
                  {SCALE_ORDER.map((s) => {
                    const val = (row as any)[s] as number
                    return (
                      <td key={s} className="py-1 px-2">
                        <div
                          className="rounded px-2 py-1 text-center font-mono"
                          style={{
                            backgroundColor: varianceToColor(val, heatmapMaxVar),
                            color: val / heatmapMaxVar > 0.5 ? '#fff' : '#1f2937',
                          }}
                          title={`${row.question} / ${s}: ${val.toFixed(6)}`}
                        >
                          {val.toFixed(4)}
                        </div>
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Finding callout */}
        <div className="mt-4 bg-green-50 border border-green-200 rounded-lg p-4 text-sm">
          <div className="flex items-center space-x-2 mb-1">
            <CheckCircle className="w-4 h-4 text-green-600" />
            <p className="font-semibold text-green-800">Finding: Reliability varies dramatically by feature type</p>
          </div>
          <p className="text-green-700">
            <strong>Perfectly stable</strong> (zero variance): Named Entities, Temporal, First Person, Modality, Imperative, Normative.
            <br />
            <strong>Unstable</strong>: Identity (0.0308 at quaternary), Causality (0.024 binary), Dialogue (0.0228 binary),
            Emotion (0.0170 quaternary), Concreteness (0.0150 quaternary).
          </p>
        </div>
      </div>

      {/* ── Variance Degradation by Text Type (line chart) ─────────────────── */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-2">
          <TrendingUp className="w-5 h-5 text-blue-600" />
          <h3 className="text-lg font-semibold text-gray-900">Variance Degradation by Text Type</h3>
        </div>
        <p className="text-sm text-gray-500 mb-4">
          How each text&rsquo;s mean variance changes across scales (binary &rarr; continuous).
        </p>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={DEGRADATION_DATA}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="scale" />
            <YAxis tickFormatter={(v: number) => v.toFixed(4)} />
            <Tooltip formatter={(v: number) => v.toFixed(6)} />
            <Legend />
            {TEXT_IDS.map((id, idx) => (
              <Line
                key={id}
                type="monotone"
                dataKey={id}
                stroke={TEXT_PALETTE[idx % TEXT_PALETTE.length]}
                strokeWidth={2}
                dot={{ r: 3 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* ── Text Difficulty Ranking ────────────────────────────────────────── */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-2">
          <Target className="w-5 h-5 text-red-600" />
          <h3 className="text-lg font-semibold text-gray-900">Text Difficulty Ranking</h3>
        </div>
        <p className="text-sm text-gray-500 mb-4">
          Texts ranked by overall mean variance. Longer bar = more unstable.
        </p>
        <ResponsiveContainer width="100%" height={360}>
          <BarChart data={TEXT_DIFFICULTY} layout="vertical" margin={{ left: 140 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" tickFormatter={(v: number) => v.toFixed(4)} />
            <YAxis type="category" dataKey="id" width={130} tick={{ fontSize: 12 }} />
            <Tooltip formatter={(v: number) => v.toFixed(6)} />
            <Bar dataKey="meanVariance" name="Mean Variance">
              {TEXT_DIFFICULTY.map((entry) => {
                const maxVar = TEXT_DIFFICULTY[0].meanVariance
                const ratio = entry.meanVariance / maxVar
                return (
                  <Cell
                    key={entry.id}
                    fill={ratio < 0.33 ? '#22c55e' : ratio < 0.66 ? '#f59e0b' : '#ef4444'}
                  />
                )
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4 text-sm text-red-800">
          <p className="font-semibold mb-1">Finding: Ambiguous text remains hardest to grade reliably</p>
          <p>
            Ambiguous text (0.0099) is 11.5&times; more unstable than abstract philosophical (0.0009).
            Narrative paragraphs (0.0068) also remain high-variance, consistent with the first run.
          </p>
        </div>
      </div>

      {/* ── Mode Consistency by Text and Scale ─────────────────────────────── */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-2">
          <BarChart3 className="w-5 h-5 text-green-600" />
          <h3 className="text-lg font-semibold text-gray-900">Mode Consistency by Text and Scale</h3>
        </div>
        <p className="text-sm text-gray-500 mb-4">
          Mean mode consistency across 20 questions. 1.0 = all 20 samples identical.
        </p>
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={MODE_CONSISTENCY_DATA}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="id" tick={{ fontSize: 11 }} angle={-30} textAnchor="end" height={60} />
            <YAxis domain={[0.85, 1.0]} tickFormatter={(v: number) => v.toFixed(2)} />
            <Tooltip formatter={(v: number) => (v * 100).toFixed(2) + '%'} />
            <Legend />
            {SCALE_ORDER.map((scale) => (
              <Bar key={scale} dataKey={scale} name={scale} fill={SCALE_COLORS[scale]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* ── Per-Text Scale Metrics ─────────────────────────────────────────── */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <Layers className="w-5 h-5 text-gray-600" />
          <h3 className="text-lg font-semibold text-gray-900">Per-Text Scale Metrics</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-2 pr-3 font-medium text-gray-500">Text</th>
                <th className="text-left py-2 px-3 font-medium text-gray-500">Scale</th>
                <th className="text-right py-2 px-3 font-medium text-gray-500">Avg Variance</th>
                <th className="text-right py-2 px-3 font-medium text-gray-500">Avg Consistency</th>
                <th className="text-right py-2 px-3 font-medium text-gray-500">Avg Entropy</th>
                <th className="text-right py-2 pl-3 font-medium text-gray-500">Unique Vectors</th>
              </tr>
            </thead>
            <tbody>
              {PER_TEXT_TABLE.map((row, idx) => (
                <tr key={`${row.textId}-${row.scale}`} className={idx % 4 === 3 ? 'border-b border-gray-200' : ''}>
                  {idx % 4 === 0 && (
                    <td className="py-1.5 pr-3 font-medium text-gray-700 font-mono text-xs" rowSpan={4}>
                      {row.textId}
                    </td>
                  )}
                  <td className="py-1.5 px-3 capitalize" style={{ color: SCALE_COLORS[row.scale] }}>
                    {row.scale}
                  </td>
                  <td className="text-right py-1.5 px-3 font-mono">{row.avgVariance.toFixed(4)}</td>
                  <td className="text-right py-1.5 px-3 font-mono">{(row.avgConsistency * 100).toFixed(2)}%</td>
                  <td className="text-right py-1.5 px-3 font-mono">{row.avgEntropy.toFixed(4)}</td>
                  <td className="text-right py-1.5 pl-3 font-mono">{row.uniqueVectors}/20</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Advanced Statistical Analysis ─────────────────────────────────── */}
      <div className="card bg-gradient-to-br from-purple-50 to-indigo-50 border-l-4 border-purple-500">
        <div className="flex items-center space-x-3 mb-3">
          <FlaskConical className="w-7 h-7 text-purple-600" />
          <h2 className="text-2xl font-bold text-gray-900">Advanced Statistical Analysis</h2>
        </div>
        <p className="text-gray-600 leading-relaxed">
          Three tiers of reliability metrics computed from logprobs-enabled second run.
          These go beyond simple variance to established psychometric and agreement measures.
        </p>
      </div>

      {/* ── Tier 1: Reliability Metrics ───────────────────────────────────── */}
      <div className="card">
        <button
          className="flex items-center justify-between w-full"
          onClick={() => setShowTier1(!showTier1)}
        >
          <div className="flex items-center space-x-2">
            <Layers className="w-5 h-5 text-purple-600" />
            <h3 className="text-lg font-semibold text-gray-900">Tier 1: Reliability Metrics</h3>
            <span className="px-2 py-0.5 bg-purple-100 text-purple-700 rounded-full text-xs font-medium">Inter-Rater Agreement</span>
          </div>
          {showTier1 ? <ChevronUp className="w-5 h-5 text-gray-400" /> : <ChevronDown className="w-5 h-5 text-gray-400" />}
        </button>

        {showTier1 && (
          <div className="mt-4">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-2 pr-4 font-medium text-gray-500">Scale</th>
                    <th className="text-right py-2 px-3 font-medium text-gray-500">Cronbach&rsquo;s &alpha;</th>
                    <th className="text-right py-2 px-3 font-medium text-gray-500">ICC</th>
                    <th className="text-right py-2 px-3 font-medium text-gray-500">Cohen&rsquo;s &kappa;</th>
                    <th className="text-right py-2 pl-3 font-medium text-gray-500">Krippendorff&rsquo;s &alpha;</th>
                  </tr>
                </thead>
                <tbody>
                  {TIER1_DATA.map((row) => (
                    <tr key={row.scale} className="border-b border-gray-100">
                      <td className="py-2 pr-4 capitalize font-medium" style={{ color: SCALE_COLORS[row.scale] }}>
                        {row.scale}
                      </td>
                      <td className="text-right py-2 px-3 font-mono">
                        <span className={row.cronbachsAlpha < 0 ? 'text-red-600' : row.cronbachsAlpha < 0.5 ? 'text-amber-600' : 'text-green-600'}>
                          {row.cronbachsAlpha.toFixed(4)}
                        </span>
                      </td>
                      <td className="text-right py-2 px-3 font-mono">{row.icc.toFixed(4)}</td>
                      <td className="text-right py-2 px-3 font-mono">
                        {row.cohensKappa !== null ? row.cohensKappa.toFixed(4) : <span className="text-gray-400">N/A</span>}
                      </td>
                      <td className="text-right py-2 pl-3 font-mono">{row.krippendorffAlpha.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mt-4 bg-purple-50 border border-purple-200 rounded-lg p-4 text-sm text-purple-800">
              <p className="font-semibold mb-1">Interpretation</p>
              <p>
                <strong>ICC &gt; 0.95</strong> across all scales indicates excellent absolute agreement between repeated runs.
                <strong> Cronbach&rsquo;s &alpha;</strong> is low or negative for non-binary scales because it measures internal consistency
                across items, not agreement &mdash; when nearly all items are identical, between-item variance collapses.
                <strong> Cohen&rsquo;s &kappa;</strong> is not applicable for continuous scales (no discrete categories).
                <strong> Krippendorff&rsquo;s &alpha; &gt; 0.95</strong> across all scales confirms strong reliability.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* ── Tier 2: Test-Retest + Bland-Altman ────────────────────────────── */}
      <div className="card">
        <button
          className="flex items-center justify-between w-full"
          onClick={() => setShowTier2(!showTier2)}
        >
          <div className="flex items-center space-x-2">
            <TrendingUp className="w-5 h-5 text-blue-600" />
            <h3 className="text-lg font-semibold text-gray-900">Tier 2: Test-Retest Stability</h3>
            <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded-full text-xs font-medium">Temporal Consistency</span>
          </div>
          {showTier2 ? <ChevronUp className="w-5 h-5 text-gray-400" /> : <ChevronDown className="w-5 h-5 text-gray-400" />}
        </button>

        {showTier2 && (
          <div className="mt-4 space-y-6">
            {/* Test-Retest Correlations */}
            <div>
              <p className="text-sm font-semibold text-gray-700 mb-2">Test-Retest Correlations (Split-Half)</p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200">
                      <th className="text-left py-2 pr-4 font-medium text-gray-500">Scale</th>
                      <th className="text-right py-2 px-3 font-medium text-gray-500">Pearson r</th>
                      <th className="text-right py-2 pl-3 font-medium text-gray-500">Spearman &rho;</th>
                    </tr>
                  </thead>
                  <tbody>
                    {TIER2_TEST_RETEST.map((row) => (
                      <tr key={row.scale} className="border-b border-gray-100">
                        <td className="py-2 pr-4 capitalize font-medium" style={{ color: SCALE_COLORS[row.scale] }}>
                          {row.scale}
                        </td>
                        <td className="text-right py-2 px-3 font-mono">{row.pearsonR.toFixed(4)}</td>
                        <td className="text-right py-2 pl-3 font-mono">{row.spearmanRho.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Bland-Altman */}
            <div>
              <p className="text-sm font-semibold text-gray-700 mb-2">Bland-Altman Analysis (Mean Bias &amp; Limits of Agreement)</p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200">
                      <th className="text-left py-2 pr-4 font-medium text-gray-500">Scale</th>
                      <th className="text-right py-2 px-3 font-medium text-gray-500">Mean Bias</th>
                      <th className="text-right py-2 pl-3 font-medium text-gray-500">Std of Differences</th>
                    </tr>
                  </thead>
                  <tbody>
                    {TIER2_BLAND_ALTMAN.map((row) => (
                      <tr key={row.scale} className="border-b border-gray-100">
                        <td className="py-2 pr-4 capitalize font-medium" style={{ color: SCALE_COLORS[row.scale] }}>
                          {row.scale}
                        </td>
                        <td className="text-right py-2 px-3 font-mono">
                          <span className={Math.abs(row.meanBias) < 0.001 ? 'text-green-600' : 'text-amber-600'}>
                            {row.meanBias.toFixed(4)}
                          </span>
                        </td>
                        <td className="text-right py-2 pl-3 font-mono">{row.meanStdDiff.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Regression callout */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-sm text-blue-800">
              <p className="font-semibold mb-1">Variance-Over-Time Regression</p>
              <p>
                Linear regression across 20 sequential samples: slope = <span className="font-mono">{REGRESSION.slope.toFixed(6)}</span>,
                R&sup2; = <span className="font-mono">{REGRESSION.rSquared.toFixed(4)}</span>,
                p = <span className="font-mono">{REGRESSION.pValue.toFixed(5)}</span>.
                <strong> No significant temporal drift detected</strong> (p = 0.893) &mdash; variance is stable across repeated runs, not increasing or decreasing.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* ── Tier 3: Bootstrap CIs + Friedman + Effect Sizes ───────────────── */}
      <div className="card">
        <button
          className="flex items-center justify-between w-full"
          onClick={() => setShowTier3(!showTier3)}
        >
          <div className="flex items-center space-x-2">
            <Target className="w-5 h-5 text-emerald-600" />
            <h3 className="text-lg font-semibold text-gray-900">Tier 3: Confidence Intervals &amp; Effect Sizes</h3>
            <span className="px-2 py-0.5 bg-emerald-100 text-emerald-700 rounded-full text-xs font-medium">Bootstrap &amp; Non-parametric</span>
          </div>
          {showTier3 ? <ChevronUp className="w-5 h-5 text-gray-400" /> : <ChevronDown className="w-5 h-5 text-gray-400" />}
        </button>

        {showTier3 && (
          <div className="mt-4 space-y-6">
            {/* Bootstrap CI Table */}
            <div>
              <p className="text-sm font-semibold text-gray-700 mb-2">Bootstrap 95% Confidence Intervals (1000 resamples)</p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200">
                      <th className="text-left py-2 pr-4 font-medium text-gray-500">Scale</th>
                      <th className="text-right py-2 px-2 font-medium text-gray-500">Variance (point)</th>
                      <th className="text-right py-2 px-2 font-medium text-gray-500">Var 95% CI</th>
                      <th className="text-right py-2 px-2 font-medium text-gray-500">Consistency (point)</th>
                      <th className="text-right py-2 pl-2 font-medium text-gray-500">Con 95% CI</th>
                    </tr>
                  </thead>
                  <tbody>
                    {TIER3_BOOTSTRAP.map((row) => (
                      <tr key={row.scale} className="border-b border-gray-100">
                        <td className="py-2 pr-4 capitalize font-medium" style={{ color: SCALE_COLORS[row.scale] }}>
                          {row.scale}
                        </td>
                        <td className="text-right py-2 px-2 font-mono">{row.varPoint.toFixed(4)}</td>
                        <td className="text-right py-2 px-2 font-mono text-gray-500">
                          [{row.varLower.toFixed(4)}, {row.varUpper.toFixed(4)}]
                        </td>
                        <td className="text-right py-2 px-2 font-mono">{(row.conPoint * 100).toFixed(2)}%</td>
                        <td className="text-right py-2 pl-2 font-mono text-gray-500">
                          [{(row.conLower * 100).toFixed(2)}%, {(row.conUpper * 100).toFixed(2)}%]
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Friedman Test callout */}
            <div className={`border rounded-lg p-4 text-sm ${FRIEDMAN.significant ? 'bg-amber-50 border-amber-200 text-amber-800' : 'bg-green-50 border-green-200 text-green-800'}`}>
              <div className="flex items-center space-x-2 mb-1">
                {FRIEDMAN.significant ? <AlertTriangle className="w-4 h-4 text-amber-600" /> : <CheckCircle className="w-4 h-4 text-green-600" />}
                <p className="font-semibold">Friedman Test (Non-parametric ANOVA for Repeated Measures)</p>
              </div>
              <p>
                &chi;&sup2; = <span className="font-mono">{FRIEDMAN.chiSquared.toFixed(4)}</span>,
                p = <span className="font-mono">{FRIEDMAN.pValue.toExponential(2)}</span>.
                {FRIEDMAN.significant
                  ? ' Significant differences exist between scales. Scale choice materially affects reliability.'
                  : ' No significant differences between scales.'}
              </p>
            </div>

            {/* Eta-Squared callout with horizontal bar */}
            <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4 text-sm text-indigo-800">
              <p className="font-semibold mb-3">Effect Sizes (&eta;&sup2;) &mdash; What Drives Variance?</p>
              <div className="space-y-3">
                {[
                  { label: 'By Question', value: ETA_SQUARED.byQuestion, color: '#8b5cf6' },
                  { label: 'By Text', value: ETA_SQUARED.byText, color: '#3b82f6' },
                  { label: 'By Scale', value: ETA_SQUARED.byScale, color: '#10b981' },
                ].map((factor) => {
                  const maxEta = Math.max(ETA_SQUARED.byScale, ETA_SQUARED.byText, ETA_SQUARED.byQuestion)
                  const pct = (factor.value / maxEta) * 100
                  return (
                    <div key={factor.label} className="flex items-center gap-3">
                      <span className="w-28 text-sm font-medium text-gray-700 flex-shrink-0">{factor.label}</span>
                      <div className="flex-1 bg-gray-200 rounded-full h-5 overflow-hidden">
                        <div
                          className="h-full rounded-full flex items-center justify-end pr-2"
                          style={{ width: `${pct}%`, backgroundColor: factor.color }}
                        >
                          <span className="text-white text-xs font-mono font-bold">
                            {factor.value.toFixed(4)}
                          </span>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
              <p className="mt-3 text-xs text-indigo-600">
                <strong>Question type</strong> explains 8&times; more variance than scale choice.
                The feature being measured matters far more than how finely you measure it.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
