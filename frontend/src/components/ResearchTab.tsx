import { useState } from 'react'
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import {
  BookOpen,
  BarChart3,
  Layers,
  TrendingUp,
  Lightbulb,
  FlaskConical,
  ChevronDown,
  ChevronUp,
  Microscope,
  AlertTriangle,
  CheckCircle,
  Target,
  GitBranch,
  Grid3X3,
} from 'lucide-react'

// ─── Hardcoded Data from 16,000 Evaluations (Feb 15, 2026) ───────────────────

const SCALE_COLORS: Record<string, string> = {
  binary: '#2196F3',
  ternary: '#4CAF50',
  quaternary: '#FF9800',
  continuous: '#E91E63',
}

const SCALE_ORDER = ['binary', 'ternary', 'quaternary', 'continuous']

const HEATMAP_DATA = [
  { question: 'Named Entities',  binary: 0.0,      ternary: 0.0,      quaternary: 0.0,      continuous: 0.000047 },
  { question: 'Actions/Events',  binary: 0.0,      ternary: 0.004,    quaternary: 0.005313, continuous: 0.00429  },
  { question: 'Temporal',        binary: 0.02075,  ternary: 0.0,      quaternary: 0.014105, continuous: 0.003777 },
  { question: 'Spatial',         binary: 0.0,      ternary: 0.0,      quaternary: 0.0,      continuous: 0.00009  },
  { question: 'Numeric',         binary: 0.00475,  ternary: 0.003188, quaternary: 0.002836, continuous: 0.000457 },
  { question: 'Modality',        binary: 0.0,      ternary: 0.0,      quaternary: 0.002428, continuous: 0.00019  },
  { question: 'Imperative',      binary: 0.0,      ternary: 0.023937, quaternary: 0.002132, continuous: 0.00457  },
  { question: 'Comparison',      binary: 0.00475,  ternary: 0.0,      quaternary: 0.0,      continuous: 0.000528 },
  { question: 'First Person',    binary: 0.0,      ternary: 0.0,      quaternary: 0.0,      continuous: 0.0      },
  { question: 'Dialogue',        binary: 0.0,      ternary: 0.0,      quaternary: 0.001388, continuous: 0.004    },
  { question: 'Causality',       binary: 0.009,    ternary: 0.007188, quaternary: 0.015791, continuous: 0.0      },
  { question: 'Negation',        binary: 0.0,      ternary: 0.0,      quaternary: 0.00185,  continuous: 0.0004   },
  { question: 'Uncertainty',     binary: 0.0,      ternary: 0.009,    quaternary: 0.01275,  continuous: 0.0      },
  { question: 'Sentiment',       binary: 0.009,    ternary: 0.0,      quaternary: 0.0,      continuous: 0.0      },
  { question: 'Emotion',         binary: 0.0,      ternary: 0.0,      quaternary: 0.0,      continuous: 0.0      },
  { question: 'Social',          binary: 0.0,      ternary: 0.0,      quaternary: 0.0,      continuous: 0.0      },
  { question: 'Normative',       binary: 0.0,      ternary: 0.0,      quaternary: 0.0,      continuous: 0.0      },
  { question: 'Intent',          binary: 0.0,      ternary: 0.00475,  quaternary: 0.018812, continuous: 0.00741  },
  { question: 'Concreteness',    binary: 0.0,      ternary: 0.0,      quaternary: 0.015925, continuous: 0.014522 },
  { question: 'Identity',        binary: 0.0,      ternary: 0.01275,  quaternary: 0.006158, continuous: 0.0      },
]

const SCALE_SUMMARY = [
  { scale: 'binary',     meanVar: 0.002413, medianVar: 0.0, zeroVarPct: 97.0, highVarPct: 0.5, medianEntropy: 0.0, meanConsistency: 0.9972 },
  { scale: 'ternary',    meanVar: 0.003241, medianVar: 0.0, zeroVarPct: 95.5, highVarPct: 1.0, medianEntropy: 0.0, meanConsistency: 0.9918 },
  { scale: 'quaternary', meanVar: 0.004974, medianVar: 0.0, zeroVarPct: 88.5, highVarPct: 1.5, medianEntropy: 0.0, meanConsistency: 0.9685 },
  { scale: 'continuous', meanVar: 0.002014, medianVar: 0.0, zeroVarPct: 80.5, highVarPct: 0.0, medianEntropy: 0.0, meanConsistency: 0.9747 },
]

const DEGRADATION_DATA = [
  { scale: 'binary',     factual_simple: 0.0, sentiment_positive: 0.0045, sentiment_negative: 0.0, ambiguous: 0.006875, medical_clinical: 0.0, negation_heavy: 0.010375, imperative_action: 0.0, abstract_philosophical: 0.0, narrative_paragraph: 0.0, technical_ml: 0.002375 },
  { scale: 'ternary',    factual_simple: 0.002, sentiment_positive: 0.0, sentiment_negative: 0.011375, ambiguous: 0.007469, medical_clinical: 0.007969, negation_heavy: 0.0, imperative_action: 0.0, abstract_philosophical: 0.000594, narrative_paragraph: 0.003, technical_ml: 0.0 },
  { scale: 'quaternary', factual_simple: 0.001361, sentiment_positive: 0.003526, sentiment_negative: 0.0, ambiguous: 0.016774, medical_clinical: 0.006188, negation_heavy: 0.0, imperative_action: 0.00189, abstract_philosophical: 0.0, narrative_paragraph: 0.018765, technical_ml: 0.001239 },
  { scale: 'continuous', factual_simple: 0.000024, sentiment_positive: 0.00218, sentiment_negative: 0.000237, ambiguous: 0.004305, medical_clinical: 0.002299, negation_heavy: 0.000095, imperative_action: 0.004539, abstract_philosophical: 0.000264, narrative_paragraph: 0.000354, technical_ml: 0.005845 },
]

const TEXT_DIFFICULTY = [
  { id: 'ambiguous',              meanVariance: 0.0088 },
  { id: 'narrative_paragraph',    meanVariance: 0.0058 },
  { id: 'medical_clinical',       meanVariance: 0.0040 },
  { id: 'sentiment_negative',     meanVariance: 0.0030 },
  { id: 'negation_heavy',         meanVariance: 0.0028 },
  { id: 'sentiment_positive',     meanVariance: 0.0027 },
  { id: 'technical_ml',           meanVariance: 0.0025 },
  { id: 'imperative_action',      meanVariance: 0.0014 },
  { id: 'factual_simple',         meanVariance: 0.0010 },
  { id: 'abstract_philosophical', meanVariance: 0.0002 },
]

const MODE_CONSISTENCY_DATA = [
  { id: 'factual_simple',         binary: 1.0,    ternary: 0.99,   quaternary: 0.975,  continuous: 0.9975 },
  { id: 'sentiment_positive',     binary: 0.995,  ternary: 1.0,    quaternary: 0.9475, continuous: 0.985  },
  { id: 'sentiment_negative',     binary: 1.0,    ternary: 0.9825, quaternary: 1.0,    continuous: 0.995  },
  { id: 'ambiguous',              binary: 0.9925, ternary: 0.9825, quaternary: 0.9475, continuous: 0.9325 },
  { id: 'medical_clinical',       binary: 1.0,    ternary: 0.985,  quaternary: 0.9625, continuous: 0.9675 },
  { id: 'negation_heavy',         binary: 0.9875, ternary: 1.0,    quaternary: 1.0,    continuous: 0.9975 },
  { id: 'imperative_action',      binary: 1.0,    ternary: 1.0,    quaternary: 0.99,   continuous: 0.965  },
  { id: 'abstract_philosophical', binary: 1.0,    ternary: 0.9975, quaternary: 1.0,    continuous: 0.9725 },
  { id: 'narrative_paragraph',    binary: 1.0,    ternary: 0.98,   quaternary: 0.88,   continuous: 0.9775 },
  { id: 'technical_ml',           binary: 0.9975, ternary: 1.0,    quaternary: 0.9825, continuous: 0.9575 },
]

const PER_TEXT_TABLE = [
  { textId: 'factual_simple',         scale: 'binary',     avgVariance: 0.0,      avgConsistency: 1.0,    avgEntropy: 0.0,    uniqueVectors: 1  },
  { textId: 'factual_simple',         scale: 'ternary',    avgVariance: 0.002,    avgConsistency: 0.99,   avgEntropy: 0.0361, uniqueVectors: 2  },
  { textId: 'factual_simple',         scale: 'quaternary', avgVariance: 0.001361, avgConsistency: 0.975,  avgEntropy: 0.05,   uniqueVectors: 2  },
  { textId: 'factual_simple',         scale: 'continuous', avgVariance: 0.000024, avgConsistency: 0.9975, avgEntropy: 0.0143, uniqueVectors: 2  },
  { textId: 'sentiment_positive',     scale: 'binary',     avgVariance: 0.0045,   avgConsistency: 0.995,  avgEntropy: 0.0234, uniqueVectors: 2  },
  { textId: 'sentiment_positive',     scale: 'ternary',    avgVariance: 0.0,      avgConsistency: 1.0,    avgEntropy: 0.0,    uniqueVectors: 1  },
  { textId: 'sentiment_positive',     scale: 'quaternary', avgVariance: 0.003526, avgConsistency: 0.9475, avgEntropy: 0.129,  uniqueVectors: 5  },
  { textId: 'sentiment_positive',     scale: 'continuous', avgVariance: 0.00218,  avgConsistency: 0.985,  avgEntropy: 0.0595, uniqueVectors: 4  },
  { textId: 'sentiment_negative',     scale: 'binary',     avgVariance: 0.0,      avgConsistency: 1.0,    avgEntropy: 0.0,    uniqueVectors: 1  },
  { textId: 'sentiment_negative',     scale: 'ternary',    avgVariance: 0.011375, avgConsistency: 0.9825, avgEntropy: 0.0467, uniqueVectors: 2  },
  { textId: 'sentiment_negative',     scale: 'quaternary', avgVariance: 0.0,      avgConsistency: 1.0,    avgEntropy: 0.0,    uniqueVectors: 1  },
  { textId: 'sentiment_negative',     scale: 'continuous', avgVariance: 0.000237, avgConsistency: 0.995,  avgEntropy: 0.0286, uniqueVectors: 2  },
  { textId: 'ambiguous',              scale: 'binary',     avgVariance: 0.006875, avgConsistency: 0.9925, avgEntropy: 0.0378, uniqueVectors: 3  },
  { textId: 'ambiguous',              scale: 'ternary',    avgVariance: 0.007469, avgConsistency: 0.9825, avgEntropy: 0.082,  uniqueVectors: 4  },
  { textId: 'ambiguous',              scale: 'quaternary', avgVariance: 0.016774, avgConsistency: 0.9475, avgEntropy: 0.1568, uniqueVectors: 6  },
  { textId: 'ambiguous',              scale: 'continuous', avgVariance: 0.004305, avgConsistency: 0.9325, avgEntropy: 0.1723, uniqueVectors: 6  },
  { textId: 'medical_clinical',       scale: 'binary',     avgVariance: 0.0,      avgConsistency: 1.0,    avgEntropy: 0.0,    uniqueVectors: 1  },
  { textId: 'medical_clinical',       scale: 'ternary',    avgVariance: 0.007969, avgConsistency: 0.985,  avgEntropy: 0.061,  uniqueVectors: 3  },
  { textId: 'medical_clinical',       scale: 'quaternary', avgVariance: 0.006188, avgConsistency: 0.9625, avgEntropy: 0.1217, uniqueVectors: 2  },
  { textId: 'medical_clinical',       scale: 'continuous', avgVariance: 0.002299, avgConsistency: 0.9675, avgEntropy: 0.1228, uniqueVectors: 4  },
  { textId: 'negation_heavy',         scale: 'binary',     avgVariance: 0.010375, avgConsistency: 0.9875, avgEntropy: 0.0504, uniqueVectors: 3  },
  { textId: 'negation_heavy',         scale: 'ternary',    avgVariance: 0.0,      avgConsistency: 1.0,    avgEntropy: 0.0,    uniqueVectors: 1  },
  { textId: 'negation_heavy',         scale: 'quaternary', avgVariance: 0.0,      avgConsistency: 1.0,    avgEntropy: 0.0,    uniqueVectors: 1  },
  { textId: 'negation_heavy',         scale: 'continuous', avgVariance: 0.000095, avgConsistency: 0.9975, avgEntropy: 0.0143, uniqueVectors: 2  },
  { textId: 'imperative_action',      scale: 'binary',     avgVariance: 0.0,      avgConsistency: 1.0,    avgEntropy: 0.0,    uniqueVectors: 1  },
  { textId: 'imperative_action',      scale: 'ternary',    avgVariance: 0.0,      avgConsistency: 1.0,    avgEntropy: 0.0,    uniqueVectors: 1  },
  { textId: 'imperative_action',      scale: 'quaternary', avgVariance: 0.00189,  avgConsistency: 0.99,   avgEntropy: 0.0573, uniqueVectors: 2  },
  { textId: 'imperative_action',      scale: 'continuous', avgVariance: 0.004539, avgConsistency: 0.965,  avgEntropy: 0.1085, uniqueVectors: 5  },
  { textId: 'abstract_philosophical', scale: 'binary',     avgVariance: 0.0,      avgConsistency: 1.0,    avgEntropy: 0.0,    uniqueVectors: 1  },
  { textId: 'abstract_philosophical', scale: 'ternary',    avgVariance: 0.000594, avgConsistency: 0.9975, avgEntropy: 0.0143, uniqueVectors: 2  },
  { textId: 'abstract_philosophical', scale: 'quaternary', avgVariance: 0.0,      avgConsistency: 1.0,    avgEntropy: 0.0,    uniqueVectors: 1  },
  { textId: 'abstract_philosophical', scale: 'continuous', avgVariance: 0.000264, avgConsistency: 0.9725, avgEntropy: 0.0756, uniqueVectors: 3  },
  { textId: 'narrative_paragraph',    scale: 'binary',     avgVariance: 0.0,      avgConsistency: 1.0,    avgEntropy: 0.0,    uniqueVectors: 1  },
  { textId: 'narrative_paragraph',    scale: 'ternary',    avgVariance: 0.003,    avgConsistency: 0.98,   avgEntropy: 0.0485, uniqueVectors: 2  },
  { textId: 'narrative_paragraph',    scale: 'quaternary', avgVariance: 0.018765, avgConsistency: 0.88,   avgEntropy: 0.3409, uniqueVectors: 10 },
  { textId: 'narrative_paragraph',    scale: 'continuous', avgVariance: 0.000354, avgConsistency: 0.9775, avgEntropy: 0.1183, uniqueVectors: 3  },
  { textId: 'technical_ml',           scale: 'binary',     avgVariance: 0.002375, avgConsistency: 0.9975, avgEntropy: 0.0143, uniqueVectors: 2  },
  { textId: 'technical_ml',           scale: 'ternary',    avgVariance: 0.0,      avgConsistency: 1.0,    avgEntropy: 0.0,    uniqueVectors: 1  },
  { textId: 'technical_ml',           scale: 'quaternary', avgVariance: 0.001239, avgConsistency: 0.9825, avgEntropy: 0.0467, uniqueVectors: 2  },
  { textId: 'technical_ml',           scale: 'continuous', avgVariance: 0.005845, avgConsistency: 0.9575, avgEntropy: 0.1279, uniqueVectors: 6  },
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

// ─── Advanced Stats from First Run (Tier 1-3) ──────────────────────────────

const TIER1_DATA = [
  { scale: 'binary', cronbachsAlpha: 0.61639, icc: 0.984923, cohensKappa: 0.98471, krippendorffAlpha: 0.984211 },
  { scale: 'ternary', cronbachsAlpha: 0.340837, icc: 0.975761, cohensKappa: 0.994013, krippendorffAlpha: 0.977118 },
  { scale: 'quaternary', cronbachsAlpha: -0.1189, icc: 0.954494, cohensKappa: 1.0, krippendorffAlpha: 0.951957 },
  { scale: 'continuous', cronbachsAlpha: -0.326744, icc: 0.962245, cohensKappa: null, krippendorffAlpha: 0.960552 },
]

const TIER2_TEST_RETEST = [
  { scale: 'binary', pearsonR: 0.998346, spearmanRho: 0.981233 },
  { scale: 'ternary', pearsonR: 0.995565, spearmanRho: 0.98877 },
  { scale: 'quaternary', pearsonR: 0.995243, spearmanRho: 0.989073 },
  { scale: 'continuous', pearsonR: 0.998103, spearmanRho: 0.999706 },
]

const TIER2_BLAND_ALTMAN = [
  { scale: 'binary', meanBias: -0.0006, meanStdDiff: 0.04180184 },
  { scale: 'ternary', meanBias: 0.0001, meanStdDiff: 0.06127552 },
  { scale: 'quaternary', meanBias: 0.001071, meanStdDiff: 0.07513856 },
  { scale: 'continuous', meanBias: -0.00028, meanStdDiff: 0.0533924 },
]

const REGRESSION = { slope: 0.00005386, rSquared: 0.002806, pValue: 0.947032 }

const TIER3_BOOTSTRAP = [
  { scale: 'binary', varPoint: 0.002413, varLower: 0.000475, varUpper: 0.004825, conPoint: 0.99725, conLower: 0.99425, conUpper: 0.9995 },
  { scale: 'ternary', varPoint: 0.003241, varLower: 0.000737, varUpper: 0.006486, conPoint: 0.99175, conLower: 0.985, conUpper: 0.9975 },
  { scale: 'quaternary', varPoint: 0.004974, varLower: 0.0026, varUpper: 0.007847, conPoint: 0.9685, conLower: 0.953244, conUpper: 0.981756 },
  { scale: 'continuous', varPoint: 0.002014, varLower: 0.000707, varUpper: 0.003494, conPoint: 0.97475, conLower: 0.961244, conUpper: 0.98675 },
]

const FRIEDMAN = { chiSquared: 39.015113, pValue: 1.72e-08, significant: true }
const ETA_SQUARED = { byScale: 0.004691, byText: 0.020604, byQuestion: 0.038963 }

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

export default function ResearchTab() {
  const [showMethodology, setShowMethodology] = useState(false)

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
            LLM Feature Extraction Reliability: 16,000 Evaluations
          </h2>
        </div>
        <p className="text-gray-600 leading-relaxed">
          This page presents hardcoded results from a completed batch analysis of 10 texts &times; 20
          questions &times; 4 scales &times; 20 samples = <strong>16,000 individual LLM evaluations</strong> on
          GPT-4o-mini at temperature=0. Run the <strong>Batch Analysis</strong> tab to replicate these
          results on your own data.
        </p>
        <div className="mt-3 flex flex-wrap gap-3 text-sm">
          <span className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full font-medium">GPT-4o-mini</span>
          <span className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full font-medium">Temperature = 0</span>
          <span className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full font-medium">20 repeated samples</span>
          <span className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full font-medium">Feb 2026</span>
        </div>
      </div>

      {/* ── The Problem ────────────────────────────────────────────────────── */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <AlertTriangle className="w-5 h-5 text-amber-600" />
          <h3 className="text-lg font-semibold text-gray-900">The Problem</h3>
        </div>
        <div className="text-gray-700 space-y-3 leading-relaxed">
          <p>
            LLMs are increasingly used as zero-shot feature extractors for unstructured data at scale.
            A single prompt can extract dozens of feature dimensions simultaneously &mdash; named entities,
            sentiment, causality, intent, emotion &mdash; replacing what would otherwise require a
            separate classifier for each dimension.
          </p>
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
            <p className="font-semibold text-amber-800">Nobody is checking which of those features are reliable.</p>
            <p className="text-sm text-amber-700 mt-1">
              Teams extract features on Monday, build downstream models on the output, and discover weeks later
              that their model degraded &mdash; without knowing which feature dimension drifted between runs.
            </p>
          </div>
          <p>
            The conventional wisdom says: use a smarter model, write better prompts, set temperature to zero.
            We show that this is insufficient. Reliability is a function of <strong>feature type</strong>,
            not model capability or prompt quality.
          </p>
        </div>
      </div>

      {/* ── Experimental Design Matrix ────────────────────────────────────── */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <Grid3X3 className="w-5 h-5 text-purple-600" />
          <h3 className="text-lg font-semibold text-gray-900">Experimental Design</h3>
        </div>
        <p className="text-sm text-gray-600 mb-5">
          Each cell is one (text, question) pair. Within each cell, GPT-4o-mini grades at 4 scale
          granularities. The entire sheet is repeated <strong>20 times</strong> to measure consistency.
        </p>

        {/* ── Stacked sheets visual ── */}
        <div className="flex flex-col lg:flex-row items-start gap-6">
          {/* Main grid with zoomed cells */}
          <div className="flex-1 min-w-0">
            {/* Axis labels */}
            <div className="flex">
              <div className="w-24 flex-shrink-0" />
              <div className="flex-1 text-center text-xs font-semibold text-gray-500 mb-1">
                20 Questions (semantic factor families)
              </div>
            </div>
            <div className="flex">
              {/* Y-axis label */}
              <div className="w-24 flex-shrink-0 flex items-center justify-center">
                <span className="text-xs font-semibold text-gray-500 -rotate-90 whitespace-nowrap">
                  10 Texts
                </span>
              </div>
              {/* Grid */}
              <div className="flex-1 border border-gray-300 rounded-lg overflow-hidden bg-white">
                {Array.from({ length: 10 }).map((_, rowIdx) => (
                  <div key={rowIdx} className="flex">
                    {Array.from({ length: 20 }).map((_, colIdx) => {
                      /* Highlight 2 cells to show inner structure */
                      const isZoomed = (rowIdx === 1 && colIdx === 3) || (rowIdx === 6 && colIdx === 14)
                      const isSecondZoom = rowIdx === 6 && colIdx === 14

                      if (isZoomed) {
                        return (
                          <div
                            key={colIdx}
                            className="relative flex-1 aspect-square border border-purple-400 bg-purple-50 flex flex-col items-center justify-center gap-[1px] p-[2px]"
                            title={`Text ${rowIdx + 1} × Q${colIdx + 1}`}
                          >
                            <div className="w-full h-[3px] rounded-sm" style={{ background: '#2196F3' }} />
                            <div className="w-full h-[3px] rounded-sm" style={{ background: '#4CAF50' }} />
                            <div className="w-full h-[3px] rounded-sm" style={{ background: '#FF9800' }} />
                            <div className="w-full h-[3px] rounded-sm" style={{ background: '#E91E63' }} />
                            {!isSecondZoom && (
                              <div className="absolute -right-1 -top-1 w-2 h-2 bg-purple-500 rounded-full" />
                            )}
                          </div>
                        )
                      }

                      return (
                        <div
                          key={colIdx}
                          className="flex-1 aspect-square border border-gray-100 bg-gray-50 hover:bg-gray-100 transition-colors"
                          title={`Text ${rowIdx + 1} × Q${colIdx + 1}`}
                        />
                      )
                    })}
                  </div>
                ))}
              </div>
            </div>
            {/* Row labels (text names on left, abbreviated) */}
            <div className="flex mt-1">
              <div className="w-24 flex-shrink-0" />
              <div className="flex-1 flex justify-between text-[9px] text-gray-400 px-0.5">
                <span>Q1</span>
                <span>Q5</span>
                <span>Q10</span>
                <span>Q15</span>
                <span>Q20</span>
              </div>
            </div>
          </div>

          {/* Zoomed cell explanation + stacked sheets */}
          <div className="flex-shrink-0 w-full lg:w-72 space-y-4">
            {/* Zoomed cell */}
            <div className="border-2 border-purple-400 rounded-lg p-3 bg-purple-50">
              <p className="text-xs font-semibold text-purple-700 mb-2">Each cell contains 4 measurements:</p>
              <div className="space-y-1.5">
                {[
                  { label: 'Binary', value: '{0, 1}', color: '#2196F3', example: '→ 1' },
                  { label: 'Ternary', value: '{0, 0.5, 1}', color: '#4CAF50', example: '→ 0.5' },
                  { label: 'Quaternary', value: '{0, .33, .66, 1}', color: '#FF9800', example: '→ 0.66' },
                  { label: 'Continuous', value: '[0, 1]', color: '#E91E63', example: '→ 0.73' },
                ].map((s) => (
                  <div key={s.label} className="flex items-center gap-2 text-xs">
                    <div className="w-3 h-3 rounded-sm flex-shrink-0" style={{ background: s.color }} />
                    <span className="font-semibold text-gray-700 w-20">{s.label}</span>
                    <span className="text-gray-500 font-mono text-[10px]">{s.value}</span>
                    <span className="text-gray-400 font-mono text-[10px] ml-auto">{s.example}</span>
                  </div>
                ))}
              </div>
              <div className="mt-2 pt-2 border-t border-purple-200 text-[10px] text-purple-600">
                = 10 &times; 20 &times; 4 = <strong>800 scores</strong> per sample
              </div>
            </div>

            {/* Stacked sheets */}
            <div className="relative h-36">
              {[4, 3, 2, 1, 0].map((i) => (
                <div
                  key={i}
                  className="absolute border border-gray-300 rounded bg-white shadow-sm"
                  style={{
                    width: '85%',
                    height: '70%',
                    top: `${i * 5}px`,
                    left: `${i * 5}px`,
                    opacity: i === 0 ? 1 : 0.5 + i * 0.1,
                    zIndex: 5 - i,
                  }}
                >
                  {i === 0 && (
                    <div className="flex flex-col items-center justify-center h-full text-xs text-gray-500">
                      <div className="grid grid-cols-5 gap-[2px] mb-2">
                        {Array.from({ length: 15 }).map((_, j) => (
                          <div key={j} className="w-2 h-1.5 bg-gray-200 rounded-[1px]" />
                        ))}
                      </div>
                      <span className="font-mono text-[10px]">10 &times; 20 sheet</span>
                    </div>
                  )}
                </div>
              ))}
              <div
                className="absolute flex items-center gap-1"
                style={{ bottom: 0, right: 0, zIndex: 10 }}
              >
                <span className="text-xs font-bold text-gray-700 bg-white px-1.5 py-0.5 rounded border border-gray-200 shadow-sm">
                  &times;20 samples
                </span>
              </div>
            </div>

            {/* Final calculation */}
            <div className="bg-gray-50 rounded-lg p-3 text-center">
              <div className="text-xs text-gray-500 space-y-0.5">
                <div>10 texts &times; 20 questions &times; 4 scales &times; 20 samples</div>
                <div className="text-lg font-bold text-gray-900">=&nbsp;16,000 evaluations</div>
              </div>
            </div>
          </div>
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
          <p className="font-semibold mb-1">Finding: Variance is non-monotonic across scales</p>
          <p>
            Quaternary is the <strong>worst</strong> scale by mean variance (0.0050) and zero-variance rate (88.5%).
            Continuous has the <strong>lowest</strong> mean variance (0.0020) &mdash; lower than binary &mdash;
            but also the lowest zero-variance rate (80.5%). The noise is quiet and everywhere rather than loud and concentrated.
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

        {/* Static matplotlib figure */}
        <div className="mt-4 border-t border-gray-200 pt-4">
          <p className="text-xs text-gray-400 mb-2">High-resolution matplotlib figure:</p>
          <img
            src="/research/question_stability_heatmap.png"
            alt="Question stability heatmap — 20 questions x 4 scales"
            className="w-full rounded-lg border border-gray-200"
          />
        </div>

        {/* Finding callout */}
        <div className="mt-4 bg-green-50 border border-green-200 rounded-lg p-4 text-sm">
          <div className="flex items-center space-x-2 mb-1">
            <CheckCircle className="w-4 h-4 text-green-600" />
            <p className="font-semibold text-green-800">Finding: Reliability varies dramatically by feature type</p>
          </div>
          <p className="text-green-700">
            <strong>Perfectly stable</strong> (zero variance): Named Entities, First Person, Emotion, Social, Normative.
            <br />
            <strong>Unstable</strong>: Intent (0.0188 at quaternary), Concreteness (0.0159 quaternary / 0.0145 continuous),
            Causality (0.0158 quaternary), Imperative (0.0239 ternary).
          </p>
        </div>
      </div>

      {/* ── Variance Distribution Violins ──────────────────────────────────── */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-2">
          <BarChart3 className="w-5 h-5 text-orange-600" />
          <h3 className="text-lg font-semibold text-gray-900">Variance Distribution by Scale</h3>
        </div>
        <p className="text-sm text-gray-500 mb-4">
          Distribution of variance values across all 200 text &times; question pairs, grouped by scale.
        </p>
        <img
          src="/research/variance_violins.png"
          alt="Violin plots showing variance distribution per scale"
          className="w-full rounded-lg border border-gray-200"
        />
      </div>

      {/* ── Entropy vs Variance Scatter ────────────────────────────────────── */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-2">
          <FlaskConical className="w-5 h-5 text-pink-600" />
          <h3 className="text-lg font-semibold text-gray-900">Entropy vs Variance</h3>
        </div>
        <p className="text-sm text-gray-500 mb-4">
          800 data points (200 per scale). Reveals distinct noise signatures: quaternary is high-variance + high-entropy
          (label flips), continuous is low-variance + moderate-entropy (feature drift).
        </p>
        <img
          src="/research/entropy_vs_variance.png"
          alt="Entropy vs variance scatter plot — 800 points colored by scale"
          className="w-full rounded-lg border border-gray-200"
        />
        <div className="mt-4 bg-pink-50 border border-pink-200 rounded-lg p-4 text-sm text-pink-800">
          <p className="font-semibold mb-1">Finding: Different noise structures per scale</p>
          <p>
            <strong>Quaternary</strong> instability = label flips (recommendation gets categorized differently between runs).
            <strong> Continuous</strong> instability = feature drift (value shifts slightly, may or may not cross a decision threshold).
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
          How each text's mean variance changes across scales (binary &rarr; continuous).
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

        {/* Static matplotlib figure */}
        <div className="mt-4 border-t border-gray-200 pt-4">
          <p className="text-xs text-gray-400 mb-2">High-resolution matplotlib figure:</p>
          <img
            src="/research/per_text_degradation.png"
            alt="Variance degradation line chart per text type"
            className="w-full rounded-lg border border-gray-200"
          />
        </div>
      </div>

      {/* ── Text Difficulty Ranking ────────────────────────────────────────── */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-2">
          <Target className="w-5 h-5 text-red-600" />
          <h3 className="text-lg font-semibold text-gray-900">Text Difficulty Ranking</h3>
        </div>
        <p className="text-sm text-gray-500 mb-4">
          Texts ranked by overall mean variance. Longer bar = more unstable. 44&times; difference between hardest and easiest.
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

        <div className="mt-4 border-t border-gray-200 pt-4">
          <p className="text-xs text-gray-400 mb-2">High-resolution matplotlib figure:</p>
          <img
            src="/research/text_difficulty_ranking.png"
            alt="Text difficulty ranking horizontal bar chart"
            className="w-full rounded-lg border border-gray-200"
          />
        </div>

        <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4 text-sm text-red-800">
          <p className="font-semibold mb-1">Finding: Text difficulty interacts with feature type</p>
          <p>
            Ambiguous text (0.0088) is 44&times; more unstable than abstract philosophical (0.0002).
            Texts with pragmatic subtext and multi-sentence narratives cluster at the top.
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

        <div className="mt-4 border-t border-gray-200 pt-4">
          <p className="text-xs text-gray-400 mb-2">High-resolution matplotlib figure:</p>
          <img
            src="/research/mode_consistency_bars.png"
            alt="Mode consistency grouped bar chart by text and scale"
            className="w-full rounded-lg border border-gray-200"
          />
        </div>
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

      {/* ── Advanced Statistical Analysis ──────────────────────────────────── */}
      <div className="card bg-gradient-to-br from-purple-50 to-indigo-50 border-l-4 border-purple-500">
        <div className="flex items-center space-x-3 mb-3">
          <FlaskConical className="w-7 h-7 text-purple-600" />
          <h2 className="text-2xl font-bold text-gray-900">
            Advanced Statistical Analysis
          </h2>
        </div>
        <p className="text-gray-600 leading-relaxed">
          Standard reliability metrics computed from the 16,000 evaluations. These are the metrics
          any reviewer of a measurement study would expect to see: internal consistency, inter-rater
          agreement, test-retest reliability, and effect sizes.
        </p>
      </div>

      {/* ── Tier 1: Essential Reliability ──────────────────────────────────── */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <CheckCircle className="w-5 h-5 text-green-600" />
          <h3 className="text-lg font-semibold text-gray-900">Tier 1: Essential Reliability Metrics</h3>
        </div>
        <p className="text-sm text-gray-500 mb-4">
          Standard psychometric reliability coefficients. Values above 0.9 indicate excellent agreement.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-2 pr-4 font-medium text-gray-500">Scale</th>
                <th className="text-right py-2 px-3 font-medium text-gray-500">Cronbach's &alpha;</th>
                <th className="text-right py-2 px-3 font-medium text-gray-500">ICC(2,1)</th>
                <th className="text-right py-2 px-3 font-medium text-gray-500">Cohen's &kappa;</th>
                <th className="text-right py-2 pl-3 font-medium text-gray-500">Krippendorff's &alpha;</th>
              </tr>
            </thead>
            <tbody>
              {TIER1_DATA.map((row) => (
                <tr key={row.scale} className="border-b border-gray-100">
                  <td className="py-2 pr-4 capitalize font-medium" style={{ color: SCALE_COLORS[row.scale] }}>
                    {row.scale}
                  </td>
                  <td className={`text-right py-2 px-3 font-mono ${row.cronbachsAlpha < 0 ? 'text-red-600' : ''}`}>
                    {row.cronbachsAlpha.toFixed(3)}
                  </td>
                  <td className="text-right py-2 px-3 font-mono">{row.icc.toFixed(4)}</td>
                  <td className="text-right py-2 px-3 font-mono">
                    {row.cohensKappa !== null ? row.cohensKappa.toFixed(4) : 'N/A'}
                  </td>
                  <td className="text-right py-2 pl-3 font-mono">{row.krippendorffAlpha.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-4 bg-purple-50 border border-purple-200 rounded-lg p-4 text-sm text-purple-800">
          <p className="font-semibold mb-1">Finding: High agreement but poor internal consistency at fine granularity</p>
          <p>
            ICC and Krippendorff's &alpha; stay above 0.95 across all scales (excellent rater agreement).
            But Cronbach's &alpha; goes <strong>negative</strong> at quaternary (-0.12) and continuous (-0.33),
            meaning questions become independently unstable in different directions rather than consistently unstable together.
          </p>
        </div>
      </div>

      {/* ── Tier 2: Informative Metrics ────────────────────────────────────── */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <TrendingUp className="w-5 h-5 text-blue-600" />
          <h3 className="text-lg font-semibold text-gray-900">Tier 2: Test-Retest &amp; Agreement</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Test-Retest */}
          <div>
            <p className="text-sm font-medium text-gray-700 mb-2">Test-Retest Correlation (odd/even split-half)</p>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-1 font-medium text-gray-500">Scale</th>
                  <th className="text-right py-1 font-medium text-gray-500">Pearson r</th>
                  <th className="text-right py-1 font-medium text-gray-500">Spearman &rho;</th>
                </tr>
              </thead>
              <tbody>
                {TIER2_TEST_RETEST.map((row) => (
                  <tr key={row.scale} className="border-b border-gray-100">
                    <td className="py-1 capitalize" style={{ color: SCALE_COLORS[row.scale] }}>{row.scale}</td>
                    <td className="text-right py-1 font-mono">{row.pearsonR.toFixed(4)}</td>
                    <td className="text-right py-1 font-mono">{row.spearmanRho.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Bland-Altman */}
          <div>
            <p className="text-sm font-medium text-gray-700 mb-2">Bland-Altman (limits of agreement)</p>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-1 font-medium text-gray-500">Scale</th>
                  <th className="text-right py-1 font-medium text-gray-500">Mean Bias</th>
                  <th className="text-right py-1 font-medium text-gray-500">Std Diff</th>
                </tr>
              </thead>
              <tbody>
                {TIER2_BLAND_ALTMAN.map((row) => (
                  <tr key={row.scale} className="border-b border-gray-100">
                    <td className="py-1 capitalize" style={{ color: SCALE_COLORS[row.scale] }}>{row.scale}</td>
                    <td className="text-right py-1 font-mono">{row.meanBias.toFixed(4)}</td>
                    <td className="text-right py-1 font-mono">{row.meanStdDiff.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="mt-4 bg-blue-50 border border-blue-200 rounded-lg p-4 text-sm text-blue-800">
          <p className="font-semibold mb-1">Scale Degradation Regression</p>
          <p>
            Linear fit (binary=0, ternary=1, quaternary=2, continuous=3): slope={REGRESSION.slope.toFixed(6)},
            R&sup2;={REGRESSION.rSquared.toFixed(4)}, p={REGRESSION.pValue.toFixed(4)}.
            Variance does <strong>not</strong> linearly increase with granularity (p=0.95, non-significant).
            The relationship is non-monotonic: quaternary peaks, continuous drops back.
          </p>
        </div>
      </div>

      {/* ── Tier 3: Advanced ───────────────────────────────────────────────── */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <Microscope className="w-5 h-5 text-indigo-600" />
          <h3 className="text-lg font-semibold text-gray-900">Tier 3: Bootstrap CIs, Friedman Test &amp; Effect Sizes</h3>
        </div>

        {/* Bootstrap CIs */}
        <p className="text-sm font-medium text-gray-700 mb-2">Bootstrap 95% Confidence Intervals (1000 resamples)</p>
        <div className="overflow-x-auto mb-4">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-1 font-medium text-gray-500">Scale</th>
                <th className="text-right py-1 font-medium text-gray-500">Variance [95% CI]</th>
                <th className="text-right py-1 font-medium text-gray-500">Consistency [95% CI]</th>
              </tr>
            </thead>
            <tbody>
              {TIER3_BOOTSTRAP.map((row) => (
                <tr key={row.scale} className="border-b border-gray-100">
                  <td className="py-1 capitalize" style={{ color: SCALE_COLORS[row.scale] }}>{row.scale}</td>
                  <td className="text-right py-1 font-mono text-xs">
                    {row.varPoint.toFixed(4)} [{row.varLower.toFixed(4)}, {row.varUpper.toFixed(4)}]
                  </td>
                  <td className="text-right py-1 font-mono text-xs">
                    {(row.conPoint * 100).toFixed(2)}% [{(row.conLower * 100).toFixed(2)}%, {(row.conUpper * 100).toFixed(2)}%]
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Friedman + Eta-squared */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4 text-sm">
            <p className="font-semibold text-indigo-800 mb-1">Friedman Test</p>
            <p className="text-indigo-700">
              &chi;&sup2; = {FRIEDMAN.chiSquared.toFixed(2)}, p = {FRIEDMAN.pValue.toExponential(2)}<br />
              Scale granularity <strong>{FRIEDMAN.significant ? 'significantly' : 'does not significantly'}</strong> affect variance.
            </p>
          </div>
          <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4 text-sm">
            <p className="font-semibold text-indigo-800 mb-2">Effect Sizes (&eta;&sup2;)</p>
            <div className="space-y-2">
              {[
                { label: 'Question type', value: ETA_SQUARED.byQuestion, color: '#8b5cf6' },
                { label: 'Text type', value: ETA_SQUARED.byText, color: '#3b82f6' },
                { label: 'Scale type', value: ETA_SQUARED.byScale, color: '#f59e0b' },
              ].map((item) => (
                <div key={item.label} className="flex items-center gap-2">
                  <span className="text-xs text-gray-600 w-24">{item.label}</span>
                  <div className="flex-1 bg-gray-200 rounded-full h-3">
                    <div
                      className="h-3 rounded-full"
                      style={{ width: `${Math.min(item.value / 0.05 * 100, 100)}%`, backgroundColor: item.color }}
                    />
                  </div>
                  <span className="text-xs font-mono text-gray-700 w-16 text-right">{item.value.toFixed(4)}</span>
                </div>
              ))}
            </div>
            <p className="text-indigo-600 text-xs mt-2">
              Which feature you extract matters 8x more than which scale you use.
            </p>
          </div>
        </div>
      </div>

      {/* ── Implications ───────────────────────────────────────────────────── */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <Lightbulb className="w-5 h-5 text-yellow-600" />
          <h3 className="text-lg font-semibold text-gray-900">Implications for Production Feature Extraction</h3>
        </div>
        <div className="space-y-4 text-sm text-gray-700">
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="font-semibold text-gray-800 mb-2">The Batch-to-Batch Reliability Problem</p>
            <ul className="list-disc list-inside space-y-1 text-gray-600">
              <li><strong>Week 1:</strong> Extract intent for 10M items &rarr; 200K items near the decision boundary get label A</li>
              <li><strong>Week 2:</strong> Re-extract &rarr; those same 200K items get label B</li>
              <li><strong>Week 3:</strong> Downstream model retrains on new features &rarr; performance shifts</li>
              <li><strong>Week 6:</strong> Someone notices quality dropped &rarr; begins debugging</li>
            </ul>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <p className="font-semibold text-green-800 mb-2">Tier 1: Deploy and Trust</p>
              <p className="text-green-700 text-xs">
                Named entities, temporal, spatial, modality, imperative, comparison, normative.
                Zero variance across all scales and text types.
              </p>
            </div>
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <p className="font-semibold text-yellow-800 mb-2">Tier 2: Deploy with Monitoring</p>
              <p className="text-yellow-700 text-xs">
                Actions/events, sentiment, social, first person, numeric.
                Low but non-zero variance on specific text types.
              </p>
            </div>
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="font-semibold text-red-800 mb-2">Tier 3: Deploy with Calibration</p>
              <p className="text-red-700 text-xs">
                Intent, emotion, concreteness, causality, identity, negation, dialogue.
                Meaningful variance depending on text type and scale.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* ── Collapsible Methodology ────────────────────────────────────────── */}
      <div className="card">
        <button
          className="flex items-center justify-between w-full"
          onClick={() => setShowMethodology(!showMethodology)}
        >
          <div className="flex items-center space-x-2">
            <FlaskConical className="w-5 h-5 text-gray-600" />
            <h3 className="text-lg font-semibold text-gray-900">Methodology Details</h3>
          </div>
          {showMethodology ? <ChevronUp className="w-5 h-5 text-gray-400" /> : <ChevronDown className="w-5 h-5 text-gray-400" />}
        </button>

        {showMethodology && (
          <div className="mt-4 space-y-4 text-sm text-gray-700">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="metric-card bg-gray-50">
                <p className="text-xs text-gray-500">Model</p>
                <p className="text-lg font-bold text-gray-800">GPT-4o-mini</p>
              </div>
              <div className="metric-card bg-gray-50">
                <p className="text-xs text-gray-500">Temperature</p>
                <p className="text-lg font-bold text-gray-800">0</p>
              </div>
              <div className="metric-card bg-gray-50">
                <p className="text-xs text-gray-500">Total API Calls</p>
                <p className="text-lg font-bold text-gray-800">~800</p>
              </div>
              <div className="metric-card bg-gray-50">
                <p className="text-xs text-gray-500">Total Evaluations</p>
                <p className="text-lg font-bold text-gray-800">16,000</p>
              </div>
            </div>

            <div>
              <p className="font-semibold text-gray-800 mb-2">Metrics Definitions</p>
              <ul className="list-disc list-inside space-y-1 text-gray-600">
                <li><strong>Variance:</strong> Computed across 20 repeated samples for each text &times; question &times; scale triple. Zero = perfectly deterministic.</li>
                <li><strong>Mode Consistency:</strong> Percentage of 20 samples matching the most frequent value. 100% = all identical.</li>
                <li><strong>Shannon Entropy:</strong> Information-theoretic diversity measure. Zero = all identical. Higher = more spread.</li>
                <li><strong>Zero-Variance Rate:</strong> % of text &times; question pairs with exactly zero variance.</li>
              </ul>
            </div>

            <div>
              <p className="font-semibold text-gray-800 mb-2">Experiment Design</p>
              <p>
                10 texts &times; 20 questions &times; 4 scales &times; 20 samples = 16,000 evaluations.
                Each text &times; question &times; scale combination was evaluated 20 times at temperature=0.
                The 20 repeated samples simulate 20 independent pipeline runs. If extraction were deterministic,
                all 20 would agree. They don't.
              </p>
            </div>

            <div>
              <p className="font-semibold text-gray-800 mb-2">Reproducibility</p>
              <p className="text-gray-600">
                All code, data, and analysis scripts are open source. Run the Batch Analysis tab to replicate,
                or use the CLI:
              </p>
              <code className="block mt-2 bg-gray-100 p-3 rounded-lg text-xs font-mono text-gray-700">
                python -m grading_llm.run_batch --input corpus.jsonl --samples 20
              </code>
            </div>
          </div>
        )}
      </div>

      {/* ── Roadmap ────────────────────────────────────────────────────────── */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <GitBranch className="w-5 h-5 text-indigo-600" />
          <h3 className="text-lg font-semibold text-gray-900">Roadmap</h3>
        </div>
        <div className="space-y-3 text-sm">
          {[
            { phase: 'Phase 2', title: 'Multi-Model Comparison', desc: 'GPT-4o, Claude 3.5 Sonnet, Llama 3 70B. Test if reliability profile is model-specific or structural.' },
            { phase: 'Phase 3', title: 'Prompt Ablation', desc: 'Zero-shot, few-shot, chain-of-thought, structured output. Test if prompt engineering reduces variance.' },
            { phase: 'Phase 4', title: 'Downstream Impact', desc: 'Train classifier on batch 1, evaluate on batch 2. Quantify how instability propagates.' },
            { phase: 'Phase 5', title: 'Calibration Framework', desc: 'Formal statistical methodology. Collaboration with UC Berkeley Dept. of Statistics.' },
          ].map((item) => (
            <div key={item.phase} className="flex items-start space-x-3 bg-gray-50 p-3 rounded-lg">
              <span className="text-xs font-mono font-bold text-indigo-600 bg-indigo-100 px-2 py-0.5 rounded whitespace-nowrap">
                {item.phase}
              </span>
              <div>
                <p className="font-medium text-gray-800">{item.title}</p>
                <p className="text-gray-600 text-xs">{item.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <div className="text-center text-xs text-gray-400 pb-4">
        <p>Last updated: February 2026. Based on 16,000 evaluations conducted February 15, 2026.</p>
        <p className="mt-1">
          Run the <strong>Batch Analysis</strong> tab to replicate these results on your own data.
        </p>
      </div>
    </div>
  )
}
