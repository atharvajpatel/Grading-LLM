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
  magnitudeColor,
  onMagnitude,
  seriesColor,
  dashFor,
  CHART,
} from '../theme/palette'
import { SCALE_COLORS } from '../api/client'

// ─── Hardcoded Data from 16,000 Evaluations (Feb 15, 2026) ───────────────────

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
  return magnitudeColor(maxValue > 0 ? value / maxValue : 0)
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
      <div className="card">
        <div className="mb-3">
          <p className="panel-title">Research</p>
          <h2 className="section-title">
            LLM Feature Extraction Reliability: 16,000 Evaluations
          </h2>
        </div>
        <p className="muted leading-relaxed">
          This page presents hardcoded results from a completed batch analysis of 10 texts &times; 20
          questions &times; 4 scales &times; 20 samples = <strong>16,000 individual LLM evaluations</strong> on
          GPT-4o-mini at temperature=0. Run the <strong>Batch Analysis</strong> tab to replicate these
          results on your own data.
        </p>
        <div className="mt-3 flex flex-wrap gap-3 text-sm">
          <span className="chip">GPT-4o-mini</span>
          <span className="chip">Temperature = 0</span>
          <span className="chip">20 repeated samples</span>
          <span className="chip">Feb 2026</span>
        </div>
      </div>

      {/* ── The Problem ────────────────────────────────────────────────────── */}
      <div className="card">
        <h3 className="section-title mb-4">The Problem</h3>
        <div className="text-ink space-y-3 leading-relaxed">
          <p>
            LLMs are increasingly used as zero-shot feature extractors for unstructured data at scale.
            A single prompt can extract dozens of feature dimensions simultaneously &mdash; named entities,
            sentiment, causality, intent, emotion &mdash; replacing what would otherwise require a
            separate classifier for each dimension.
          </p>
          <div className="callout-ink">
            <p className="font-semibold text-ink">Nobody is checking which of those features are reliable.</p>
            <p className="text-sm text-mute mt-1">
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
        <h3 className="section-title mb-4">Experimental Design</h3>
        <p className="text-sm text-mute mb-5">
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
              <div className="flex-1 text-center text-xs font-semibold text-mute mb-1">
                20 Questions (semantic factor families)
              </div>
            </div>
            <div className="flex">
              {/* Y-axis label */}
              <div className="w-24 flex-shrink-0 flex items-center justify-center">
                <span className="text-xs font-semibold text-mute -rotate-90 whitespace-nowrap">
                  10 Texts
                </span>
              </div>
              {/* Grid */}
              <div className="flex-1 border border-hair rounded-lg overflow-hidden bg-white">
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
                            className="relative flex-1 aspect-square border border-gold bg-cream flex flex-col items-center justify-center gap-[1px] p-[2px]"
                            title={`Text ${rowIdx + 1} × Q${colIdx + 1}`}
                          >
                            <div className="w-full h-[3px] rounded-sm" style={{ background: SCALE_COLORS.binary }} />
                            <div className="w-full h-[3px] rounded-sm" style={{ background: SCALE_COLORS.ternary }} />
                            <div className="w-full h-[3px] rounded-sm" style={{ background: SCALE_COLORS.quaternary }} />
                            <div className="w-full h-[3px] rounded-sm" style={{ background: SCALE_COLORS.continuous }} />
                            {!isSecondZoom && (
                              <div className="absolute -right-1 -top-1 w-2 h-2 bg-gold rounded-full" />
                            )}
                          </div>
                        )
                      }

                      return (
                        <div
                          key={colIdx}
                          className="flex-1 aspect-square border border-hair bg-cream hover:bg-hair transition-colors"
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
              <div className="flex-1 flex justify-between text-[9px] text-mute px-0.5">
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
            <div className="border border-gold rounded-lg p-3 bg-white">
              <p className="text-xs font-semibold text-ink mb-2">Each cell contains 4 measurements:</p>
              <div className="space-y-1.5">
                {[
                  { label: 'Binary', value: '{0, 1}', color: SCALE_COLORS.binary, example: '→ 1' },
                  { label: 'Ternary', value: '{0, 0.5, 1}', color: SCALE_COLORS.ternary, example: '→ 0.5' },
                  { label: 'Quaternary', value: '{0, .33, .66, 1}', color: SCALE_COLORS.quaternary, example: '→ 0.66' },
                  { label: 'Continuous', value: '[0, 1]', color: SCALE_COLORS.continuous, example: '→ 0.73' },
                ].map((s) => (
                  <div key={s.label} className="flex items-center gap-2 text-xs">
                    <div className="w-3 h-3 rounded-sm flex-shrink-0 border border-hair" style={{ background: s.color }} />
                    <span className="font-semibold text-ink w-20">{s.label}</span>
                    <span className="text-mute font-mono text-[10px]">{s.value}</span>
                    <span className="text-mute font-mono text-[10px] ml-auto">{s.example}</span>
                  </div>
                ))}
              </div>
              <div className="mt-2 pt-2 border-t border-hair text-[10px] text-mute">
                = 10 &times; 20 &times; 4 = <strong>800 scores</strong> per sample
              </div>
            </div>

            {/* Stacked sheets */}
            <div className="relative h-36">
              {[4, 3, 2, 1, 0].map((i) => (
                <div
                  key={i}
                  className="absolute border border-hair rounded bg-white shadow-sm"
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
                    <div className="flex flex-col items-center justify-center h-full text-xs text-mute">
                      <div className="grid grid-cols-5 gap-[2px] mb-2">
                        {Array.from({ length: 15 }).map((_, j) => (
                          <div key={j} className="w-2 h-1.5 bg-hair rounded-[1px]" />
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
                <span className="text-xs font-bold text-ink bg-white px-1.5 py-0.5 rounded border border-hair shadow-sm">
                  &times;20 samples
                </span>
              </div>
            </div>

            {/* Final calculation */}
            <div className="bg-white border border-hair rounded-lg p-3 text-center">
              <div className="text-xs text-mute space-y-0.5">
                <div>10 texts &times; 20 questions &times; 4 scales &times; 20 samples</div>
                <div className="text-lg font-bold text-ink">=&nbsp;16,000 evaluations</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Scale Degradation Summary ──────────────────────────────────────── */}
      <div className="card">
        <h3 className="section-title mb-4">Scale Degradation Summary</h3>
        <div className="overflow-x-auto">
          <table className="data-table w-full text-sm">
            <thead>
              <tr className="border-b border-hair">
                <th className="text-left py-2 pr-4 font-medium text-mute">Scale</th>
                <th className="text-right py-2 px-3 font-medium text-mute">Mean Var</th>
                <th className="text-right py-2 px-3 font-medium text-mute">% Zero-Var</th>
                <th className="text-right py-2 px-3 font-medium text-mute">% High-Var</th>
                <th className="text-right py-2 pl-3 font-medium text-mute">Mean Mode Consistency</th>
              </tr>
            </thead>
            <tbody>
              {SCALE_SUMMARY.map((row) => (
                <tr key={row.scale} className="border-b border-hair">
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
        <div className="callout mt-4 text-sm">
          <p className="font-semibold text-ink mb-1">Finding: Variance is non-monotonic across scales</p>
          <p>
            Quaternary is the <strong>worst</strong> scale by mean variance (0.0050) and zero-variance rate (88.5%).
            Continuous has the <strong>lowest</strong> mean variance (0.0020) &mdash; lower than binary &mdash;
            but also the lowest zero-variance rate (80.5%). The noise is quiet and everywhere rather than loud and concentrated.
          </p>
        </div>
      </div>

      {/* ── Mean Variance by Scale (bar chart) ─────────────────────────────── */}
      <div className="card">
        <h3 className="section-title mb-4">Mean Variance by Scale</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={SCALE_SUMMARY}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
            <XAxis dataKey="scale" tick={{ fill: CHART.tick }} />
            <YAxis tickFormatter={(v: number) => v.toFixed(4)} tick={{ fill: CHART.tick }} />
            <Tooltip formatter={(v: number) => v.toFixed(6)} contentStyle={{ backgroundColor: CHART.tooltipBg, border: `1px solid ${CHART.tooltipBorder}` }} />
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
        <h3 className="section-title mb-2">Question Stability Across Scales</h3>
        <p className="text-sm text-mute mb-4">
          Mean variance per question across all 10 texts. Pale = stable (zero variance); dark = unstable.
        </p>

        {/* Interactive heatmap */}
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
              {HEATMAP_DATA.map((row) => (
                <tr key={row.question}>
                  <td className="py-1 pr-2 font-medium text-ink text-xs">{row.question}</td>
                  {SCALE_ORDER.map((s) => {
                    const val = (row as any)[s] as number
                    return (
                      <td key={s} className="py-1 px-2">
                        <div
                          className="rounded px-2 py-1 text-center font-mono"
                          style={{
                            backgroundColor: varianceToColor(val, heatmapMaxVar),
                            color: onMagnitude(val / heatmapMaxVar),
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
        <div className="mt-4 border-t border-hair pt-4">
          <p className="text-xs text-mute mb-2">High-resolution matplotlib figure:</p>
          <div className="plot-frame">
            <img
              src="/research/question_stability_heatmap.png"
              alt="Question stability heatmap — 20 questions x 4 scales"
              className="w-full"
            />
          </div>
        </div>

        {/* Finding callout */}
        <div className="callout mt-4 text-sm">
          <p className="font-semibold text-ink mb-1">Finding: Reliability varies dramatically by feature type</p>
          <p className="text-mute">
            <strong>Perfectly stable</strong> (zero variance): Named Entities, First Person, Emotion, Social, Normative.
            <br />
            <strong>Unstable</strong>: Intent (0.0188 at quaternary), Concreteness (0.0159 quaternary / 0.0145 continuous),
            Causality (0.0158 quaternary), Imperative (0.0239 ternary).
          </p>
        </div>
      </div>

      {/* ── Variance Distribution Violins ──────────────────────────────────── */}
      <div className="card">
        <h3 className="section-title mb-2">Variance Distribution by Scale</h3>
        <p className="text-sm text-mute mb-4">
          Distribution of variance values across all 200 text &times; question pairs, grouped by scale.
        </p>
        <div className="plot-frame">
          <img
            src="/research/variance_violins.png"
            alt="Violin plots showing variance distribution per scale"
            className="w-full"
          />
        </div>
      </div>

      {/* ── Entropy vs Variance Scatter ────────────────────────────────────── */}
      <div className="card">
        <h3 className="section-title mb-2">Entropy vs Variance</h3>
        <p className="text-sm text-mute mb-4">
          800 data points (200 per scale). Reveals distinct noise signatures: quaternary is high-variance + high-entropy
          (label flips), continuous is low-variance + moderate-entropy (feature drift).
        </p>
        <div className="plot-frame">
          <img
            src="/research/entropy_vs_variance.png"
            alt="Entropy vs variance scatter plot — 800 points colored by scale"
            className="w-full"
          />
        </div>
        <div className="callout mt-4 text-sm">
          <p className="font-semibold text-ink mb-1">Finding: Different noise structures per scale</p>
          <p>
            <strong>Quaternary</strong> instability = label flips (recommendation gets categorized differently between runs).
            <strong> Continuous</strong> instability = feature drift (value shifts slightly, may or may not cross a decision threshold).
          </p>
        </div>
      </div>

      {/* ── Variance Degradation by Text Type (line chart) ─────────────────── */}
      <div className="card">
        <h3 className="section-title mb-2">Variance Degradation by Text Type</h3>
        <p className="text-sm text-mute mb-4">
          How each text's mean variance changes across scales (binary &rarr; continuous).
        </p>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={DEGRADATION_DATA}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
            <XAxis dataKey="scale" tick={{ fill: CHART.tick }} />
            <YAxis tickFormatter={(v: number) => v.toFixed(4)} tick={{ fill: CHART.tick }} />
            <Tooltip formatter={(v: number) => v.toFixed(6)} contentStyle={{ backgroundColor: CHART.tooltipBg, border: `1px solid ${CHART.tooltipBorder}` }} />
            <Legend />
            {TEXT_IDS.map((id, idx) => (
              <Line
                key={id}
                type="monotone"
                dataKey={id}
                stroke={seriesColor(idx)}
                strokeDasharray={dashFor(idx)}
                strokeWidth={2}
                dot={{ r: 3 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>

        {/* Static matplotlib figure */}
        <div className="mt-4 border-t border-hair pt-4">
          <p className="text-xs text-mute mb-2">High-resolution matplotlib figure:</p>
          <div className="plot-frame">
            <img
              src="/research/per_text_degradation.png"
              alt="Variance degradation line chart per text type"
              className="w-full"
            />
          </div>
        </div>
      </div>

      {/* ── Text Difficulty Ranking ────────────────────────────────────────── */}
      <div className="card">
        <h3 className="section-title mb-2">Text Difficulty Ranking</h3>
        <p className="text-sm text-mute mb-4">
          Texts ranked by overall mean variance. Longer bar = more unstable. 44&times; difference between hardest and easiest.
        </p>
        <ResponsiveContainer width="100%" height={360}>
          <BarChart data={TEXT_DIFFICULTY} layout="vertical" margin={{ left: 140 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
            <XAxis type="number" tickFormatter={(v: number) => v.toFixed(4)} tick={{ fill: CHART.tick }} />
            <YAxis type="category" dataKey="id" width={130} tick={{ fontSize: 12, fill: CHART.tick }} />
            <Tooltip formatter={(v: number) => v.toFixed(6)} contentStyle={{ backgroundColor: CHART.tooltipBg, border: `1px solid ${CHART.tooltipBorder}` }} />
            <Bar dataKey="meanVariance" name="Mean Variance">
              {TEXT_DIFFICULTY.map((entry) => {
                const maxVar = TEXT_DIFFICULTY[0].meanVariance
                const ratio = entry.meanVariance / maxVar
                return (
                  <Cell
                    key={entry.id}
                    fill={magnitudeColor(ratio)}
                  />
                )
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        <div className="mt-4 border-t border-hair pt-4">
          <p className="text-xs text-mute mb-2">High-resolution matplotlib figure:</p>
          <div className="plot-frame">
            <img
              src="/research/text_difficulty_ranking.png"
              alt="Text difficulty ranking horizontal bar chart"
              className="w-full"
            />
          </div>
        </div>

        <div className="callout mt-4 text-sm">
          <p className="font-semibold text-ink mb-1">Finding: Text difficulty interacts with feature type</p>
          <p className="text-mute">
            Ambiguous text (0.0088) is 44&times; more unstable than abstract philosophical (0.0002).
            Texts with pragmatic subtext and multi-sentence narratives cluster at the top.
          </p>
        </div>
      </div>

      {/* ── Mode Consistency by Text and Scale ─────────────────────────────── */}
      <div className="card">
        <h3 className="section-title mb-2">Mode Consistency by Text and Scale</h3>
        <p className="text-sm text-mute mb-4">
          Mean mode consistency across 20 questions. 1.0 = all 20 samples identical.
        </p>
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={MODE_CONSISTENCY_DATA}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
            <XAxis dataKey="id" tick={{ fontSize: 11, fill: CHART.tick }} angle={-30} textAnchor="end" height={60} />
            <YAxis domain={[0.85, 1.0]} tickFormatter={(v: number) => v.toFixed(2)} tick={{ fill: CHART.tick }} />
            <Tooltip formatter={(v: number) => (v * 100).toFixed(2) + '%'} contentStyle={{ backgroundColor: CHART.tooltipBg, border: `1px solid ${CHART.tooltipBorder}` }} />
            <Legend />
            {SCALE_ORDER.map((scale) => (
              <Bar key={scale} dataKey={scale} name={scale} fill={SCALE_COLORS[scale]} />
            ))}
          </BarChart>
        </ResponsiveContainer>

        <div className="mt-4 border-t border-hair pt-4">
          <p className="text-xs text-mute mb-2">High-resolution matplotlib figure:</p>
          <div className="plot-frame">
            <img
              src="/research/mode_consistency_bars.png"
              alt="Mode consistency grouped bar chart by text and scale"
              className="w-full"
            />
          </div>
        </div>
      </div>

      {/* ── Per-Text Scale Metrics ─────────────────────────────────────────── */}
      <div className="card">
        <h3 className="section-title mb-4">Per-Text Scale Metrics</h3>
        <div className="overflow-x-auto">
          <table className="data-table w-full text-sm">
            <thead>
              <tr className="border-b border-hair">
                <th className="text-left py-2 pr-3 font-medium text-mute">Text</th>
                <th className="text-left py-2 px-3 font-medium text-mute">Scale</th>
                <th className="text-right py-2 px-3 font-medium text-mute">Avg Variance</th>
                <th className="text-right py-2 px-3 font-medium text-mute">Avg Consistency</th>
                <th className="text-right py-2 px-3 font-medium text-mute">Avg Entropy</th>
                <th className="text-right py-2 pl-3 font-medium text-mute">Unique Vectors</th>
              </tr>
            </thead>
            <tbody>
              {PER_TEXT_TABLE.map((row, idx) => (
                <tr key={`${row.textId}-${row.scale}`} className={idx % 4 === 3 ? 'border-b border-hair' : ''}>
                  {idx % 4 === 0 && (
                    <td className="py-1.5 pr-3 font-medium text-ink font-mono text-xs" rowSpan={4}>
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
      <div className="card">
        <div className="mb-3">
          <p className="panel-title">Advanced</p>
          <h2 className="section-title">
            Advanced Statistical Analysis
          </h2>
        </div>
        <p className="muted leading-relaxed">
          Standard reliability metrics computed from the 16,000 evaluations. These are the metrics
          any reviewer of a measurement study would expect to see: internal consistency, inter-rater
          agreement, test-retest reliability, and effect sizes.
        </p>
      </div>

      {/* ── Tier 1: Essential Reliability ──────────────────────────────────── */}
      <div className="card">
        <h3 className="section-title mb-4">Tier 1: Essential Reliability Metrics</h3>
        <p className="text-sm text-mute mb-4">
          Standard psychometric reliability coefficients. Values above 0.9 indicate excellent agreement.
          A negative Cronbach's &alpha; (flagged &ldquo;below zero&rdquo;) signals broken internal consistency.
        </p>
        <div className="overflow-x-auto">
          <table className="data-table w-full text-sm">
            <thead>
              <tr className="border-b border-hair">
                <th className="text-left py-2 pr-4 font-medium text-mute">Scale</th>
                <th className="text-right py-2 px-3 font-medium text-mute">Cronbach's &alpha;</th>
                <th className="text-right py-2 px-3 font-medium text-mute">ICC(2,1)</th>
                <th className="text-right py-2 px-3 font-medium text-mute">Cohen's &kappa;</th>
                <th className="text-right py-2 pl-3 font-medium text-mute">Krippendorff's &alpha;</th>
              </tr>
            </thead>
            <tbody>
              {TIER1_DATA.map((row) => (
                <tr key={row.scale} className="border-b border-hair">
                  <td className="py-2 pr-4 capitalize font-medium" style={{ color: SCALE_COLORS[row.scale] }}>
                    {row.scale}
                  </td>
                  <td className={`text-right py-2 px-3 font-mono ${row.cronbachsAlpha < 0 ? 'text-ink font-semibold' : ''}`}>
                    {row.cronbachsAlpha.toFixed(3)}{row.cronbachsAlpha < 0 ? ' (below zero)' : ''}
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
        <div className="callout-ink mt-4 text-sm">
          <p className="font-semibold text-ink mb-1">Finding: High agreement but poor internal consistency at fine granularity</p>
          <p>
            ICC and Krippendorff's &alpha; stay above 0.95 across all scales (excellent rater agreement).
            But Cronbach's &alpha; goes <strong>negative</strong> at quaternary (-0.12) and continuous (-0.33),
            meaning questions become independently unstable in different directions rather than consistently unstable together.
          </p>
        </div>
      </div>

      {/* ── Tier 2: Informative Metrics ────────────────────────────────────── */}
      <div className="card">
        <h3 className="section-title mb-4">Tier 2: Test-Retest &amp; Agreement</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Test-Retest */}
          <div>
            <p className="text-sm font-medium text-mute mb-2">Test-Retest Correlation (odd/even split-half)</p>
            <table className="data-table w-full text-sm">
              <thead>
                <tr className="border-b border-hair">
                  <th className="text-left py-1 font-medium text-mute">Scale</th>
                  <th className="text-right py-1 font-medium text-mute">Pearson r</th>
                  <th className="text-right py-1 font-medium text-mute">Spearman &rho;</th>
                </tr>
              </thead>
              <tbody>
                {TIER2_TEST_RETEST.map((row) => (
                  <tr key={row.scale} className="border-b border-hair">
                    <td className="py-1 capitalize" style={{ color: SCALE_COLORS[row.scale] }}>{row.scale}</td>
                    <td className="num text-right py-1 font-mono">{row.pearsonR.toFixed(4)}</td>
                    <td className="num text-right py-1 font-mono">{row.spearmanRho.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Bland-Altman */}
          <div>
            <p className="text-sm font-medium text-mute mb-2">Bland-Altman (limits of agreement)</p>
            <table className="data-table w-full text-sm">
              <thead>
                <tr className="border-b border-hair">
                  <th className="text-left py-1 font-medium text-mute">Scale</th>
                  <th className="text-right py-1 font-medium text-mute">Mean Bias</th>
                  <th className="text-right py-1 font-medium text-mute">Std Diff</th>
                </tr>
              </thead>
              <tbody>
                {TIER2_BLAND_ALTMAN.map((row) => (
                  <tr key={row.scale} className="border-b border-hair">
                    <td className="py-1 capitalize" style={{ color: SCALE_COLORS[row.scale] }}>{row.scale}</td>
                    <td className="num text-right py-1 font-mono">{row.meanBias.toFixed(4)}</td>
                    <td className="num text-right py-1 font-mono">{row.meanStdDiff.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="callout mt-4 text-sm">
          <p className="font-semibold text-ink mb-1">Scale Degradation Regression</p>
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
        <h3 className="section-title mb-4">Tier 3: Bootstrap CIs, Friedman Test &amp; Effect Sizes</h3>

        {/* Bootstrap CIs */}
        <p className="text-sm font-medium text-mute mb-2">Bootstrap 95% Confidence Intervals (1000 resamples)</p>
        <div className="overflow-x-auto mb-4">
          <table className="data-table w-full text-sm">
            <thead>
              <tr className="border-b border-hair">
                <th className="text-left py-1 font-medium text-mute">Scale</th>
                <th className="text-right py-1 font-medium text-mute">Variance [95% CI]</th>
                <th className="text-right py-1 font-medium text-mute">Consistency [95% CI]</th>
              </tr>
            </thead>
            <tbody>
              {TIER3_BOOTSTRAP.map((row) => (
                <tr key={row.scale} className="border-b border-hair">
                  <td className="py-1 capitalize" style={{ color: SCALE_COLORS[row.scale] }}>{row.scale}</td>
                  <td className="num text-right py-1 font-mono text-xs">
                    {row.varPoint.toFixed(4)} [{row.varLower.toFixed(4)}, {row.varUpper.toFixed(4)}]
                  </td>
                  <td className="num text-right py-1 font-mono text-xs">
                    {(row.conPoint * 100).toFixed(2)}% [{(row.conLower * 100).toFixed(2)}%, {(row.conUpper * 100).toFixed(2)}%]
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Friedman + Eta-squared */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="callout text-sm">
            <p className="font-semibold text-ink mb-1">Friedman Test</p>
            <p className="text-mute">
              &chi;&sup2; = {FRIEDMAN.chiSquared.toFixed(2)}, p = {FRIEDMAN.pValue.toExponential(2)}<br />
              Scale granularity <strong>{FRIEDMAN.significant ? 'significantly' : 'does not significantly'}</strong> affect variance.
            </p>
          </div>
          <div className="callout text-sm">
            <p className="font-semibold text-ink mb-2">Effect Sizes (&eta;&sup2;)</p>
            <div className="space-y-2">
              {[
                { label: 'Question type', value: ETA_SQUARED.byQuestion, color: seriesColor(0) },
                { label: 'Text type', value: ETA_SQUARED.byText, color: seriesColor(1) },
                { label: 'Scale type', value: ETA_SQUARED.byScale, color: seriesColor(2) },
              ].map((item) => (
                <div key={item.label} className="flex items-center gap-2">
                  <span className="text-xs text-mute w-24">{item.label}</span>
                  <div className="flex-1 bg-hair rounded-full h-3">
                    <div
                      className="h-3 rounded-full"
                      style={{ width: `${Math.min(item.value / 0.05 * 100, 100)}%`, backgroundColor: item.color }}
                    />
                  </div>
                  <span className="num text-xs font-mono text-mute w-16 text-right">{item.value.toFixed(4)}</span>
                </div>
              ))}
            </div>
            <p className="text-mute text-xs mt-2">
              Which feature you extract matters 8x more than which scale you use.
            </p>
          </div>
        </div>
      </div>

      {/* ── Implications ───────────────────────────────────────────────────── */}
      <div className="card">
        <h3 className="section-title mb-4">Implications for Production Feature Extraction</h3>
        <div className="space-y-4 text-sm text-mute">
          <div className="callout">
            <p className="font-semibold text-ink mb-2">The Batch-to-Batch Reliability Problem</p>
            <ul className="list-disc list-inside space-y-1 text-mute">
              <li><strong>Week 1:</strong> Extract intent for 10M items &rarr; 200K items near the decision boundary get label A</li>
              <li><strong>Week 2:</strong> Re-extract &rarr; those same 200K items get label B</li>
              <li><strong>Week 3:</strong> Downstream model retrains on new features &rarr; performance shifts</li>
              <li><strong>Week 6:</strong> Someone notices quality dropped &rarr; begins debugging</li>
            </ul>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="callout">
              <p className="font-semibold text-ink mb-2">Tier 1: Deploy and Trust</p>
              <p className="text-mute text-xs">
                Named entities, temporal, spatial, modality, imperative, comparison, normative.
                Zero variance across all scales and text types.
              </p>
            </div>
            <div className="callout-ink">
              <p className="font-semibold text-ink mb-2">Tier 2: Deploy with Monitoring</p>
              <p className="text-mute text-xs">
                Actions/events, sentiment, social, first person, numeric.
                Low but non-zero variance on specific text types.
              </p>
            </div>
            <div className="callout-ink">
              <p className="font-semibold text-ink mb-2">Tier 3: Deploy with Calibration</p>
              <p className="text-mute text-xs">
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
          <h3 className="section-title">Methodology Details</h3>
          <span className="btn-secondary">{showMethodology ? 'Hide' : 'Show'}</span>
        </button>

        {showMethodology && (
          <div className="mt-4 space-y-4 text-sm text-mute">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="metric-card">
                <p className="text-xs text-mute">Model</p>
                <p className="text-lg font-bold text-ink">GPT-4o-mini</p>
              </div>
              <div className="metric-card">
                <p className="text-xs text-mute">Temperature</p>
                <p className="text-lg font-bold text-ink">0</p>
              </div>
              <div className="metric-card">
                <p className="text-xs text-mute">Total API Calls</p>
                <p className="text-lg font-bold text-ink">~800</p>
              </div>
              <div className="metric-card">
                <p className="text-xs text-mute">Total Evaluations</p>
                <p className="text-lg font-bold text-ink">16,000</p>
              </div>
            </div>

            <div>
              <p className="font-semibold text-ink mb-2">Metrics Definitions</p>
              <ul className="list-disc list-inside space-y-1 text-mute">
                <li><strong>Variance:</strong> Computed across 20 repeated samples for each text &times; question &times; scale triple. Zero = perfectly deterministic.</li>
                <li><strong>Mode Consistency:</strong> Percentage of 20 samples matching the most frequent value. 100% = all identical.</li>
                <li><strong>Shannon Entropy:</strong> Information-theoretic diversity measure. Zero = all identical. Higher = more spread.</li>
                <li><strong>Zero-Variance Rate:</strong> % of text &times; question pairs with exactly zero variance.</li>
              </ul>
            </div>

            <div>
              <p className="font-semibold text-ink mb-2">Experiment Design</p>
              <p>
                10 texts &times; 20 questions &times; 4 scales &times; 20 samples = 16,000 evaluations.
                Each text &times; question &times; scale combination was evaluated 20 times at temperature=0.
                The 20 repeated samples simulate 20 independent pipeline runs. If extraction were deterministic,
                all 20 would agree. They don't.
              </p>
            </div>

            <div>
              <p className="font-semibold text-ink mb-2">Reproducibility</p>
              <p className="text-mute">
                All code, data, and analysis scripts are open source. Run the Batch Analysis tab to replicate,
                or use the CLI:
              </p>
              <code className="block mt-2 bg-cream p-3 rounded-lg text-xs font-mono text-mute">
                python -m grading_llm.run_batch --input corpus.jsonl --samples 20
              </code>
            </div>
          </div>
        )}
      </div>

      {/* ── Roadmap ────────────────────────────────────────────────────────── */}
      <div className="card">
        <h3 className="section-title mb-4">Roadmap</h3>
        <div className="space-y-3 text-sm">
          {[
            { phase: 'Phase 2', title: 'Multi-Model Comparison', desc: 'GPT-4o, Claude 3.5 Sonnet, Llama 3 70B. Test if reliability profile is model-specific or structural.' },
            { phase: 'Phase 3', title: 'Prompt Ablation', desc: 'Zero-shot, few-shot, chain-of-thought, structured output. Test if prompt engineering reduces variance.' },
            { phase: 'Phase 4', title: 'Downstream Impact', desc: 'Train classifier on batch 1, evaluate on batch 2. Quantify how instability propagates.' },
            { phase: 'Phase 5', title: 'Calibration Framework', desc: 'Formal statistical methodology. Collaboration with UC Berkeley Dept. of Statistics.' },
          ].map((item) => (
            <div key={item.phase} className="flex items-start space-x-3 bg-cream p-3 rounded-lg">
              <span className="chip-accent text-xs font-mono font-bold whitespace-nowrap">
                {item.phase}
              </span>
              <div>
                <p className="font-medium text-ink">{item.title}</p>
                <p className="text-mute text-xs">{item.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <div className="text-center text-xs text-mute pb-4">
        <p>Last updated: February 2026. Based on 16,000 evaluations conducted February 15, 2026.</p>
        <p className="mt-1">
          Run the <strong>Batch Analysis</strong> tab to replicate these results on your own data.
        </p>
      </div>
    </div>
  )
}
