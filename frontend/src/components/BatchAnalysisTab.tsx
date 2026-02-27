import { useState, useRef, useEffect } from 'react'
import {
  startJob,
  getJobStatus,
  cancelJob,
  validateApiKey,
  getAppConfig,
  getQuestions,
  SCALE_COLORS,
  SCALE_ORDER,
} from '../api/client'
import {
  Play,
  Square,
  Download,
  RotateCcw,
  ChevronDown,
  ChevronUp,
  Edit3,
  List,
  Key,
  CheckCircle,
  AlertCircle,
  Loader2,
} from 'lucide-react'
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

// ─── Default AI Capstone Corpus (10 texts) ───────────────────────────────────

const DEFAULT_TEXTS: Array<{ id: string; text: string }> = [
  {
    id: 'factual_simple',
    text: 'The Eiffel Tower was completed in 1889 and stands 330 meters tall in Paris, France.',
  },
  {
    id: 'sentiment_positive',
    text: 'The community garden transformed our neighborhood \u2014 strangers became friends, children learned patience, and even the most skeptical residents admitted the fresh tomatoes were worth the effort.',
  },
  {
    id: 'sentiment_negative',
    text: 'The project was a catastrophic failure that wasted millions in taxpayer money, displaced vulnerable families, and left behind nothing but empty promises and crumbling infrastructure.',
  },
  {
    id: 'ambiguous',
    text: "She said she was fine with the decision, but everyone in the room knew that wasn't the whole story.",
  },
  {
    id: 'medical_clinical',
    text: 'Patient presents with acute onset chest pain radiating to the left arm, diaphoresis, and shortness of breath. ECG shows ST-elevation in leads II, III, and aVF. Troponin levels pending.',
  },
  {
    id: 'negation_heavy',
    text: 'The study found no significant evidence that the treatment was neither ineffective nor harmful, leaving researchers unable to draw any definitive conclusions.',
  },
  {
    id: 'imperative_action',
    text: 'Immediately evacuate the building through the nearest emergency exit. Do not use the elevators. Proceed to the designated assembly point and wait for further instructions from emergency personnel.',
  },
  {
    id: 'abstract_philosophical',
    text: 'Consciousness may be less like a light switch and more like a dimmer \u2014 not something that is simply present or absent, but something that exists in degrees across a spectrum we barely understand.',
  },
  {
    id: 'narrative_paragraph',
    text: 'Maria had been working at the clinic for twelve years when the new director arrived. Within weeks, he had restructured the scheduling system, fired two senior nurses, and implemented a policy requiring all staff to log their breaks. Morale plummeted. Patients noticed the tension. By March, three more staff members had quietly submitted their resignations.',
  },
  {
    id: 'technical_ml',
    text: 'The transformer architecture uses multi-head self-attention to compute weighted representations of input tokens, where attention weights are derived from scaled dot-product similarity between query and key projections, enabling the model to capture long-range dependencies without recurrence.',
  },
]

const TEXT_PALETTE = [
  '#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
  '#ec4899', '#14b8a6', '#f97316', '#6366f1', '#84cc16',
]

const FALLBACK_QUESTION_LABELS = [
  'Named Entities', 'Actions/Events', 'Temporal', 'Spatial', 'Numeric',
  'Modality', 'Imperative', 'Comparison', 'First Person', 'Dialogue',
  'Causality', 'Negation', 'Uncertainty', 'Sentiment', 'Emotion',
  'Social', 'Normative', 'Intent', 'Concreteness', 'Identity',
]

// ─── Utility Functions ────────────────────────────────────────────────────────

function computeVariance(values: number[]): number {
  if (values.length === 0) return 0
  const mean = values.reduce((s, v) => s + v, 0) / values.length
  return values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length
}

function computeEntropy(values: number[]): number {
  if (values.length === 0) return 0
  const counts: Record<string, number> = {}
  for (const v of values) {
    const key = String(v)
    counts[key] = (counts[key] || 0) + 1
  }
  let entropy = 0
  for (const key in counts) {
    const p = counts[key] / values.length
    if (p > 0) entropy -= p * Math.log2(p)
  }
  return entropy
}

function computeModeConsistency(values: number[]): number {
  if (values.length === 0) return 0
  const counts: Record<string, number> = {}
  for (const v of values) {
    const key = String(v)
    counts[key] = (counts[key] || 0) + 1
  }
  const maxCount = Math.max(...Object.values(counts))
  return maxCount / values.length
}

/** Extract column j from a 2D matrix (samples x questions) */
function getColumn(matrix: number[][], col: number): number[] {
  return matrix.map((row) => row[col]).filter((v) => v !== undefined)
}

/** Green → Yellow → Red color interpolation based on ratio */
function varianceToColor(value: number, maxValue: number): string {
  if (maxValue === 0) return '#22c55e'
  const ratio = Math.min(value / maxValue, 1)
  if (ratio < 0.5) {
    const t = ratio * 2
    const r = Math.round(34 + t * (245 - 34))
    const g = Math.round(197 + t * (158 - 197))
    const b = Math.round(94 + t * (11 - 94))
    return `rgb(${r},${g},${b})`
  }
  const t = (ratio - 0.5) * 2
  const r = Math.round(245 + t * (239 - 245))
  const g = Math.round(158 + t * (68 - 158))
  const b = Math.round(11 + t * (68 - 11))
  return `rgb(${r},${g},${b})`
}

// ─── Types ────────────────────────────────────────────────────────────────────

interface TextEntry {
  id: string
  text: string
}

interface TextResult {
  id: string
  text: string
  rawScores: Record<string, number[][]>
  scaleMetrics: Record<string, { avg_variance: number; avg_consistency: number; avg_entropy: number }>
  usage: { totalCost: number; apiCalls: number; inputTokens: number; outputTokens: number }
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function BatchAnalysisTab() {
  // Config
  const [requireApiKey, setRequireApiKey] = useState<boolean | null>(null)
  const [isLocalDev, setIsLocalDev] = useState(false)

  // API key
  const [apiKey, setApiKey] = useState('')
  const [isKeyValidated, setIsKeyValidated] = useState(false)
  const [isValidatingKey, setIsValidatingKey] = useState(false)
  const [keyError, setKeyError] = useState<string | null>(null)

  // Text selection
  const [textMode, setTextMode] = useState<'default' | 'custom'>('default')
  const [customTexts, setCustomTexts] = useState<TextEntry[]>([
    { id: '', text: '' },
    { id: '', text: '' },
  ])
  const [showDefaultTexts, setShowDefaultTexts] = useState(false)

  // Batch execution
  const [isRunning, setIsRunning] = useState(false)
  const [currentTextIndex, setCurrentTextIndex] = useState(0)
  const [currentTextProgress, setCurrentTextProgress] = useState(0)
  const [results, setResults] = useState<TextResult[]>([])
  const [error, setError] = useState<string | null>(null)
  const cancelRef = useRef(false)
  const currentJobIdRef = useRef<string | null>(null)

  // Question labels for heatmap
  const [questionLabels, setQuestionLabels] = useState<string[]>([])

  // ─── Init ───────────────────────────────────────────────────────────────────

  useEffect(() => {
    const init = async () => {
      try {
        const config = await getAppConfig()
        setRequireApiKey(config.require_api_key)
        setIsLocalDev(config.mode.startsWith('local_dev'))
        if (!config.require_api_key) setIsKeyValidated(true)
      } catch {
        setRequireApiKey(true)
      }
      try {
        const questions = await getQuestions('mech')
        setQuestionLabels(questions.map((q) => q.family || q.id))
      } catch {
        setQuestionLabels(FALLBACK_QUESTION_LABELS)
      }
    }
    init()
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancelRef.current = true
      const jobId = currentJobIdRef.current
      if (jobId) {
        const apiBaseUrl = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/+$/, '')
        navigator.sendBeacon(`${apiBaseUrl}/api/jobs/${jobId}/cancel`, '')
      }
    }
  }, [])

  // ─── API Key Handlers ───────────────────────────────────────────────────────

  const handleValidateKey = async () => {
    if (!apiKey.trim()) {
      setKeyError('Please enter your OpenAI API key')
      return
    }
    setIsValidatingKey(true)
    setKeyError(null)
    try {
      const response = await validateApiKey(apiKey.trim())
      if (response.valid) {
        setIsKeyValidated(true)
        setKeyError(null)
      } else {
        setKeyError(response.error || 'Invalid API key')
        setIsKeyValidated(false)
      }
    } catch (err: any) {
      setKeyError(
        err.response?.data?.detail || err.message || 'Could not validate key. Is the backend running?'
      )
      setIsKeyValidated(false)
    } finally {
      setIsValidatingKey(false)
    }
  }

  const handleResetKey = () => {
    setApiKey('')
    setIsKeyValidated(false)
    setKeyError(null)
  }

  // ─── Text Management ───────────────────────────────────────────────────────

  const getTexts = (): TextEntry[] => {
    if (textMode === 'default') return DEFAULT_TEXTS
    return customTexts.filter((t) => t.id.trim() && t.text.trim())
  }

  const addCustomRow = () => setCustomTexts([...customTexts, { id: '', text: '' }])

  const removeCustomRow = (idx: number) => {
    if (customTexts.length <= 1) return
    setCustomTexts(customTexts.filter((_, i) => i !== idx))
  }

  const updateCustomText = (idx: number, field: 'id' | 'text', value: string) => {
    const updated = [...customTexts]
    updated[idx] = { ...updated[idx], [field]: value }
    setCustomTexts(updated)
  }

  // ─── Batch Execution ───────────────────────────────────────────────────────

  const runSingleJob = (text: TextEntry): Promise<TextResult> => {
    return new Promise(async (resolve, reject) => {
      try {
        const { job_id } = await startJob({
          statement: text.text,
          api_key: apiKey || undefined,
          n_samples: 20,
          question_mode: 'mech',
        })
        currentJobIdRef.current = job_id

        const poll = setInterval(async () => {
          if (cancelRef.current) {
            clearInterval(poll)
            try { await cancelJob(job_id) } catch { /* ignore */ }
            reject(new Error('Cancelled'))
            return
          }
          try {
            const status = await getJobStatus(job_id)
            setCurrentTextProgress(status.progress || 0)

            if (status.status === 'completed' && status.result) {
              clearInterval(poll)
              currentJobIdRef.current = null
              const res = status.result as any
              const raw = res.raw_scores || res.embeddings_raw || res.metrics?.embeddings_raw || {}
              resolve({
                id: text.id,
                text: text.text,
                rawScores: raw,
                scaleMetrics: res.scale_metrics || {},
                usage: {
                  totalCost: status.usage?.cost_usd || 0,
                  apiCalls: status.usage?.api_calls || 0,
                  inputTokens: status.usage?.input_tokens || 0,
                  outputTokens: status.usage?.output_tokens || 0,
                },
              })
            } else if (status.status === 'failed') {
              clearInterval(poll)
              currentJobIdRef.current = null
              reject(new Error(status.error || 'Job failed'))
            } else if (status.status === 'cancelled') {
              clearInterval(poll)
              currentJobIdRef.current = null
              reject(new Error('Cancelled'))
            }
          } catch (err: any) {
            if (err.response?.status === 404) {
              clearInterval(poll)
              reject(new Error('Job not found'))
            }
          }
        }, 1500)
      } catch (err: any) {
        reject(err)
      }
    })
  }

  const runBatch = async () => {
    const texts = getTexts()
    if (texts.length === 0) return

    setIsRunning(true)
    setError(null)
    setResults([])
    cancelRef.current = false

    for (let i = 0; i < texts.length; i++) {
      if (cancelRef.current) break
      setCurrentTextIndex(i)
      setCurrentTextProgress(0)
      try {
        const result = await runSingleJob(texts[i])
        setResults((prev) => [...prev, result])
      } catch (err: any) {
        if (err.message === 'Cancelled') break
        setError(`Failed on ${texts[i].id}: ${err.message}`)
        break
      }
    }

    setIsRunning(false)
    currentJobIdRef.current = null
  }

  const handleCancel = async () => {
    cancelRef.current = true
    const jobId = currentJobIdRef.current
    if (jobId) {
      try { await cancelJob(jobId) } catch { /* ignore */ }
    }
  }

  const handleReset = () => {
    setResults([])
    setError(null)
    setCurrentTextIndex(0)
    setCurrentTextProgress(0)
  }

  // ─── Download ───────────────────────────────────────────────────────────────

  const handleDownload = () => {
    const data = results.map((r) => ({
      id: r.id,
      text: r.text,
      raw_scores: r.rawScores,
      metrics: r.scaleMetrics,
      usage: r.usage,
    }))
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `batch_results_${new Date().toISOString().split('T')[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  // ─── Computed Metrics ───────────────────────────────────────────────────────

  const computeHeatmapData = () => {
    const labels = questionLabels.length > 0 ? questionLabels : FALLBACK_QUESTION_LABELS
    return labels.map((label, qIdx) => {
      const row: Record<string, any> = { question: label }
      for (const scale of SCALE_ORDER) {
        const variances: number[] = []
        for (const r of results) {
          const matrix = r.rawScores[scale]
          if (!matrix || matrix.length === 0) continue
          const col = getColumn(matrix, qIdx)
          if (col.length > 0) variances.push(computeVariance(col))
        }
        row[scale] = variances.length > 0 ? variances.reduce((a, b) => a + b, 0) / variances.length : 0
      }
      return row
    })
  }

  const computeScaleSummary = () => {
    return SCALE_ORDER.map((scale) => {
      const allVariances: number[] = []
      const allEntropies: number[] = []
      const allConsistencies: number[] = []
      for (const r of results) {
        const matrix = r.rawScores[scale]
        if (!matrix || matrix.length === 0) continue
        const numQ = matrix[0]?.length || 0
        for (let q = 0; q < numQ; q++) {
          const col = getColumn(matrix, q)
          allVariances.push(computeVariance(col))
          allEntropies.push(computeEntropy(col))
          allConsistencies.push(computeModeConsistency(col))
        }
      }
      const n = allVariances.length
      const meanVar = n > 0 ? allVariances.reduce((a, b) => a + b, 0) / n : 0
      const sorted = [...allVariances].sort((a, b) => a - b)
      const medianVar = n > 0 ? sorted[Math.floor(n / 2)] : 0
      const zeroVarPct = n > 0 ? (allVariances.filter((v) => v === 0).length / n) * 100 : 0
      const highVarPct = n > 0 ? (allVariances.filter((v) => v > 0.1).length / n) * 100 : 0
      const sortedEnt = [...allEntropies].sort((a, b) => a - b)
      const medianEntropy = n > 0 ? sortedEnt[Math.floor(n / 2)] : 0
      const meanConsistency = n > 0 ? allConsistencies.reduce((a, b) => a + b, 0) / n : 0
      return { scale, meanVar, medianVar, zeroVarPct, highVarPct, medianEntropy, meanConsistency }
    })
  }

  const computeDegradationData = () => {
    return SCALE_ORDER.map((scale) => {
      const point: Record<string, any> = { scale }
      for (const r of results) {
        const matrix = r.rawScores[scale]
        if (!matrix || matrix.length === 0) { point[r.id] = 0; continue }
        const numQ = matrix[0]?.length || 0
        let sum = 0
        for (let q = 0; q < numQ; q++) {
          sum += computeVariance(getColumn(matrix, q))
        }
        point[r.id] = numQ > 0 ? sum / numQ : 0
      }
      return point
    })
  }

  const computeTextDifficulty = () => {
    return results
      .map((r) => {
        let totalVar = 0
        let count = 0
        for (const scale of SCALE_ORDER) {
          const matrix = r.rawScores[scale]
          if (!matrix || matrix.length === 0) continue
          const numQ = matrix[0]?.length || 0
          for (let q = 0; q < numQ; q++) {
            totalVar += computeVariance(getColumn(matrix, q))
            count++
          }
        }
        return { id: r.id, meanVariance: count > 0 ? totalVar / count : 0 }
      })
      .sort((a, b) => b.meanVariance - a.meanVariance)
  }

  const computeModeConsistencyByText = () => {
    return results.map((r) => {
      const row: Record<string, any> = { id: r.id }
      for (const scale of SCALE_ORDER) {
        const matrix = r.rawScores[scale]
        if (!matrix || matrix.length === 0) { row[scale] = 0; continue }
        const numQ = matrix[0]?.length || 0
        let sum = 0
        for (let q = 0; q < numQ; q++) {
          sum += computeModeConsistency(getColumn(matrix, q))
        }
        row[scale] = numQ > 0 ? sum / numQ : 0
      }
      return row
    })
  }

  const computePerTextTable = () => {
    const rows: Array<{
      textId: string; scale: string; avgVariance: number
      avgConsistency: number; avgEntropy: number; uniqueVectors: number
    }> = []
    for (const r of results) {
      for (const scale of SCALE_ORDER) {
        const matrix = r.rawScores[scale]
        if (!matrix || matrix.length === 0) {
          const sm = r.scaleMetrics[scale]
          rows.push({
            textId: r.id,
            scale,
            avgVariance: sm?.avg_variance || 0,
            avgConsistency: sm?.avg_consistency || 0,
            avgEntropy: sm?.avg_entropy || 0,
            uniqueVectors: 0,
          })
          continue
        }
        const numQ = matrix[0]?.length || 0
        let varSum = 0
        let consSum = 0
        let entSum = 0
        for (let q = 0; q < numQ; q++) {
          const col = getColumn(matrix, q)
          varSum += computeVariance(col)
          consSum += computeModeConsistency(col)
          entSum += computeEntropy(col)
        }
        const uniqueVecs = new Set(matrix.map((row) => JSON.stringify(row))).size
        rows.push({
          textId: r.id,
          scale,
          avgVariance: numQ > 0 ? varSum / numQ : 0,
          avgConsistency: numQ > 0 ? consSum / numQ : 0,
          avgEntropy: numQ > 0 ? entSum / numQ : 0,
          uniqueVectors: uniqueVecs,
        })
      }
    }
    return rows
  }

  // ─── Derived Values ─────────────────────────────────────────────────────────

  const texts = getTexts()
  const totalTexts = texts.length

  const totalUsage = {
    cost: results.reduce((s, r) => s + r.usage.totalCost, 0),
    apiCalls: results.reduce((s, r) => s + r.usage.apiCalls, 0),
    inputTokens: results.reduce((s, r) => s + r.usage.inputTokens, 0),
    outputTokens: results.reduce((s, r) => s + r.usage.outputTokens, 0),
  }

  // ─── Loading State ──────────────────────────────────────────────────────────

  if (requireApiKey === null) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
      </div>
    )
  }

  // ─── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6">
      {/* ── API Key ────────────────────────────────────────────────────────── */}
      {requireApiKey && (
        <div className="card">
          <div className="flex items-center space-x-2 mb-4">
            <Key className="w-5 h-5 text-blue-600" />
            <h2 className="text-lg font-semibold text-gray-900">OpenAI API Key</h2>
          </div>
          {!isKeyValidated ? (
            <div className="space-y-4">
              <p className="text-sm text-gray-600">
                Enter your OpenAI API key. Your key is sent directly to OpenAI and is not stored.
              </p>
              <input
                type="password"
                className="input-field"
                placeholder="sk-..."
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                disabled={isValidatingKey}
              />
              <button
                className="btn-primary flex items-center space-x-2"
                onClick={handleValidateKey}
                disabled={!apiKey.trim() || isValidatingKey}
              >
                {isValidatingKey ? (
                  <><Loader2 className="w-5 h-5 animate-spin" /><span>Validating...</span></>
                ) : (
                  <><Key className="w-5 h-5" /><span>Validate</span></>
                )}
              </button>
              {keyError && (
                <div className="flex items-center space-x-2 text-red-600 bg-red-50 p-3 rounded-lg">
                  <AlertCircle className="w-5 h-5" /><span>{keyError}</span>
                </div>
              )}
            </div>
          ) : (
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-green-600">
                <CheckCircle className="w-5 h-5" />
                <span className="font-medium">API key validated</span>
              </div>
              <button className="text-sm text-gray-500 hover:text-gray-700 underline" onClick={handleResetKey}>
                Use different key
              </button>
            </div>
          )}
        </div>
      )}

      {/* Local dev notice */}
      {isLocalDev && !requireApiKey && (
        <div className="card bg-blue-50 border-blue-200">
          <div className="flex items-center space-x-2 text-blue-700">
            <CheckCircle className="w-5 h-5" />
            <span className="font-medium">Local Development Mode — using environment API key</span>
          </div>
        </div>
      )}

      {/* ── Text Selection ─────────────────────────────────────────────────── */}
      {isKeyValidated && (
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Text Corpus</h2>

          {/* Toggle */}
          <div className="flex space-x-2 mb-4">
            <button
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg border-2 text-sm font-medium transition-all ${
                textMode === 'default'
                  ? 'border-blue-500 bg-blue-50 text-blue-700'
                  : 'border-gray-200 text-gray-600 hover:border-gray-300'
              }`}
              onClick={() => setTextMode('default')}
              disabled={isRunning}
            >
              <List className="w-4 h-4" />
              <span>Default ({DEFAULT_TEXTS.length})</span>
            </button>
            <button
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg border-2 text-sm font-medium transition-all ${
                textMode === 'custom'
                  ? 'border-purple-500 bg-purple-50 text-purple-700'
                  : 'border-gray-200 text-gray-600 hover:border-gray-300'
              }`}
              onClick={() => setTextMode('custom')}
              disabled={isRunning}
            >
              <Edit3 className="w-4 h-4" />
              <span>Custom</span>
            </button>
          </div>

          {/* Default texts (collapsible) */}
          {textMode === 'default' && (
            <div>
              <button
                className="flex items-center space-x-1 text-sm text-blue-600 hover:text-blue-800 mb-2"
                onClick={() => setShowDefaultTexts(!showDefaultTexts)}
              >
                {showDefaultTexts ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                <span>{showDefaultTexts ? 'Hide' : 'Show'} {DEFAULT_TEXTS.length} default texts</span>
              </button>
              {showDefaultTexts && (
                <div className="space-y-2 max-h-80 overflow-y-auto">
                  {DEFAULT_TEXTS.map((t) => (
                    <div key={t.id} className="bg-gray-50 rounded-lg p-3">
                      <span className="text-xs font-mono font-medium text-gray-500">{t.id}</span>
                      <p className="text-sm text-gray-700 mt-1">{t.text}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Custom texts */}
          {textMode === 'custom' && (
            <div className="space-y-3">
              {customTexts.map((entry, idx) => (
                <div key={idx} className="flex space-x-2 items-start">
                  <input
                    className="input-field w-40 text-sm"
                    placeholder="text_id"
                    value={entry.id}
                    onChange={(e) => updateCustomText(idx, 'id', e.target.value)}
                    disabled={isRunning}
                  />
                  <textarea
                    className="input-field flex-1 text-sm h-16"
                    placeholder="Enter text..."
                    value={entry.text}
                    onChange={(e) => updateCustomText(idx, 'text', e.target.value)}
                    disabled={isRunning}
                  />
                  <button
                    className="text-red-400 hover:text-red-600 mt-2 text-lg font-bold"
                    onClick={() => removeCustomRow(idx)}
                    disabled={isRunning || customTexts.length <= 1}
                    title="Remove row"
                  >
                    &times;
                  </button>
                </div>
              ))}
              <button
                className="text-sm text-blue-600 hover:text-blue-800 font-medium"
                onClick={addCustomRow}
                disabled={isRunning}
              >
                + Add text
              </button>
            </div>
          )}
        </div>
      )}

      {/* ── Controls + Progress ────────────────────────────────────────────── */}
      {isKeyValidated && (
        <div className="card">
          <div className="flex items-center space-x-4 flex-wrap gap-y-2">
            {!isRunning ? (
              <button
                className="btn-primary flex items-center space-x-2"
                onClick={runBatch}
                disabled={getTexts().length === 0}
              >
                <Play className="w-5 h-5" />
                <span>Run Batch Analysis</span>
              </button>
            ) : (
              <button className="btn-danger flex items-center space-x-2" onClick={handleCancel}>
                <Square className="w-5 h-5" />
                <span>Stop</span>
              </button>
            )}
            {results.length > 0 && !isRunning && (
              <>
                <button
                  className="flex items-center space-x-2 px-4 py-3 bg-gray-100 text-gray-700 font-medium rounded-lg hover:bg-gray-200 transition-colors"
                  onClick={handleDownload}
                >
                  <Download className="w-5 h-5" />
                  <span>Download JSON</span>
                </button>
                <button
                  className="flex items-center space-x-2 px-4 py-3 text-gray-500 font-medium rounded-lg hover:bg-gray-100 transition-colors"
                  onClick={handleReset}
                >
                  <RotateCcw className="w-5 h-5" />
                  <span>Reset</span>
                </button>
              </>
            )}
          </div>

          {/* Progress */}
          {isRunning && (
            <div className="mt-4 space-y-2">
              <div className="flex justify-between text-sm text-gray-600">
                <span className="font-medium">
                  Text {currentTextIndex + 1} of {totalTexts}: <span className="font-mono">{texts[currentTextIndex]?.id}</span>
                </span>
                <span>{results.length} / {totalTexts} completed</span>
              </div>
              {/* Macro progress (texts) */}
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${((results.length + currentTextProgress / 100) / totalTexts) * 100}%` }}
                />
              </div>
              {/* Micro progress (current text) */}
              <div className="flex items-center space-x-2 text-sm text-gray-500">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Current text: {currentTextProgress.toFixed(1)}%</span>
              </div>
            </div>
          )}

          {error && (
            <div className="mt-4 flex items-center space-x-2 text-red-600 bg-red-50 p-3 rounded-lg">
              <AlertCircle className="w-5 h-5" /><span>{error}</span>
            </div>
          )}
        </div>
      )}

      {/* ── Results ────────────────────────────────────────────────────────── */}
      {results.length > 0 && (
        <>
          {/* Cost Summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="metric-card bg-green-50">
              <p className="text-sm text-green-600">Total Cost</p>
              <p className="text-2xl font-bold text-green-700">
                ${totalUsage.cost < 0.01 ? totalUsage.cost.toFixed(4) : totalUsage.cost.toFixed(2)}
              </p>
            </div>
            <div className="metric-card bg-blue-50">
              <p className="text-sm text-blue-600">API Calls</p>
              <p className="text-2xl font-bold text-blue-700">{totalUsage.apiCalls.toLocaleString()}</p>
            </div>
            <div className="metric-card bg-purple-50">
              <p className="text-sm text-purple-600">Input Tokens</p>
              <p className="text-2xl font-bold text-purple-700">{totalUsage.inputTokens.toLocaleString()}</p>
            </div>
            <div className="metric-card bg-orange-50">
              <p className="text-sm text-orange-600">Output Tokens</p>
              <p className="text-2xl font-bold text-orange-700">{totalUsage.outputTokens.toLocaleString()}</p>
            </div>
          </div>

          {/* Scale Degradation Summary Table */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Scale Degradation Summary</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-2 pr-4 font-medium text-gray-500">Scale</th>
                    <th className="text-right py-2 px-3 font-medium text-gray-500">Mean Var</th>
                    <th className="text-right py-2 px-3 font-medium text-gray-500">Median Var</th>
                    <th className="text-right py-2 px-3 font-medium text-gray-500">% Zero-Var</th>
                    <th className="text-right py-2 px-3 font-medium text-gray-500">% High-Var</th>
                    <th className="text-right py-2 px-3 font-medium text-gray-500">Median Entropy</th>
                    <th className="text-right py-2 pl-3 font-medium text-gray-500">Mean Mode Consistency</th>
                  </tr>
                </thead>
                <tbody>
                  {computeScaleSummary().map((row) => (
                    <tr key={row.scale} className="border-b border-gray-100">
                      <td className="py-2 pr-4 capitalize font-medium" style={{ color: SCALE_COLORS[row.scale] }}>
                        {row.scale}
                      </td>
                      <td className="text-right py-2 px-3 font-mono">{row.meanVar.toFixed(4)}</td>
                      <td className="text-right py-2 px-3 font-mono">{row.medianVar.toFixed(4)}</td>
                      <td className="text-right py-2 px-3 font-mono">{row.zeroVarPct.toFixed(1)}%</td>
                      <td className="text-right py-2 px-3 font-mono">{row.highVarPct.toFixed(1)}%</td>
                      <td className="text-right py-2 px-3 font-mono">{row.medianEntropy.toFixed(4)}</td>
                      <td className="text-right py-2 pl-3 font-mono">{(row.meanConsistency * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Question Stability Heatmap */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Question Stability Across Scales</h3>
            <p className="text-sm text-gray-500 mb-4">
              Mean variance per question across all texts. Green = stable (zero variance). Red = unstable.
            </p>
            {(() => {
              const heatmap = computeHeatmapData()
              const maxVar = Math.max(
                ...heatmap.flatMap((row) => SCALE_ORDER.map((s) => row[s] as number)),
                0.001
              )
              return (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr>
                        <th className="text-left py-1 pr-2 font-medium text-gray-500 w-36">Question</th>
                        {SCALE_ORDER.map((s) => (
                          <th key={s} className="py-1 px-2 font-medium text-gray-500 capitalize text-center w-28">
                            {s}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {heatmap.map((row) => (
                        <tr key={row.question}>
                          <td className="py-1 pr-2 font-medium text-gray-700 truncate max-w-[9rem]" title={row.question}>
                            {row.question}
                          </td>
                          {SCALE_ORDER.map((s) => {
                            const val = row[s] as number
                            return (
                              <td key={s} className="py-1 px-2">
                                <div
                                  className="rounded px-2 py-1 text-center font-mono"
                                  style={{
                                    backgroundColor: varianceToColor(val, maxVar),
                                    color: val / maxVar > 0.5 ? '#fff' : '#1f2937',
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
              )
            })()}
          </div>

          {/* Mean Variance by Scale (bar chart) */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Mean Variance by Scale</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={computeScaleSummary()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="scale" tick={{ textTransform: 'capitalize' } as any} />
                <YAxis tickFormatter={(v: number) => v.toFixed(4)} />
                <Tooltip formatter={(v: number) => v.toFixed(6)} />
                <Bar dataKey="meanVar" name="Mean Variance">
                  {computeScaleSummary().map((entry) => (
                    <Cell key={entry.scale} fill={SCALE_COLORS[entry.scale]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Variance Degradation by Text Type (line chart) */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Variance Degradation by Text Type</h3>
            <p className="text-sm text-gray-500 mb-4">
              How each text's mean variance changes across scales (binary &rarr; continuous).
            </p>
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={computeDegradationData()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="scale" />
                <YAxis tickFormatter={(v: number) => v.toFixed(4)} />
                <Tooltip formatter={(v: number) => v.toFixed(6)} />
                <Legend />
                {results.map((r, idx) => (
                  <Line
                    key={r.id}
                    type="monotone"
                    dataKey={r.id}
                    stroke={TEXT_PALETTE[idx % TEXT_PALETTE.length]}
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Text Difficulty Ranking (horizontal bars) */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Text Difficulty Ranking</h3>
            <p className="text-sm text-gray-500 mb-4">
              Texts ranked by overall mean variance. Longer bar = more unstable.
            </p>
            {(() => {
              const difficulty = computeTextDifficulty()
              const maxVar = difficulty.length > 0 ? difficulty[0].meanVariance : 0.001
              return (
                <ResponsiveContainer width="100%" height={Math.max(difficulty.length * 36, 120)}>
                  <BarChart data={difficulty} layout="vertical" margin={{ left: 120 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" tickFormatter={(v: number) => v.toFixed(4)} />
                    <YAxis type="category" dataKey="id" width={110} tick={{ fontSize: 12 }} />
                    <Tooltip formatter={(v: number) => v.toFixed(6)} />
                    <Bar dataKey="meanVariance" name="Mean Variance">
                      {difficulty.map((entry) => (
                        <Cell
                          key={entry.id}
                          fill={
                            maxVar === 0
                              ? '#22c55e'
                              : entry.meanVariance / maxVar < 0.33
                                ? '#22c55e'
                                : entry.meanVariance / maxVar < 0.66
                                  ? '#f59e0b'
                                  : '#ef4444'
                          }
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )
            })()}
          </div>

          {/* Mode Consistency by Text and Scale (grouped bars) */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Mode Consistency by Text and Scale</h3>
            <p className="text-sm text-gray-500 mb-4">
              Mean mode consistency across 20 questions. 1.0 = all 20 samples identical.
            </p>
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={computeModeConsistencyByText()}>
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

          {/* Per-Text Scale Metrics (raw table) */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Per-Text Scale Metrics</h3>
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
                  {computePerTextTable().map((row, idx) => (
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
        </>
      )}
    </div>
  )
}
