import { useState, useEffect, useRef, useCallback } from 'react'
import {
  startJob,
  getJobStatus,
  cancelJob,
  validateApiKey,
  getAppConfig,
  AnalysisResult,
  JobStatusResponse,
  UsageStats,
  SCALE_COLORS,
  SCALE_ORDER,
  QuestionMode,
} from '../api/client'
import PCAPlot3D from './PCAPlot3D'
import MetricsTable from './MetricsTable'
import VectorHeatmap from './VectorHeatmap'
import ValueDistribution from './ValueDistribution'

const POLL_INTERVAL = 1000 // 1 second

export default function AnalyzeTab() {
  // Config state - whether API key is required
  const [requireApiKey, setRequireApiKey] = useState<boolean | null>(null) // null = loading
  const [isLocalDev, setIsLocalDev] = useState(false)

  const [apiKey, setApiKey] = useState('')
  const [isKeyValidated, setIsKeyValidated] = useState(false)
  const [isValidatingKey, setIsValidatingKey] = useState(false)
  const [keyError, setKeyError] = useState<string | null>(null)

  const [statement, setStatement] = useState('')
  const [questionMode, setQuestionMode] = useState<QuestionMode>('mech')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [usage, setUsage] = useState<UsageStats | null>(null)

  // Job tracking
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<JobStatusResponse | null>(null)
  const pollIntervalRef = useRef<number | null>(null)

  // Use ref to track current job ID for cleanup (avoids stale closure issues)
  const jobIdRef = useRef<string | null>(null)

  // Keep ref in sync with state
  useEffect(() => {
    jobIdRef.current = jobId
  }, [jobId])

  // Cleanup function to cancel job and stop polling
  const cleanup = useCallback(async () => {
    // Stop polling first
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current)
      pollIntervalRef.current = null
    }

    // Cancel the job using ref (always has latest value)
    const id = jobIdRef.current
    if (id) {
      try {
        await cancelJob(id)
        console.log(`Cleanup: Cancelled job ${id}`)
      } catch {
        // Ignore errors on cleanup
      }
      jobIdRef.current = null
    }
  }, [])

  // Load app config on mount
  useEffect(() => {
    const loadConfig = async () => {
      try {
        const config = await getAppConfig()
        setRequireApiKey(config.require_api_key)
        setIsLocalDev(config.mode.startsWith('local_dev'))
        // If API key not required, auto-validate
        if (!config.require_api_key) {
          setIsKeyValidated(true)
        }
      } catch {
        // Default to requiring API key if config fails
        setRequireApiKey(true)
      }
    }
    loadConfig()
  }, [])

  // Handle page unload (refresh, close tab, navigate away)
  useEffect(() => {
    const handleBeforeUnload = () => {
      // Use ref to get current job ID (avoids stale closure)
      const currentJobId = jobIdRef.current
      if (currentJobId) {
        // Use sendBeacon for reliable cleanup on page close/refresh
        const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
        navigator.sendBeacon(`${API_BASE_URL}/api/jobs/${currentJobId}/cancel`, '')
        console.log(`beforeunload: Sent cancel for job ${currentJobId}`)
      }
    }

    // Handle visibility change (tab switching, minimizing)
    const handleVisibilityChange = () => {
      // Don't cancel on visibility change - only on actual unload
    }

    window.addEventListener('beforeunload', handleBeforeUnload)
    document.addEventListener('visibilitychange', handleVisibilityChange)

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload)
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      // Cleanup on component unmount (e.g., navigating to different tab in app)
      cleanup()
    }
  }, [cleanup])

  // Poll for job status
  const pollJobStatus = useCallback(async (id: string) => {
    try {
      const status = await getJobStatus(id)
      setJobStatus(status)

      if (status.status === 'completed' && status.result) {
        setResult(status.result)
        setUsage(status.usage)
        setIsLoading(false)
        setJobId(null)
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current)
          pollIntervalRef.current = null
        }
      } else if (status.status === 'failed') {
        setError(status.error || 'Analysis failed')
        setIsLoading(false)
        setJobId(null)
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current)
          pollIntervalRef.current = null
        }
      } else if (status.status === 'cancelled') {
        setIsLoading(false)
        setJobId(null)
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current)
          pollIntervalRef.current = null
        }
      }
    } catch (err: any) {
      // Job might have been cancelled/deleted
      if (err.response?.status === 404) {
        setIsLoading(false)
        setJobId(null)
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current)
          pollIntervalRef.current = null
        }
      }
    }
  }, [])

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
      setKeyError(err.response?.data?.detail || err.message || 'Failed to validate API key')
      setIsKeyValidated(false)
    } finally {
      setIsValidatingKey(false)
    }
  }

  const handleResetKey = () => {
    setApiKey('')
    setIsKeyValidated(false)
    setKeyError(null)
    setResult(null)
    setUsage(null)
  }

  const handleAnalyze = async () => {
    if (!statement.trim() || !isKeyValidated) return

    setIsLoading(true)
    setError(null)
    setResult(null)
    setUsage(null)
    setJobStatus(null)

    try {
      const response = await startJob({
        statement: statement.trim(),
        api_key: apiKey || undefined,  // Only send if provided
        n_samples: 20,
        question_mode: questionMode,
      })

      setJobId(response.job_id)

      // Start polling
      pollIntervalRef.current = window.setInterval(() => {
        pollJobStatus(response.job_id)
      }, POLL_INTERVAL)

      // Initial poll
      pollJobStatus(response.job_id)
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to start analysis')
      setIsLoading(false)
    }
  }

  const handleCancel = async () => {
    if (!jobId) return

    try {
      await cancelJob(jobId)
    } catch {
      // Ignore errors
    }

    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current)
      pollIntervalRef.current = null
    }

    setIsLoading(false)
    setJobId(null)
    setJobStatus(null)
  }

  // Calculate evaluations completed (20 questions × samples completed)
  const getEvaluationsCompleted = () => {
    if (!jobStatus) return 0
    const { current_scale, current_sample } = jobStatus
    const scaleIndex = SCALE_ORDER.indexOf(current_scale)
    if (scaleIndex === -1) return 0
    // Each completed sample = 20 evaluations (all questions)
    const completedSamples = scaleIndex * 20 + current_sample
    return completedSamples * 20
  }

  const TOTAL_EVALUATIONS = 1600 // 20 questions × 20 samples × 4 scales

  // Still loading config
  if (requireApiKey === null) {
    return (
      <div className="flex items-center justify-center py-12">
        <span className="muted">Loading…</span>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* API Key Section - Only show if API key is required */}
      {requireApiKey && (
        <div className="card">
          <div className="flex items-center space-x-2 mb-4">
            <h2 className="panel-title">OpenAI API Key</h2>
          </div>

          {!isKeyValidated ? (
            <div className="space-y-4">
              <p className="muted text-sm">
                Enter your OpenAI API key to use this tool. Your key is sent directly to OpenAI and is not stored on our servers.
              </p>

              <div>
                <input
                  type="password"
                  className="input-field"
                  placeholder="sk-..."
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  disabled={isValidatingKey}
                />
              </div>

              <button
                className="btn-primary"
                onClick={handleValidateKey}
                disabled={!apiKey.trim() || isValidatingKey}
              >
                {isValidatingKey ? 'Validating…' : 'Validate Key'}
              </button>

              {keyError && (
                <div className="callout-ink">
                  <span className="font-semibold">Error:</span> {keyError}
                </div>
              )}
            </div>
          ) : (
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <span className="font-semibold text-ink">Validated — API key ready</span>
              </div>
              <button
                className="text-sm text-mute hover:text-ink underline"
                onClick={handleResetKey}
              >
                Use different key
            </button>
          </div>
        )}
        </div>
      )}

      {/* Local Dev Mode Notice */}
      {isLocalDev && !requireApiKey && (
        <div className="callout">
          <div className="flex items-center space-x-2">
            <span className="font-semibold text-ink">Local Development Mode</span>
          </div>
          <p className="muted text-sm mt-1">
            Using API key from environment. No key entry required.
          </p>
        </div>
      )}

      {/* Problem Statement */}
      <div className="callout">
        <div className="flex items-center space-x-2 mb-4">
          <h2 className="section-title">The Problem</h2>
        </div>

        <div className="prose prose-sm max-w-none text-ink space-y-4">
          <p className="text-lg">
            <strong>Where does the statistical boundary lie for a calibrated LLM as a judge/labeller?</strong>
          </p>

          <p>
            When using LLMs to evaluate or label data, we face a critical question: at what granularity
            can we trust the model's judgments? We know binary labels (yes/no) are generally accurate,
            but at what point is multi-class rating (low/medium/high) or continuous scoring (0.0-1.0)
            actually calibrated versus just noise?
          </p>

          <div className="card">
            <p className="font-semibold text-ink mb-2">This tool helps you find that boundary by:</p>
            <ul className="list-disc list-inside space-y-1 text-ink">
              <li>Testing LLM consistency across <strong>binary → ternary → quaternary → continuous</strong> scales</li>
              <li>Measuring variance and entropy at each granularity level</li>
              <li>Visualizing where the model's responses start to diverge</li>
              <li>Identifying which semantic dimensions are most unstable</li>
            </ul>
          </div>

          <p className="muted text-sm italic">
            Use this to calibrate your LLM as a judge/labeller pipeline and choose the right labeling
            granularity for your specific use case.
          </p>
        </div>
      </div>

      {/* Analysis Section - Only shown after key is validated */}
      {isKeyValidated && (
        <div className="card">
          <h2 className="section-title mb-4">Analyze Statement</h2>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-ink mb-1">
                Statement to Analyze
              </label>
              <textarea
                className="input-field h-24"
                placeholder="Enter the statement to analyze"
                value={statement}
                onChange={(e) => setStatement(e.target.value)}
                disabled={isLoading}
              />
            </div>

            {/* Question Mode Selector */}
            <div>
              <label className="block text-sm font-medium text-ink mb-2">
                Question Mode
              </label>
              <div className="flex space-x-4">
                <label className={`flex items-center space-x-2 p-3 rounded-ctl border-2 cursor-pointer transition-all ${
                  questionMode === 'mech'
                    ? 'border-gold bg-gold-badge font-semibold'
                    : 'border-hair hover:border-gold'
                } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}>
                  <input
                    type="radio"
                    name="questionMode"
                    value="mech"
                    checked={questionMode === 'mech'}
                    onChange={(e) => setQuestionMode(e.target.value as QuestionMode)}
                    disabled={isLoading}
                    className="accent-gold"
                  />
                  <div>
                    <span className={questionMode === 'mech' ? 'font-semibold text-ink' : 'text-mute'}>Mechanistic</span>
                    <p className="text-xs text-mute">Explicit linguistic features</p>
                  </div>
                </label>
                <label className={`flex items-center space-x-2 p-3 rounded-ctl border-2 cursor-pointer transition-all ${
                  questionMode === 'interp'
                    ? 'border-gold bg-gold-badge font-semibold'
                    : 'border-hair hover:border-gold'
                } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}>
                  <input
                    type="radio"
                    name="questionMode"
                    value="interp"
                    checked={questionMode === 'interp'}
                    onChange={(e) => setQuestionMode(e.target.value as QuestionMode)}
                    disabled={isLoading}
                    className="accent-gold"
                  />
                  <div>
                    <span className={questionMode === 'interp' ? 'font-semibold text-ink' : 'text-mute'}>Interpretability</span>
                    <p className="text-xs text-mute">Implicit meaning & inference</p>
                  </div>
                </label>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {!isLoading ? (
                <button
                  className="btn-primary"
                  onClick={handleAnalyze}
                  disabled={!statement.trim()}
                >
                  Run Analysis
                </button>
              ) : (
                <button
                  className="btn-danger"
                  onClick={handleCancel}
                >
                  Cancel
                </button>
              )}
            </div>
          </div>

          {/* Progress Bar */}
          {isLoading && jobStatus && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm text-mute">
                <span className="font-medium tabular-nums">
                  {getEvaluationsCompleted()} / {TOTAL_EVALUATIONS} evaluations
                </span>
                <span className="tabular-nums">
                  ({jobStatus.progress}%)
                </span>
              </div>
              <div className="w-full bg-hair rounded-full h-3">
                <div
                  className="bg-gold h-3 rounded-full transition-all duration-300"
                  style={{ width: `${(getEvaluationsCompleted() / TOTAL_EVALUATIONS) * 100}%` }}
                />
              </div>
              <div className="flex items-center space-x-2 text-sm text-mute">
                <span>Running…</span>
                <span>
                  Scale: <span className="font-medium capitalize">{jobStatus.current_scale || 'starting'}</span>
                  {jobStatus.current_sample > 0 && (
                    <span> • Sample {jobStatus.current_sample}/20</span>
                  )}
                </span>
              </div>
            </div>
          )}

          {error && (
            <div className="callout-ink">
              <span className="font-semibold">Error:</span> {error}
            </div>
          )}
        </div>
      )}

      {/* Results Section */}
      {result && (
        <>
          {/* Statement + Mode + Usage Row */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Statement */}
            <div className="card lg:col-span-2">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-medium text-mute">Analyzed Statement</h3>
                {result.question_mode && (
                  <span className="chip-accent">
                    {result.question_mode === 'mech' ? 'Mechanistic' : 'Interpretability'}
                  </span>
                )}
              </div>
              <p className="text-lg text-ink italic">"{result.statement}"</p>
            </div>

            {/* Usage Stats */}
            {usage && (
              <div className="card">
                <div className="flex items-center space-x-2 mb-3">
                  <h3 className="panel-title">API Usage</h3>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-mute">Cost</span>
                    <span className="font-bold text-ink tabular-nums">
                      ${usage.cost_usd < 0.01 ? usage.cost_usd.toFixed(4) : usage.cost_usd.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-mute">Total Tokens</span>
                    <span className="font-medium text-ink tabular-nums">
                      {usage.total_tokens.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between text-xs text-mute">
                    <span>Input: {usage.input_tokens.toLocaleString()}</span>
                    <span>Output: {usage.output_tokens.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between pt-1 border-t border-hair">
                    <span className="text-mute">API Calls</span>
                    <span className="font-medium text-ink tabular-nums">{usage.api_calls}</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Unique Vectors by Scale */}
          <div className="card">
            <h3 className="section-title mb-2">Distinct Vector Patterns</h3>
            <p className="muted text-sm mb-4">
              Number of distinct patterns among 20 samples. 1 = all identical (most consistent), 20 = all different (least consistent).
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {SCALE_ORDER.map((scale) => {
                const scaleData = result.identical_counts?.per_scale?.[scale]
                const uniqueCount = scaleData?.unique_vectors ?? '?'
                const totalCount = scaleData?.total_samples ?? 20
                return (
                  <div
                    key={scale}
                    className="metric-card"
                    style={{ backgroundColor: `${SCALE_COLORS[scale]}15` }}
                  >
                    <p className="metric-label capitalize" style={{ color: SCALE_COLORS[scale] }}>
                      {scale}
                    </p>
                    <p className="stat-num" style={{ color: SCALE_COLORS[scale] }}>
                      {uniqueCount}/{totalCount}
                    </p>
                    <p className="text-xs text-mute">distinct patterns</p>
                  </div>
                )
              })}
            </div>
            {result.identical_counts && (
              <div className="mt-4 text-center text-sm text-mute">
                Overall: <span className="font-semibold text-ink">{result.identical_counts.total_unique}</span> distinct patterns out of <span className="font-semibold text-ink">{result.identical_counts.total_samples}</span> total samples
              </div>
            )}
          </div>

          {/* Value Distribution */}
          {result.embeddings_raw && (
            <div className="card">
              <ValueDistribution embeddings={result.embeddings_raw} />
            </div>
          )}

          {/* 3D PCA Plot */}
          <div className="card">
            <h3 className="section-title mb-4">
              3D PCA Visualization
            </h3>
            <p className="muted text-sm mb-4">
              Each point represents one embedding sample. Colors indicate grading scale.
              Tight clusters = consistent, spread = uncertain.
            </p>
            <div className="h-[500px] plot-frame overflow-hidden">
              <PCAPlot3D
                coords={result.pca.coords}
                scaleLabels={result.pca.scale_labels}
                explainedVariance={result.pca.explained_variance_ratio}
              />
            </div>
            <div className="mt-4 flex justify-center space-x-6">
              {SCALE_ORDER.map((scale) => (
                <div key={scale} className="flex items-center space-x-2">
                  <div
                    className="w-4 h-4 rounded-full border border-hair"
                    style={{ backgroundColor: SCALE_COLORS[scale] }}
                  />
                  <span className="text-sm text-mute capitalize">{scale}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Vector Heatmap */}
          {result.embeddings_raw && (
            <div className="card">
              <VectorHeatmap embeddings={result.embeddings_raw} />
            </div>
          )}

          {/* Metrics Table */}
          <div className="card">
            <h3 className="section-title mb-4">
              Detailed Metrics
            </h3>
            <MetricsTable scaleMetrics={result.scale_metrics} />
          </div>

          {/* Top Loading Questions */}
          <div className="card">
            <h3 className="section-title mb-4">
              Top Contributing Questions per PC
            </h3>

            {/* PCA Explanation */}
            <div className="callout text-sm mb-4">
              <p className="font-semibold text-ink mb-2">How PCA Works:</p>
              <ul className="text-ink space-y-1 list-disc list-inside">
                <li>Each Principal Component (PC) is a <strong>linear combination of ALL 20 questions</strong></li>
                <li>Every question has a "loading" (coefficient) showing its weight in that PC</li>
                <li>We show the <strong>top 5 questions with highest absolute loadings</strong> - these have the strongest influence on that PC</li>
                <li><span className="text-gold font-mono">+</span> loading = question score increases along this PC direction</li>
                <li><span className="text-ink font-mono">−</span> loading = question score decreases along this PC direction</li>
              </ul>
              <p className="mt-2 muted text-xs">
                Example: PC1 = 0.4×Q3 + 0.35×Q7 + 0.25×Q12 + ... (all 20 questions contribute, but Q3, Q7, Q12 dominate)
              </p>
            </div>
            <div className="space-y-6">
              {['PC1', 'PC2', 'PC3'].map((pc, idx) => (
                <div key={pc}>
                  <h4 className="font-semibold text-ink mb-2">
                    {pc} ({(result.pca.explained_variance_ratio[idx] * 100).toFixed(1)}% variance)
                  </h4>
                  <div className="space-y-2">
                    {result.pca.top_loading_questions[pc]?.slice(0, 5).map((q, i) => (
                      <div
                        key={q.id}
                        className="flex items-start space-x-3 text-sm bg-cream border border-hair p-2 rounded-ctl"
                      >
                        <span className="text-mute w-4 tabular-nums">{i + 1}.</span>
                        <span className="flex-1 text-ink">{q.question}</span>
                        <span
                          className={`font-mono tabular-nums ${
                            q.loading > 0 ? 'text-gold' : 'text-ink'
                          }`}
                        >
                          {q.loading > 0 ? '+' : ''}{q.loading.toFixed(3)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
