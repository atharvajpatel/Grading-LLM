import axios from 'axios'

// Use environment variable for API URL, fallback to localhost for dev
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface Question {
  id: string
  family: string
  question: string
  rubric?: Record<string, string>
  minimal_pairs?: {
    positive: string
    negative: string
  }
}

export interface ScaleMetrics {
  avg_variance: number
  avg_consistency: number
  avg_entropy: number
}

export interface PCAData {
  coords: number[][]
  scale_labels: string[]
  explained_variance_ratio: number[]
  top_loading_questions: Record<string, Array<{
    id: string
    question: string
    loading: number
  }>>
}

export interface ScaleIdenticalCounts {
  total_samples: number
  unique_vectors: number
  identical_count: number
}

export interface IdenticalCounts {
  per_scale: Record<string, ScaleIdenticalCounts>
  total_unique: number
  total_samples: number
}

export interface AnalysisResult {
  run_id: string
  statement: string
  question_mode?: QuestionMode
  question_mode_description?: string
  scale_metrics: Record<string, ScaleMetrics>
  pca: PCAData
  aggregate: {
    total_samples: number
    n_questions: number
    n_scales: number
    overall_avg_variance: number
    overall_avg_consistency: number
  }
  identical_counts?: IdenticalCounts
  embeddings_raw?: Record<string, number[][]>  // scale -> n_samples x n_questions
}

export type JobStatus = 'pending' | 'running' | 'completed' | 'cancelled' | 'failed'

export interface UsageStats {
  input_tokens: number
  output_tokens: number
  total_tokens: number
  api_calls: number
  cost_usd: number
}

export interface JobStatusResponse {
  job_id: string
  status: JobStatus
  progress: number
  current_scale: string
  current_sample: number
  total_samples: number
  result: AnalysisResult | null
  error: string | null
  usage: UsageStats | null
}

export interface ValidateKeyRequest {
  api_key: string
}

export interface ValidateKeyResponse {
  valid: boolean
  error?: string
}

export type QuestionMode = 'mech' | 'interp'

export interface StartJobRequest {
  statement: string
  api_key?: string  // Required in production, optional in local dev
  n_samples?: number
  question_mode?: QuestionMode
}

export interface QuestionModesResponse {
  modes: QuestionMode[]
  default: QuestionMode
  descriptions: Record<QuestionMode, string>
}

export interface AppConfig {
  require_api_key: boolean
  mode: 'production' | 'local_dev' | 'local_dev_no_key'
}

export interface StartJobResponse {
  job_id: string
  status: string
}

// API Key Validation
export async function validateApiKey(apiKey: string): Promise<ValidateKeyResponse> {
  const response = await api.post<ValidateKeyResponse>('/validate-key', { api_key: apiKey })
  return response.data
}

// Job-based API
export async function startJob(request: StartJobRequest): Promise<StartJobResponse> {
  const response = await api.post<StartJobResponse>('/jobs/start', request)
  return response.data
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const response = await api.get<JobStatusResponse>(`/jobs/${jobId}`)
  return response.data
}

export async function cancelJob(jobId: string): Promise<{ status: string }> {
  const response = await api.post<{ status: string }>(`/jobs/${jobId}/cancel`)
  return response.data
}

export async function deleteJob(jobId: string): Promise<{ status: string }> {
  const response = await api.delete<{ status: string }>(`/jobs/${jobId}`)
  return response.data
}

// Questions API
export async function getQuestions(mode?: QuestionMode): Promise<Question[]> {
  const params = mode ? { mode } : {}
  const response = await api.get<{ questions: Question[] }>('/questions', { params })
  return response.data.questions
}

// Question modes API
export async function getQuestionModes(): Promise<QuestionModesResponse> {
  const response = await api.get<QuestionModesResponse>('/question-modes')
  return response.data
}

// Health check
export async function checkHealth(): Promise<{ status: string }> {
  const response = await api.get<{ status: string }>('/health')
  return response.data
}

// App config (check if API key is required)
export async function getAppConfig(): Promise<AppConfig> {
  const response = await api.get<AppConfig>('/config')
  return response.data
}

export const SCALE_COLORS: Record<string, string> = {
  binary: '#2196F3',
  ternary: '#4CAF50',
  quaternary: '#FF9800',
  continuous: '#E91E63',
}

export const SCALE_ORDER = ['binary', 'ternary', 'quaternary', 'continuous']
