const BASE_URL = (window as any).ada?.backendUrl ?? (import.meta.env?.VITE_BACKEND_URL || 'http://127.0.0.1:8000')

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.error || body.detail || `Request to ${path} failed (${res.status})`)
  }
  return res.json()
}

export interface HealthStatus {
  status: 'ok' | 'degraded'
  model_backend_reachable: boolean
  detail: string
  chat_model: string
  vision_model: string
  embed_model: string
}

export function getHealth() {
  return request<HealthStatus>('/health')
}

// --- Chat sessions ---------------------------------------------------------

export interface ChatSession {
  id: string
  title: string
  updated_at: number
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface ChatResult {
  answer: string
  unverified_citations: string[]
}

export function listSessions() {
  return request<{ sessions: ChatSession[] }>('/sessions')
}

export function createSession() {
  return request<ChatSession>('/sessions', { method: 'POST' })
}

export function getSessionMessages(sessionId: string) {
  return request<{ messages: ChatMessage[] }>(`/sessions/${sessionId}/messages`)
}

export function renameSession(sessionId: string, title: string) {
  return request<ChatSession>(`/sessions/${sessionId}`, {
    method: 'PATCH',
    body: JSON.stringify({ title }),
  })
}

export function deleteSession(sessionId: string) {
  return request<{ deleted: string }>(`/sessions/${sessionId}`, { method: 'DELETE' })
}

export function sendChat(sessionId: string, message: string) {
  return request<ChatResult>('/chat', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, message }),
  })
}

/** One event from the agent loop. `t` discriminates; the rest depends on it. */
export interface AgentEvent {
  t: 'start' | 'step' | 'token' | 'tool_start' | 'tool_end' | 'verify' | 'artifact' | 'done' | 'error'
  d?: string
  n?: number
  of?: number
  name?: string
  summary?: string
  ms?: number
  read_only?: boolean
  answer?: string
  reason?: string
  citations?: string[]
  artifacts?: string[]
  message?: string
}

/**
 * Streams a chat turn over SSE. The local model generates at roughly 18 tokens/second,
 * so waiting for the whole answer means several seconds of blank screen — long enough
 * that users assume it has hung. Tool activity streams too, which is what tells someone
 * the agent is searching the right thing and lets them stop it early if it isn't.
 */
export async function streamChat(
  sessionId: string,
  message: string,
  onEvent: (e: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${BASE_URL}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message }),
    signal,
  })
  if (!res.ok || !res.body) throw new Error(`Chat failed (${res.status})`)

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    // SSE frames are separated by a blank line, and a frame can straddle two chunks —
    // so only complete frames are consumed and the remainder stays buffered.
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''
    for (const frame of frames) {
      const line = frame.split('\n').find((l) => l.startsWith('data: '))
      if (!line) continue
      try {
        onEvent(JSON.parse(line.slice(6)) as AgentEvent)
      } catch {
        // A malformed frame is not worth aborting a live answer over.
      }
    }
  }
}

// --- Document upload ---------------------------------------------------------

export interface IngestResult {
  filename: string
  state: 'indexed' | 'skipped' | 'unsupported' | 'error'
  pages: number
  chunks: number
  error?: string | null
  doc_id?: string
}

export async function ingestFiles(files: File[]): Promise<{ ingested: IngestResult[] }> {
  const form = new FormData()
  for (const file of files) form.append('files', file)
  const res = await fetch(`${BASE_URL}/ingest`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Ingest failed (${res.status})`)
  return res.json()
}

// --- Folders -----------------------------------------------------------------
// Adding a folder does everything server-side in one call: grant, scan, index its
// content for search, and generate organization suggestions.

export interface ScanResult {
  root: string
  queued: number
  removed: number
  total_files: number
  pending: number
  seconds: number
}

/** Live state of the background indexer. Indexing is a queue that drains on its own, so
 *  this is progress to display, not a result to wait for. */
export interface IndexStatus {
  running: boolean
  current_file: string | null
  pending: number
  indexed: number
  unsupported: number
  error: number
  skipped: number
  total: number
  chunks: number
}

export interface Suggestion {
  path: string
  suggested_destination: string
  reason: string
}

export interface FolderRefreshResult {
  scan: ScanResult
  index: IndexStatus
  suggestions: Suggestion[]
  suggestions_total: number
  suggestions_truncated: boolean
  dedup_skipped: boolean
}

export function getFolders() {
  return request<{ granted: string[]; index: IndexStatus }>('/folders')
}

export function getIndexStatus(folder?: string) {
  const q = folder ? `?folder=${encodeURIComponent(folder)}` : ''
  return request<IndexStatus>(`/index/status${q}`)
}

export function pickFolder() {
  return request<{ picked: string | null; granted: string[] }>('/folders/pick', { method: 'POST' })
}

export function refreshFolder(folder: string) {
  return request<FolderRefreshResult>('/folders/refresh', {
    method: 'POST',
    body: JSON.stringify({ folder }),
  })
}

export function revokeFolder(folder: string) {
  return request<{ granted: string[] }>(`/folders?folder=${encodeURIComponent(folder)}`, {
    method: 'DELETE',
  })
}

export interface ApplyResult {
  applied: number
  skipped: number
  errors: string[]
}

export function applyMoves(rows: Suggestion[]) {
  return request<ApplyResult>('/apply', { method: 'POST', body: JSON.stringify({ rows }) })
}

// --- RAG Evaluation Benchmark ------------------------------------------------

export interface RagEvalCase {
  id: string
  category: string
  query: string
  reference_answer?: string
  generated_answer?: string
  citations?: string[]
  retrieval?: {
    hit_at_1?: number
    hit_at_3?: number
    mrr?: number
    is_out_of_corpus?: boolean
  }
  correctness?: {
    correctness?: number
    fact_recall?: number
    token_f1?: number
  }
  faithfulness?: {
    faithfulness?: number
    hallucination_rate?: number
  }
  timing_ms?: {
    total?: number
    retrieval?: number
    generation?: number
  }
}

export interface RagEvalReport {
  status: string
  message?: string
  benchmark_summary?: {
    source_pdf?: string
    total_queries?: number
    mode?: string
    timestamp?: string
  }
  retrieval_accuracy?: {
    hit_rate_at_1?: number
    hit_rate_at_3?: number
    mean_mrr?: number
  }
  correctness?: {
    overall_correctness?: number
    mean_fact_recall?: number
    mean_token_f1?: number
  }
  faithfulness?: {
    mean_faithfulness?: number
    mean_hallucination_rate?: number
    total_unsupported_numbers?: number
  }
  response_time_ms?: {
    total?: {
      mean_ms?: number
      median_ms?: number
    }
    retrieval?: {
      mean_ms?: number
    }
  }
  individual_cases?: RagEvalCase[]
}

export function getEvalRag(forceRefresh = false) {
  const q = forceRefresh ? '?force_refresh=true' : ''
  return request<RagEvalReport>(`/eval/rag${q}`)
}

export async function uploadEvalPdf(file: File): Promise<RagEvalReport> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE_URL}/eval/rag/upload`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Upload and eval failed (${res.status})`)
  return res.json()
}
