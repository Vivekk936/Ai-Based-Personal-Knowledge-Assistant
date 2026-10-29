import { useEffect, useRef, useState, type JSX } from 'react'
import { getSessionMessages, ingestFiles, streamChat, type AgentEvent, type ChatMessage } from '../api'

interface ActivityItem {
  id: number
  label: string
  status: 'running' | 'done'
  ms?: number
}

interface DisplayMessage extends ChatMessage {
  citations?: string[]
  artifacts?: string[]
  reason?: string
  toolTrail?: ActivityItem[]
}

/** What the agent is doing right now, phrased for a person rather than a log. */
const TOOL_LABELS: Record<string, string> = {
  search_documents: 'Searching documents',
  read_chunk: 'Reading passage',
  list_folder: 'Listing folder',
  read_file: 'Reading file',
  grep_files: 'Searching files',
  write_note: 'Saving a note',
  write_deliverable: 'Writing file',
}

const draftKey = (sessionId: string) => `ada:draft:${sessionId}`

/** Minimal markdown-ish rendering: fenced code, inline code, bold, lists, headings.
 *  No markdown dependency — this is the whole formatting surface the model actually uses. */
function renderContent(text: string): JSX.Element {
  const blocks = text.split(/```/)
  return (
    <>
      {blocks.map((block, i) =>
        i % 2 === 1 ? (
          <pre key={i} className="my-1.5 overflow-x-auto rounded bg-black/40 p-2 text-xs">
            <code>{block.replace(/^[^\n]*\n/, '')}</code>
          </pre>
        ) : (
          <span key={i}>{renderParagraphs(block)}</span>
        ),
      )}
    </>
  )
}

function renderParagraphs(text: string): JSX.Element[] {
  return text.split(/\n{2,}/).map((para, pi) => {
    const lines = para.split('\n').filter(Boolean)
    if (lines.length && lines.every((l) => /^[-*]\s+/.test(l))) {
      return (
        <ul key={pi} className="my-1 list-disc pl-5">
          {lines.map((l, li) => (
            <li key={li}>{renderInline(l.replace(/^[-*]\s+/, ''))}</li>
          ))}
        </ul>
      )
    }
    if (lines.length && lines.every((l) => /^\d+\.\s+/.test(l))) {
      return (
        <ol key={pi} className="my-1 list-decimal pl-5">
          {lines.map((l, li) => (
            <li key={li}>{renderInline(l.replace(/^\d+\.\s+/, ''))}</li>
          ))}
        </ol>
      )
    }
    const heading = para.match(/^(#{1,3})\s+(.*)$/)
    if (heading) {
      const level = heading[1].length
      const cls = 'mt-2 mb-1 font-semibold ' + (level === 1 ? 'text-base' : 'text-sm')
      return (
        <p key={pi} className={cls}>
          {renderInline(heading[2])}
        </p>
      )
    }
    return (
      <p key={pi} className={pi > 0 ? 'mt-2' : ''}>
        {para.split('\n').map((line, li) => (
          <span key={li}>
            {li > 0 && <br />}
            {renderInline(line)}
          </span>
        ))}
      </p>
    )
  })
}

function renderInline(text: string): (string | JSX.Element)[] {
  return text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g).map((part, i) => {
    if (part.startsWith('`') && part.endsWith('`')) {
      return (
        <code key={i} className="rounded bg-black/30 px-1 py-0.5 text-xs">
          {part.slice(1, -1)}
        </code>
      )
    }
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i}>{part.slice(2, -2)}</strong>
    }
    return part
  })
}

/** The agent's tool/reasoning trail for one turn, rendered as a small timeline
 *  instead of a flat log. `live` styles the still-running item differently. */
function ActivityTrail({ items, live }: { items: ActivityItem[]; live?: boolean }) {
  if (!items.length) return null
  return (
    <ul className="flex flex-col gap-1 border-l border-neutral-700 pl-2.5">
      {items.map((a) => (
        <li key={a.id} className="flex items-baseline justify-between gap-3 text-xs">
          <span className="flex items-center gap-1.5">
            {a.status === 'running' && (
              <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-accent" />
            )}
            <span className={a.status === 'running' ? 'text-neutral-300' : 'text-neutral-500'}>
              {a.status === 'running' && live ? `${a.label}…` : a.label}
            </span>
          </span>
          {a.ms != null && <span className="shrink-0 text-neutral-600">{(a.ms / 1000).toFixed(1)}s</span>}
        </li>
      ))}
    </ul>
  )
}

function TypingIndicator() {
  return (
    <span className="flex items-center gap-1 py-0.5">
      <span className="typing-dot h-1.5 w-1.5 rounded-full bg-neutral-400" style={{ animationDelay: '0ms' }} />
      <span className="typing-dot h-1.5 w-1.5 rounded-full bg-neutral-400" style={{ animationDelay: '150ms' }} />
      <span className="typing-dot h-1.5 w-1.5 rounded-full bg-neutral-400" style={{ animationDelay: '300ms' }} />
    </span>
  )
}

function Avatar({ role }: { role: 'user' | 'assistant' }) {
  if (role === 'assistant') {
    return (
      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-accent text-xs font-semibold text-neutral-950">
        A
      </div>
    )
  }
  return (
    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-neutral-700 text-xs font-semibold text-neutral-300">
      U
    </div>
  )
}

export function ChatPanel({ sessionId, onFirstMessage }: { sessionId: string; onFirstMessage: () => void }) {
  const [messages, setMessages] = useState<DisplayMessage[]>([])
  const [input, setInput] = useState(() => {
    try {
      return sessionStorage.getItem(draftKey(sessionId)) ?? ''
    } catch {
      return ''
    }
  })
  const [sending, setSending] = useState(false)
  const [live, setLive] = useState('')
  const [activity, setActivity] = useState<ActivityItem[]>([])
  const [ingestStatus, setIngestStatus] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    getSessionMessages(sessionId).then((r) => setMessages(r.messages))
  }, [sessionId])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, live, activity])

  // A run in flight belongs to the session it started in; switching away should not
  // leave a stream writing into a conversation the user is no longer looking at.
  useEffect(() => () => abortRef.current?.abort(), [sessionId])

  function handleInputChange(value: string) {
    setInput(value)
    try {
      if (value) sessionStorage.setItem(draftKey(sessionId), value)
      else sessionStorage.removeItem(draftKey(sessionId))
    } catch {
      // Private browsing / storage disabled — losing the draft is fine, not fatal.
    }
  }

  async function handleSend() {
    const text = input.trim()
    if (!text || sending) return
    const wasEmpty = messages.length === 0
    setMessages((prev) => [...prev, { role: 'user', content: text }])
    handleInputChange('')
    setSending(true)
    setLive('')
    setActivity([])

    const controller = new AbortController()
    abortRef.current = controller
    let answer = ''
    let trail: ActivityItem[] = []
    let stepId = 0

    const onEvent = (e: AgentEvent) => {
      switch (e.t) {
        case 'token':
          answer += e.d ?? ''
          setLive(answer)
          break
        case 'tool_start':
          trail = [...trail, { id: stepId++, label: TOOL_LABELS[e.name ?? ''] ?? e.name ?? 'Working', status: 'running' }]
          setActivity(trail)
          break
        case 'tool_end':
          trail = trail.map((a, i) => (i === trail.length - 1 ? { ...a, status: 'done', ms: e.ms } : a))
          setActivity(trail)
          break
        case 'verify':
          trail = [...trail, { id: stepId++, label: 'Checking the answer against the sources', status: 'done' }]
          setActivity(trail)
          break
        case 'artifact':
          trail = [...trail, { id: stepId++, label: e.summary ?? 'File written', status: 'done' }]
          setActivity(trail)
          break
        case 'done':
          setMessages((prev) => [
            ...prev,
            {
              role: 'assistant',
              content: e.answer ?? answer,
              citations: e.citations,
              artifacts: e.artifacts,
              reason: e.reason,
              toolTrail: trail,
            },
          ])
          setLive('')
          break
        case 'error':
          setMessages((prev) => [
            ...prev,
            { role: 'assistant', content: `Something went wrong: ${e.message}`, toolTrail: trail },
          ])
          setLive('')
          break
      }
    }

    try {
      await streamChat(sessionId, text, onEvent, controller.signal)
      if (wasEmpty) onFirstMessage()
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: `Something went wrong: ${(err as Error).message}` },
        ])
      }
    } finally {
      setSending(false)
      setActivity([])
      abortRef.current = null
    }
  }

  function handleStop() {
    abortRef.current?.abort()
  }

  async function handleFiles(fileList: FileList | null) {
    if (!fileList || fileList.length === 0) return
    setIngestStatus('Reading files...')
    try {
      const { ingested } = await ingestFiles(Array.from(fileList))
      setIngestStatus(
        ingested
          .map((f) => {
            if (f.state === 'unsupported') return `${f.filename}: unsupported file type`
            if (f.state === 'error') return `${f.filename}: ${f.error}`
            if (f.state === 'skipped') return `${f.filename}: ${f.error ?? 'skipped'}`
            return `${f.filename}: ${f.pages} pages, ${f.chunks} chunks indexed`
          })
          .join(' · '),
      )
    } catch (err) {
      setIngestStatus(`Ingest failed: ${(err as Error).message}`)
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-neutral-800/80 px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold text-neutral-100">Document Q&amp;A</h2>
          <p className="text-xs text-neutral-500">Ask anything about what you've shared with Ada.</p>
        </div>
        <label className="flex cursor-pointer items-center gap-1.5 rounded-lg border border-neutral-700 px-3 py-1.5 text-xs text-neutral-200 transition hover:border-accent hover:text-accent active:scale-[0.98]">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 3v12m0-12 4 4m-4-4-4 4M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
          </svg>
          Upload files
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.xlsx,.xlsm,.pptx,.csv,.tsv,.txt,.md,.json,.html,.png,.jpg,.jpeg"
            multiple
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
        </label>
      </div>
      {ingestStatus && <div className="border-b border-neutral-800/80 px-4 py-2 text-xs text-neutral-500">{ingestStatus}</div>}

      <div className="flex-1 overflow-y-auto px-4 py-4">
        {messages.length === 0 && !sending && (
          <div className="mt-12 flex flex-col items-center gap-3 text-center">
            <div className="flex h-11 w-11 items-center justify-center rounded-full bg-accent-soft text-accent">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
            </div>
            <p className="max-w-xs text-sm leading-relaxed text-neutral-400">
              Ask about anything in your uploaded files or granted folders. Every answer cites the
              document and page it came from, and says so when the documents don't cover your question.
            </p>
          </div>
        )}
        <div className="flex flex-col gap-4">
          {messages.map((m, i) => (
            <div key={i} className={`flex items-end gap-2 ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              {m.role === 'assistant' && <Avatar role="assistant" />}
              <div
                className={`max-w-[75%] rounded-2xl px-3.5 py-2.5 text-sm shadow-sm ${
                  m.role === 'user'
                    ? 'bg-accent text-neutral-950 whitespace-pre-wrap'
                    : 'bg-neutral-800/90 text-neutral-100'
                }`}
              >
                {!!m.toolTrail?.length && (
                  <details className="mb-2">
                    <summary className="cursor-pointer select-none text-xs text-neutral-500 hover:text-neutral-300">
                      {m.toolTrail.length} step{m.toolTrail.length > 1 ? 's' : ''}
                    </summary>
                    <div className="mt-1">
                      <ActivityTrail items={m.toolTrail} />
                    </div>
                  </details>
                )}
                {m.role === 'user' ? m.content : renderContent(m.content)}
                {!!m.artifacts?.length && (
                  <div className="mt-2 rounded-lg bg-emerald-900/40 px-2 py-1 text-xs text-emerald-300">
                    {m.artifacts.join(' · ')}
                  </div>
                )}
                {!!m.citations?.length && (
                  <div className="mt-2 text-xs text-neutral-400">Sources: {m.citations.join(' ')}</div>
                )}
                {m.reason && m.reason !== 'complete' && (
                  <div className="mt-2 rounded-lg bg-amber-900/40 px-2 py-1 text-xs text-amber-300">
                    Stopped early ({m.reason.replace(/_/g, ' ')}) — this answer may be incomplete.
                  </div>
                )}
              </div>
              {m.role === 'user' && <Avatar role="user" />}
            </div>
          ))}

          {/* The answer as it is generated, plus what the agent is doing to produce it.
              A local model is slow enough that a bare spinner reads as a hang. */}
          {sending && (
            <div className="flex items-end justify-start gap-2">
              <Avatar role="assistant" />
              <div className="max-w-[75%] rounded-2xl bg-neutral-800/90 px-3.5 py-2.5 text-sm text-neutral-100 shadow-sm">
                {activity.length > 0 && (
                  <div className="mb-1.5">
                    <ActivityTrail items={activity} live />
                  </div>
                )}
                {live ? renderContent(live) : <TypingIndicator />}
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      {ingestStatus && (
        <div className="border-t border-neutral-800 bg-neutral-900/80 px-4 py-1.5 text-xs text-neutral-300">
          {ingestStatus}
        </div>
      )}
      <div className="flex items-center gap-2 border-t border-neutral-800/80 p-3">
        <input
          type="file"
          ref={fileInputRef}
          multiple
          onChange={(e) => handleFiles(e.target.files)}
          className="hidden"
          accept=".pdf,.xlsx,.xls,.csv,.docx,.pptx,.txt,.md,.py,.json"
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          aria-label="Upload documents"
          title="Upload documents to chat (PDF, XLSX, DOCX, etc.)"
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-neutral-700 bg-neutral-900 text-neutral-400 transition hover:border-neutral-500 hover:text-neutral-100 active:scale-[0.96]"
        >
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
          </svg>
        </button>
        <input
          value={input}
          onChange={(e) => handleInputChange(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Ask about your documents, or attach files..."
          className="flex-1 rounded-full border border-neutral-700 bg-neutral-900 px-4 py-2.5 text-sm text-neutral-100 outline-none transition focus:border-accent"
        />
        <button
          onClick={sending ? handleStop : handleSend}
          aria-label={sending ? 'Stop' : 'Send'}
          className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-full transition active:scale-[0.96] ${
            sending ? 'bg-neutral-700 text-white hover:bg-neutral-600' : 'bg-accent text-neutral-950 hover:bg-accent-strong'
          }`}
        >
          {sending ? (
            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
              <rect x="5" y="5" width="14" height="14" rx="2" />
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 19V5M5 12l7-7 7 7" />
            </svg>
          )}
        </button>
      </div>
    </div>
  )
}
