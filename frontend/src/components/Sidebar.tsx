import { useState } from 'react'
import type { ChatSession } from '../api'

interface Props {
  sessions: ChatSession[]
  activeSessionId: string | null
  organizeActive: boolean
  evalActive: boolean
  onNewChat: () => void
  onSelectSession: (id: string) => void
  onDeleteSession: (id: string) => void
  onRenameSession: (id: string, title: string) => void
  onSelectOrganize: () => void
  onSelectEval: () => void
}

export function Sidebar({
  sessions,
  activeSessionId,
  organizeActive,
  evalActive,
  onNewChat,
  onSelectSession,
  onDeleteSession,
  onRenameSession,
  onSelectOrganize,
  onSelectEval,
}: Props) {
  const [editingId, setEditingId] = useState<string | null>(null)
  const [draft, setDraft] = useState('')

  function startEditing(s: ChatSession) {
    setEditingId(s.id)
    setDraft(s.title)
  }

  function commitEdit() {
    if (editingId && draft.trim()) onRenameSession(editingId, draft.trim())
    setEditingId(null)
  }

  return (
    <nav className="flex w-64 flex-col border-r border-neutral-800/80 bg-neutral-950 p-3">
      <div className="mb-5 flex items-center gap-2 px-1">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent text-sm font-semibold text-neutral-950">
          A
        </div>
        <div>
          <h1 className="text-sm font-semibold leading-none text-neutral-100">Ada</h1>
          <p className="mt-0.5 text-[11px] leading-none text-neutral-600">your documents, searchable</p>
        </div>
      </div>

      <button
        onClick={onNewChat}
        className="mb-1.5 flex items-center gap-1.5 rounded-lg bg-accent px-3 py-2 text-left text-sm font-medium text-neutral-950 transition hover:bg-accent-strong active:scale-[0.98]"
      >
        <span className="text-base leading-none">+</span> New chat
      </button>
      <button
        onClick={onSelectOrganize}
        className={`mb-1.5 rounded-lg px-3 py-2 text-left text-sm transition ${
          organizeActive
            ? 'bg-neutral-800/80 text-neutral-100'
            : 'text-neutral-400 hover:bg-neutral-900 hover:text-neutral-200'
        }`}
      >
        Organize files
      </button>
      <button
        onClick={onSelectEval}
        className={`mb-4 rounded-lg px-3 py-2 text-left text-sm transition ${
          evalActive
            ? 'bg-neutral-800/80 text-neutral-100'
            : 'text-neutral-400 hover:bg-neutral-900 hover:text-neutral-200'
        }`}
      >
        RAG Benchmark
      </button>

      <p className="mb-1.5 px-2 text-[11px] font-medium tracking-wide text-neutral-600 uppercase">Chats</p>
      <div className="flex-1 overflow-y-auto">
        {sessions.length === 0 && (
          <p className="px-2 py-1 text-xs leading-relaxed text-neutral-600">
            No chats yet — start one above.
          </p>
        )}
        {sessions.map((s) => {
          const active = !organizeActive && activeSessionId === s.id
          return (
            <div
              key={s.id}
              className={`group relative mb-0.5 flex items-center rounded-lg px-2 py-1.5 text-sm transition ${
                active ? 'bg-neutral-800/80 text-neutral-100' : 'text-neutral-400 hover:bg-neutral-900 hover:text-neutral-200'
              }`}
            >
              {active && <span className="absolute left-0 h-4 w-0.5 rounded-full bg-accent" />}
              {editingId === s.id ? (
                <input
                  autoFocus
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onBlur={commitEdit}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') commitEdit()
                    if (e.key === 'Escape') setEditingId(null)
                  }}
                  className="flex-1 rounded bg-neutral-900 px-1 text-sm text-white outline-none"
                />
              ) : (
                <button
                  onClick={() => onSelectSession(s.id)}
                  onDoubleClick={() => startEditing(s)}
                  className="flex-1 truncate text-left"
                  title="Double-click to rename"
                >
                  {s.title}
                </button>
              )}
              {editingId !== s.id && (
                <button
                  onClick={() => onDeleteSession(s.id)}
                  className="ml-1 hidden shrink-0 rounded p-0.5 text-neutral-600 transition hover:bg-neutral-800 hover:text-red-400 group-hover:block"
                  title="Delete chat"
                >
                  ×
                </button>
              )}
            </div>
          )
        })}
      </div>
    </nav>
  )
}
