import { useEffect, useState } from 'react'
import { createSession, deleteSession, listSessions, renameSession, type ChatSession } from './api'
import { ChatPanel } from './components/ChatPanel'
import { EvalPanel } from './components/EvalPanel'
import { LoadingScreen } from './components/LoadingScreen'
import { OrganizePanel } from './components/OrganizePanel'
import { Sidebar } from './components/Sidebar'
import { StatusBar } from './components/StatusBar'
import { useBackendStatus } from './hooks/useBackendStatus'

type View = { kind: 'chat'; sessionId: string } | { kind: 'organize' } | { kind: 'eval' }

export default function App() {
  const { state, health } = useBackendStatus()
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [view, setView] = useState<View | null>(null)

  const ready = state === 'connected' || state === 'degraded'

  useEffect(() => {
    if (!ready) return
    refreshSessions()
  }, [ready])

  async function refreshSessions() {
    const { sessions } = await listSessions()
    setSessions(sessions)
    setView((current) => {
      if (current) return current
      return sessions.length > 0 ? { kind: 'chat', sessionId: sessions[0].id } : null
    })
  }

  async function handleNewChat() {
    const session = await createSession()
    setSessions((prev) => [session, ...prev])
    setView({ kind: 'chat', sessionId: session.id })
  }

  async function handleRenameSession(id: string, title: string) {
    const updated = await renameSession(id, title)
    setSessions((prev) => prev.map((s) => (s.id === id ? updated : s)))
  }

  async function handleDeleteSession(id: string) {
    await deleteSession(id)
    const next = sessions.filter((s) => s.id !== id)
    setSessions(next)
    setView((current) => {
      if (current?.kind === 'chat' && current.sessionId === id) {
        return next.length > 0 ? { kind: 'chat', sessionId: next[0].id } : null
      }
      return current
    })
  }

  return (
    <div className="flex h-screen bg-neutral-950 text-neutral-100">
      <Sidebar
        sessions={sessions}
        activeSessionId={view?.kind === 'chat' ? view.sessionId : null}
        organizeActive={view?.kind === 'organize'}
        evalActive={view?.kind === 'eval'}
        onNewChat={handleNewChat}
        onSelectSession={(id) => setView({ kind: 'chat', sessionId: id })}
        onDeleteSession={handleDeleteSession}
        onRenameSession={handleRenameSession}
        onSelectOrganize={() => setView({ kind: 'organize' })}
        onSelectEval={() => setView({ kind: 'eval' })}
      />

      <div className="flex flex-1 flex-col">
        {!ready ? (
          <LoadingScreen variant={state === 'crashed' ? 'crashed' : state === 'reconnecting' ? 'reconnecting' : 'connecting'} />
        ) : view?.kind === 'organize' ? (
          <OrganizePanel />
        ) : view?.kind === 'eval' ? (
          <EvalPanel />
        ) : view?.kind === 'chat' ? (
          <ChatPanel key={view.sessionId} sessionId={view.sessionId} onFirstMessage={refreshSessions} />
        ) : (
          <div className="flex flex-1 items-center justify-center text-sm text-neutral-600">
            Start a new chat to ask about your documents.
          </div>
        )}
        <StatusBar state={state} health={health} />
      </div>
    </div>
  )
}
