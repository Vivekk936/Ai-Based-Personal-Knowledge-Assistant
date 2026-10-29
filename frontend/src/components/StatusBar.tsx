import type { ConnectionState } from '../hooks/useBackendStatus'
import type { HealthStatus } from '../api'

const LABELS: Record<ConnectionState, string> = {
  connecting: 'Starting Ada...',
  connected: 'Connected',
  degraded: 'Model backend unreachable',
  reconnecting: 'Reconnecting...',
  crashed: 'Backend crashed',
}

const DOT_COLOR: Record<ConnectionState, string> = {
  connecting: 'bg-amber-400',
  connected: 'bg-emerald-400',
  degraded: 'bg-amber-400',
  reconnecting: 'bg-amber-400',
  crashed: 'bg-red-400',
}

export function StatusBar({ state, health }: { state: ConnectionState; health: HealthStatus | null }) {
  return (
    <div className="flex items-center gap-2 border-t border-neutral-800/80 px-4 py-2 text-xs text-neutral-400">
      <span className={`h-2 w-2 rounded-full ${DOT_COLOR[state]} ${state === 'connected' ? '' : 'animate-pulse'}`} />
      <span>{LABELS[state]}</span>
      {health && <span className="text-neutral-600">· {health.chat_model}</span>}
    </div>
  )
}
