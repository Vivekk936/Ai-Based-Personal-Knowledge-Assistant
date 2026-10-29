import { useEffect, useRef, useState } from 'react'
import { getHealth, type HealthStatus } from '../api'

export type ConnectionState = 'connecting' | 'connected' | 'degraded' | 'reconnecting' | 'crashed'

/** Merges two signals: HTTP /health polling (is the server answering, and is the
 * model backend behind it reachable) and the Electron main process's IPC events
 * for its own process supervision (backend restarted, or gave up after too many
 * crashes). Either one alone is an incomplete picture. */
export function useBackendStatus(pollMs = 4000) {
  const [state, setState] = useState<ConnectionState>('connecting')
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const everConnected = useRef(false)

  useEffect(() => {
    ;(window as any).ada?.onBackendStatus?.((s: { status: string; attempt?: number }) => {
      setState(s.status === 'crashed' ? 'crashed' : s.status === 'connected' ? 'connected' : 'reconnecting')
    })
  }, [])

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>

    async function poll() {
      try {
        const result = await getHealth()
        if (cancelled) return
        everConnected.current = true
        setHealth(result)
        setState(result.status === 'ok' ? 'connected' : 'degraded')
      } catch {
        if (cancelled) return
        setState((prev) => (prev === 'crashed' ? prev : everConnected.current ? 'reconnecting' : 'connecting'))
      } finally {
        if (!cancelled) timer = setTimeout(poll, pollMs)
      }
    }

    poll()
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [pollMs])

  return { state, health }
}
