import { useEffect, useRef, useState } from 'react'
import type { EpisodeProgress } from '@/types/api'

const MAX_RECONNECT_DELAY = 30000 // 30 seconds
const INITIAL_RECONNECT_DELAY = 1000 // 1 second

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [progressMap, setProgressMap] = useState<Map<number, EpisodeProgress>>(() => new Map())
  const reconnectTimeoutRef = useRef<number | undefined>(undefined)
  const reconnectDelayRef = useRef(INITIAL_RECONNECT_DELAY)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true

    const connect = () => {
      // Don't connect if unmounted or already connected
      if (!mountedRef.current) return
      if (wsRef.current?.readyState === WebSocket.OPEN) return
      if (wsRef.current?.readyState === WebSocket.CONNECTING) return

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = `${protocol}//${window.location.host}/ws/progress`

      try {
        const ws = new WebSocket(wsUrl)

        ws.onopen = () => {
          if (!mountedRef.current) {
            ws.close()
            return
          }
          setIsConnected(true)
          reconnectDelayRef.current = INITIAL_RECONNECT_DELAY // Reset delay on successful connection
        }

        ws.onmessage = (event) => {
          if (!mountedRef.current) return
          try {
            const data = JSON.parse(event.data) as EpisodeProgress
            setProgressMap((prev) => {
              const next = new Map(prev)
              next.set(data.episode_id, data)
              return next
            })
          } catch {
            // Ignore parse errors
          }
        }

        ws.onclose = () => {
          if (!mountedRef.current) return
          setIsConnected(false)
          wsRef.current = null

          // Exponential backoff for reconnection
          reconnectTimeoutRef.current = window.setTimeout(() => {
            reconnectDelayRef.current = Math.min(
              reconnectDelayRef.current * 2,
              MAX_RECONNECT_DELAY
            )
            connect()
          }, reconnectDelayRef.current)
        }

        ws.onerror = () => {
          // Let onclose handle reconnection
          ws.close()
        }

        wsRef.current = ws
      } catch {
        // Failed to create WebSocket, try again later
        if (mountedRef.current) {
          reconnectTimeoutRef.current = window.setTimeout(() => {
            reconnectDelayRef.current = Math.min(
              reconnectDelayRef.current * 2,
              MAX_RECONNECT_DELAY
            )
            connect()
          }, reconnectDelayRef.current)
        }
      }
    }

    connect()

    return () => {
      mountedRef.current = false
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, []) // Empty dependency array - only run once

  return {
    isConnected,
    progressMap,
  }
}
