import { useState, useRef, useCallback, useEffect } from 'react'
import type { AnimationEvent } from '../types/animation'

export function useAnimationQueue() {
  const [activeEvent, setActiveEvent] = useState<AnimationEvent | null>(null)
  const queueRef = useRef<AnimationEvent[]>([])
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const onEmptyRef = useRef<(() => void) | null>(null)

  const processNext = useCallback(() => {
    if (queueRef.current.length === 0) {
      setActiveEvent(null)
      onEmptyRef.current?.()
      return
    }
    const next = queueRef.current.shift()!
    setActiveEvent(next)
    timerRef.current = setTimeout(processNext, next.durationMs)
  }, [])

  const enqueue = useCallback((events: AnimationEvent[], onComplete: () => void) => {
    queueRef.current = [...events]
    onEmptyRef.current = onComplete
    if (timerRef.current) clearTimeout(timerRef.current)
    processNext()
  }, [processNext])

  const clearQueue = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current)
    queueRef.current = []
    setActiveEvent(null)
  }, [])

  useEffect(() => () => { if (timerRef.current) clearTimeout(timerRef.current) }, [])

  return { activeEvent, enqueue, clearQueue }
}
