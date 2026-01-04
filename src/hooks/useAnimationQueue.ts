import { useState, useRef, useCallback, useEffect } from 'react'
import type { AnimationEvent } from '../types/animation'

export interface AnimationStep {
  event: AnimationEvent
  onStepComplete: () => void
}

export function useAnimationQueue() {
  const [activeEvent, setActiveEvent] = useState<AnimationEvent | null>(null)
  const queueRef = useRef<AnimationStep[]>([])
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const onAllCompleteRef = useRef<(() => void) | null>(null)

  const processNext = useCallback(() => {
    if (queueRef.current.length === 0) {
      setActiveEvent(null)
      onAllCompleteRef.current?.()
      return
    }
    const next = queueRef.current.shift()!
    setActiveEvent(next.event)
    timerRef.current = setTimeout(() => {
      next.onStepComplete()
      processNext()
    }, next.event.durationMs)
  }, [])

  const enqueue = useCallback((steps: AnimationStep[], onAllComplete: () => void) => {
    queueRef.current = [...steps]
    onAllCompleteRef.current = onAllComplete
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
