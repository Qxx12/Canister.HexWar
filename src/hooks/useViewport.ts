import { useState, useCallback, useRef } from 'react'

export interface ViewportState {
  zoom: number
  panX: number
  panY: number
}

const MIN_ZOOM = 0.4
const MAX_ZOOM = 2.5
const ZOOM_STEP = 0.2
const DRAG_THRESHOLD = 6

export function useViewport() {
  const [viewport, setViewport] = useState<ViewportState>({ zoom: 1, panX: 0, panY: 0 })
  const isPanning = useRef(false)
  const lastPos = useRef({ x: 0, y: 0 })
  const startPos = useRef({ x: 0, y: 0 })
  const pendingCapture = useRef<{ pointerId: number; target: Element } | null>(null)

  const zoomIn = useCallback(() => {
    setViewport(v => ({ ...v, zoom: Math.min(MAX_ZOOM, v.zoom + ZOOM_STEP) }))
  }, [])

  const zoomOut = useCallback(() => {
    setViewport(v => ({ ...v, zoom: Math.max(MIN_ZOOM, v.zoom - ZOOM_STEP) }))
  }, [])

  const resetViewport = useCallback(() => {
    setViewport({ zoom: 1, panX: 0, panY: 0 })
  }, [])

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    if (e.button !== 0) return
    isPanning.current = false
    startPos.current = { x: e.clientX, y: e.clientY }
    lastPos.current = { x: e.clientX, y: e.clientY }
    pendingCapture.current = { pointerId: e.pointerId, target: e.currentTarget }
  }, [])

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!pendingCapture.current && !isPanning.current) return
    const dx = e.clientX - startPos.current.x
    const dy = e.clientY - startPos.current.y
    const dist = Math.sqrt(dx * dx + dy * dy)

    if (!isPanning.current && dist > DRAG_THRESHOLD) {
      isPanning.current = true
      if (pendingCapture.current) {
        pendingCapture.current.target.setPointerCapture(pendingCapture.current.pointerId)
        pendingCapture.current = null
      }
    }

    if (isPanning.current) {
      const moveDx = e.clientX - lastPos.current.x
      const moveDy = e.clientY - lastPos.current.y
      setViewport(v => ({ ...v, panX: v.panX + moveDx, panY: v.panY + moveDy }))
    }

    lastPos.current = { x: e.clientX, y: e.clientY }
  }, [])

  const onPointerUp = useCallback(() => {
    isPanning.current = false
    pendingCapture.current = null
  }, [])

  return { viewport, zoomIn, zoomOut, resetViewport, onPointerDown, onPointerMove, onPointerUp }
}
