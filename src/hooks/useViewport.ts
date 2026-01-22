import { useState, useCallback, useRef } from 'react'

export interface ViewportState {
  zoom: number
  panX: number
  panY: number
}

const MIN_ZOOM = 0.4
const MAX_ZOOM = 2.5
const DRAG_THRESHOLD = 6

function clampZoom(z: number) {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z))
}

function zoomToward(v: ViewportState, newZoom: number, cx: number, cy: number): ViewportState {
  const clamped = clampZoom(newZoom)
  const factor = clamped / v.zoom
  return {
    zoom: clamped,
    panX: cx - factor * (cx - v.panX),
    panY: cy - factor * (cy - v.panY),
  }
}

export function useViewport() {
  const [viewport, setViewport] = useState<ViewportState>({ zoom: 1, panX: 0, panY: 0 })
  const isPanning = useRef(false)
  const lastPos = useRef({ x: 0, y: 0 })
  const startPos = useRef({ x: 0, y: 0 })
  const pendingCapture = useRef<{ pointerId: number; target: Element } | null>(null)

  // Multi-touch pinch tracking
  const activePointers = useRef<Map<number, { x: number; y: number }>>(new Map())
  const lastPinchDist = useRef<number | null>(null)
  const lastPinchMid = useRef<{ x: number; y: number } | null>(null)

  const centerBoard = useCallback((boardW: number, boardH: number) => {
    setViewport({
      zoom: 1,
      panX: (window.innerWidth - boardW) / 2,
      panY: (window.innerHeight - boardH) / 2,
    })
  }, [])

  const resetViewport = useCallback(() => {
    setViewport({ zoom: 1, panX: 0, panY: 0 })
  }, [])

  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    const factor = e.deltaY < 0 ? 1.2 : 1 / 1.2
    const cx = window.innerWidth / 2
    const cy = window.innerHeight / 2
    setViewport(v => zoomToward(v, v.zoom * factor, cx, cy))
  }, [])

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    if (e.button !== 0 && e.pointerType === 'mouse') return
    activePointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY })

    if (activePointers.current.size === 1) {
      isPanning.current = false
      startPos.current = { x: e.clientX, y: e.clientY }
      lastPos.current = { x: e.clientX, y: e.clientY }
      pendingCapture.current = { pointerId: e.pointerId, target: e.currentTarget }
      lastPinchDist.current = null
      lastPinchMid.current = null
    } else if (activePointers.current.size === 2) {
      // Cancel any single-pointer pan/capture when second finger arrives
      pendingCapture.current = null
      isPanning.current = false
      const pts = [...activePointers.current.values()]
      const dx = pts[1].x - pts[0].x
      const dy = pts[1].y - pts[0].y
      lastPinchDist.current = Math.sqrt(dx * dx + dy * dy)
      lastPinchMid.current = { x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2 }
    }
  }, [])

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    activePointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY })

    if (activePointers.current.size === 2) {
      const pts = [...activePointers.current.values()]
      const dx = pts[1].x - pts[0].x
      const dy = pts[1].y - pts[0].y
      const dist = Math.sqrt(dx * dx + dy * dy)
      const mid = { x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2 }

      if (lastPinchDist.current !== null && lastPinchMid.current !== null) {
        const factor = dist / lastPinchDist.current
        const prevMid = lastPinchMid.current
        const cx = window.innerWidth / 2
        const cy = window.innerHeight / 2
        setViewport(v => {
          const zoomed = zoomToward(v, v.zoom * factor, cx, cy)
          return {
            ...zoomed,
            panX: zoomed.panX + (mid.x - prevMid.x),
            panY: zoomed.panY + (mid.y - prevMid.y),
          }
        })
      }

      lastPinchDist.current = dist
      lastPinchMid.current = mid
      return
    }

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

  const onPointerUp = useCallback((e: React.PointerEvent) => {
    activePointers.current.delete(e.pointerId)
    isPanning.current = false
    pendingCapture.current = null
    lastPinchDist.current = null
    lastPinchMid.current = null
  }, [])

  return { viewport, centerBoard, resetViewport, onWheel, onPointerDown, onPointerMove, onPointerUp }
}
