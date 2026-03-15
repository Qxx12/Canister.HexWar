import { describe, it, expect } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useViewport } from '../hooks/useViewport'

// jsdom defaults — make them explicit so tests don't depend on env defaults
Object.defineProperty(window, 'innerWidth',  { writable: true, configurable: true, value: 1024 })
Object.defineProperty(window, 'innerHeight', { writable: true, configurable: true, value: 768 })

const mockTarget = { setPointerCapture: () => {} } as unknown as Element

function makePointer(id: number, x: number, y: number, overrides: Partial<React.PointerEvent> = {}): React.PointerEvent {
  return {
    pointerId: id,
    clientX: x,
    clientY: y,
    button: 0,
    pointerType: 'touch',
    currentTarget: mockTarget,
    preventDefault: () => {},
    ...overrides,
  } as unknown as React.PointerEvent
}

describe('useViewport — panning', () => {
  it('pans after dragging past threshold', () => {
    const { result } = renderHook(() => useViewport())
    act(() => result.current.centerBoard(500, 500))
    const startX = result.current.viewport.panX

    act(() => result.current.onPointerDown(makePointer(1, 100, 100)))
    act(() => result.current.onPointerMove(makePointer(1, 108, 100))) // 8px > threshold
    act(() => result.current.onPointerMove(makePointer(1, 118, 100))) // extra 10px

    expect(result.current.viewport.panX).toBeGreaterThan(startX)
  })

  it('does not pan before drag threshold', () => {
    const { result } = renderHook(() => useViewport())
    act(() => result.current.centerBoard(500, 500))
    const startX = result.current.viewport.panX

    act(() => result.current.onPointerDown(makePointer(1, 100, 100)))
    act(() => result.current.onPointerMove(makePointer(1, 104, 100))) // 4px < threshold

    expect(result.current.viewport.panX).toBe(startX)
  })

  it('stops panning after pointerUp', () => {
    const { result } = renderHook(() => useViewport())
    act(() => result.current.centerBoard(500, 500))

    act(() => result.current.onPointerDown(makePointer(1, 100, 100)))
    act(() => result.current.onPointerMove(makePointer(1, 120, 100))) // starts pan
    act(() => result.current.onPointerUp(makePointer(1, 120, 100)))

    const panAfterUp = result.current.viewport.panX
    act(() => result.current.onPointerMove(makePointer(1, 160, 100))) // should be ignored
    expect(result.current.viewport.panX).toBe(panAfterUp)
  })
})

describe('useViewport — pointercancel', () => {
  it('clears all pointer state so next single-finger touch pans correctly', () => {
    const { result } = renderHook(() => useViewport())
    act(() => result.current.centerBoard(500, 500))
    const startPan = result.current.viewport.panX

    // Simulate a two-finger pinch
    act(() => result.current.onPointerDown(makePointer(1, 100, 100)))
    act(() => result.current.onPointerDown(makePointer(2, 200, 100)))
    // Browser cancels the gesture
    act(() => result.current.onPointerCancel())

    // Single-finger pan should work normally now
    act(() => result.current.onPointerDown(makePointer(3, 100, 100)))
    act(() => result.current.onPointerMove(makePointer(3, 120, 100))) // past threshold
    act(() => result.current.onPointerMove(makePointer(3, 140, 100)))

    expect(result.current.viewport.panX).toBeGreaterThan(startPan)
  })

  it('does not move viewport on pointer move after cancel without new pointerDown', () => {
    const { result } = renderHook(() => useViewport())
    act(() => result.current.centerBoard(500, 500))

    act(() => result.current.onPointerDown(makePointer(1, 100, 100)))
    act(() => result.current.onPointerMove(makePointer(1, 120, 100)))
    act(() => result.current.onPointerCancel())

    const panAfterCancel = result.current.viewport.panX
    act(() => result.current.onPointerMove(makePointer(1, 200, 100))) // stale move — should be ignored
    expect(result.current.viewport.panX).toBe(panAfterCancel)
  })
})

describe('useViewport — post-pinch pan', () => {
  it('allows panning with remaining finger after one finger lifts from pinch', () => {
    const { result } = renderHook(() => useViewport())
    act(() => result.current.centerBoard(500, 500))
    const startPan = result.current.viewport.panX

    // Two fingers down
    act(() => result.current.onPointerDown(makePointer(1, 100, 100)))
    act(() => result.current.onPointerDown(makePointer(2, 200, 100)))
    // One finger lifts
    act(() => result.current.onPointerUp(makePointer(2, 200, 100)))

    // Remaining finger (1) drags — should pan without needing a new pointerDown
    act(() => result.current.onPointerMove(makePointer(1, 108, 100))) // > threshold from 100
    act(() => result.current.onPointerMove(makePointer(1, 130, 100)))

    expect(result.current.viewport.panX).toBeGreaterThan(startPan)
  })

  it('allows panning after lifting both fingers and starting fresh', () => {
    const { result } = renderHook(() => useViewport())
    act(() => result.current.centerBoard(500, 500))
    const startPan = result.current.viewport.panX

    // Pinch then lift both
    act(() => result.current.onPointerDown(makePointer(1, 100, 100)))
    act(() => result.current.onPointerDown(makePointer(2, 200, 100)))
    act(() => result.current.onPointerUp(makePointer(1, 100, 100)))
    act(() => result.current.onPointerUp(makePointer(2, 200, 100)))

    // Fresh single-finger pan
    act(() => result.current.onPointerDown(makePointer(3, 100, 100)))
    act(() => result.current.onPointerMove(makePointer(3, 120, 100)))
    act(() => result.current.onPointerMove(makePointer(3, 140, 100)))

    expect(result.current.viewport.panX).toBeGreaterThan(startPan)
  })
})

describe('useViewport — mouse button handling', () => {
  it('pans with RMB (button 2) on mouse', () => {
    const { result } = renderHook(() => useViewport())
    act(() => result.current.centerBoard(500, 500))
    const startX = result.current.viewport.panX

    act(() => result.current.onPointerDown(makePointer(1, 100, 100, { button: 2, pointerType: 'mouse' })))
    act(() => result.current.onPointerMove(makePointer(1, 110, 100, { button: 2, pointerType: 'mouse' })))
    act(() => result.current.onPointerMove(makePointer(1, 130, 100, { button: 2, pointerType: 'mouse' })))

    expect(result.current.viewport.panX).toBeGreaterThan(startX)
  })

  it('does not pan with LMB (button 0) on mouse', () => {
    const { result } = renderHook(() => useViewport())
    act(() => result.current.centerBoard(500, 500))
    const startX = result.current.viewport.panX

    act(() => result.current.onPointerDown(makePointer(1, 100, 100, { button: 0, pointerType: 'mouse' })))
    act(() => result.current.onPointerMove(makePointer(1, 150, 100, { button: 0, pointerType: 'mouse' })))

    expect(result.current.viewport.panX).toBe(startX)
  })

  it('onContextMenu calls preventDefault', () => {
    const { result } = renderHook(() => useViewport())
    let prevented = false
    const fakeEvent = { preventDefault: () => { prevented = true } } as unknown as React.MouseEvent
    act(() => result.current.onContextMenu(fakeEvent))
    expect(prevented).toBe(true)
  })
})

describe('useViewport — pinch zoom', () => {
  it('zooms in when fingers spread apart', () => {
    const { result } = renderHook(() => useViewport())
    act(() => result.current.centerBoard(500, 500))
    const startZoom = result.current.viewport.zoom

    act(() => result.current.onPointerDown(makePointer(1, 200, 300)))
    act(() => result.current.onPointerDown(makePointer(2, 300, 300))) // dist = 100
    act(() => result.current.onPointerMove(makePointer(2, 400, 300))) // dist = 200 → 2x zoom

    expect(result.current.viewport.zoom).toBeGreaterThan(startZoom)
  })

  it('zoom is clamped to MAX_ZOOM', () => {
    const { result } = renderHook(() => useViewport())
    act(() => result.current.centerBoard(500, 500))

    act(() => result.current.onPointerDown(makePointer(1, 250, 300)))
    act(() => result.current.onPointerDown(makePointer(2, 260, 300))) // dist = 10
    act(() => result.current.onPointerMove(makePointer(2, 2000, 300))) // huge spread

    expect(result.current.viewport.zoom).toBeLessThanOrEqual(2.5)
  })
})
