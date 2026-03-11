import { useState, useEffect, useLayoutEffect } from 'react'
import type { AnimationEvent } from '@hexwar/engine'
import type { Board } from '@hexwar/engine'
import { axialToPixel } from '@hexwar/engine'
import { PLAYER_COLORS } from '@hexwar/engine'
import styles from './AnimationLayer.module.scss'

interface AnimationLayerProps {
  activeEvent: AnimationEvent | null
  board: Board
  hexSize: number
  playerIndex: (id: string) => number
}

export function AnimationLayer({ activeEvent, board, hexSize, playerIndex }: AnimationLayerProps) {
  const [progress, setProgress] = useState(0)

  // eslint-disable-next-line react-hooks/set-state-in-effect
  useLayoutEffect(() => { setProgress(0) }, [activeEvent])

  useEffect(() => {
    if (!activeEvent) return
    const start = performance.now()
    let rafId: number

    const animate = (now: number) => {
      const elapsed = now - start
      const p = Math.min(1, elapsed / activeEvent.durationMs)
      setProgress(p)
      if (p < 1) rafId = requestAnimationFrame(animate)
    }
    rafId = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(rafId)
  }, [activeEvent])

  if (!activeEvent) return null

  const fromTile = board.get(activeEvent.fromKey)
  const toTile = board.get(activeEvent.toKey)
  if (!fromTile || !toTile) return null

  const from = axialToPixel(fromTile.coord, hexSize)
  const to = axialToPixel(toTile.coord, hexSize)

  const color = PLAYER_COLORS[playerIndex(activeEvent.playerId)] ?? '#fff'
  const isFight = activeEvent.kind === 'fight' || activeEvent.kind === 'conquer'

  const IMPACT_START = 0.65
  const LAND_START = 0.6
  // ease-out quad: decelerates as it approaches destination
  const easeOut = (t: number) => 1 - (1 - t) * (1 - t)
  const moveProgress = isFight
    ? Math.min(1, progress / IMPACT_START)
    : easeOut(progress)
  const impactProgress = isFight ? Math.max(0, (progress - IMPACT_START) / (1 - IMPACT_START)) : 0

  const cx = from.x + (to.x - from.x) * moveProgress
  const cy = from.y + (to.y - from.y) * moveProgress

  const landT = !isFight && progress > LAND_START
    ? (progress - LAND_START) / (1 - LAND_START)
    : 0
  const dotRadius = isFight ? 8 : 6 * (1 - landT * 0.7)
  const dotOpacity = isFight
    ? Math.max(0, 1 - impactProgress * 3)
    : 0.9 * (1 - landT)
  const burstRadius = hexSize * 0.2 + hexSize * impactProgress * 1.2
  const burstOpacity = impactProgress > 0 ? Math.max(0, 0.7 - impactProgress * 0.7) : 0
  const ringRadius = hexSize * impactProgress * 1.6
  const ringOpacity = impactProgress > 0 ? Math.max(0, 0.5 - impactProgress * 0.5) : 0

  return (
    <g className={styles.layer}>
      {isFight && impactProgress > 0 && (
        <>
          <circle cx={to.x} cy={to.y} r={ringRadius} fill="none"
            stroke={color} strokeWidth={2} opacity={ringOpacity} />
          <circle cx={to.x} cy={to.y} r={burstRadius}
            fill={color} opacity={burstOpacity} />
        </>
      )}
      {dotOpacity > 0 && (
        <circle
          cx={cx} cy={cy}
          r={dotRadius}
          fill={color}
          stroke="#fff"
          strokeWidth={1.5}
          opacity={dotOpacity}
        />
      )}
      {dotOpacity > 0 && (
        <text
          x={cx} y={cy}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize={8}
          fill="#fff"
          fontWeight="bold"
        >
          {activeEvent.units}
        </text>
      )}
    </g>
  )
}
