import { useState, useEffect } from 'react'
import type { AnimationEvent } from '../../types/animation'
import type { Board } from '../../types/board'
import { axialToPixel } from '../../types/hex'
import { PLAYER_COLORS } from '../../types/player'
import styles from './AnimationLayer.module.scss'

interface AnimationLayerProps {
  activeEvent: AnimationEvent | null
  board: Board
  hexSize: number
  playerIndex: (id: string) => number
}

export function AnimationLayer({ activeEvent, board, hexSize, playerIndex }: AnimationLayerProps) {
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    if (!activeEvent) return
    setProgress(0)
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

  const cx = from.x + (to.x - from.x) * progress
  const cy = from.y + (to.y - from.y) * progress

  const color = PLAYER_COLORS[playerIndex(activeEvent.playerId)] ?? '#fff'
  const isFight = activeEvent.kind === 'fight' || activeEvent.kind === 'conquer'
  const radius = isFight ? 8 : 6

  return (
    <g className={styles.layer}>
      {isFight && (
        <circle
          cx={cx} cy={cy}
          r={radius * 1.8}
          fill={color}
          opacity={0.2}
        />
      )}
      <circle
        cx={cx} cy={cy}
        r={radius}
        fill={color}
        stroke="#fff"
        strokeWidth={1.5}
        opacity={0.9}
      />
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
    </g>
  )
}
