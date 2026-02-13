import { useState, useEffect, useLayoutEffect } from 'react'
import type { AnimationEvent } from '../../types/animation'
import type { Board } from '../../types/board'
import { axialToPixel } from '../../types/hex'
import { PLAYER_COLORS } from '../../types/player'
import { SURFACE_Y } from './HexTile3D'

interface AnimationLayer3DProps {
  activeEvent: AnimationEvent | null
  board: Board
  hexSize: number
  playerIndex: (id: string) => number
}

export function AnimationLayer3D({ activeEvent, board, hexSize, playerIndex }: AnimationLayer3DProps) {
  const [progress, setProgress] = useState(0)

  useLayoutEffect(() => { setProgress(0) }, [activeEvent])

  useEffect(() => {
    if (!activeEvent) return
    const start = performance.now()
    let rafId: number
    const animate = (now: number) => {
      const p = Math.min(1, (now - start) / activeEvent.durationMs)
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
  const easeOut = (t: number) => 1 - (1 - t) * (1 - t)

  const moveProgress = isFight ? Math.min(1, progress / IMPACT_START) : easeOut(progress)
  const impactProgress = isFight ? Math.max(0, (progress - IMPACT_START) / (1 - IMPACT_START)) : 0

  const cx = from.x + (to.x - from.x) * moveProgress
  const cz = from.y + (to.y - from.y) * moveProgress

  const landT = !isFight && progress > LAND_START
    ? (progress - LAND_START) / (1 - LAND_START) : 0
  const dotR = (isFight ? 8 : 6) * (1 - landT * 0.7)
  const dotOpacity = isFight ? Math.max(0, 1 - impactProgress * 3) : 0.9 * (1 - landT)

  const burstR = hexSize * 0.2 + hexSize * impactProgress * 1.2
  const burstOpacity = impactProgress > 0 ? Math.max(0, 0.7 - impactProgress * 0.7) : 0
  const ringR = hexSize * impactProgress * 1.6
  const ringOpacity = impactProgress > 0 ? Math.max(0, 0.5 - impactProgress * 0.5) : 0

  return (
    <group>
      {dotOpacity > 0 && (
        <mesh position={[cx, SURFACE_Y + dotR, cz]}>
          <sphereGeometry args={[dotR, 12, 8]} />
          <meshBasicMaterial color={color} transparent opacity={dotOpacity} />
        </mesh>
      )}
      {isFight && burstOpacity > 0 && (
        <mesh position={[to.x, SURFACE_Y + 4, to.y]}>
          <sphereGeometry args={[burstR, 12, 8]} />
          <meshBasicMaterial color={color} transparent opacity={burstOpacity} />
        </mesh>
      )}
      {isFight && ringOpacity > 0 && (
        <mesh position={[to.x, SURFACE_Y + 1, to.y]} rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[ringR, 2, 8, 32]} />
          <meshBasicMaterial color={color} transparent opacity={ringOpacity} />
        </mesh>
      )}
    </group>
  )
}
