import type { Tile } from '../../types/board'
import type { PlayerId } from '../../types/player'
import { axialToPixel, hexCorners } from '../../types/hex'
import { PLAYER_COLORS } from '../../types/player'
import { TERRAIN_COLORS } from '../../types/board'
import styles from './HexTile.module.scss'

interface HexTileProps {
  tile: Tile
  hexSize: number
  playerIndex: (id: PlayerId) => number
  isSelected: boolean
  isValidDestination: boolean
  onClick: () => void
}

function fourPointedStar(cx: number, cy: number, outer: number, inner: number): string {
  const pts: string[] = []
  for (let i = 0; i < 8; i++) {
    const angle = (Math.PI / 4) * i - Math.PI / 2
    const r = i % 2 === 0 ? outer : inner
    pts.push(`${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`)
  }
  return pts.join(' ')
}

export function HexTile({ tile, hexSize, playerIndex, isSelected, isValidDestination, onClick }: HexTileProps) {
  const { x: cx, y: cy } = axialToPixel(tile.coord, hexSize)
  const corners = hexCorners(cx, cy, hexSize - 1) // -1 for gap
  const points = corners.map(c => `${c.x},${c.y}`).join(' ')

  const fillColor = TERRAIN_COLORS[tile.terrain] ?? '#d8d8d8'

  const strokeColor = isSelected ? '#111' : isValidDestination ? '#555' : '#b0b0b0'
  const strokeWidth = isSelected || isValidDestination ? 2 : 1

  const starOuter = hexSize * 0.12
  const starInner = hexSize * 0.05
  const starY = cy - hexSize * 0.42
  const starColor = tile.startOwner ? PLAYER_COLORS[playerIndex(tile.startOwner)] : '#666'

  return (
    <g className={styles.hexGroup} onClick={onClick} style={{ cursor: 'pointer' }}>
      <polygon
        points={points}
        fill={fillColor}
        stroke={strokeColor}
        strokeWidth={strokeWidth}
        className={styles.hexPoly}
      />
      {tile.owner !== null && (
        <text
          x={cx}
          y={cy}
          textAnchor="middle"
          dominantBaseline="middle"
          className={styles.unitCount}
          fontSize={hexSize * 0.35}
          fill={PLAYER_COLORS[playerIndex(tile.owner)]}
          stroke="white"
          strokeWidth={2.5}
          paintOrder="stroke fill"
        >
          {tile.units}
        </text>
      )}
      {tile.isStartTile && (
        <polygon
          points={fourPointedStar(cx, starY, starOuter, starInner)}
          fill={starColor}
          stroke="white"
          strokeWidth={1}
          strokeLinejoin="round"
        />
      )}
    </g>
  )
}
