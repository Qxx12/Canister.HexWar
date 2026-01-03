import type { Tile } from '../../types/board'
import type { PlayerId } from '../../types/player'
import { axialToPixel, hexCorners } from '../../types/hex'
import { PLAYER_COLORS } from '../../types/player'
import styles from './HexTile.module.scss'

interface HexTileProps {
  tile: Tile
  hexSize: number
  playerIndex: (id: PlayerId) => number
  isSelected: boolean
  isValidDestination: boolean
  onClick: () => void
}

export function HexTile({ tile, hexSize, playerIndex, isSelected, isValidDestination, onClick }: HexTileProps) {
  const { x: cx, y: cy } = axialToPixel(tile.coord, hexSize)
  const corners = hexCorners(cx, cy, hexSize - 1) // -1 for gap
  const points = corners.map(c => `${c.x},${c.y}`).join(' ')

  const fillColor = tile.owner !== null
    ? PLAYER_COLORS[playerIndex(tile.owner)]
    : '#d8d8d8'

  const strokeColor = isSelected ? '#111' : isValidDestination ? '#555' : '#b0b0b0'
  const strokeWidth = isSelected || isValidDestination ? 2 : 1

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
          y={tile.isStartTile ? cy + 6 : cy + 5}
          textAnchor="middle"
          dominantBaseline="middle"
          className={styles.unitCount}
          fontSize={hexSize * 0.35}
        >
          {tile.units}
        </text>
      )}
      {tile.isStartTile && (
        <text
          x={cx}
          y={cy - hexSize * 0.25}
          textAnchor="middle"
          dominantBaseline="middle"
          className={styles.crown}
          fontSize={hexSize * 0.3}
          fill={tile.startOwner ? PLAYER_COLORS[playerIndex(tile.startOwner)] : '#666'}
        >
          &#9819;
        </text>
      )}
    </g>
  )
}
