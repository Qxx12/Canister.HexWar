import type { Tile } from '../../types/board'
import type { Player } from '../../types/player'
import styles from './TileTooltip.module.scss'

interface TileTooltipProps {
  tile: Tile
  players: Player[]
  x: number
  y: number
}

export function TileTooltip({ tile, players, x, y }: TileTooltipProps) {
  const owner = tile.owner ? players.find(p => p.id === tile.owner) : null
  const terrain = tile.terrain.charAt(0).toUpperCase() + tile.terrain.slice(1)

  const offsetX = 14
  const offsetY = 14
  const clampedX = Math.min(x + offsetX, window.innerWidth - 160)
  const clampedY = Math.min(y + offsetY, window.innerHeight - 120)

  return (
    <div className={styles.tooltip} style={{ left: clampedX, top: clampedY }}>
      <div className={styles.owner}>
        {owner ? (
          <>
            <span className={styles.dot} style={{ background: owner.color }} />
            <span>{owner.name}</span>
          </>
        ) : (
          <span className={styles.muted}>Unconquered</span>
        )}
      </div>
      <div className={styles.terrain}>{terrain}</div>
      {tile.owner !== null && (
        <div className={styles.row}>
          <span className={styles.muted}>Units</span>
          <span>{tile.units}</span>
        </div>
      )}
    </div>
  )
}
