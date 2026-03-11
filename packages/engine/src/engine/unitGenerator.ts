import type { Board } from '../types/board'
import type { PlayerStats } from '../types/stats'
import type { PlayerId } from '../types/player'

export function generateUnits(
  board: Board,
  runningStats: Map<PlayerId, PlayerStats>,
): Board {
  const newBoard = new Map(board)
  for (const [key, tile] of newBoard) {
    if (tile.owner !== null) {
      if (tile.newlyConquered) {
        newBoard.set(key, { ...tile, newlyConquered: false })
      } else {
        newBoard.set(key, { ...tile, units: tile.units + 1 })
        const stats = runningStats.get(tile.owner)
        if (stats) {
          runningStats.set(tile.owner, { ...stats, unitsGenerated: stats.unitsGenerated + 1 })
        }
      }
    }
  }
  return newBoard
}
