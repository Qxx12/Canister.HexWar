import type { Board } from '../types/board'
import type { Player, PlayerId } from '../types/player'

export function checkWin(board: Board, players: Player[]): PlayerId | null {
  const activePlayers = players.filter(p => !p.isEliminated)

  for (const player of activePlayers) {
    // Must own their own start tile
    const ownStartTile = [...board.values()].find(
      t => t.isStartTile && t.startOwner === player.id
    )
    if (!ownStartTile || ownStartTile.owner !== player.id) continue

    // Must own all other start tiles
    const allStartTiles = [...board.values()].filter(t => t.isStartTile)
    const ownsAll = allStartTiles.every(t => t.owner === player.id)
    if (ownsAll) return player.id
  }

  return null
}

export function checkEliminations(board: Board, players: Player[]): PlayerId[] {
  const eliminated: PlayerId[] = []
  for (const player of players) {
    if (player.isEliminated) continue
    const hasTiles = [...board.values()].some(t => t.owner === player.id)
    if (!hasTiles) eliminated.push(player.id)
  }
  return eliminated
}
