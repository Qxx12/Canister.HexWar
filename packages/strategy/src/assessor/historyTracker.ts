import type { Board, PlayerId } from '@hexwar/engine'

const MAX_HISTORY = 5

/**
 * Ring buffer of recent turn snapshots.
 * Used by the assessor to compute momentum (tile count trends) and
 * infer whether a player is engaged in conflict elsewhere.
 */
export class HistoryTracker {
  private readonly snapshots: Map<PlayerId, number>[] = []

  recordTurn(board: Board): void {
    const counts = new Map<PlayerId, number>()
    for (const tile of board.values()) {
      if (tile.owner !== null) {
        counts.set(tile.owner, (counts.get(tile.owner) ?? 0) + 1)
      }
    }
    this.snapshots.push(counts)
    if (this.snapshots.length > MAX_HISTORY) this.snapshots.shift()
  }

  /** Net tile change from oldest to newest snapshot. Negative = shrinking. */
  getMomentumDelta(playerId: PlayerId): number {
    if (this.snapshots.length < 2) return 0
    const oldest = this.snapshots[0].get(playerId) ?? 0
    const newest = this.snapshots[this.snapshots.length - 1].get(playerId) ?? 0
    return newest - oldest
  }

  isLosingTiles(playerId: PlayerId, threshold = -1): boolean {
    return this.getMomentumDelta(playerId) <= threshold
  }

  reset(): void {
    this.snapshots.length = 0
  }
}
