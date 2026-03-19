import type { GeopoliticalSnapshot } from '../types.ts'
import type { PartialDirective } from './registry.ts'

/**
 * S12 – Common Enemy
 *
 * When one neighbor is notably larger than the rest, focus everything on them
 * and soft-ignore the smallest neighbor (implicit non-aggression).
 * Prevents the largest player from snowballing while we fight on two fronts.
 */
export function commonEnemy(snapshot: GeopoliticalSnapshot): PartialDirective[] {
  if (snapshot.neighbors.length < 2) return []

  const sorted = [...snapshot.neighbors].sort((a, b) => b.tileCount - a.tileCount)
  const largest = sorted[0]
  const smallest = sorted[sorted.length - 1]
  if (largest.playerId === smallest.playerId) return []

  // Only trigger if the largest is notably bigger (50% more tiles than average)
  const avgTiles = snapshot.neighbors.reduce((s, n) => s + n.tileCount, 0) / snapshot.neighbors.length
  if (largest.tileCount < avgTiles * 1.5) return []

  // Don't ignore a neighbor that's almost eliminated (we want them gone)
  if (smallest.isNearlyEliminated) return []

  return [
    {
      neighborId: largest.playerId,
      stance: 'INVADE',
      unitBudgetFraction: 0.9,
      priority: 4,
      rationale: 'common-enemy-focus',
    },
    {
      neighborId: smallest.playerId,
      stance: 'IGNORE',
      unitBudgetFraction: 0,
      priority: 4,
      rationale: 'common-enemy-ignore-small',
    },
  ]
}

/**
 * S13 – Don't Finish the Decoy
 *
 * A nearly-eliminated neighbor is already being pounded by other players.
 * Let them waste units finishing that player — we spend ours elsewhere.
 */
export function dontFinishDecoy(snapshot: GeopoliticalSnapshot): PartialDirective[] {
  return snapshot.neighbors
    .filter(n => n.isNearlyEliminated && n.isEngagedElsewhere)
    .map(n => ({
      neighborId: n.playerId,
      stance: 'IGNORE' as const,
      unitBudgetFraction: 0,
      priority: 3,
      rationale: 'decoy-ignore',
    }))
}
