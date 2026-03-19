import type { GeopoliticalSnapshot } from '../types.ts'
import type { PartialDirective } from './registry.ts'

/**
 * S14 – Turtle Mode
 *
 * We're shrinking on multiple fronts simultaneously. Stop all offensives,
 * hold every border, and let units regenerate before attempting a counter.
 */
export function turtleMode(snapshot: GeopoliticalSnapshot): PartialDirective[] {
  const losingBadly = snapshot.myMomentumDelta <= -3
  const multipleWeakFronts =
    snapshot.neighbors.filter(n => n.relativeStrength < 0.9).length >= 2
  if (!losingBadly || !multipleWeakFronts) return []

  return snapshot.neighbors.map(n => ({
    neighborId: n.playerId,
    stance: 'HOLD' as const,
    unitBudgetFraction: 0.4,
    priority: 7,
    rationale: 'turtle-mode',
  }))
}
