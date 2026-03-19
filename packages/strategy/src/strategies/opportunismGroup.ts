import type { GeopoliticalSnapshot } from '../types.ts'
import type { PartialDirective } from './registry.ts'

/**
 * S5 – Vulture Strike
 *
 * Two adjacent enemies are fighting each other and both losing tiles.
 * Hold both borders and wait. Once either hits near-elimination, finish them.
 * Saves our units while they deplete each other.
 */
export function vultureStrike(snapshot: GeopoliticalSnapshot): PartialDirective[] {
  const bleeding = snapshot.neighbors.filter(
    n => n.isEngagedElsewhere && n.momentumDelta <= -2,
  )
  if (bleeding.length < 2) return []

  return bleeding.map(n => ({
    neighborId: n.playerId,
    stance: n.isNearlyEliminated ? 'INVADE' : 'HOLD',
    unitBudgetFraction: n.isNearlyEliminated ? 1.0 : 0.3,
    priority: n.isNearlyEliminated ? 8 : 5,
    rationale: n.isNearlyEliminated ? 'vulture-finish' : 'vulture-wait',
  }))
}

/**
 * S6 – Collapse Exploitation
 *
 * A neighbor is nearly eliminated (≤ 3 tiles). Rush them immediately before
 * another player absorbs their territory. Highest priority override.
 */
export function collapseExploitation(snapshot: GeopoliticalSnapshot): PartialDirective[] {
  return snapshot.neighbors
    .filter(n => n.isNearlyEliminated)
    .map(n => ({
      neighborId: n.playerId,
      stance: 'INVADE' as const,
      unitBudgetFraction: 1.0,
      priority: 10,
      rationale: 'collapse-exploitation',
    }))
}

/**
 * S7 – Wounded Enemy
 *
 * A neighbor recently lost many tiles (likely from another front) but isn't
 * near elimination yet. Escalate while their defenses are thin.
 */
export function woundedEnemy(snapshot: GeopoliticalSnapshot): PartialDirective[] {
  return snapshot.neighbors
    .filter(n => n.momentumDelta <= -3 && !n.isNearlyEliminated && n.relativeStrength >= 0.7)
    .map(n => ({
      neighborId: n.playerId,
      stance: 'INVADE' as const,
      unitBudgetFraction: 0.9,
      priority: 6,
      rationale: 'wounded-enemy',
    }))
}

/**
 * S8 – Distraction Exploit
 *
 * A neighbor pulled units away to fight someone else. Their border facing us
 * is weaker than usual. Press the opportunity.
 */
export function distractionExploit(snapshot: GeopoliticalSnapshot): PartialDirective[] {
  return snapshot.neighbors
    .filter(n => n.isEngagedElsewhere && n.relativeStrength >= 1.0 && !n.isNearlyEliminated)
    .map(n => ({
      neighborId: n.playerId,
      stance: 'EXPAND' as const,
      unitBudgetFraction: 0.8,
      priority: 5,
      rationale: 'distraction-exploit',
    }))
}
