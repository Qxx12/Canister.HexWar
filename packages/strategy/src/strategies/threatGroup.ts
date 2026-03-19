import type { GeopoliticalSnapshot } from '../types.ts'
import type { PartialDirective } from './registry.ts'

/**
 * S1 – Deterrence Wall
 *
 * When we face a strong enemy (they outgun our border) alongside a weak one,
 * mass units against the strong neighbor to deter invasion while freely
 * expanding against the weak neighbor. Reduces active conflict tiles by ~50%.
 */
export function deterrenceWall(snapshot: GeopoliticalSnapshot): PartialDirective[] {
  if (snapshot.neighbors.length < 2) return []

  const strong = snapshot.neighbors.filter(n => n.relativeStrength < 0.8)
  const weak = snapshot.neighbors.filter(n => n.relativeStrength >= 1.3 && !n.isNearlyEliminated)
  if (strong.length === 0 || weak.length === 0) return []

  const result: PartialDirective[] = []
  for (const n of strong) {
    result.push({
      neighborId: n.playerId,
      stance: 'DETER',
      unitBudgetFraction: 0.9,
      priority: 4,
      rationale: 'deterrence-wall',
    })
  }
  for (const n of weak) {
    result.push({
      neighborId: n.playerId,
      stance: 'EXPAND',
      unitBudgetFraction: 0.8,
      priority: 3,
      rationale: 'deterrence-exploit-weak',
    })
  }
  return result
}

/**
 * S4 – Flanking Deterrent
 *
 * When surrounded on 3+ sides and outgunned on all of them, deter the
 * two strongest and focus the weakest front for a breakout attempt.
 */
export function flankingDeterrent(snapshot: GeopoliticalSnapshot): PartialDirective[] {
  if (snapshot.neighbors.length < 3) return []

  const allOutgunned = snapshot.neighbors.every(n => n.relativeStrength < 1.0)
  if (!allOutgunned) return []

  // Sort by their tile count ascending — smallest = most breakable
  const sorted = [...snapshot.neighbors].sort((a, b) => a.tileCount - b.tileCount)
  const breakout = sorted[0]

  return snapshot.neighbors.map(n => ({
    neighborId: n.playerId,
    stance: n.playerId === breakout.playerId ? 'HOLD' : 'DETER',
    unitBudgetFraction: n.playerId === breakout.playerId ? 0.7 : 0.95,
    priority: 3,
    rationale: 'flanking-deterrent',
  }))
}
