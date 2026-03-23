import type { PlayerId } from '@hexwar/engine'
import type { GeopoliticalSnapshot, FrontDirective, StrategicPlan } from '../types.ts'

export type PartialDirective = { neighborId: PlayerId } & Partial<Omit<FrontDirective, 'neighborId'>>
export type StrategyFn = (snapshot: GeopoliticalSnapshot) => PartialDirective[]

function defaultDirective(neighborId: PlayerId): FrontDirective {
  return {
    neighborId,
    stance: 'EXPAND',
    unitBudgetFraction: 0.75,
    priority: 1,
    rationale: 'default-expand',
  }
}

/**
 * Applies all strategy functions to the snapshot and merges their outputs.
 *
 * Merge rule: higher `priority` wins. Equal priority: last writer wins (later
 * strategies in the array take precedence, so order the array from low to high
 * priority strategies).
 *
 * After merging, enforces two-front avoidance: keeps at most MAX_OFFENSIVE_FRONTS
 * simultaneous INVADE/EXPAND directives, demoting the rest to HOLD.
 */
export function buildStrategicPlan(
  snapshot: GeopoliticalSnapshot,
  strategies: StrategyFn[],
): StrategicPlan {
  const directives = new Map<PlayerId, FrontDirective>()
  for (const n of snapshot.neighbors) {
    directives.set(n.playerId, defaultDirective(n.playerId))
  }

  for (const strategy of strategies) {
    for (const partial of strategy(snapshot)) {
      const { neighborId, ...rest } = partial
      const current = directives.get(neighborId)
      if (!current) continue
      const incomingPriority = rest.priority ?? 1
      if (incomingPriority >= current.priority) {
        directives.set(neighborId, { ...current, ...rest, neighborId })
      }
    }
  }

  // Two-front avoidance: cap simultaneous offensive fronts.
  // Only applies when actively at war (INVADE directive exists) — pure neutral
  // expansion (EXPAND only) should not be artificially capped, as that just
  // hands free tiles to Greedy-style opponents.
  const offensiveFronts = [...directives.values()]
    .filter(d => d.stance === 'INVADE' || d.stance === 'EXPAND')
    .sort((a, b) => b.priority - a.priority)

  const hasActiveWar = offensiveFronts.some(d => d.stance === 'INVADE')
  const maxOffensive = !hasActiveWar ? Infinity
    : snapshot.totalActivePlayers <= 2 ? 1
    : 2

  for (const d of offensiveFronts.slice(maxOffensive)) {
    directives.set(d.neighborId, {
      ...d,
      stance: 'HOLD',
      rationale: d.rationale + '+two-front-cap',
    })
  }

  // Allow neutral expansion unless we're fully committed to an INVADE
  const hasInvade = [...directives.values()].some(d => d.stance === 'INVADE')
  const neutralExpansionEnabled = !hasInvade || snapshot.myTileCount < 12

  return { directives, neutralExpansionEnabled }
}
