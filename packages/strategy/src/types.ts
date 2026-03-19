import type { PlayerId } from '@hexwar/engine'

// ─── Strategic layer types ────────────────────────────────────────────────────

export type FrontStance = 'DETER' | 'HOLD' | 'EXPAND' | 'INVADE' | 'IGNORE'

/** Assessment of one neighboring player from our perspective. */
export interface NeighborAssessment {
  playerId: PlayerId
  /** Our tiles that share an edge with this neighbor's tiles. */
  sharedBorderTiles: string[]
  ourBorderUnits: number
  theirBorderUnits: number
  /** ourBorderUnits / max(theirBorderUnits, 1), capped at 3. >1 = we're stronger. */
  relativeStrength: number
  tileCount: number
  /** True if this neighbor is actively fighting another player (inferred from tile loss). */
  isEngagedElsewhere: boolean
  /** Net tile count change over the tracked history window. Negative = shrinking. */
  momentumDelta: number
  /** Tile count ≤ 3 — nearly out of the game. */
  isNearlyEliminated: boolean
}

/** Full geopolitical picture for one player at one point in time. */
export interface GeopoliticalSnapshot {
  myPlayerId: PlayerId
  myTileCount: number
  myTotalUnits: number
  myMomentumDelta: number
  neighbors: NeighborAssessment[]
  totalActivePlayers: number
}

// ─── Plan types ───────────────────────────────────────────────────────────────

/** High command's decision for one border. */
export interface FrontDirective {
  neighborId: PlayerId
  stance: FrontStance
  /** Fraction of border units allowed to commit to orders (0.0–1.0). */
  unitBudgetFraction: number
  /** Higher priority wins when strategies conflict. */
  priority: number
  /** Debug label: which strategy fired. */
  rationale: string
}

export interface StrategicPlan {
  directives: Map<PlayerId, FrontDirective>
  /** Whether to expand into neutral tiles this turn. */
  neutralExpansionEnabled: boolean
}

// ─── Operational layer types ─────────────────────────────────────────────────

/** Per-source-tile constraints for the tactical layer. */
export interface TileConstraint {
  sourceKey: string
  /** Keys the tactical layer may target (filtered by stance). */
  allowedTargetKeys: string[]
  /** Cap on fraction of units that may be sent. */
  maxUnitsFraction: number
  /** Whether any target crosses a border into enemy/neutral territory. */
  crossBorderAllowed: boolean
}
