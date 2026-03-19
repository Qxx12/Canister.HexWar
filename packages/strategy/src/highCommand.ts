import type { Board, PlayerId, Player, OrderMap } from '@hexwar/engine'
import { HistoryTracker } from './assessor/historyTracker.ts'
import { buildSnapshot } from './assessor/neighborAssessor.ts'
import { buildStrategicPlan } from './strategies/registry.ts'
import { deterrenceWall, flankingDeterrent } from './strategies/threatGroup.ts'
import {
  collapseExploitation,
  woundedEnemy,
  vultureStrike,
  distractionExploit,
} from './strategies/opportunismGroup.ts'
import { commonEnemy, dontFinishDecoy } from './strategies/allianceGroup.ts'
import { turtleMode } from './strategies/consolidationGroup.ts'
import { buildFrontMask } from './operational/frontMask.ts'
import { buildInteriorRoutes } from './operational/interiorRouter.ts'
import { computeConstrainedOrders } from './tactical/constrainedGreedy.ts'
import type { GeopoliticalSnapshot, StrategicPlan } from './types.ts'

// Strategy list — applied in order; higher-priority strategies run later so
// they can override lower-priority ones with equal or higher priority values.
const STRATEGIES = [
  // Background defaults (lowest priority)
  deterrenceWall,
  flankingDeterrent,
  // Opportunism
  distractionExploit,
  commonEnemy,
  dontFinishDecoy,
  vultureStrike,
  woundedEnemy,
  collapseExploitation,
  // Defensive override (highest priority — runs last, overrides everything)
  turtleMode,
]

/**
 * High Command AI — the main entry point for the strategic agent.
 *
 * Each instance is bound to one player and maintains history across turns.
 * Create one instance per AI player and reuse it each turn.
 */
export class HighCommandAI {
  private readonly history = new HistoryTracker()

  computeOrders(
    board: Board,
    playerId: PlayerId,
    _currentOrders: OrderMap,
    allPlayers: Player[],
  ): OrderMap {
    // Record board state before this player's turn for momentum tracking
    this.history.recordTurn(board)

    // Layer 1 — Strategic assessment
    const snapshot = buildSnapshot(board, playerId, allPlayers, this.history)

    // Layer 2 — Strategy selection → directive map
    const plan = buildStrategicPlan(snapshot, STRATEGIES)

    // Layer 3 — Operational: tile constraints + interior routing
    const constraints = buildFrontMask(board, playerId, plan)
    const activeFrontTiles = resolveActiveFrontTiles(snapshot, plan)
    const interiorRoutes = buildInteriorRoutes(board, playerId, activeFrontTiles)

    // Layer 4 — Tactical: constrained greedy order generation
    return computeConstrainedOrders(board, playerId, allPlayers, constraints, interiorRoutes)
  }

  reset(): void {
    this.history.reset()
  }
}

/**
 * Determines which frontier tiles are "active" for interior unit routing.
 *
 * Active = tiles on the border of an EXPAND or INVADE front.
 * Fallback: if no offensive fronts, include all border tiles so units still
 * flow toward DETER/HOLD positions (accumulation for deterrence).
 */
function resolveActiveFrontTiles(
  snapshot: GeopoliticalSnapshot,
  plan: StrategicPlan,
): Set<string> {
  const tiles = new Set<string>()

  for (const neighbor of snapshot.neighbors) {
    const directive = plan.directives.get(neighbor.playerId)
    if (directive && (directive.stance === 'EXPAND' || directive.stance === 'INVADE')) {
      for (const key of neighbor.sharedBorderTiles) tiles.add(key)
    }
  }

  // Fallback: route toward all borders (for DETER accumulation)
  if (tiles.size === 0) {
    for (const neighbor of snapshot.neighbors) {
      for (const key of neighbor.sharedBorderTiles) tiles.add(key)
    }
  }

  return tiles
}
