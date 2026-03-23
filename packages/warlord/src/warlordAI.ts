import type { Board, PlayerId, Player, OrderMap, Tile } from '@hexwar/engine'
import { hexNeighbors, hexToKey } from '@hexwar/engine'

// ─── Constants ────────────────────────────────────────────────────────────────

const GARRISON_FACTOR = 1.0

/**
 * Below this tile count: skip garrison entirely — pure expansion.
 * Early-game all players are roughly equal so deterrence just wastes tiles.
 * Set higher than classic 10 to keep aggressive expansion going longer.
 */
const EARLY_PHASE_TILES = 20

/**
 * Only seed the routing BFS from the top K attack fronts by score.
 * Concentrates interior unit flow toward breakthrough points.
 */
const MAX_ROUTING_FRONTS = 6

// ─── Scoring ──────────────────────────────────────────────────────────────────

function scoreTarget(myUnits: number, target: Tile, earlyPhase: boolean): number {
  const startBonus = target.isStartTile ? 45 : 0

  if (target.owner === null) {
    // Early game: prioritise neutral expansion (snowball territory before enemies meet).
    // Late game: match Greedy (prefer beatable enemies over neutrals).
    return (earlyPhase ? 55 : 35) + startBonus
  }

  // Enemy tile we can't beat — only gamble on start tiles (matches Greedy)
  if (myUnits <= target.units) return target.isStartTile ? 18 : -1

  return 50 + (myUnits - target.units) * 3 + startBonus
}

function unitsToSend(available: number, target: Tile): number {
  if (target.owner === null) return available
  if (available <= target.units) return available // gamble: commit all
  return Math.min(available, Math.max(target.units + 1, Math.ceil(available * 0.85)))
}

// ─── Routing ──────────────────────────────────────────────────────────────────

/**
 * Multi-source BFS outward from `seeds` through friendly territory.
 * Returns: interiorKey → nextHop one step closer to the nearest seed.
 * Seeds are processed in the order given — pass high-priority seeds first
 * to bias routing toward the best attack fronts.
 */
function buildRoutes(
  board: Board,
  playerId: PlayerId,
  seeds: readonly string[],
): Map<string, string> {
  const routes = new Map<string, string>()
  const visited = new Set<string>(seeds)
  const queue: string[] = [...seeds]
  let qi = 0

  while (qi < queue.length) {
    const key = queue[qi++]
    const tile = board.get(key)
    if (!tile) continue
    for (const nCoord of hexNeighbors(tile.coord)) {
      const nKey = hexToKey(nCoord)
      if (visited.has(nKey)) continue
      const nTile = board.get(nKey)
      if (!nTile || nTile.owner !== playerId) continue
      visited.add(nKey)
      routes.set(nKey, key)
      queue.push(nKey)
    }
  }
  return routes
}

// ─── WarlordAI ────────────────────────────────────────────────────────────────

/**
 * WarlordAI — aggressive AI with garrison, smart routing, and start-tile awareness.
 *
 * Layers beyond Greedy:
 *
 *   1. Garrison       — frontier tiles keep ceil(maxAdjEnemy × 1.0) units to deter
 *                       counter-attacks during neutral expansion; surplus routes to fronts.
 *   2. Full routing   — all attack fronts (score-sorted; top-K priority) + stuck
 *                       frontier tiles as BFS seeds, so every border receives units.
 *   3. Start-tile bonus (+45) — win condition requires all start tiles; always
 *                       prioritise capturing them, even gambling losing fights.
 *   4. Extended early phase (20 tiles) — no garrison during the critical land-grab.
 *   5. Neutral-adj bonus (+8 per adjacent neutral) — prefer attacking enemy tiles
 *                       that border neutral territory to open up future expansion.
 *   6. Adaptive neutral score (55 early / 35 late) — aggressive territory snowball
 *                       in early game; switch to enemy focus in late game.
 */
export class WarlordAI {
  computeOrders(
    board: Board,
    playerId: PlayerId,
    _currentOrders: OrderMap,
    _allPlayers: Player[],
  ): OrderMap {
    // ── Phase detection ───────────────────────────────────────────────────────
    let myTileCount = 0
    for (const tile of board.values()) {
      if (tile.owner === playerId) myTileCount++
    }
    const earlyPhase = myTileCount < EARLY_PHASE_TILES

    // ── Per-tile analysis ─────────────────────────────────────────────────────

    type AttackPlan = { targetKey: string; score: number; units: number }

    const attackPlans = new Map<string, AttackPlan>()
    const routingTiles = new Map<string, number>() // key → available units to route
    const frontierKeys: string[] = []

    for (const [key, tile] of board) {
      if (tile.owner !== playerId || tile.units === 0) continue

      let maxAdjEnemy = 0
      let isFrontier = false

      for (const nCoord of hexNeighbors(tile.coord)) {
        const nTile = board.get(hexToKey(nCoord))
        if (!nTile || nTile.owner === playerId) continue
        isFrontier = true
        if (nTile.owner !== null) maxAdjEnemy = Math.max(maxAdjEnemy, nTile.units)
      }

      if (isFrontier) frontierKeys.push(key)

      // Score targets with full tile.units — same as Greedy, avoids missing
      // winnable fights that a garrison-reduced budget would skip.
      let bestTargetKey: string | null = null
      let bestScore = 0

      for (const nCoord of hexNeighbors(tile.coord)) {
        const nKey = hexToKey(nCoord)
        const nTile = board.get(nKey)
        if (!nTile || nTile.owner === playerId) continue

        let s = scoreTarget(tile.units, nTile, earlyPhase)
        // Bonus for enemy tiles adjacent to neutral territory — capturing opens expansion.
        if (nTile.owner !== null) {
          for (const nnCoord of hexNeighbors(nTile.coord)) {
            if (board.get(hexToKey(nnCoord))?.owner === null) s += 8
          }
        }
        if (s > bestScore) { bestScore = s; bestTargetKey = nKey }
      }

      const garrison = earlyPhase ? 0 : Math.ceil(maxAdjEnemy * GARRISON_FACTOR)
      const available = Math.max(0, tile.units - garrison)

      if (bestTargetKey !== null) {
        const targetTile = board.get(bestTargetKey)!
        // For neutral attacks: only send surplus over garrison (keeps source tile
        // defended while expanding). For enemy attacks: use full tile.units so we
        // don't under-commit on winnable fights.
        const sendUnits = targetTile.owner === null
          ? available
          : unitsToSend(tile.units, targetTile)

        if (sendUnits > 0) {
          attackPlans.set(key, { targetKey: bestTargetKey, score: bestScore, units: sendUnits })
        } else {
          routingTiles.set(key, 0)
        }
      } else {
        routingTiles.set(key, available)
      }
    }

    // ── Build routing table ───────────────────────────────────────────────────
    // Primary seeds: top-K attack fronts sorted by score DESC — concentrates
    //   interior unit flow toward the highest-leverage breakthrough points.
    // Secondary seeds: stuck frontier tiles — ensures isolated borders still
    //   receive reinforcements, preventing permanent stalemates.

    let routingSeeds: string[]

    if (attackPlans.size > 0) {
      const sortedFronts = [...attackPlans.entries()]
        .sort((a, b) => b[1].score - a[1].score)
        .map(([k]) => k)
      const topFronts = sortedFronts.slice(0, MAX_ROUTING_FRONTS)
      const lowerFronts = sortedFronts.slice(MAX_ROUTING_FRONTS)
      const stuckFrontiers = frontierKeys.filter(k => !attackPlans.has(k))
      routingSeeds = [...topFronts, ...lowerFronts, ...stuckFrontiers]
    } else if (frontierKeys.length > 0) {
      routingSeeds = frontierKeys
    } else {
      // Pre-contact: find frontier tiles directly from the board
      routingSeeds = []
      for (const [key, tile] of board) {
        if (tile.owner !== playerId) continue
        for (const nCoord of hexNeighbors(tile.coord)) {
          const nTile = board.get(hexToKey(nCoord))
          if (nTile && nTile.owner !== playerId) { routingSeeds.push(key); break }
        }
      }
    }

    const routes = buildRoutes(board, playerId, routingSeeds)

    // ── Issue orders ──────────────────────────────────────────────────────────
    const orders: OrderMap = new Map()

    for (const [key, plan] of attackPlans) {
      orders.set(key, { fromKey: key, toKey: plan.targetKey, requestedUnits: plan.units })
    }

    for (const [key, units] of routingTiles) {
      if (units <= 0) continue
      const next = routes.get(key)
      if (!next) continue
      orders.set(key, { fromKey: key, toKey: next, requestedUnits: units })
    }

    return orders
  }

  reset(): void {}
}
