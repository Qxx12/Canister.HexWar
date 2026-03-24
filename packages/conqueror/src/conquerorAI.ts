import type { Board, PlayerId, Player, OrderMap, Tile } from '@hexwar/engine'
import { hexNeighbors, hexToKey } from '@hexwar/engine'

// ─── Constants ────────────────────────────────────────────────────────────────

/** Below this tile count the AI skips garrison entirely — pure expansion. */
const EARLY_PHASE_TILES = 20

/** Garrison fraction for ordinary tiles (late phase). */
const GARRISON_FACTOR = 0.6

/**
 * Garrison fraction for our own start tile.
 * Losing our start tile makes winning IMPOSSIBLE (win condition requires owning it).
 * We never deplete it for a neutral grab regardless of adjacent threat.
 */
const OWN_START_GARRISON_FACTOR = 1.0

/**
 * Players at or below this tile count become elimination targets.
 * Removing a player collapses their front and eliminates their unit income permanently.
 */
const ELIM_THRESHOLD = 4

// ─── Global analysis ──────────────────────────────────────────────────────────

/**
 * Multi-source BFS from every unowned start tile through the whole board.
 * Returns tileKey → distance to the nearest unowned start tile.
 *
 * Scoring uses this for path bonuses; routing uses it to order seeds.
 */
function computeStartGradient(board: Board, playerId: PlayerId): Map<string, number> {
  const dist = new Map<string, number>()
  const queue: string[] = []

  for (const [key, tile] of board) {
    if (tile.isStartTile && tile.owner !== playerId) {
      dist.set(key, 0)
      queue.push(key)
    }
  }

  let qi = 0
  while (qi < queue.length) {
    const key = queue[qi++]
    const tile = board.get(key)
    if (!tile) continue
    const d = dist.get(key)!
    for (const nCoord of hexNeighbors(tile.coord)) {
      const nKey = hexToKey(nCoord)
      if (dist.has(nKey) || !board.has(nKey)) continue
      dist.set(nKey, d + 1)
      queue.push(nKey)
    }
  }

  return dist
}

/** Players with ≤ ELIM_THRESHOLD tiles are near elimination. */
function findNearElimPlayers(board: Board, playerId: PlayerId): Set<PlayerId> {
  const counts = new Map<PlayerId, number>()
  for (const tile of board.values()) {
    if (tile.owner && tile.owner !== playerId) {
      counts.set(tile.owner, (counts.get(tile.owner) ?? 0) + 1)
    }
  }
  const result = new Set<PlayerId>()
  for (const [id, count] of counts) {
    if (count <= ELIM_THRESHOLD) result.add(id)
  }
  return result
}

// ─── Scoring ──────────────────────────────────────────────────────────────────

function pathBonus(dist: number | undefined): number {
  if (dist === undefined) return 0
  if (dist === 1) return 20
  if (dist === 2) return 10
  if (dist === 3) return 5
  return 0
}

function scoreTarget(
  myUnits: number,
  target: Tile,
  targetKey: string,
  earlyPhase: boolean,
  isNearElim: boolean,
  startGradient: Map<string, number>,
  board: Board,
): number {
  const startBonus = target.isStartTile ? 45 : 0

  if (target.owner === null) {
    return (earlyPhase ? 55 : 35) + startBonus + pathBonus(startGradient.get(targetKey))
  }

  if (myUnits <= target.units) return target.isStartTile ? 18 : -1

  const elimBonus = isNearElim ? 60 : 0
  let s = 50 + (myUnits - target.units) * 3 + startBonus + elimBonus
  s += pathBonus(startGradient.get(targetKey))

  for (const nnCoord of hexNeighbors(target.coord)) {
    if (board.get(hexToKey(nnCoord))?.owner === null) s += 8
  }
  return s
}

function unitsToSend(myUnits: number, target: Tile): number {
  if (target.owner === null) return myUnits
  if (myUnits <= target.units) return myUnits
  return Math.min(myUnits, Math.max(target.units + 1, Math.ceil(myUnits * 0.85)))
}

// ─── Coordinated attacks ──────────────────────────────────────────────────────

/**
 * Detects "bleed + kill" joint attacks across a player's turn.
 *
 * The turn resolver processes orders SEQUENTIALLY and mutates the board between
 * each order (see turnResolver.ts). If a "bleeder" tile sends its units into a
 * stronger enemy first, the enemy's HP drops. A "killer" tile whose order is
 * inserted immediately after then sees the weakened enemy and can conquer it —
 * even though neither tile could win alone.
 *
 * Only tiles with NO viable solo attack are eligible (those in blockedFrontier).
 * For each enemy tile with 2+ eligible adjacent friendlies:
 *   bleeder = tile with the smallest available units (sacrificed to weaken enemy)
 *   killer  = tile with the largest tile.units     (finishes the fight)
 *
 * Returns a list of joint attacks ordered by priority. The INSERTION ORDER of
 * bleeders-then-killers in the final OrderMap determines execution order.
 */
type JointAttack = {
  enemyKey: string
  bleederKey: string
  killerKey: string
  bleederSend: number
  killerSend: number
  score: number
  seedPriority: number
}

function detectJointAttacks(
  board: Board,
  playerId: PlayerId,
  blockedFrontier: Map<string, { tile: Tile; available: number }>,
  nearElimPlayers: Set<PlayerId>,
  startGradient: Map<string, number>,
): JointAttack[] {
  // Group blocked tiles by the enemy tile they are adjacent to
  const byEnemy = new Map<string, Array<{ key: string; tile: Tile; available: number }>>()

  for (const [key, { tile, available }] of blockedFrontier) {
    for (const nCoord of hexNeighbors(tile.coord)) {
      const nKey = hexToKey(nCoord)
      const nTile = board.get(nKey)
      if (!nTile || nTile.owner === null || nTile.owner === playerId) continue
      if (!byEnemy.has(nKey)) byEnemy.set(nKey, [])
      byEnemy.get(nKey)!.push({ key, tile, available })
    }
  }

  const attacks: JointAttack[] = []
  const assigned = new Set<string>()

  for (const [enemyKey, friendlies] of byEnemy) {
    if (friendlies.length < 2) continue
    const enemyTile = board.get(enemyKey)!

    const eligible = friendlies.filter(f => !assigned.has(f.key))
    if (eligible.length < 2) continue

    // Smallest available → bleeder (sacrificed); largest tile.units → killer
    eligible.sort((a, b) => a.available - b.available)
    const bleeder = eligible[0]
    const killer = eligible[eligible.length - 1]

    if (bleeder.available <= 0) continue

    // After bleeder sends all available units, the enemy has this many left
    const enemyAfterBleed = Math.max(0, enemyTile.units - bleeder.available)

    // Killer must be able to finish the weakened enemy
    if (killer.tile.units <= enemyAfterBleed) continue

    const killerSend = Math.min(
      killer.tile.units,
      Math.max(enemyAfterBleed + 1, Math.ceil(killer.tile.units * 0.85)),
    )

    const isNearElim = nearElimPlayers.has(enemyTile.owner!)
    const startBonus = enemyTile.isStartTile ? 45 : 0
    const elimBonus = isNearElim ? 60 : 0
    const targetDist = startGradient.get(enemyKey) ?? Infinity
    const score = 50 + (killer.tile.units - enemyTile.units) * 3
      + startBonus + elimBonus + pathBonus(startGradient.get(enemyKey))

    let seedPriority = 3
    if (isNearElim) seedPriority = 0
    else if (enemyTile.isStartTile) seedPriority = 1
    else if (targetDist <= 2) seedPriority = 2

    attacks.push({ enemyKey, bleederKey: bleeder.key, killerKey: killer.key, bleederSend: bleeder.available, killerSend, score, seedPriority })
    assigned.add(bleeder.key)
    assigned.add(killer.key)
  }

  return attacks
}

// ─── Routing ──────────────────────────────────────────────────────────────────

/**
 * Multi-source BFS outward from `seeds` through friendly territory.
 * Returns: interiorKey → nextHop one step closer to the nearest seed.
 * Seeds processed in the order given — high-priority seeds first biases flow.
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

// ─── ConquerorAI ──────────────────────────────────────────────────────────────

/**
 * ConquerorAI — objective-aware AI with coordinated multi-tile attacks.
 *
 * Core mechanisms:
 *
 *   1. Start-tile gradient   — BFS from every unowned start tile gives each tile
 *                              a distance-to-objective. Path bonuses (+5/+10/+20)
 *                              steer attacks toward the win condition; routing
 *                              seeds are ordered by this gradient.
 *
 *   2. Elimination priority  — Players with ≤ 4 tiles earn +60 on attack scoring.
 *                              Finishing them collapses a full front.
 *
 *   3. Coordinated attacks   — The turn resolver processes orders sequentially,
 *                              mutating the board between each. Tiles that cannot
 *                              win alone are detected and paired: the "bleeder"
 *                              (smaller available) is inserted first into the
 *                              OrderMap so it fires first, weakening the enemy;
 *                              the "killer" (larger units) fires second and
 *                              conquers the now-reachable target.
 *
 *   4. Own start tile        — Uses OWN_START_GARRISON_FACTOR = 1.0 (vs 0.6 for
 *                              ordinary tiles). Losing the start tile makes
 *                              winning impossible; we never deplete it for a
 *                              neutral grab.
 *
 *   5. Objective-aware seeds — Routing BFS seeds ordered:
 *                              elim fronts → start-tile fronts → near-gradient
 *                              → other → stuck, including joint-attack killer
 *                              fronts so interior units pipeline correctly.
 *
 *   6. Neutral-adjacent bonus — +8 per neutral bordering an enemy target.
 */
export class ConquerorAI {
  computeOrders(
    board: Board,
    playerId: PlayerId,
    _currentOrders: OrderMap,
    _allPlayers: Player[],
  ): OrderMap {
    // ── Global analysis ───────────────────────────────────────────────────────
    let myTileCount = 0
    for (const tile of board.values()) {
      if (tile.owner === playerId) myTileCount++
    }
    const earlyPhase = myTileCount < EARLY_PHASE_TILES
    const startGradient = computeStartGradient(board, playerId)
    const nearElimPlayers = findNearElimPlayers(board, playerId)

    // ── Per-tile analysis ─────────────────────────────────────────────────────
    type AttackPlan = { targetKey: string; score: number; units: number; seedPriority: number }

    const attackPlans = new Map<string, AttackPlan>()
    const routingTiles = new Map<string, number>()
    const frontierKeys: string[] = []

    // Tiles at the frontier with no viable solo attack — candidates for joint attacks
    const blockedFrontier = new Map<string, { tile: Tile; available: number }>()

    for (const [key, tile] of board) {
      if (tile.owner !== playerId || tile.units === 0) continue

      const ownStart = tile.isStartTile    // tile is our own start tile

      let maxAdjEnemy = 0
      let hasAdjEnemy = false
      let isFrontier = false

      for (const nCoord of hexNeighbors(tile.coord)) {
        const nTile = board.get(hexToKey(nCoord))
        if (!nTile || nTile.owner === playerId) continue
        isFrontier = true
        if (nTile.owner !== null) {
          hasAdjEnemy = true
          maxAdjEnemy = Math.max(maxAdjEnemy, nTile.units)
        }
      }

      if (isFrontier) frontierKeys.push(key)

      let bestTargetKey: string | null = null
      let bestScore = 0

      for (const nCoord of hexNeighbors(tile.coord)) {
        const nKey = hexToKey(nCoord)
        const nTile = board.get(nKey)
        if (!nTile || nTile.owner === playerId) continue

        const isNearElim = nTile.owner !== null && nearElimPlayers.has(nTile.owner)
        const s = scoreTarget(tile.units, nTile, nKey, earlyPhase, isNearElim, startGradient, board)
        if (s > bestScore) { bestScore = s; bestTargetKey = nKey }
      }

      // Own start tile uses a higher garrison so we never deplete it for neutrals
      const garrisonFactor = ownStart ? OWN_START_GARRISON_FACTOR : GARRISON_FACTOR
      const garrison = earlyPhase ? 0 : Math.ceil(maxAdjEnemy * garrisonFactor)
      const available = Math.max(0, tile.units - garrison)

      if (bestTargetKey !== null) {
        const targetTile = board.get(bestTargetKey)!
        const sendUnits = targetTile.owner === null
          ? available
          : unitsToSend(tile.units, targetTile)

        if (sendUnits > 0) {
          const isNearElimTarget = targetTile.owner !== null && nearElimPlayers.has(targetTile.owner)
          const targetDist = startGradient.get(bestTargetKey) ?? Infinity

          let seedPriority = 3
          if (isNearElimTarget) seedPriority = 0
          else if (targetTile.isStartTile) seedPriority = 1
          else if (targetDist <= 2) seedPriority = 2

          attackPlans.set(key, { targetKey: bestTargetKey, score: bestScore, units: sendUnits, seedPriority })
        } else {
          routingTiles.set(key, 0)
        }
      } else {
        routingTiles.set(key, available)

        // Frontier tile with adjacent enemies but no viable solo target — pool for joint attacks
        if (hasAdjEnemy && available > 0) {
          blockedFrontier.set(key, { tile, available })
        }
      }
    }

    // ── Coordinated (joint) attacks ───────────────────────────────────────────
    // Detect pairs of blocked tiles that can together defeat an enemy neither
    // could beat alone, exploiting sequential order resolution within a turn.
    const jointAttacks = detectJointAttacks(board, playerId, blockedFrontier, nearElimPlayers, startGradient)

    // Remove joint attack participants from the routing pool
    for (const ja of jointAttacks) {
      routingTiles.delete(ja.bleederKey)
      routingTiles.delete(ja.killerKey)
    }

    // ── Build routing table ───────────────────────────────────────────────────
    // Collect all active attack fronts (solo + joint killers) sorted by priority.
    // Interior units pipeline toward the highest-value front via BFS from seeds.
    type FrontEntry = { key: string; score: number; prio: number }
    const frontEntries: FrontEntry[] = []

    for (const [k, plan] of attackPlans) {
      frontEntries.push({ key: k, score: plan.score, prio: plan.seedPriority })
    }
    for (const ja of jointAttacks) {
      // Killer tile is the effective front (it advances into enemy territory)
      frontEntries.push({ key: ja.killerKey, score: ja.score, prio: ja.seedPriority })
    }

    frontEntries.sort((a, b) => a.prio !== b.prio ? a.prio - b.prio : b.score - a.score)
    const stuckFrontiers = frontierKeys.filter(k => !attackPlans.has(k) && !jointAttacks.some(ja => ja.killerKey === k))
    const routingSeeds: string[] = [...frontEntries.map(e => e.key), ...stuckFrontiers]

    if (routingSeeds.length === 0) {
      // Pre-contact: seed from any tile touching non-friendly territory
      for (const [key, tile] of board) {
        if (tile.owner !== playerId) continue
        for (const nCoord of hexNeighbors(tile.coord)) {
          const nTile = board.get(hexToKey(nCoord))
          if (nTile && nTile.owner !== playerId) { routingSeeds.push(key); break }
        }
      }
    }

    const routes = buildRoutes(board, playerId, routingSeeds)

    // ── Issue orders (insertion order controls sequential execution!) ──────────
    //
    // The turn resolver iterates the OrderMap in insertion order. For joint
    // attacks we MUST insert all bleeders before all killers so the bleed
    // completes before the kill is attempted.
    //
    //   1. Bleeders  — weakens the enemy this turn
    //   2. Killers   — finishes the weakened enemy in the same turn
    //   3. Solo attacks
    //   4. Routing

    const orders: OrderMap = new Map()

    // 1. Bleeders first
    for (const ja of jointAttacks) {
      orders.set(ja.bleederKey, { fromKey: ja.bleederKey, toKey: ja.enemyKey, requestedUnits: ja.bleederSend })
    }

    // 2. Killers second (board has already been updated by the bleeders)
    for (const ja of jointAttacks) {
      orders.set(ja.killerKey, { fromKey: ja.killerKey, toKey: ja.enemyKey, requestedUnits: ja.killerSend })
    }

    // 3. Solo attacks
    for (const [key, plan] of attackPlans) {
      orders.set(key, { fromKey: key, toKey: plan.targetKey, requestedUnits: plan.units })
    }

    // 4. Routing
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
