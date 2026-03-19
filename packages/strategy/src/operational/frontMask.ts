import type { Board, PlayerId } from '@hexwar/engine'
import { hexNeighbors, hexToKey } from '@hexwar/engine'
import type { StrategicPlan, TileConstraint } from '../types.ts'

/**
 * Translates a StrategicPlan into per-tile constraints for the tactical layer.
 *
 * For each owned tile:
 *  - Friendly neighbor targets are always allowed (interior routing).
 *  - Neutral targets are allowed when neutralExpansionEnabled.
 *  - Enemy targets are allowed only when the directive stance is EXPAND or INVADE.
 *  - DETER tiles accumulate units by setting maxUnitsFraction = 0 (no orders).
 *  - HOLD tiles can route half their units toward active fronts via interior moves.
 *  - IGNORE neighbors are excluded from allowedTargetKeys entirely.
 */
export function buildFrontMask(
  board: Board,
  myPlayerId: PlayerId,
  plan: StrategicPlan,
): Map<string, TileConstraint> {
  const constraints = new Map<string, TileConstraint>()

  for (const [key, tile] of board) {
    if (tile.owner !== myPlayerId || tile.units === 0) continue

    const allowedTargetKeys: string[] = []
    let crossBorderAllowed = false
    let bestOffensiveBudget = 0
    let facingDeterOnly = false
    let facingAnyEnemy = false

    for (const nCoord of hexNeighbors(tile.coord)) {
      const nKey = hexToKey(nCoord)
      const nTile = board.get(nKey)
      if (!nTile) continue

      if (nTile.owner === myPlayerId) {
        // Always include friendly tiles for interior routing
        allowedTargetKeys.push(nKey)
        continue
      }

      if (nTile.owner === null) {
        if (plan.neutralExpansionEnabled) {
          allowedTargetKeys.push(nKey)
          crossBorderAllowed = true
          bestOffensiveBudget = Math.max(bestOffensiveBudget, 0.8)
        }
        continue
      }

      // Enemy tile — check directive
      const directive = plan.directives.get(nTile.owner)
      if (!directive) continue

      facingAnyEnemy = true

      switch (directive.stance) {
        case 'INVADE':
        case 'EXPAND':
          allowedTargetKeys.push(nKey)
          crossBorderAllowed = true
          bestOffensiveBudget = Math.max(bestOffensiveBudget, directive.unitBudgetFraction)
          facingDeterOnly = false
          break
        case 'DETER':
          // Don't cross — hold position. Mark so we can set budget = 0.
          if (!crossBorderAllowed) facingDeterOnly = true
          break
        case 'HOLD':
          // Don't attack this turn; allow interior routing through
          facingDeterOnly = false
          break
        case 'IGNORE':
          // Exclude this neighbor's tiles from targets entirely
          break
      }
    }

    // Determine unit budget for this tile
    let maxUnitsFraction: number
    if (crossBorderAllowed) {
      // Offensive tile — spend up to the strategic budget
      maxUnitsFraction = bestOffensiveBudget
    } else if (facingAnyEnemy && facingDeterOnly) {
      // Pure deterrence tile — hold all units here (no outgoing orders)
      maxUnitsFraction = 0
    } else {
      // Interior or HOLD frontier — can route units toward active fronts
      maxUnitsFraction = 1.0
    }

    if (allowedTargetKeys.length > 0) {
      constraints.set(key, {
        sourceKey: key,
        allowedTargetKeys,
        maxUnitsFraction,
        crossBorderAllowed,
      })
    }
  }

  return constraints
}
