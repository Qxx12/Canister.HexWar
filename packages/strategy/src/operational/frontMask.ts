import type { Board, PlayerId } from '@hexwar/engine'
import { hexNeighbors, hexToKey } from '@hexwar/engine'
import type { StrategicPlan, TileConstraint } from '../types.ts'

/**
 * Translates a StrategicPlan into per-tile constraints for the tactical layer.
 *
 * For each owned tile we separately track:
 *  - hasOffensiveCrossing: an INVADE or EXPAND enemy is adjacent
 *  - hasDeterEnemy: a DETER enemy is adjacent (accumulate here, never cross)
 *  - hasHoldEnemy: a HOLD enemy is adjacent
 *  - hasNeutralTarget: a neutral tile is adjacent (with neutralExpansionEnabled)
 *
 * Resolution priority:
 *  1. Offensive (INVADE/EXPAND) → crossBorderAllowed, directive budget (neutral is free on top)
 *  2. DETER → never cross, budget=0 — even if neutral tiles are adjacent
 *  3. Neutral only (HOLD or no enemies) → cross, 0.5 budget when facing HOLD enemy, 1.0 otherwise
 *  4. Interior / HOLD frontier with no neutral → no crossing, budget=1.0 for interior routing
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
    let hasNeutralTarget = false
    let hasOffensiveCrossing = false
    let hasDeterEnemy = false
    let hasHoldEnemy = false
    let offensiveBudget = 0

    for (const nCoord of hexNeighbors(tile.coord)) {
      const nKey = hexToKey(nCoord)
      const nTile = board.get(nKey)
      if (!nTile) continue

      if (nTile.owner === myPlayerId) {
        // Always allow friendly tiles for interior routing
        allowedTargetKeys.push(nKey)
        continue
      }

      if (nTile.owner === null) {
        if (plan.neutralExpansionEnabled) {
          allowedTargetKeys.push(nKey)
          hasNeutralTarget = true
        }
        continue
      }

      // Enemy tile — check directive
      const directive = plan.directives.get(nTile.owner)
      if (!directive) continue

      switch (directive.stance) {
        case 'INVADE':
        case 'EXPAND':
          allowedTargetKeys.push(nKey)
          hasOffensiveCrossing = true
          offensiveBudget = Math.max(offensiveBudget, directive.unitBudgetFraction)
          break
        case 'DETER':
          hasDeterEnemy = true
          break
        case 'HOLD':
          hasHoldEnemy = true
          break
        case 'IGNORE':
          // Excluded from targets entirely
          break
      }
    }

    // Resolve crossing permission and unit budget
    let crossBorderAllowed: boolean
    let maxUnitsFraction: number

    if (hasOffensiveCrossing) {
      // Offensive front — attack enemies; also grab adjacent neutrals for free
      crossBorderAllowed = true
      maxUnitsFraction = hasNeutralTarget
        ? Math.max(offensiveBudget, 1.0)
        : offensiveBudget
    } else if (hasDeterEnemy) {
      // DETER: accumulate units here — do NOT cross even if neutral tiles are adjacent
      crossBorderAllowed = false
      maxUnitsFraction = 0
    } else if (hasNeutralTarget) {
      // HOLD frontier or interior facing neutral targets — expand, but conservatively
      // when we have a defensive obligation against a HOLD enemy
      crossBorderAllowed = true
      maxUnitsFraction = hasHoldEnemy ? 0.5 : 1.0
    } else {
      // Interior or HOLD frontier with no neutral — route toward active fronts
      crossBorderAllowed = false
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
