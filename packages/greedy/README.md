# @hexwar/greedy

Standalone greedy AI for HexWar, plus the shared scoring functions used by `@hexwar/strategy`'s tactical layer.

---

## Contents

| Export | Purpose |
| ------ | ------- |
| `GreedyAI` | Self-contained AI: classifies tiles, routes units, scores targets — no strategic layer |
| `scoreTarget` | Score a potential attack target given attacker unit count and target tile |
| `unitsToSend` | How many units to commit to an attack given attacker count and target tile |
| `AIStrategy` | Interface satisfied by `GreedyAI` (and `HighCommandAI`) |

---

## Scoring heuristics

### `scoreTarget(myUnits, target)`

| Situation | Score |
| --------- | ----- |
| Neutral tile | 35 |
| Neutral capital tile | 80 (35 + 45) |
| Enemy tile, losing or tied | −1 (skip) |
| Enemy capital tile, losing or tied | 18 (gamble) |
| Enemy tile, winning | 50 + advantage × 3 |
| Enemy capital tile, winning | 95 + advantage × 3 (50 + 3adv + 45) |

Tiles with a score ≤ 0 are never attacked.

### `unitsToSend(myUnits, target)`

| Situation | Units sent |
| --------- | --------- |
| Neutral target | All units (`myUnits`) |
| Losing or tied against enemy | All units (desperate) |
| Winning against enemy | `max(target.units + 1, floor(myUnits × 0.85))` |

---

## How `GreedyAI` works

1. **Classify** — scan all owned tiles; mark any tile adjacent to a non-friendly tile as a **frontier** tile. The rest are **interior**.
2. **Route** — BFS outward from frontier tiles through friendly territory, building a `nextTowardFront` table for interior tiles.
3. **Order**
   - Frontier tiles: score all non-friendly neighbours with `scoreTarget`; issue an order to the best one (if score > 0).
   - Interior tiles: follow `nextTowardFront` and pipeline all units toward the nearest frontier.

No garrison is enforced — `GreedyAI` commits freely. Capital garrison logic lives in `@hexwar/strategy`'s constrained tactical layer.

---

## Usage

```ts
import { GreedyAI } from '@hexwar/greedy'

const ai = new GreedyAI()
const orders = ai.computeOrders(board, playerId, currentOrders, allPlayers)
```

### Using scoring functions directly

```ts
import { scoreTarget, unitsToSend } from '@hexwar/greedy'

const score = scoreTarget(myTile.units, targetTile)
if (score > 0) {
  const send = unitsToSend(myTile.units, targetTile)
}
```

---

## Relationship to `@hexwar/strategy`

`@hexwar/strategy` imports `scoreTarget` and `unitsToSend` from this package for its constrained tactical layer. This keeps the scoring logic in one place rather than duplicated across both AIs.

```text
@hexwar/greedy          @hexwar/strategy
─────────────           ────────────────
GreedyAI   ◄──────      HighCommandAI
scoreTarget ──────►      constrainedGreedy.ts
unitsToSend ──────►
```
