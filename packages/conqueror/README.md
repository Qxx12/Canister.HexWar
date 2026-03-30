# @hexwar/conqueror

ConquerorAI is the primary AI player in HexWar. It beats both WarlordAI and GreedyAI by organising every decision around the actual win condition: capture all start tiles.

---

## Why it beats WarlordAI

WarlordAI is myopic — each tile independently picks the best adjacent target with no awareness of where the finish line is. ConquerorAI adds four layers of global awareness:

1. **Start-tile gradient** — BFS from every unowned start tile gives each board tile a distance-to-objective. Attacks on "gateway" tiles (one hop from a start tile) earn a path bonus, and interior unit routing concentrates flow along the shortest corridor to the objective.

2. **Elimination priority** — any player with ≤ 4 tiles earns a large bonus on attacks against their tiles. Finishing a player collapses their entire front and removes their future unit income permanently.

3. **Coordinated (bleed + kill) attacks** — the turn resolver processes orders *sequentially*, mutating the board between each. ConquerorAI exploits this: a "bleeder" tile (smaller available units) is inserted into the `OrderMap` first, sacrificing its available units to weaken an enemy that neither tile could beat alone. The "killer" tile fires second in the same turn and conquers the now-reachable enemy. This unlocks territory that would otherwise be permanently blocked.

4. **Objective-aware routing seeds** — the BFS that pipelines interior units toward fronts is seeded in priority order: elimination fronts → start-tile fronts → near-gradient fronts → other fronts. Interior units no longer spread evenly; they funnel toward whichever breakthrough matters most.

---

## Scoring

| Target type | Score |
| --- | --- |
| Neutral (early phase, < 20 tiles) | 55 |
| Neutral (late phase) | 35 |
| Neutral start tile (early) | 55 + 45 = 100 |
| Beatable enemy | 50 + (advantage × 3) |
| Beatable enemy start tile | 50 + (advantage × 3) + 45 |
| Beatable near-elimination enemy | 50 + (advantage × 3) + 60 |
| Unbeatable enemy start tile | 18 (gamble) |
| Unbeatable enemy | −1 (skip) |
| Path bonus — adjacent to unowned start tile | +20 |
| Path bonus — 2 hops from unowned start tile | +10 |
| Path bonus — 3 hops from unowned start tile | +5 |
| Per neutral adjacent to enemy target | +8 |

---

## Routing seed priority

Interior units follow a BFS routing table whose seeds are ordered:

| Priority | Seed type |
| --- | --- |
| 0 — highest | Fronts attacking a near-elimination player |
| 1 | Fronts attacking an enemy start tile |
| 2 | Fronts attacking a tile ≤ 2 hops from an unowned start tile |
| 3 | All other attack fronts |
| 4 — lowest | Stuck frontier tiles (can't attack, need reinforcement) |

Within each priority tier, seeds are ordered by score descending.

---

## Garrison

| Phase | Tile type | Formula |
| --- | --- | --- |
| Early (< 20 tiles) | all | 0 — no garrison, pure expansion |
| Late | ordinary tile | `ceil(maxAdjacentEnemy × 0.6)` |
| Late | **own start tile** | `ceil(maxAdjacentEnemy × 1.0)` |

The 0.6 factor for ordinary tiles is deliberately leaner than WarlordAI (1.0) — over-garrisoning stalls the attack. The start tile uses 1.0 because losing it makes winning *impossible* (`checkWin` requires owning your own start tile): we never deplete it chasing a neutral.

---

## Benchmark

Tested in 2 400-game 6-player FFA runs (2 ConquerorAI vs 2 WarlordAI vs 2 GreedyAI, shuffled start positions):

| Agent | Win rate |
| --- | --- |
| ConquerorAI | ~38% |
| WarlordAI | ~32% |
| GreedyAI | ~30% |

Expected if equal strength: 33.3% each.

---

## Usage

```typescript
import { ConquerorAI } from '@hexwar/conqueror'

const ai = new ConquerorAI()

// Call once per turn for a given player.
const orders = ai.computeOrders(board, playerId, currentOrders, allPlayers)
```

`computeOrders` returns an `OrderMap` (`Map<string, Order>`) compatible with the HexWar engine's `resolvePlayerTurn`.

One instance per player is recommended (the AI is currently stateless between turns, but `reset()` is available for future use).

In the HexWar app, `aiController.ts` selects the agent class based on the difficulty chosen on the Settings screen (Soldier → HighCommandAI, Commander → ConquerorAI, Warlord → WarlordAI). It maintains one instance per player and clears them with `resetAiAgents()` on restart.
