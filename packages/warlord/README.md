# @hexwar/warlord

WarlordAI is the primary AI player in HexWar. It consistently outperforms GreedyAI in free-for-all benchmarks (>50% collective win rate in 3v3 FFA).

---

## How it works

### 1. Phase detection

The AI tracks how many tiles it owns. Below **20 tiles** it is in **early phase** — pure expansion with no garrison. Above that it switches to **late phase** — balanced attack with defensive garrisons.

### 2. Target scoring

Each frontier tile scores all adjacent non-friendly targets:

| Target type | Score |
| --- | --- |
| Neutral (early phase) | 55 |
| Neutral (late phase) | 35 |
| Beatable enemy | 50 + (advantage × 3) |
| Unbeatable enemy start tile | 18 (gamble) |
| Unbeatable enemy | −1 (skip) |
| Start tile bonus (any) | +45 |
| Per neutral adjacent to enemy target | +8 |

The neutral-adjacent bonus is the key differentiator: it steers attacks toward enemy tiles that border open territory, keeping expansion options alive after the capture.

### 3. Garrison

In late phase, frontier tiles reserve `ceil(maxAdjacentEnemy × 1.0)` units before sending the surplus forward. This prevents counter-attacks from immediately reclaiming newly captured tiles.

### 4. Routing

Interior tiles (no non-friendly neighbour) have nowhere to attack directly. WarlordAI runs a multi-source BFS outward from all active attack fronts (sorted by score, top 6 prioritised) plus any stuck frontier tiles. Each interior tile follows the route one hop toward the nearest front.

### 5. Unit commitment

| Situation | Units sent |
| --- | --- |
| Neutral target | All available (garrison already subtracted) |
| Winning against enemy | `max(enemy + 1, ceil(myUnits × 0.85))` |
| Unbeatable start tile (gamble) | All units |

---

## Benchmark

Tested in 4 800-game 6-player FFA runs (3 WarlordAI vs 3 GreedyAI, shuffled start positions):

| Agent | Win rate |
| --- | --- |
| WarlordAI | ~50–51% |
| GreedyAI | ~49–50% |

In a 3-way benchmark (WarlordAI / HighCommandAI / GreedyAI):

| Agent | Win rate |
| --- | --- |
| WarlordAI | ~44% |
| GreedyAI | ~40% |
| HighCommandAI | ~16% |

---

## Usage

```typescript
import { WarlordAI } from '@hexwar/warlord'

const ai = new WarlordAI()

// Call once per turn for a given player.
const orders = ai.computeOrders(board, playerId, currentOrders, allPlayers)
```

`computeOrders` returns an `OrderMap` (`Map<string, Order>`) compatible with the HexWar engine's `resolvePlayerTurn`.

One instance per player is recommended (the AI is currently stateless between turns, but `reset()` is available for future use).

WarlordAI has been superseded by `@hexwar/conqueror` as the primary in-game AI. It remains useful as a benchmark baseline and for understanding the design evolution.
