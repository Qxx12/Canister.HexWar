# @hexwar/strategy

Hierarchical AI for HexWar. A **High Command** layer reasons about geopolitics — which enemies to fight, which to deter, which to ignore — and hands a reduced problem to a **tactical layer** that generates individual unit orders.

The design keeps the four layers fully decoupled so any tactical backend (greedy, MCTS, learned) can be plugged in under the same strategic shell.

---

## Architecture

```
Board + Players + TurnHistory
          │
          ▼
┌──────────────────────┐
│   STRATEGIC LAYER    │  Reads the full board once; produces per-border directives
│   (High Command)     │  "Deter Rome, invade Egypt, ignore Greece"
└─────────┬────────────┘
          │  StrategicPlan  (Map<PlayerId, FrontDirective>)
          ▼
┌──────────────────────┐
│  OPERATIONAL LAYER   │  Translates directives into tile-level constraints
│  (Theater Command)   │  Which tiles may attack, how many units, where to route
└─────────┬────────────┘
          │  Map<string, TileConstraint> + interiorRoutes
          ▼
┌──────────────────────┐
│    TACTICAL LAYER    │  Greedy scorer that operates within the constraints
│ (Constrained Greedy) │  Identical scoring logic to GreedyAI, smaller search space
└──────────────────────┘
          │  OrderMap
          ▼
     Turn Resolver
```

---

## Strategic Directives

Every neighbor gets exactly one `FrontDirective` per turn:

| Stance | Meaning | Unit budget |
|--------|---------|-------------|
| `INVADE` | Full offensive — commit everything | 100% of border units |
| `EXPAND` | Push forward, take tiles when possible | 80% |
| `HOLD` | Don't attack; route excess units to other fronts | 40% |
| `DETER` | Mass units on this border to discourage invasion | 0% (accumulate in place) |
| `IGNORE` | Issue no orders toward this neighbor at all | 0% |

---

## Strategy Catalog

Strategies are pure functions `(GeopoliticalSnapshot) → PartialDirective[]`. They run in sequence; the registry merges outputs by priority (higher wins). The **two-front cap** (registry post-process) ensures at most 2 simultaneous offensive fronts.

### Threat group

**S1 – Deterrence Wall**
We face a strong enemy (they outgun our border) and a weak one simultaneously. Mass units against the strong neighbor — their high risk of losses deters invasion — while freely expanding against the weak one. Reduces active conflict tiles by roughly half.

**S4 – Flanking Deterrent**
Surrounded on 3+ sides and outgunned everywhere. Deter the two strongest borders; probe only the weakest for a breakout. Prevents being simultaneously crushed from all directions.

### Opportunism group

**S5 – Vulture Strike**
Two neighbors are fighting each other and both shrinking (`momentumDelta ≤ -2`). Hold both borders until one drops to near-elimination, then rush in. Saves our units entirely while they deplete each other.

**S6 – Collapse Exploitation** *(priority 10 — highest)*
Any neighbor reaches ≤ 3 tiles. Invade immediately at full budget before another player absorbs their territory. Eliminating a player removes one entire front.

**S7 – Wounded Enemy**
A neighbor recently lost many tiles (`momentumDelta ≤ -3`) but is not yet near elimination. Escalate to `INVADE` while their interior is thin and their units are spent.

**S8 – Distraction Exploit**
A neighbor is fighting someone else and losing tiles there. Their border facing us is weaker than usual. Upgrade to `EXPAND` and take advantage of the gap.

### Alliance group

**S12 – Common Enemy**
One neighbor is 50%+ larger than the average by tile count. Focus everything on them; soft-ignore the smallest neighbor (implicit non-aggression for that turn). Prevents the dominant player from snowballing unchecked.

**S13 – Don't Finish the Decoy**
A nearly-eliminated player is already being attacked by others. Let them waste units on the kill; `IGNORE` that border and spend our budget elsewhere.

### Consolidation group

**S14 – Turtle Mode** *(priority 7)*
We are shrinking on two or more fronts simultaneously (`myMomentumDelta ≤ -3`). Set all borders to `HOLD` and stop spending units until the situation stabilises.

---

## Operational Layer

**Front Mask** (`frontMask.ts`)
Converts directives into `TileConstraint` per owned tile:
- `INVADE`/`EXPAND` tiles: `crossBorderAllowed = true`, enemy targets included
- `DETER` tiles: `maxUnitsFraction = 0` — units accumulate in place
- `HOLD` / interior tiles: free to route toward active fronts via friendly hops
- `IGNORE` neighbors: their tiles excluded from `allowedTargetKeys` entirely

**Interior Router** (`interiorRouter.ts`)
BFS outward from active front tiles through friendly territory. Produces a routing table `interiorKey → nextHopKey` so deep interior tiles pipeline their units to the front automatically.

---

## Usage

```typescript
import { HighCommandAI } from '@hexwar/strategy'

// Create one instance per AI player; reuse each turn (history persists)
const agent = new HighCommandAI()

const orders = agent.computeOrders(board, playerId, currentOrders, allPlayers)

// On game reset:
agent.reset()
```

In the HexWar app, `aiController.ts` maintains a `Map<PlayerId, HighCommandAI>` and calls `resetAiAgents()` on restart.

---

## Package layout

```
src/
├── types.ts                        — All plan/directive/assessment types
├── assessor/
│   ├── historyTracker.ts           — 5-turn ring buffer of tile counts per player
│   └── neighborAssessor.ts         — Single-pass board scan → GeopoliticalSnapshot
├── strategies/
│   ├── registry.ts                 — Merge engine + two-front cap
│   ├── threatGroup.ts              — S1 Deterrence Wall, S4 Flanking Deterrent
│   ├── opportunismGroup.ts         — S5 Vulture, S6 Collapse, S7 Wounded, S8 Distraction
│   ├── allianceGroup.ts            — S12 Common Enemy, S13 Decoy Ignore
│   └── consolidationGroup.ts       — S14 Turtle Mode
├── operational/
│   ├── frontMask.ts                — Plan → TileConstraint per source tile
│   └── interiorRouter.ts           — BFS routing table for interior units
├── tactical/
│   └── constrainedGreedy.ts        — Greedy scorer + capital garrison enforcement
└── highCommand.ts                  — HighCommandAI: wires all four layers
```

---

## Extending with new strategies

1. Add a function `myStrategy(snapshot: GeopoliticalSnapshot): PartialDirective[]` in the appropriate group file (or a new one).
2. Import it in `highCommand.ts` and add it to the `STRATEGIES` array. Later entries take precedence at equal priority, so order matters.
3. Write a unit test in `src/tests/strategies.test.ts` with a handcrafted `GeopoliticalSnapshot`.

Priority guide: 1–3 background, 4–6 opportunism, 7–8 defensive overrides, 9–10 emergency (collapse).
