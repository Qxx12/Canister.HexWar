# HexWar

A turn-based hexagonal grid strategy game. Command units across a procedurally generated map, battle five AI opponents simultaneously, and win by capturing all enemy capital tiles.

© 2025 nonaction.net

---

## Gameplay

- The map is a connected hexagonal grid with varied terrain (grassland, plains, desert, tundra)
- Each player starts on a **capital tile** with a small garrison
- Each turn, all owned tiles generate +1 unit (except tiles captured that same turn)
- Issue **movement orders** to send units from one tile to an adjacent tile
- **Combat**: attacker and defender lose equal units; attacker conquers if their units outnumber the defender's
- **Win** by eliminating all other players (capturing their capitals)

### Orders

- Click an owned tile to select it, then click an adjacent tile to set a movement order
- Choose how many units to send using the stepper (+/−/Max)
- Enable **Repeat each turn** to make the order a standing order that auto-replays every turn
- With repeat enabled, choose **∞** to always send all available units from that tile
- Standing orders appear as hollow arrows on the board

### Controls

#### 2D view

| Input | Action |
| ----- | ------ |
| Left-click tile | Select / set order destination |
| Right-drag | Pan the board |
| Scroll / Pinch | Zoom |
| `Enter` | End turn |
| `Esc` | Open menu |

#### 3D view

| Input | Action |
| ----- | ------ |
| Click tile | Select / set order destination |
| Right-drag | Pan the board along the ground plane |
| Scroll | Zoom |
| `Enter` | End turn |
| `Esc` | Open menu |

Toggle between 2D and 3D using the **2D / 3D** button in the bottom-left corner. In 3D mode two additional buttons appear:

| Button | Action |
| ------ | ------ |
| ☀ | Toggle directional sunlight on/off (resets to on each time you enter 3D) |
| ◐ | Toggle tree/object shadows on/off |

---

## Architecture

### Tech Stack

| Layer | Technology |
| ----- | ---------- |
| UI | React 19, TypeScript |
| 3D rendering | Three.js, React Three Fiber, Drei |
| Styling | SCSS Modules |
| Build | Vite 8 |
| Unit testing | Vitest 3, jsdom, Testing Library |
| E2E testing | Playwright |

### Project Structure

```text
packages/
├── engine/             # @hexwar/engine — shared game logic (npm package)
│   ├── src/
│   │   ├── types/          # Shared type definitions (Board, Player, Orders, etc.)
│   │   └── engine/         # Pure game logic (boardGenerator, combat, turnResolver, …)
│   ├── scripts/
│   │   └── generate-fixtures.ts  # Generate cross-engine parity test fixtures
│   └── fixtures/           # Deterministic game traces (committed; used by Python tests)
├── greedy/             # @hexwar/greedy — standalone GreedyAI + shared scoring (npm package)
│   └── src/
│       ├── scoring.ts      # scoreTarget / unitsToSend (shared with @hexwar/strategy)
│       ├── greedyAI.ts     # Pure greedy AI (no strategic layer)
│       └── aiStrategy.ts   # AIStrategy interface
└── strategy/           # @hexwar/strategy — hierarchical AI (npm package)
    └── src/
        ├── assessor/       # Geopolitical snapshot + momentum history
        ├── strategies/     # Strategy catalog (threat, opportunism, alliance, consolidation)
        ├── operational/    # Front mask + interior unit routing
        ├── tactical/       # Constrained greedy order generation (imports scoring from @hexwar/greedy)
        └── highCommand.ts  # HighCommandAI — wires all four layers
src/
├── engine/             # React-side game engine (imports from @hexwar/engine)
│   └── gameEngine.ts      # State transitions, turn management
├── ai/
│   └── aiController.ts    # Per-player AI instances (HighCommandAI / ConquerorAI / WarlordAI); setAiDifficulty + reset
├── hooks/
│   ├── useGameState.ts    # Reducer-based game state management
│   ├── useViewport.ts     # Pan/zoom with pointer and pinch support
│   └── useAnimationQueue.ts # Sequential animation playback
├── components/
│   ├── game/              # In-game UI (board, HUD, modals, tooltips)
│   │   ├── GameBoard.tsx          # 2D SVG board
│   │   ├── GameBoard3D.tsx        # 3D board (Three.js); territory borders, hex grid
│   │   ├── HexTile3D.tsx          # Individual hex prism with terrain texture
│   │   ├── MovementArrow3D.tsx    # Flat arrow overlay for movement orders
│   │   ├── AnimationLayer3D.tsx   # 3D animation playback
│   │   ├── terrainTextures.ts     # Procedural canvas textures per terrain type
│   │   └── ...
│   └── screens/           # Start, Settings, and end screens
└── tests/                 # Vitest unit tests (19 suites)
e2e/                       # Playwright end-to-end tests (42 tests)
research/                  # Python ML research (engine port + AI agents + PPO/GNN training)
├── hexwar/
│   ├── engine/            # Python port of game engine (board, combat, turn resolver)
│   ├── agents/
│   │   ├── greedy_agent.py        # 14-feature linear scoring agent (CMA-ES target)
│   │   ├── evolutionary/          # CMA-ES weight optimiser (cmaes_train.py)
│   │   └── neural/
│   │       ├── gnn_model.py       # HexWarGNN — GATv2Conv, 4 layers, 4 heads (legacy)
│   │       ├── ppo_agent.py       # PPOAgent with K=5 frame stacking (legacy)
│   │       ├── strategist_model.py # StrategistGNN — per-tile GRUCell + global attention
│   │       └── strategist_agent.py # StrategistAgent — maintains per-tile GRU hidden state
│   ├── training/
│   │   ├── rollout_buffer.py      # Transition storage + GAE return computation
│   │   ├── self_play.py           # SelfPlayCollector — population-based self-play (legacy)
│   │   ├── ppo_trainer.py         # PPOTrainer (legacy)
│   │   ├── ppo_train.py           # Legacy PPO training entry point
│   │   ├── league.py              # League — opponent pool (self-play + snapshots + greedy)
│   │   ├── behavioural_cloning.py # BC warm-start: imitate GreedyAgent before PPO
│   │   ├── strategist_collect.py  # StrategistCollector + parallel worker (spawn-safe)
│   │   └── strategist_train.py    # Strategist training entry point (Phase A/B/C)
│   └── evaluation/
│       └── eval_agent.py          # Tournament evaluator vs GreedyAgent baseline
└── tests/                 # Python tests (engine, agents, training)
```

### Game State Flow

```text
playerTurn → [human issues orders] → aiTurn → [AI 1..5 resolve] → generateUnits → playerTurn
```

State lives in a React reducer (`useGameState`). The engine is pure functions — all state mutations return new state objects. Animations are decoupled: `App.tsx` maintains a `displayBoard` that lags one step behind `state.board`, advancing as each animation completes.

### Board Generation

1. Grow a random blob (~166 tiles) from the center using a randomized BFS
2. Punch ~22% of tiles as holes using layered positional noise + rng jitter (while maintaining full connectivity)
3. Assign terrain by latitude (r-axis) with slight sine-wave noise
4. Place player start tiles by maximising minimum pairwise distance across 200 random candidates

### Terrain Textures (3D)

Each terrain type has a procedurally generated canvas texture drawn with a seeded RNG (deterministic, no external assets):

| Terrain | Visual |
| ------- | ------ |
| Grassland | Sparse grass tufts, tiny white flowers |
| Plains | Dry grass strokes, small red/yellow wildflowers |
| Desert | Wavy dune ripples, pebble ellipses |
| Tundra | Snow blobs, olive moss patches, ice-crystal sparkles |

Textures are created once per terrain type and cached for reuse.

### 3D Scene Lighting

The 3D view uses a three-layer lighting model:

- **Ambient light** — low-intensity base fill so surfaces are never fully black
- **Directional sun light** — positioned at the specular mirror of the camera's default viewpoint, so oasis pools catch a specular highlight when the sun is on. Toggled via the ☀ button.
- **Shadow map** — 2048×2048 PCFSoft shadow map; trees, palms, rocks, and deer cast shadows onto tile surfaces. Toggled via the ◐ button.

Tree and palm materials use a small emissive fill (≈20% of their base colour) so the camera-facing sides don't go dark under side lighting.

### Oasis (Desert tiles)

Rare desert tiles (~6% chance) spawn an oasis: an organic water pool with border rocks and 1–2 leaning palm trees. The pool uses a `meshStandardMaterial` with `roughness=0` so it acts as a mirror surface, catching a specular highlight from the directional sun.

### Combat Model

```text
casualties     = min(unitsSent, defendingUnits)
remainingAttackers = unitsSent - casualties
conquered      = remainingAttackers > 0 AND remainingDefenders == 0
```

Orders use the **initial board snapshot** at turn start — units that arrive at a tile mid-turn cannot be chained into another order that same turn.

### AI Strategy

The AI uses a four-layer hierarchical architecture (`@hexwar/strategy`):

```text
GeopoliticalSnapshot → StrategicPlan → TileConstraints → OrderMap
      (assessor)         (strategies)     (operational)   (tactical)
```

#### Strategic layer — High Command
Reads the board once and produces a `FrontDirective` per neighboring player. Each directive carries a `stance` and a unit budget fraction:

| Stance | Meaning |
| ------ | ------- |
| `INVADE` | Full offensive — commit everything |
| `EXPAND` | Push forward when advantageous |
| `HOLD` | Defend; route surplus to other fronts |
| `DETER` | Mass units here to discourage invasion |
| `IGNORE` | Issue no orders toward this neighbor |

**Strategy catalog** (applied in priority order, higher wins):

| Group | Strategies |
| ----- | ---------- |
| Opportunism | Collapse Exploitation (priority 10), Wounded Enemy (6), Vulture Strike (5), Distraction Exploit (5) |
| Alliance | Common Enemy (4), Don't Finish the Decoy (3) |
| Threat | Deterrence Wall (4), Flanking Deterrent (3) |
| Consolidation | Turtle Mode (7 — defensive override) |

A **two-front cap** in the registry limits simultaneous offensive fronts to prevent unit overcommitment.

#### Operational layer
Translates directives into per-tile `TileConstraint` objects. DETER tiles accumulate units in place; interior tiles are BFS-routed toward active fronts through friendly territory.

#### Tactical layer
Greedy scorer that operates only within the allowed targets and unit budgets. Imports `scoreTarget` / `unitsToSend` from `@hexwar/greedy` (neutral: 35, winnable enemy: 50 + advantage × 3, capital bonus: +45), so the scoring logic is shared and not duplicated. A capital garrison of 1 unit is enforced on enemy-targeted orders; neutral-targeted orders are exempt since neutral tiles cannot counter-attack.

See [`packages/strategy/README.md`](packages/strategy/README.md) for the full design and extension guide.

---

## Development

```bash
npm install
npm run dev          # dev server at localhost:5173
npm run build        # production build → dist/
npm run lint         # ESLint
npm run test         # unit tests (watch mode)
npm run test:run     # unit tests (once)
npm run test:e2e     # end-to-end tests (headless, builds first)
npm run test:e2e:ui  # end-to-end tests with Playwright interactive UI
npm run gen-fixtures # regenerate Python cross-engine parity fixtures
```

### ML Research (Python)

```bash
cd research
uv sync                      # install base deps
uv sync --extra neural       # include PyTorch + torch_geometric

# CMA-ES weight optimisation (CPU, works on any machine)
uv run python -m hexwar.agents.evolutionary.cmaes_train \
  --sigma0 0.5 --games 120 --popsize 48 --output runs/cmaes_v3

# StrategistGNN training (GPU recommended)
uv run python -m hexwar.training.strategist_train \
  --device cuda --n-episodes 128 --n-workers 16 --run-dir runs/strategist_v3  # WSL2 / Linux with CUDA
uv run python -m hexwar.training.strategist_train \
  --device mps  --n-episodes 32  --run-dir runs/strategist_v3  # Apple Silicon (no workers)

# Reuse an existing BC checkpoint (skips re-collecting games)
uv run python -m hexwar.training.strategist_train \
  --device cuda --n-episodes 128 --n-workers 16 --run-dir runs/strategist_v3 \
  --weights runs/strategist_v2/ckpt_bc.pt

# Evaluate a checkpoint
uv run python -m hexwar.evaluation.eval_agent \
  --checkpoint runs/strategist_v3/ckpt_iter0050.pt --baseline

# Python tests
uv run pytest tests/ -v
```

**Training curriculum** (`strategist_train.py`):

| Phase | Description | Exit condition |
| ----- | ----------- | -------------- |
| BC — Warm-start | Behavioural cloning: 200 games collected, up to 10K examples used | Fixed, runs once |
| A — Bootstrap | Learner vs 5 × GreedyAgent opponents | Win rate ≥ 25% or 50 iterations |
| B — League self-play | Opponents drawn from snapshots + greedy pool | 500 iterations |
| C — Competitive | Same as B, reduced entropy — encourages decisive play | 200 iterations |

**Neural model** (`StrategistGNN`): GATv2Conv (4 layers, 4 heads, 128-dim) + per-tile GRUCell for infinite temporal memory + one MultiheadAttention layer for global board visibility. Node features are 18-dim (current frame only — GRU accumulates history). Per-order PPO clipping: each move logit + Beta fraction clipped independently.
