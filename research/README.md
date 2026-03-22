# hexwar-ai

A deep machine-learning study for building a strong AI for [HexWar](../Canister.HexWar) — a 6-player hex-grid strategy game.

This project contains:

- A faithful Python port of the HexWar game engine (for headless simulation)
- AI agents in four tiers: random → greedy → evolutionary (CMA-ES) → neural (PPO + GNN)
- Training infrastructure for self-play and evaluation
- A tournament/Elo evaluation framework

The trained neural AI will eventually be exported (ONNX) and imported back into the browser game.

---

## Architecture

```text
hexwar/
├── engine/          # Faithful Python port of the JS game engine
│   ├── types.py         — Core dataclasses (Tile, Board, Player, etc.)
│   ├── hex_utils.py     — Axial coordinate math
│   ├── board_generator.py — Procedural map generation
│   ├── combat.py        — Deterministic combat resolution
│   ├── unit_generator.py — End-of-round unit generation
│   ├── win_condition.py — Win / elimination detection
│   ├── turn_resolver.py — Per-player order execution (with snapshot cap)
│   └── game_engine.py   — Headless HexWarEnv simulation loop
│
├── agents/
│   ├── base_agent.py    — Abstract base class
│   ├── random_agent.py  — Uniformly random moves (baseline)
│   ├── greedy_agent.py  — Linear feature scoring (tunable weights)
│   ├── evolutionary/
│   │   ├── fitness.py       — Multi-game fitness evaluation
│   │   └── cmaes_train.py   — CMA-ES optimiser for greedy weights
│   └── neural/
│       ├── history_buffer.py — Circular buffer of K board snapshots (frame stacking)
│       ├── encoder.py       — Board → PyG Data (node/edge/global features)
│       ├── gnn_model.py     — GATv2Conv policy + value network
│       └── ppo_agent.py     — PPO agent wrapping the GNN
│
├── training/
│   ├── reward.py        — Dense + sparse reward shaping
│   ├── rollout_buffer.py — GAE rollout buffer
│   ├── self_play.py     — Self-play rollout collection
│   └── ppo_trainer.py   — PPO-clip policy update
│
└── evaluation/
    ├── tournament.py    — Round-robin tournament
    └── elo.py           — Multi-player Elo rating system
```

---

## Game Engine Port

The Python engine is a line-by-line port of the TypeScript source in `src/engine/`. Key fidelity guarantees:

| Concept | JS source | Python port |
| - | - | - |
| Hex coordinates | `src/types/hex.ts` | `hexwar/engine/hex_utils.py` |
| Board generation | `src/engine/boardGenerator.ts` | `hexwar/engine/board_generator.py` |
| Combat model | `src/engine/combat.ts` | `hexwar/engine/combat.py` |
| Turn resolution | `src/engine/turnResolver.ts` | `hexwar/engine/turn_resolver.py` |
| Unit generation | `src/engine/unitGenerator.ts` | `hexwar/engine/unit_generator.py` |
| Win condition | `src/engine/winCondition.ts` | `hexwar/engine/win_condition.py` |

### Cross-engine parity

Deterministic game traces (zero-order play — only unit generation runs) are generated from the TS engine and committed as JSON fixtures:

```bash
# In monorepo root — regenerate after TS engine changes
npm run gen-fixtures
```

The Python test suite replays these fixtures and asserts identical board state after every player turn and unit generation step:

```bash
uv run pytest tests/test_engine_parity.py
```

Fixture files live at `packages/engine/fixtures/` and should be committed alongside any TS engine change that affects game logic.

### Combat model

```text
Neutral / friendly tile (owner is None or same player):
  → No casualties. Attacker moves in and stacks units.
  → conquered = True only for neutral tiles.

Hostile tile (different owner):
  casualties = min(units_sent, defending_units)
  remaining_attackers = units_sent − casualties
  remaining_defenders = defending_units − casualties
  conquered = remaining_attackers > 0 AND remaining_defenders == 0
```

### Snapshot cap (anti-chaining)

Each player's orders are capped to units present **at the start of their turn** (not units that arrived via a previous order this turn). This mirrors the JS `initialBoard` snapshot and prevents chain-moving.

---

## AI Tiers

### Tier 1: Random Agent

Issues a random valid move from every owned tile each turn. Win rate ≈ 1/N_players. Used as a baseline.

### Tier 2: Greedy Agent

Scores each candidate move (source tile → adjacent target) with a linear combination of 8 hand-crafted features and picks the highest-scoring move per tile.

**Features:**

| Index | Name | Description |
| - | - | - |
| 0 | can_conquer | 1 if we have more units than defender |
| 1 | is_start_tile | 1 if target is a start tile |
| 2 | expand_neutral | 1 if target is unowned |
| 3 | attack_enemy | 1 if target is an enemy tile |
| 4 | units_advantage | `(sent − defending) / sent` |
| 5 | relative_tile_count | `(ours − theirs) / total` |
| 6 | border_exposure | fraction of source's neighbors that are enemy |
| 7 | reinforce_friendly | 1 if target is owned by us |

**Key design rules:**

- Never leave own start tile with 0 units (retains ≥ 1 unit defensively)
- Only reinforces friendly tiles that are on the frontier (has non-owned neighbors) — prevents oscillation between interior tiles
- Default weights tuned so `attack_enemy` alone (non-winning attack) scores below neutral expansion

### Tier 3: Evolutionary Agent (CMA-ES)

Optimises the Greedy Agent's 8-dimensional weight vector using [CMA-ES](https://arxiv.org/abs/1604.00772) (Covariance Matrix Adaptation Evolution Strategy). Evaluation fitness is the greedy agent's win rate over 60 games against 5 default-greedy opponents, parallelised across CPU cores. The candidate rotates through all 6 player slots evenly to remove positional bias.

```bash
python scripts/train_cmaes.py --generations 200 --games 20 --output runs/cmaes
```

CMA-ES is well-suited because:

- The search space is continuous, low-dimensional (8-dim), and noisy
- No gradient is available (game outcomes are stochastic)
- CMA-ES handles non-separable, ill-conditioned landscapes

### Tier 4: Neural Agent (PPO + GNN)

A graph neural network (GNN) policy trained with Proximal Policy Optimisation (PPO) via self-play.

#### Temporal frame stacking

The model sees the last **K = 5** board states stacked per tile, giving it an explicit view of unit movement trajectories across turns. This lets it detect:

- **Massing attacks** — a tile whose units grow from 0 → 3 → 9 → 22 over 4 turns
- **Evacuations** — a tile whose units fall from 15 → 11 → 6 → 2 → 0 (units moving toward the border)
- **Recent conquests** — a tile whose owner changes from enemy → neutral → mine

A `HistoryBuffer` (circular deque, `maxlen=K`) stores deep-copied board snapshots and is pre-filled with the initial state so that turn 1 already has K frames.

#### Graph representation

Each turn is encoded as a graph where:

- **Nodes** = tiles with a **`18 + (K−1) × 4`-dim** feature vector per tile
  - *Current frame (18-dim):* unit count, ownership, terrain one-hot, hex coordinates, neighbor pressure, max enemy-neighbor units
  - *Each past frame (4-dim):* unit count, is-mine, is-enemy, is-neutral — oldest history last
- **Edges** = adjacency (3-dim: same-owner, enemy-border, unit-strength proxy)
- **Global features** = tile/unit fractions per player (12-dim)

At **K = 5**: node feature dimension = 18 + 4 × 4 = **34-dim**.

**Node feature layout:**

| Columns | Content | Dim |
| - | - | - |
| 0–17 | Current frame (full features) | 18 |
| 18–21 | Frame t−1 (units, owner×3) | 4 |
| 22–25 | Frame t−2 | 4 |
| 26–29 | Frame t−3 | 4 |
| 30–33 | Frame t−4 | 4 |

#### Model architecture

```text
K board snapshots (oldest → newest)
        │
        ▼ HistoryBuffer.get_frames()
Node features [N, 18 + (K-1)×4]
        │
        ▼
Node encoder MLP → hidden_dim
        │
        ▼  + Global broadcast (12-dim → hidden_dim)
4× GATv2Conv (multi-head, skip connection, edge_dim=3)
        │
       ┌┴──────────────────────────┐
       ▼                           ▼
Edge action head              Value head
  move_logit (1-dim)          mean-pool (all nodes)
  Beta(α, β) for fraction   + mean-pool (acting-player nodes)
                              → scalar V(s)
```

#### Action space

For each directed edge (owned source → any adjacent target), the model outputs:

1. **move_logit** — selection score (Categorical over valid edges, invalid edges masked to −∞)
2. **(α, β)** — Beta distribution parameters for fraction of units to send

At inference: sample edge ∝ softmax(logits), sample fraction ~ Beta(α, β).
At evaluation: argmax edge, fraction = mode of Beta.

#### PPO agent

`PPOAgent` maintains a per-game `HistoryBuffer`:

```python
agent = PPOAgent(history_k=5)
agent.reset(initial_board)   # pre-fill buffer at game start

# Each turn:
orders = agent(board, player_id, players, stats)
# Internally: push board to buffer, encode K-frame stack, run GNN
```

#### Training

```bash
python scripts/train_ppo.py --iterations 1000 --episodes 16 --output runs/ppo
```

Self-play setup:

- 6-player games; learner is assigned to a random player slot per episode
- Other 5 slots filled from a checkpoint pool for diversity
- GAE (λ=0.95, γ=0.99) for advantage estimation
- PPO-clip (ε=0.2), 4 epochs per update
- Dense rewards: tile progress + conquest bonus + elimination bonus
- Terminal rewards: +1 win, −1 loss

---

## Reward Shaping

```text
terminal win:           +1.0
terminal loss:          −1.0
eliminated mid-game:    −0.5
tile progress per tile: ±0.01
conquest (enemy tile):  +0.02
fight (partial kill):   +0.005 × units_killed
elimination bonus:      +0.5 × n_players_eliminated
```

All dense rewards are smaller than the terminal ±1 signal to ensure the agent prioritises winning over farming.

---

## Evaluation

### Tournament

```bash
python scripts/run_tournament.py --games 100
```

Runs a round-robin between registered agents and reports win rates, average turns, and average tiles held.

### Elo ratings

The `EloSystem` tracks per-agent Elo ratings using all-pairs comparison from multi-player games.

---

## Installation

This project lives at `research/` inside the [Canister.HexWar](../) monorepo. It uses [uv](https://docs.astral.sh/uv/) for isolated environment management.

```bash
cd research

# Install uv (once)
brew install uv

# Create venv and install all dependencies
uv sync

# Neural agent extras (requires PyTorch + torch_geometric)
uv sync --extra neural
```

### Dependencies

| Package | Purpose |
| - | - |
| `cma` | CMA-ES optimisation |
| `torch` | PyTorch (neural agents) |
| `torch_geometric` | Graph neural networks |
| `pytest` | Testing |
| `ruff` | Linting + formatting |

---

## Quick Start

```python
from hexwar.engine.game_engine import HexWarEnv
from hexwar.engine.types import PLAYER_IDS
from hexwar.agents.greedy_agent import GreedyAgent
from hexwar.agents.random_agent import RandomAgent

# Run a game: 1 greedy vs 5 random
agents = {pid: RandomAgent(seed=i) for i, pid in enumerate(PLAYER_IDS)}
agents["p1"] = GreedyAgent()

env = HexWarEnv(agents=agents, seed=42, max_turns=300)
result = env.run()

print(f"Winner: {result.winner_id}")
print(f"Turns played: {result.turns_played}")
```

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check .

# Run tournament
uv run python scripts/run_tournament.py --games 100
```

---

## Roadmap

- [x] Python engine port (board generation, combat, turn resolution, win condition)
- [x] Random agent (baseline)
- [x] Greedy agent (hand-tuned heuristics)
- [x] CMA-ES evolutionary optimisation
- [x] GNN + PPO neural agent
- [x] Frame-stacked temporal encoding (K=5, 34-dim node features)
- [x] Self-play training infrastructure
- [x] Tournament + Elo evaluation
- [ ] ONNX export for browser integration
- [ ] Curriculum training (start against random, progress to self-play)
- [ ] Population-based training (PBT) for hyperparameter tuning
- [ ] Per-tile LSTM as an alternative to frame stacking (future enhancement)
- [ ] Integrate trained model back into Canister.HexWar
