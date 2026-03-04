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

**2D view**

| Input | Action |
|-------|--------|
| Click tile | Select / set order destination |
| Drag | Pan the board |
| Scroll / Pinch | Zoom |
| `Enter` | End turn |
| `Esc` | Open menu |

**3D view**

| Input | Action |
|-------|--------|
| Click tile | Select / set order destination |
| Right-drag | Pan the board along the ground plane |
| Scroll | Zoom |
| `Enter` | End turn |
| `Esc` | Open menu |

Toggle between 2D and 3D using the **2D / 3D** button in the bottom-left corner.

---

## Architecture

### Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | React 19, TypeScript |
| 3D rendering | Three.js, React Three Fiber, Drei |
| Styling | SCSS Modules |
| Build | Vite 8 |
| Unit testing | Vitest 3, jsdom, Testing Library |
| E2E testing | Playwright |

### Project Structure

```
src/
├── types/          # Shared type definitions
│   ├── board.ts    # Tile, Board, TerrainType
│   ├── game.ts     # GameState, GamePhase
│   ├── hex.ts      # Axial coordinates, neighbor/distance math
│   ├── orders.ts   # MovementOrder, OrderMap, UNITS_ALL
│   ├── player.ts   # Player, colors, names
│   └── ...
├── engine/         # Pure game logic (no React)
│   ├── gameEngine.ts      # State transitions, turn management
│   ├── turnResolver.ts    # Order execution, combat, animation events
│   ├── boardGenerator.ts  # Procedural map generation
│   ├── unitGenerator.ts   # Per-turn unit spawning
│   ├── combat.ts          # Casualty calculation, conquest
│   └── winCondition.ts    # Elimination and victory checks
├── ai/
│   ├── greedyAI.ts        # BFS-based frontier routing + attack scoring
│   └── aiController.ts    # AI order computation entry point
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
│   └── screens/           # Start and end screens
└── tests/                 # Vitest unit tests (91 tests across 10 suites)
e2e/                       # Playwright end-to-end tests (29 tests)
```

### Game State Flow

```
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
|---------|--------|
| Grassland | Sparse grass tufts, tiny white flowers |
| Plains | Dry grass strokes, small red/yellow wildflowers |
| Desert | Wavy dune ripples, pebble ellipses |
| Tundra | Snow blobs, olive moss patches, ice-crystal sparkles |

Textures are created once per terrain type and cached for reuse.

### Combat Model

```
casualties     = min(unitsSent, defendingUnits)
remainingAttackers = unitsSent - casualties
conquered      = remainingAttackers > 0 AND remainingDefenders == 0
```

Orders use the **initial board snapshot** at turn start — units that arrive at a tile mid-turn cannot be chained into another order that same turn.

### AI Strategy

The AI runs a two-phase decision per turn:

1. **Classify tiles** — frontier (adjacent to non-friendly) vs interior
2. **BFS routing map** — from all frontier tiles inward; each interior tile records its next step toward the nearest border
3. **Frontier tiles** evaluate attack targets scored by:
   - Neutral tile: 35
   - Winnable enemy: 50 + advantage × 3
   - Enemy capital bonus: +45
   - Losing fight: −1 (skipped), unless enemy capital (18)
4. **Interior tiles** send all units toward the front via the routing map, creating a reinforcement pipeline

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
```
