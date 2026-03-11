/**
 * Generates cross-engine parity fixtures.
 *
 * Each fixture captures a seeded game run with zero orders (no moves), which
 * exercises unit generation and win/elimination detection in isolation from
 * AI divergence. The Python engine loads the same initial board from the
 * fixture and replays the turns; any difference surfaces a rules divergence.
 *
 * Usage:
 *   node --experimental-strip-types packages/engine/scripts/generate-fixtures.ts
 *
 * Outputs JSON to packages/engine/fixtures/ (committed to the repo).
 */

import { generateBoard, resolvePlayerTurn, generateUnits, checkWin } from '../src/index.ts'
import type { Board, Player, PlayerId, PlayerStats } from '../src/index.ts'
import { writeFileSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))

// ---------------------------------------------------------------------------
// Seeded PRNG — Mulberry32
// Must match the Python implementation in hexwar-ai/tests/test_engine_parity.py
// ---------------------------------------------------------------------------
function mulberry32(seed: number): () => number {
  let s = seed >>> 0
  return function () {
    s = ((s + 0x6D2B79F5) >>> 0)
    let t = Math.imul(s ^ (s >>> 15), s | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// ---------------------------------------------------------------------------
// Serialisation
// ---------------------------------------------------------------------------
type SerializedBoard = Record<string, {
  owner: string | null
  units: number
  terrain: string
  isStartTile: boolean
  startOwner: string | null
  newlyConquered: boolean
  coord: { q: number; r: number }
}>

function serializeBoard(board: Board): SerializedBoard {
  const out: SerializedBoard = {}
  for (const [key, tile] of board) {
    out[key] = {
      owner: tile.owner,
      units: tile.units,
      terrain: tile.terrain,
      isStartTile: tile.isStartTile,
      startOwner: tile.startOwner,
      newlyConquered: tile.newlyConquered,
      coord: { q: tile.coord.q, r: tile.coord.r },
    }
  }
  return out
}

function makeStats(playerIds: PlayerId[]): Map<PlayerId, PlayerStats> {
  return new Map(playerIds.map(id => [id, {
    playerId: id,
    unitsGenerated: 0,
    unitsKilled: 0,
    tilesConquered: 0,
    tilesLost: 0,
    tilesAtEnd: 0,
  }]))
}

// ---------------------------------------------------------------------------
// Fixture runner — zero-order turns
// ---------------------------------------------------------------------------
const PLAYER_IDS: PlayerId[] = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5']
const MAX_TURNS = 30

function runFixture(seed: number) {
  const rng = mulberry32(seed)
  const initialBoard = generateBoard(PLAYER_IDS, rng)

  let board: Board = new Map(initialBoard)
  let players: Player[] = PLAYER_IDS.map((id, i) => ({
    id,
    type: i === 0 ? 'human' : 'ai',
    color: '#000000',
    name: id,
    isEliminated: false,
  }))
  const runningStats = makeStats(PLAYER_IDS)

  const turns: unknown[] = []
  let winner: PlayerId | null = null
  let turnNumber = 0

  outer: for (let t = 0; t < MAX_TURNS; t++) {
    turnNumber = t + 1
    const playerSnapshots: { playerId: string; boardAfter: SerializedBoard }[] = []

    // Each player resolves with zero orders
    for (const pid of PLAYER_IDS) {
      const player = players.find(p => p.id === pid)
      if (!player || player.isEliminated) {
        playerSnapshots.push({ playerId: pid, boardAfter: {} })
        continue
      }

      const result = resolvePlayerTurn(board, players, new Map(), pid, new Map(runningStats))
      board = result.board
      players = result.players
      for (const [k, v] of result.runningStats) runningStats.set(k, v)

      playerSnapshots.push({ playerId: pid, boardAfter: serializeBoard(board) })

      if (result.winnerId) {
        winner = result.winnerId
        turns.push({ turn: turnNumber, playerSnapshots, boardAfterUnitGen: null })
        break outer
      }
    }

    // Unit generation
    board = generateUnits(board, runningStats)

    turns.push({
      turn: turnNumber,
      playerSnapshots,
      boardAfterUnitGen: serializeBoard(board),
    })

    winner = checkWin(board, players)
    if (winner) break
  }

  return {
    seed,
    playerIds: PLAYER_IDS,
    initialBoard: serializeBoard(initialBoard),
    turns,
    winner,
    turnsPlayed: turnNumber,
  }
}

// ---------------------------------------------------------------------------
// Generate and write fixtures
// ---------------------------------------------------------------------------
const SEEDS = [42, 137, 999, 2024, 31415]
const outDir = join(__dirname, '../fixtures')
mkdirSync(outDir, { recursive: true })

for (const seed of SEEDS) {
  const fixture = runFixture(seed)
  const outPath = join(outDir, `fixture_seed_${seed}.json`)
  writeFileSync(outPath, JSON.stringify(fixture, null, 2))
  console.log(`Written: ${outPath}  (turns=${fixture.turnsPlayed}, winner=${fixture.winner ?? 'none'})`)
}

console.log('\nDone. Commit packages/engine/fixtures/ to the repo.')
