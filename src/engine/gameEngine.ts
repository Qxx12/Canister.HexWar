import type { GameState } from '../types/game'
import type { Player, PlayerId } from '../types/player'
import type { Board } from '../types/board'
import type { MovementOrder, AllOrders } from '../types/orders'
import type { PlayerStats, EndGameStats } from '../types/stats'
import type { TurnStep } from './turnResolver'
import { generateBoard } from './boardGenerator'
import { generateUnits } from './unitGenerator'
import { resolvePlayerTurn } from './turnResolver'
import { PLAYER_COLORS, PLAYER_NAMES } from '../types/player'
import { computeAiOrders } from '../ai/aiController'

const HUMAN_PLAYER_ID: PlayerId = 'p0'
const AI_COUNT = 5

function createInitialStats(playerIds: PlayerId[]): Map<PlayerId, PlayerStats> {
  return new Map(playerIds.map(id => [id, {
    playerId: id,
    unitsGenerated: 0,
    unitsKilled: 0,
    tilesConquered: 0,
    tilesLost: 0,
    tilesAtEnd: 0,
  }]))
}

export function initGame(): GameState {
  const playerIds: PlayerId[] = Array.from({ length: 1 + AI_COUNT }, (_, i) => `p${i}`)
  const players: Player[] = playerIds.map((id, i) => ({
    id,
    type: i === 0 ? 'human' : 'ai',
    color: PLAYER_COLORS[i],
    name: PLAYER_NAMES[i],
    isEliminated: false,
  }))

  const board = generateBoard(playerIds)
  const orders: AllOrders = new Map(playerIds.map(id => [id, new Map()]))
  const runningStats = createInitialStats(playerIds)

  return {
    phase: 'playerTurn',
    board,
    players,
    humanPlayerId: HUMAN_PLAYER_ID,
    orders,
    humanStandingOrders: new Map(),
    turn: { turnNumber: 1, activeAiIndex: 0 },
    winner: null,
    stats: null,
    runningStats,
  }
}

export function setHumanStandingOrder(state: GameState, order: MovementOrder): GameState {
  const newStanding = new Map(state.humanStandingOrders)
  newStanding.set(order.fromKey, order)
  return { ...state, humanStandingOrders: newStanding }
}

export function cancelHumanStandingOrder(state: GameState, fromKey: string): GameState {
  const newStanding = new Map(state.humanStandingOrders)
  newStanding.delete(fromKey)
  return { ...state, humanStandingOrders: newStanding }
}

function cleanupStandingOrders(standing: OrderMap, board: Board, humanPlayerId: PlayerId): OrderMap {
  const cleaned = new Map(standing)
  for (const [fromKey] of cleaned) {
    const tile = board.get(fromKey)
    if (!tile || tile.owner !== humanPlayerId) cleaned.delete(fromKey)
  }
  return cleaned
}

function applyStandingOrdersToFreshOrders(
  freshOrders: AllOrders,
  standing: OrderMap,
  board: Board,
  humanPlayerId: PlayerId,
): AllOrders {
  const humanOrders = new Map<string, MovementOrder>()
  for (const [fromKey, order] of standing) {
    const tile = board.get(fromKey)
    if (tile && tile.owner === humanPlayerId && tile.units > 0) {
      humanOrders.set(fromKey, order)
    }
  }
  const result = new Map(freshOrders)
  result.set(humanPlayerId, humanOrders)
  return result
}

export function applyHumanOrder(state: GameState, order: MovementOrder): GameState {
  const humanOrders = new Map(state.orders.get(state.humanPlayerId) ?? new Map())
  humanOrders.set(order.fromKey, order)
  const newOrders = new Map(state.orders)
  newOrders.set(state.humanPlayerId, humanOrders)
  return { ...state, orders: newOrders }
}

export function cancelHumanOrder(state: GameState, fromKey: string): GameState {
  const humanOrders = new Map(state.orders.get(state.humanPlayerId) ?? new Map())
  humanOrders.delete(fromKey)
  const newOrders = new Map(state.orders)
  newOrders.set(state.humanPlayerId, humanOrders)
  return { ...state, orders: newOrders }
}

export function executeHumanMoves(state: GameState): {
  newState: GameState
  steps: TurnStep[]
} {
  const humanOrders = state.orders.get(state.humanPlayerId) ?? new Map()
  const result = resolvePlayerTurn(
    state.board,
    state.players,
    humanOrders,
    state.humanPlayerId,
    new Map(state.runningStats),
  )

  if (result.winnerId) {
    return buildEndState(state, result.board, result.players, result.runningStats, result.winnerId, result.steps)
  }

  const humanPlayer = result.players.find(p => p.id === state.humanPlayerId)
  if (!humanPlayer || humanPlayer.isEliminated) {
    return buildEndState(state, result.board, result.players, result.runningStats, null, result.steps)
  }

  const cleanedStanding = cleanupStandingOrders(state.humanStandingOrders, result.board, state.humanPlayerId)
  return {
    newState: {
      ...state,
      board: result.board,
      players: result.players,
      runningStats: result.runningStats,
      humanStandingOrders: cleanedStanding,
      phase: 'playerTurn',
    },
    steps: result.steps,
  }
}

export function endHumanTurn(state: GameState): GameState {
  return {
    ...state,
    phase: 'aiTurn',
    turn: { ...state.turn, activeAiIndex: 0 },
  }
}

export function resolveAiTurn(state: GameState, aiIndex: number): {
  newState: GameState
  steps: TurnStep[]
} {
  const aiPlayers = state.players.filter(p => p.type === 'ai' && !p.isEliminated)
  if (aiIndex >= aiPlayers.length) {
    // All AIs done — generate units and start next player turn
    const updatedStats = new Map(state.runningStats)
    const newBoard = generateUnits(state.board, updatedStats)
    const cleanedStanding = cleanupStandingOrders(state.humanStandingOrders, newBoard, state.humanPlayerId)
    const freshOrders: AllOrders = new Map(state.players.map(p => [p.id, new Map()]))
    const ordersWithStanding = applyStandingOrdersToFreshOrders(freshOrders, cleanedStanding, newBoard, state.humanPlayerId)
    const nextState: GameState = {
      ...state,
      board: newBoard,
      orders: ordersWithStanding,
      humanStandingOrders: cleanedStanding,
      runningStats: updatedStats,
      phase: 'playerTurn',
      turn: { turnNumber: state.turn.turnNumber + 1, activeAiIndex: 0 },
    }
    return { newState: nextState, steps: [] }
  }

  const aiPlayer = aiPlayers[aiIndex]
  const aiOrders = state.orders.get(aiPlayer.id) ?? new Map()
  const freshAiOrders = computeAiOrders(state.board, aiPlayer.id, aiOrders, state.players)

  const newOrdersMap = new Map(state.orders)
  newOrdersMap.set(aiPlayer.id, freshAiOrders)

  const result = resolvePlayerTurn(
    state.board,
    state.players,
    freshAiOrders,
    aiPlayer.id,
    new Map(state.runningStats),
  )

  if (result.winnerId) {
    return buildEndState(state, result.board, result.players, result.runningStats, result.winnerId, result.steps)
  }

  const humanPlayer = result.players.find(p => p.id === state.humanPlayerId)
  if (!humanPlayer || humanPlayer.isEliminated) {
    return buildEndState(state, result.board, result.players, result.runningStats, null, result.steps)
  }

  return {
    newState: {
      ...state,
      board: result.board,
      players: result.players,
      runningStats: result.runningStats,
      orders: newOrdersMap,
      phase: 'aiTurn',
      turn: { ...state.turn, activeAiIndex: aiIndex + 1 },
    },
    steps: result.steps,
  }
}

function buildEndState(
  state: GameState,
  board: Board,
  players: Player[],
  runningStats: Map<PlayerId, PlayerStats>,
  winnerId: PlayerId | null,
  steps: TurnStep[],
): { newState: GameState; steps: TurnStep[] } {
  const outcome = winnerId === state.humanPlayerId ? 'win' : 'lose'

  const finalStats: EndGameStats = {
    outcome,
    winnerId,
    playerStats: players.map(p => {
      const s = runningStats.get(p.id) ?? { playerId: p.id, unitsGenerated: 0, unitsKilled: 0, tilesConquered: 0, tilesLost: 0, tilesAtEnd: 0 }
      const tilesAtEnd = [...board.values()].filter(t => t.owner === p.id).length
      return { ...s, tilesAtEnd }
    }),
  }

  return {
    newState: {
      ...state,
      board,
      players,
      runningStats,
      phase: 'end',
      winner: winnerId,
      stats: finalStats,
    },
    steps,
  }
}

export function retireGame(state: GameState): GameState {
  const finalStats: EndGameStats = {
    outcome: 'retire',
    winnerId: null,
    playerStats: state.players.map(p => {
      const s = state.runningStats.get(p.id) ?? { playerId: p.id, unitsGenerated: 0, unitsKilled: 0, tilesConquered: 0, tilesLost: 0, tilesAtEnd: 0 }
      const tilesAtEnd = [...state.board.values()].filter(t => t.owner === p.id).length
      return { ...s, tilesAtEnd }
    }),
  }
  return { ...state, phase: 'end', stats: finalStats }
}
