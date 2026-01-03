import type { GameState, GamePhase } from '../types/game'
import type { Player, PlayerId } from '../types/player'
import type { Board } from '../types/board'
import type { MovementOrder, OrderMap, AllOrders } from '../types/orders'
import type { AnimationEvent } from '../types/animation'
import type { PlayerStats, EndGameStats } from '../types/stats'
import { generateBoard } from './boardGenerator'
import { generateUnits } from './unitGenerator'
import { resolvePlayerTurn } from './turnResolver'
import { checkWin, checkEliminations } from './winCondition'
import { PLAYER_COLORS, PLAYER_NAMES } from '../types/player'
import { hexToKey, hexNeighbors } from '../types/hex'
import { computeAiOrders } from '../ai/aiController'

const HUMAN_PLAYER_ID: PlayerId = 'p0'
const AI_COUNT = 5

function createInitialStats(playerIds: PlayerId[]): Map<PlayerId, PlayerStats> {
  return new Map(playerIds.map(id => [id, {
    playerId: id,
    unitsGenerated: 6, // start with 1 unit per player × 6 players... actually track from 0
    unitsKilled: 0,
    tilesConquered: 0,
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
    turn: { turnNumber: 1, activeAiIndex: 0 },
    winner: null,
    stats: null,
    runningStats,
  }
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
  animationEvents: AnimationEvent[]
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
    return buildEndState(state, result.board, result.players, result.runningStats, result.winnerId, result.animationEvents)
  }

  const humanPlayer = result.players.find(p => p.id === state.humanPlayerId)
  if (!humanPlayer || humanPlayer.isEliminated) {
    return buildEndState(state, result.board, result.players, result.runningStats, null, result.animationEvents)
  }

  return {
    newState: {
      ...state,
      board: result.board,
      players: result.players,
      runningStats: result.runningStats,
      phase: 'playerTurn', // stay in player turn after executing moves
    },
    animationEvents: result.animationEvents,
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
  animationEvents: AnimationEvent[]
} {
  const aiPlayers = state.players.filter(p => p.type === 'ai' && !p.isEliminated)
  if (aiIndex >= aiPlayers.length) {
    // All AIs done - generate units and start next player turn with fresh orders
    const newBoard = generateUnits(state.board, new Map(state.runningStats))
    const freshOrders: AllOrders = new Map(state.players.map(p => [p.id, new Map()]))
    const nextState: GameState = {
      ...state,
      board: newBoard,
      orders: freshOrders,
      phase: 'playerTurn',
      turn: { turnNumber: state.turn.turnNumber + 1, activeAiIndex: 0 },
    }
    return { newState: nextState, animationEvents: [] }
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
    return buildEndState(state, result.board, result.players, result.runningStats, result.winnerId, result.animationEvents)
  }

  const humanPlayer = result.players.find(p => p.id === state.humanPlayerId)
  if (!humanPlayer || humanPlayer.isEliminated) {
    return buildEndState(state, result.board, result.players, result.runningStats, null, result.animationEvents)
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
    animationEvents: result.animationEvents,
  }
}

function buildEndState(
  state: GameState,
  board: Board,
  players: Player[],
  runningStats: Map<PlayerId, PlayerStats>,
  winnerId: PlayerId | null,
  animationEvents: AnimationEvent[],
): { newState: GameState; animationEvents: AnimationEvent[] } {
  const outcome = winnerId === state.humanPlayerId
    ? 'win'
    : (winnerId === null || players.find(p => p.id === state.humanPlayerId)?.isEliminated)
      ? 'lose'
      : 'lose'

  const finalStats: EndGameStats = {
    outcome,
    winnerId,
    playerStats: players.map(p => {
      const s = runningStats.get(p.id) ?? { playerId: p.id, unitsGenerated: 0, unitsKilled: 0, tilesConquered: 0, tilesAtEnd: 0 }
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
    animationEvents,
  }
}

export function retireGame(state: GameState): GameState {
  const finalStats: EndGameStats = {
    outcome: 'retire',
    winnerId: null,
    playerStats: state.players.map(p => {
      const s = state.runningStats.get(p.id) ?? { playerId: p.id, unitsGenerated: 0, unitsKilled: 0, tilesConquered: 0, tilesAtEnd: 0 }
      const tilesAtEnd = [...state.board.values()].filter(t => t.owner === p.id).length
      return { ...s, tilesAtEnd }
    }),
  }
  return { ...state, phase: 'end', stats: finalStats }
}
