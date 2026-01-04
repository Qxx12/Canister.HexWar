import { useReducer, useCallback } from 'react'
import type { GameState } from '../types/game'
import type { MovementOrder } from '../types/orders'
import type { TurnStep } from '../engine/turnResolver'
import {
  initGame,
  applyHumanOrder,
  cancelHumanOrder,
  executeHumanMoves,
  endHumanTurn,
  resolveAiTurn,
  retireGame,
} from '../engine/gameEngine'

type GameAction =
  | { type: 'START_GAME' }
  | { type: 'SET_ORDER'; order: MovementOrder }
  | { type: 'CANCEL_ORDER'; fromKey: string }
  | { type: 'EXECUTE_HUMAN_MOVES'; onSteps: (steps: TurnStep[]) => void }
  | { type: 'END_HUMAN_TURN' }
  | { type: 'RESOLVE_AI'; aiIndex: number; onSteps: (steps: TurnStep[]) => void }
  | { type: 'RETIRE' }
  | { type: 'RESET' }

function gameReducer(state: GameState | null, action: GameAction): GameState | null {
  if (action.type === 'START_GAME' || action.type === 'RESET') {
    return initGame()
  }
  if (!state) return state

  switch (action.type) {
    case 'SET_ORDER':
      return applyHumanOrder(state, action.order)
    case 'CANCEL_ORDER':
      return cancelHumanOrder(state, action.fromKey)
    case 'EXECUTE_HUMAN_MOVES': {
      const { newState, steps } = executeHumanMoves(state)
      action.onSteps(steps)
      return newState
    }
    case 'END_HUMAN_TURN':
      return endHumanTurn(state)
    case 'RESOLVE_AI': {
      const { newState, steps } = resolveAiTurn(state, action.aiIndex)
      action.onSteps(steps)
      return newState
    }
    case 'RETIRE':
      return retireGame(state)
    default:
      return state
  }
}

export function useGameState() {
  const [state, dispatch] = useReducer(gameReducer, null)

  const startGame = useCallback(() => dispatch({ type: 'START_GAME' }), [])
  const resetGame = useCallback(() => dispatch({ type: 'RESET' }), [])
  const setOrder = useCallback((order: MovementOrder) => dispatch({ type: 'SET_ORDER', order }), [])
  const cancelOrder = useCallback((fromKey: string) => dispatch({ type: 'CANCEL_ORDER', fromKey }), [])
  const executeHumanMovesAction = useCallback((onSteps: (s: TurnStep[]) => void) =>
    dispatch({ type: 'EXECUTE_HUMAN_MOVES', onSteps }), [])
  const endTurn = useCallback(() => dispatch({ type: 'END_HUMAN_TURN' }), [])
  const resolveAi = useCallback((aiIndex: number, onSteps: (s: TurnStep[]) => void) =>
    dispatch({ type: 'RESOLVE_AI', aiIndex, onSteps }), [])
  const retire = useCallback(() => dispatch({ type: 'RETIRE' }), [])

  return { state, startGame, resetGame, setOrder, cancelOrder, executeHumanMovesAction, endTurn, resolveAi, retire }
}
