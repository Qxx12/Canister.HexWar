import { useEffect, useCallback, useState } from 'react'
import { useGameState } from './hooks/useGameState'
import { useViewport } from './hooks/useViewport'
import { useAnimationQueue } from './hooks/useAnimationQueue'
import type { AnimationStep } from './hooks/useAnimationQueue'
import { GameBoard } from './components/game/GameBoard'
import { GameHUD } from './components/game/GameHUD'
import { StartScreen } from './components/screens/StartScreen'
import { EndScreen } from './components/screens/EndScreen'
import type { MovementOrder } from './types/orders'
import type { Board } from './types/board'
import type { TurnStep } from './engine/turnResolver'
import './styles/main.scss'

type AppScreen = 'start' | 'game' | 'end'

export default function App() {
  const [screen, setScreen] = useState<AppScreen>('start')
  const { state, startGame, resetGame, setOrder, cancelOrder, executeHumanMovesAction, endTurn, resolveAi, retire } = useGameState()
  const viewport = useViewport()
  const { activeEvent, enqueue, clearQueue } = useAnimationQueue()
  const [isAnimating, setIsAnimating] = useState(false)

  // Display board lags behind state.board — updated one step at a time as animations play.
  // state.board (logic board) is always fully resolved and used by the engine.
  const [displayBoard, setDisplayBoard] = useState<Board | null>(null)

  const handleStartGame = useCallback(() => {
    startGame()
    setScreen('game')
  }, [startGame])

  const handleRestart = useCallback(() => {
    resetGame()
    clearQueue()
    setDisplayBoard(null)
    setScreen('start')
  }, [resetGame, clearQueue])

  const handleSetOrder = useCallback((fromKey: string, toKey: string, units: number) => {
    if (!state) return
    const order: MovementOrder = { fromKey, toKey, requestedUnits: units }
    setOrder(order)
  }, [state, setOrder])

  // Build animation steps: each step shows the pre-move board, then updates to post-move board.
  const buildAnimationSteps = useCallback((steps: TurnStep[], boardBefore: Board): AnimationStep[] => {
    setDisplayBoard(boardBefore)
    return steps.map(step => ({
      event: step.event,
      onStepComplete: () => setDisplayBoard(step.boardAfter),
    }))
  }, [])

  const handleEndTurn = useCallback(() => {
    if (!state) return
    const boardBefore = state.board
    setIsAnimating(true)
    executeHumanMovesAction((steps: TurnStep[]) => {
      const animSteps = buildAnimationSteps(steps, boardBefore)
      if (animSteps.length > 0) {
        enqueue(animSteps, () => {
          setDisplayBoard(null)
          endTurn()
          setIsAnimating(false)
        })
      } else {
        setDisplayBoard(null)
        endTurn()
        setIsAnimating(false)
      }
    })
  }, [state, executeHumanMovesAction, buildAnimationSteps, endTurn, enqueue])

  const handleRetire = useCallback(() => {
    retire()
    setScreen('end')
  }, [retire])

  // Process AI turns sequentially
  useEffect(() => {
    if (!state || state.phase !== 'aiTurn') return
    const aiPlayers = state.players.filter(p => p.type === 'ai' && !p.isEliminated)
    const aiIndex = state.turn.activeAiIndex

    if (aiIndex >= aiPlayers.length) {
      resolveAi(aiIndex, () => {})
      return
    }

    setIsAnimating(true)
    const boardBefore = state.board
    const timer = setTimeout(() => {
      resolveAi(aiIndex, (steps: TurnStep[]) => {
        const animSteps = buildAnimationSteps(steps, boardBefore)
        if (animSteps.length > 0) {
          enqueue(animSteps, () => {
            setDisplayBoard(null)
            setIsAnimating(false)
          })
        } else {
          setDisplayBoard(null)
          setIsAnimating(false)
        }
      })
    }, 100)

    return () => clearTimeout(timer)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state?.phase, state?.turn.activeAiIndex])

  // Transition to end screen only after animations complete
  useEffect(() => {
    if (state?.phase === 'end' && screen === 'game' && !isAnimating) {
      setScreen('end')
    }
  }, [state?.phase, screen, isAnimating])

  if (screen === 'start') {
    return <StartScreen onStart={handleStartGame} />
  }

  if (screen === 'end' && state?.stats) {
    return (
      <EndScreen
        stats={state.stats}
        players={state.players}
        onRestart={handleRestart}
      />
    )
  }

  if (!state) return null

  // Use displayBoard for rendering during animation; fall back to state.board otherwise
  const renderedBoard = displayBoard ?? state.board

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <GameBoard
        gameState={{ ...state, board: renderedBoard }}
        activeAnimation={activeEvent}
        viewport={viewport.viewport}
        onPointerDown={viewport.onPointerDown}
        onPointerMove={viewport.onPointerMove}
        onPointerUp={viewport.onPointerUp}
        onSetOrder={handleSetOrder}
        onCancelOrder={cancelOrder}
      />
      <GameHUD
        gameState={state}
        onEndTurn={handleEndTurn}
        onRetire={handleRetire}
        onZoomIn={viewport.zoomIn}
        onZoomOut={viewport.zoomOut}
        isAnimating={isAnimating}
      />
    </div>
  )
}
