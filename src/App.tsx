import { useEffect, useCallback, useState } from 'react'
import { useGameState } from './hooks/useGameState'
import { useViewport } from './hooks/useViewport'
import { useAnimationQueue } from './hooks/useAnimationQueue'
import type { AnimationStep } from './hooks/useAnimationQueue'
import { GameBoard } from './components/game/GameBoard'
import { GameBoard3D } from './components/game/GameBoard3D'
import { GameHUD } from './components/game/GameHUD'
import { StartScreen } from './components/screens/StartScreen'
import { EndScreen } from './components/screens/EndScreen'
import type { MovementOrder, Board, TurnStep } from '@hexwar/engine'
import { resetAiAgents } from './ai/aiController'
import './styles/main.scss'

type AppScreen = 'start' | 'game' | 'end'

export default function App() {
  const [screen, setScreen] = useState<AppScreen>('start')
  const [viewMode, setViewMode] = useState<'2d' | '3d'>('2d')
  const [sunEnabled, setSunEnabled] = useState(true)
  const [shadowsEnabled, setShadowsEnabled] = useState(true)

  // Re-enable sun whenever switching to 3D
  const handleToggleView = useCallback(() => {
    setViewMode(v => {
      if (v === '2d') setSunEnabled(true)
      return v === '2d' ? '3d' : '2d'
    })
  }, [])
  const { state, startGame, resetGame, setOrder, cancelOrder, setStandingOrder, cancelStandingOrder, executeHumanMovesAction, endTurn, resolveAi, retire } = useGameState()
  const viewport = useViewport()
  const handleBoardReady = useCallback((w: number, h: number) => {
    viewport.centerBoard(w, h)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewport.centerBoard])
  const { activeEvent, enqueue, clearQueue } = useAnimationQueue()
  const [isAnimating, setIsAnimating] = useState(false)

  // Display board lags behind state.board — updated one step at a time as animations play.
  // state.board (logic board) is always fully resolved and used by the engine.
  const [displayBoard, setDisplayBoard] = useState<Board | null>(null)
  const [snapshotBoard, setSnapshotBoard] = useState<Board | null>(null)

  const handleStartGame = useCallback(() => {
    startGame()
    setScreen('game')
  }, [startGame])

  const handleRestart = useCallback(() => {
    resetGame()
    resetAiAgents()
    clearQueue()
    setDisplayBoard(null)
    setSnapshotBoard(null)
    setScreen('start')
  }, [resetGame, clearQueue])

  const handleSetOrder = useCallback((fromKey: string, toKey: string, units: number) => {
    if (!state) return
    const order: MovementOrder = { fromKey, toKey, requestedUnits: units }
    setOrder(order)
  }, [state, setOrder])

  const handleSetStandingOrder = useCallback((fromKey: string, toKey: string, units: number) => {
    if (!state) return
    const order: MovementOrder = { fromKey, toKey, requestedUnits: units }
    setStandingOrder(order)
  }, [state, setStandingOrder])

  const handleCancelStandingOrder = useCallback((fromKey: string) => {
    cancelStandingOrder(fromKey)
  }, [cancelStandingOrder])

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
    setSnapshotBoard(boardBefore)
    setIsAnimating(true)
    executeHumanMovesAction((steps: TurnStep[]) => {
      const animSteps = buildAnimationSteps(steps, boardBefore)
      if (animSteps.length > 0) {
        enqueue(animSteps, () => {
          setDisplayBoard(null)
          setSnapshotBoard(null)
          endTurn()
          setIsAnimating(false)
        })
      } else {
        setDisplayBoard(null)
        setSnapshotBoard(null)
        endTurn()
        setIsAnimating(false)
      }
    })
  }, [state, executeHumanMovesAction, buildAnimationSteps, endTurn, enqueue])

  const handleRetire = useCallback(() => {
    retire()
    setScreen('end')
  }, [retire])

  // Process AI turns sequentially — wait for animation to finish before advancing
  useEffect(() => {
    if (!state || state.phase !== 'aiTurn' || isAnimating) return
    const aiPlayers = state.players.filter(p => p.type === 'ai' && !p.isEliminated)
    const aiIndex = state.turn.activeAiIndex

    if (aiIndex >= aiPlayers.length) {
      resolveAi(aiIndex, () => {})
      return
    }

    const boardBefore = state.board
    const timer = setTimeout(() => {
      setIsAnimating(true)
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
  }, [state?.phase, state?.turn.activeAiIndex, isAnimating])

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

  const sharedBoardProps = {
    gameState: { ...state, board: renderedBoard },
    activeAnimation: activeEvent,
    arrowBoard: snapshotBoard ?? state.board,
    onSetOrder: handleSetOrder,
    onCancelOrder: cancelOrder,
    onSetStandingOrder: handleSetStandingOrder,
    onCancelStandingOrder: handleCancelStandingOrder,
  }

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {viewMode === '2d' ? (
        <GameBoard
          {...sharedBoardProps}
          viewport={viewport.viewport}
          onPointerDown={viewport.onPointerDown}
          onPointerMove={viewport.onPointerMove}
          onPointerUp={viewport.onPointerUp}
          onPointerCancel={viewport.onPointerCancel}
          onContextMenu={viewport.onContextMenu}
          onWheel={viewport.onWheel}
          onReady={handleBoardReady}
          wasPanning={viewport.wasPanning}
        />
      ) : (
        <GameBoard3D {...sharedBoardProps} sunEnabled={sunEnabled} shadowsEnabled={shadowsEnabled} />
      )}
      <GameHUD
        gameState={state}
        onEndTurn={handleEndTurn}
        onRetire={handleRetire}
        isAnimating={isAnimating}
        animatingPlayerId={activeEvent?.playerId ?? null}
        viewMode={viewMode}
        onToggleView={handleToggleView}
        sunEnabled={sunEnabled}
        onToggleSun={() => setSunEnabled(v => !v)}
        shadowsEnabled={shadowsEnabled}
        onToggleShadows={() => setShadowsEnabled(v => !v)}
      />
    </div>
  )
}
