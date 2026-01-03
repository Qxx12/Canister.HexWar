import { useEffect, useCallback, useState } from 'react'
import { useGameState } from './hooks/useGameState'
import { useViewport } from './hooks/useViewport'
import { useAnimationQueue } from './hooks/useAnimationQueue'
import { GameBoard } from './components/game/GameBoard'
import { GameHUD } from './components/game/GameHUD'
import { StartScreen } from './components/screens/StartScreen'
import { EndScreen } from './components/screens/EndScreen'
import type { MovementOrder } from './types/orders'
import type { AnimationEvent } from './types/animation'
import './styles/main.scss'

type AppScreen = 'start' | 'game' | 'end'

export default function App() {
  const [screen, setScreen] = useState<AppScreen>('start')
  const { state, startGame, resetGame, setOrder, cancelOrder, executeHumanMovesAction, endTurn, resolveAi, retire } = useGameState()
  const viewport = useViewport()
  const { activeEvent, enqueue, clearQueue } = useAnimationQueue()
  const [isAnimating, setIsAnimating] = useState(false)

  const handleStartGame = useCallback(() => {
    startGame()
    setScreen('game')
  }, [startGame])

  const handleRestart = useCallback(() => {
    resetGame()
    clearQueue()
    setScreen('start')
  }, [resetGame, clearQueue])

  const handleSetOrder = useCallback((fromKey: string, toKey: string, units: number) => {
    if (!state) return
    const order: MovementOrder = { fromKey, toKey, requestedUnits: units }
    setOrder(order)
  }, [state, setOrder])

  const handleEndTurn = useCallback(() => {
    if (!state) return
    setIsAnimating(true)
    executeHumanMovesAction((events: AnimationEvent[]) => {
      if (events.length > 0) {
        enqueue(events, () => {
          endTurn(() => {})
          setIsAnimating(false)
        })
      } else {
        endTurn(() => {})
        setIsAnimating(false)
      }
    })
  }, [state, executeHumanMovesAction, endTurn, enqueue])

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
      // All AIs done - call resolveAi to generate units and transition back to playerTurn
      resolveAi(aiIndex, () => {})
      return
    }

    setIsAnimating(true)
    const timer = setTimeout(() => {
      resolveAi(aiIndex, (events: AnimationEvent[]) => {
        if (events.length > 0) {
          enqueue(events, () => setIsAnimating(false))
        } else {
          setIsAnimating(false)
        }
      })
    }, 100)

    return () => clearTimeout(timer)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state?.phase, state?.turn.activeAiIndex])

  // Transition to end screen when game ends
  useEffect(() => {
    if (state?.phase === 'end' && screen === 'game') {
      setScreen('end')
    }
  }, [state?.phase, screen])

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

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <GameBoard
        gameState={state}
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
