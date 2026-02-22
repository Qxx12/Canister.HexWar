import { describe, it, expect } from 'vitest'
import {
  setHumanStandingOrder,
  cancelHumanStandingOrder,
  applyHumanOrder,
  cancelHumanOrder,
  endHumanTurn,
  retireGame,
} from '../engine/gameEngine'
import type { GameState } from '../types/game'
import type { Board } from '../types/board'
import type { Player } from '../types/player'
import type { PlayerStats } from '../types/stats'

function makeBoard(entries: [string, { owner: string | null; units: number }][]): Board {
  const board: Board = new Map()
  for (const [key, { owner, units }] of entries) {
    const [q, r] = key.split(',').map(Number)
    board.set(key, {
      coord: { q, r },
      owner,
      units,
      isStartTile: false,
      startOwner: null,
      terrain: 'plains' as const,
      newlyConquered: false,
    })
  }
  return board
}

function makeStats(ids: string[]): Map<string, PlayerStats> {
  return new Map(ids.map(id => [id, {
    playerId: id, unitsGenerated: 0, unitsKilled: 0,
    tilesConquered: 0, tilesLost: 0, tilesAtEnd: 0,
  }]))
}

function makeState(overrides: Partial<GameState> = {}): GameState {
  const players: Player[] = [
    { id: 'p0', type: 'human', color: '#fff', name: 'P0', isEliminated: false },
    { id: 'p1', type: 'ai',    color: '#f00', name: 'P1', isEliminated: false },
  ]
  return {
    phase: 'playerTurn',
    board: makeBoard([['0,0', { owner: 'p0', units: 5 }]]),
    players,
    humanPlayerId: 'p0',
    orders: new Map([['p0', new Map()], ['p1', new Map()]]),
    humanStandingOrders: new Map(),
    turn: { turnNumber: 1, activeAiIndex: 0 },
    winner: null,
    stats: null,
    runningStats: makeStats(['p0', 'p1']),
    ...overrides,
  }
}

describe('setHumanStandingOrder', () => {
  it('adds a standing order', () => {
    const state = makeState()
    const order = { fromKey: '0,0', toKey: '1,0', requestedUnits: 3 }
    const next = setHumanStandingOrder(state, order)
    expect(next.humanStandingOrders.get('0,0')).toEqual(order)
  })

  it('overwrites an existing standing order for the same source tile', () => {
    const state = makeState()
    const first  = { fromKey: '0,0', toKey: '1,0', requestedUnits: 2 }
    const second = { fromKey: '0,0', toKey: '2,0', requestedUnits: 5 }
    const next = setHumanStandingOrder(setHumanStandingOrder(state, first), second)
    expect(next.humanStandingOrders.get('0,0')).toEqual(second)
    expect(next.humanStandingOrders.size).toBe(1)
  })

  it('does not mutate the original state', () => {
    const state = makeState()
    setHumanStandingOrder(state, { fromKey: '0,0', toKey: '1,0', requestedUnits: 1 })
    expect(state.humanStandingOrders.size).toBe(0)
  })
})

describe('cancelHumanStandingOrder', () => {
  it('removes a standing order', () => {
    const order = { fromKey: '0,0', toKey: '1,0', requestedUnits: 3 }
    const state = makeState({ humanStandingOrders: new Map([['0,0', order]]) })
    const next = cancelHumanStandingOrder(state, '0,0')
    expect(next.humanStandingOrders.has('0,0')).toBe(false)
  })

  it('is a no-op for a key that does not exist', () => {
    const state = makeState()
    const next = cancelHumanStandingOrder(state, 'nonexistent')
    expect(next.humanStandingOrders.size).toBe(0)
  })
})

describe('applyHumanOrder', () => {
  it('adds an order for the human player', () => {
    const state = makeState()
    const order = { fromKey: '0,0', toKey: '1,0', requestedUnits: 2 }
    const next = applyHumanOrder(state, order)
    expect(next.orders.get('p0')!.get('0,0')).toEqual(order)
  })

  it('overwrites an existing order for the same source tile', () => {
    const state = makeState()
    const first  = { fromKey: '0,0', toKey: '1,0', requestedUnits: 1 }
    const second = { fromKey: '0,0', toKey: '2,0', requestedUnits: 4 }
    const next = applyHumanOrder(applyHumanOrder(state, first), second)
    expect(next.orders.get('p0')!.get('0,0')).toEqual(second)
  })

  it('does not affect other players orders', () => {
    const p1order = { fromKey: '1,0', toKey: '2,0', requestedUnits: 1 }
    const state = makeState({ orders: new Map([['p0', new Map()], ['p1', new Map([['1,0', p1order]])]]) })
    const next = applyHumanOrder(state, { fromKey: '0,0', toKey: '1,0', requestedUnits: 3 })
    expect(next.orders.get('p1')!.get('1,0')).toEqual(p1order)
  })
})

describe('cancelHumanOrder', () => {
  it('removes an order for the human player', () => {
    const order = { fromKey: '0,0', toKey: '1,0', requestedUnits: 3 }
    const state = makeState({ orders: new Map([['p0', new Map([['0,0', order]])], ['p1', new Map()]]) })
    const next = cancelHumanOrder(state, '0,0')
    expect(next.orders.get('p0')!.has('0,0')).toBe(false)
  })

  it('is a no-op for a key that does not exist', () => {
    const state = makeState()
    const next = cancelHumanOrder(state, 'nonexistent')
    expect(next.orders.get('p0')!.size).toBe(0)
  })
})

describe('endHumanTurn', () => {
  it('sets phase to aiTurn', () => {
    const next = endHumanTurn(makeState())
    expect(next.phase).toBe('aiTurn')
  })

  it('resets activeAiIndex to 0', () => {
    const state = makeState({ turn: { turnNumber: 3, activeAiIndex: 2 } })
    const next = endHumanTurn(state)
    expect(next.turn.activeAiIndex).toBe(0)
  })

  it('preserves turnNumber', () => {
    const state = makeState({ turn: { turnNumber: 7, activeAiIndex: 0 } })
    expect(endHumanTurn(state).turn.turnNumber).toBe(7)
  })
})

describe('retireGame', () => {
  it('sets phase to end with retire outcome', () => {
    const next = retireGame(makeState())
    expect(next.phase).toBe('end')
    expect(next.stats!.outcome).toBe('retire')
    expect(next.stats!.winnerId).toBeNull()
  })

  it('computes tilesAtEnd correctly', () => {
    const board = makeBoard([
      ['0,0', { owner: 'p0', units: 3 }],
      ['1,0', { owner: 'p0', units: 2 }],
      ['2,0', { owner: 'p1', units: 1 }],
    ])
    const next = retireGame(makeState({ board }))
    const p0stats = next.stats!.playerStats.find(s => s.playerId === 'p0')!
    const p1stats = next.stats!.playerStats.find(s => s.playerId === 'p1')!
    expect(p0stats.tilesAtEnd).toBe(2)
    expect(p1stats.tilesAtEnd).toBe(1)
  })
})
