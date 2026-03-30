import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { SettingsScreen } from '../components/screens/SettingsScreen'
import type { GameSettings } from '../types/settings'

describe('SettingsScreen', () => {
  afterEach(cleanup)

  it('renders the HexWar title', () => {
    render(<SettingsScreen onConfirm={() => {}} onBack={() => {}} />)
    expect(screen.getByRole('heading', { name: 'HexWar' })).toBeDefined()
  })

  it('shows Opponent section label', () => {
    render(<SettingsScreen onConfirm={() => {}} onBack={() => {}} />)
    expect(screen.getByRole('heading', { name: 'Opponent' })).toBeDefined()
  })

  it('renders all three difficulty options', () => {
    render(<SettingsScreen onConfirm={() => {}} onBack={() => {}} />)
    expect(screen.getByRole('button', { name: /Soldier/ })).toBeDefined()
    expect(screen.getByRole('button', { name: /Commander/ })).toBeDefined()
    expect(screen.getByRole('button', { name: /Warlord/ })).toBeDefined()
  })

  it('renders flavor text for each difficulty', () => {
    render(<SettingsScreen onConfirm={() => {}} onBack={() => {}} />)
    expect(screen.getByText(/Predictable tactics/)).toBeDefined()
    expect(screen.getByText(/Coordinates offensives/)).toBeDefined()
    expect(screen.getByText(/Merciless expansion/)).toBeDefined()
  })

  it('renders a Deploy button', () => {
    render(<SettingsScreen onConfirm={() => {}} onBack={() => {}} />)
    expect(screen.getByRole('button', { name: /Deploy/i })).toBeDefined()
  })

  it('calls onConfirm with commander difficulty by default', () => {
    const onConfirm = vi.fn()
    render(<SettingsScreen onConfirm={onConfirm} onBack={() => {}} />)
    fireEvent.click(screen.getByRole('button', { name: /Deploy/i }))
    expect(onConfirm).toHaveBeenCalledTimes(1)
    const settings: GameSettings = onConfirm.mock.calls[0][0]
    expect(settings.difficulty).toBe('commander')
  })

  it('selecting Soldier then deploying confirms soldier difficulty', () => {
    const onConfirm = vi.fn()
    render(<SettingsScreen onConfirm={onConfirm} onBack={() => {}} />)
    fireEvent.click(screen.getByRole('button', { name: /Soldier/ }))
    fireEvent.click(screen.getByRole('button', { name: /Deploy/i }))
    const settings: GameSettings = onConfirm.mock.calls[0][0]
    expect(settings.difficulty).toBe('soldier')
  })

  it('selecting Warlord then deploying confirms warlord difficulty', () => {
    const onConfirm = vi.fn()
    render(<SettingsScreen onConfirm={onConfirm} onBack={() => {}} />)
    fireEvent.click(screen.getByRole('button', { name: /Warlord/ }))
    fireEvent.click(screen.getByRole('button', { name: /Deploy/i }))
    const settings: GameSettings = onConfirm.mock.calls[0][0]
    expect(settings.difficulty).toBe('warlord')
  })

  it('switching from Warlord back to Commander produces commander', () => {
    const onConfirm = vi.fn()
    render(<SettingsScreen onConfirm={onConfirm} onBack={() => {}} />)
    fireEvent.click(screen.getByRole('button', { name: /Warlord/ }))
    fireEvent.click(screen.getByRole('button', { name: /Commander/ }))
    fireEvent.click(screen.getByRole('button', { name: /Deploy/i }))
    const settings: GameSettings = onConfirm.mock.calls[0][0]
    expect(settings.difficulty).toBe('commander')
  })

  it('calls onBack when the Back button is clicked', () => {
    const onBack = vi.fn()
    render(<SettingsScreen onConfirm={() => {}} onBack={onBack} />)
    fireEvent.click(screen.getByRole('button', { name: /Back/i }))
    expect(onBack).toHaveBeenCalledTimes(1)
  })

  it('does not call onConfirm when Back is clicked', () => {
    const onConfirm = vi.fn()
    render(<SettingsScreen onConfirm={onConfirm} onBack={() => {}} />)
    fireEvent.click(screen.getByRole('button', { name: /Back/i }))
    expect(onConfirm).not.toHaveBeenCalled()
  })

  it('Deploy button uses primary variant', () => {
    render(<SettingsScreen onConfirm={() => {}} onBack={() => {}} />)
    expect(screen.getByRole('button', { name: /Deploy/i }).className).toContain('primary')
  })

  it('Back button uses secondary variant', () => {
    render(<SettingsScreen onConfirm={() => {}} onBack={() => {}} />)
    expect(screen.getByRole('button', { name: /Back/i }).className).toContain('secondary')
  })
})
