import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { Button } from '../components/shared/Button'

describe('Button', () => {
  afterEach(cleanup)

  it('renders children', () => {
    render(<Button onClick={() => {}}>Click me</Button>)
    expect(screen.getByRole('button', { name: 'Click me' })).toBeDefined()
  })

  it('calls onClick when clicked', () => {
    const onClick = vi.fn()
    render(<Button onClick={onClick}>Go</Button>)
    fireEvent.click(screen.getByRole('button'))
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it('does not call onClick when disabled', () => {
    const onClick = vi.fn()
    render(<Button onClick={onClick} disabled>Go</Button>)
    fireEvent.click(screen.getByRole('button'))
    expect(onClick).not.toHaveBeenCalled()
  })

  it('is disabled when disabled prop is true', () => {
    render(<Button onClick={() => {}} disabled>Go</Button>)
    expect(screen.getByRole('button')).toHaveProperty('disabled', true)
  })

  it('defaults to primary variant', () => {
    render(<Button onClick={() => {}}>Go</Button>)
    expect(screen.getByRole('button').className).toContain('primary')
  })

  it('applies secondary variant class', () => {
    render(<Button onClick={() => {}} variant="secondary">Go</Button>)
    expect(screen.getByRole('button').className).toContain('secondary')
  })

  it('applies danger variant class', () => {
    render(<Button onClick={() => {}} variant="danger">Go</Button>)
    expect(screen.getByRole('button').className).toContain('danger')
  })

  it('applies an extra className when provided', () => {
    render(<Button onClick={() => {}} className="custom">Go</Button>)
    expect(screen.getByRole('button').className).toContain('custom')
  })
})
