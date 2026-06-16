import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi } from 'vitest'
import { ConversationCard } from './ConversationCard'
import type { ConversationSummary } from '../types'

const BASE: ConversationSummary = {
  id: 'conv-1',
  title: 'What is consciousness?',
  topic: 'consciousness',
  last_message_at: new Date(Date.now() - 2 * 60_000).toISOString(), // 2 minutes ago
  preview: 'Consciousness is the state of being aware of your surroundings.',
}

describe('ConversationCard', () => {
  it('renders the conversation title', () => {
    render(<ConversationCard conversation={BASE} onClick={vi.fn()} />)
    expect(screen.getByText('What is consciousness?')).toBeInTheDocument()
  })

  it('renders the topic badge', () => {
    render(<ConversationCard conversation={BASE} onClick={vi.fn()} />)
    // exact match targets the badge <span> only — title and preview contain different casing/surrounding text
    expect(screen.getByText('consciousness')).toBeInTheDocument()
  })

  it('renders the preview text', () => {
    render(<ConversationCard conversation={BASE} onClick={vi.fn()} />)
    expect(screen.getByText(/Consciousness is the state/)).toBeInTheDocument()
  })

  it('shows relative time', () => {
    render(<ConversationCard conversation={BASE} onClick={vi.fn()} />)
    expect(screen.getByText(/ago/)).toBeInTheDocument()
  })

  it('calls onClick with the conversation id', async () => {
    const onClick = vi.fn()
    render(<ConversationCard conversation={BASE} onClick={onClick} />)
    await userEvent.click(screen.getByRole('button'))
    expect(onClick).toHaveBeenCalledWith('conv-1')
  })

  it('does not render a topic badge when topic is null', () => {
    render(<ConversationCard conversation={{ ...BASE, topic: null }} onClick={vi.fn()} />)
    // badge text is exact lowercase 'consciousness'; title and preview use different text
    expect(screen.queryByText('consciousness')).not.toBeInTheDocument()
  })

  it('does not render timestamp when last_message_at is null', () => {
    render(<ConversationCard conversation={{ ...BASE, last_message_at: null }} onClick={vi.fn()} />)
    expect(screen.queryByText(/ago/)).not.toBeInTheDocument()
  })
})
