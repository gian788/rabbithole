import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { RagChat } from './RagChat'
import { sseResponse, jsonResponse, DONE_EVENT, MOCK_CONVERSATIONS, MOCK_MESSAGES } from '../test/sse'

const API_URL = 'http://localhost:8000'
const AUTH_TOKEN = 'test-token'

beforeEach(() => {
  vi.restoreAllMocks()
})

function mockFetchSequence(...responses: Response[]) {
  let i = 0
  vi.spyOn(global, 'fetch').mockImplementation(() => {
    const res = responses[i] ?? responses[responses.length - 1]
    i++
    return Promise.resolve(res)
  })
}

describe('RagChat — default state', () => {
  it('renders the chat view with empty state by default', () => {
    vi.spyOn(global, 'fetch').mockResolvedValue(jsonResponse([]))
    render(<RagChat apiUrl={API_URL} />)
    expect(screen.getByText(/ask a question/i)).toBeInTheDocument()
  })

  it('renders the header title', () => {
    vi.spyOn(global, 'fetch').mockResolvedValue(jsonResponse([]))
    render(<RagChat apiUrl={API_URL} title="My Assistant" />)
    expect(screen.getByText('My Assistant')).toBeInTheDocument()
  })

  it('does not show history button without authToken', () => {
    render(<RagChat apiUrl={API_URL} />)
    expect(screen.queryByRole('button', { name: /history/i })).not.toBeInTheDocument()
  })

  it('shows history button when authToken is provided', async () => {
    vi.spyOn(global, 'fetch').mockResolvedValue(jsonResponse([]))
    render(<RagChat apiUrl={API_URL} authToken={AUTH_TOKEN} />)
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /conversation history/i })).toBeInTheDocument()
    )
  })

  it('hides header when showHeader is false', () => {
    render(<RagChat apiUrl={API_URL} showHeader={false} />)
    expect(screen.queryByText('AI Chat')).not.toBeInTheDocument()
  })
})

describe('RagChat — sending a message', () => {
  it('displays user message and streaming assistant response', async () => {
    vi.spyOn(global, 'fetch').mockResolvedValue(
      sseResponse([{ type: 'token', content: 'Hello!' }, DONE_EVENT])
    )
    render(<RagChat apiUrl={API_URL} />)

    await userEvent.type(screen.getByRole('textbox'), 'hi{Enter}')

    await waitFor(() => expect(screen.getByText('hi')).toBeInTheDocument())
    await waitFor(() => expect(screen.getByText('Hello!')).toBeInTheDocument())
  })
})

describe('RagChat — auth error', () => {
  it('shows session expired message on 401', async () => {
    vi.spyOn(global, 'fetch').mockResolvedValue(new Response(null, { status: 401 }))
    render(<RagChat apiUrl={API_URL} authToken={AUTH_TOKEN} />)

    await userEvent.type(screen.getByRole('textbox'), 'hi{Enter}')

    await waitFor(() =>
      expect(screen.getByText(/session expired/i)).toBeInTheDocument()
    )
  })
})

describe('RagChat — history panel', () => {
  it('switches to history view when clock icon is clicked', async () => {
    mockFetchSequence(
      jsonResponse(MOCK_CONVERSATIONS), // useConversations on mount
      jsonResponse(MOCK_CONVERSATIONS), // reload on clock click
    )
    render(<RagChat apiUrl={API_URL} authToken={AUTH_TOKEN} />)

    await waitFor(() =>
      expect(screen.getByRole('button', { name: /conversation history/i })).toBeInTheDocument()
    )
    await userEvent.click(screen.getByRole('button', { name: /conversation history/i }))

    expect(screen.getByText('History')).toBeInTheDocument()
    await waitFor(() =>
      expect(screen.getByText('What is consciousness?')).toBeInTheDocument()
    )
  })

  it('back button returns to chat view', async () => {
    mockFetchSequence(
      jsonResponse(MOCK_CONVERSATIONS),
      jsonResponse(MOCK_CONVERSATIONS),
    )
    render(<RagChat apiUrl={API_URL} authToken={AUTH_TOKEN} />)

    await waitFor(() =>
      expect(screen.getByRole('button', { name: /conversation history/i })).toBeInTheDocument()
    )
    await userEvent.click(screen.getByRole('button', { name: /conversation history/i }))
    await userEvent.click(screen.getByRole('button', { name: /back/i }))

    expect(screen.getByText(/ask a question/i)).toBeInTheDocument()
  })

  it('clicking a conversation card loads it and returns to chat', async () => {
    mockFetchSequence(
      jsonResponse(MOCK_CONVERSATIONS),    // useConversations on mount
      jsonResponse(MOCK_CONVERSATIONS),    // reload triggered by clock click
      jsonResponse(MOCK_MESSAGES),         // loadConversation fetch
    )
    render(<RagChat apiUrl={API_URL} authToken={AUTH_TOKEN} />)

    // Wait for history button to appear (conversations loaded)
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /conversation history/i })).toBeInTheDocument()
    )

    // Open history panel
    await userEvent.click(screen.getByRole('button', { name: /conversation history/i }))
    await waitFor(() => expect(screen.getByText('What is consciousness?')).toBeInTheDocument())

    // Click the first conversation card
    await userEvent.click(screen.getByText('What is consciousness?'))

    // Should return to chat view and show loaded messages
    await waitFor(() =>
      expect(screen.queryByText('History')).not.toBeInTheDocument()
    )
    await waitFor(() =>
      expect(screen.getByText('Consciousness is awareness of oneself and the environment.')).toBeInTheDocument()
    )
  })
})

describe('RagChat — new chat', () => {
  it('"New chat" button is disabled when conversation is empty', () => {
    render(<RagChat apiUrl={API_URL} />)
    expect(screen.getByRole('button', { name: /new chat/i })).toBeDisabled()
  })

  it('"New chat" resets to empty state', async () => {
    vi.spyOn(global, 'fetch').mockResolvedValue(
      sseResponse([{ type: 'token', content: 'Hi there' }, DONE_EVENT])
    )
    render(<RagChat apiUrl={API_URL} />)

    await userEvent.type(screen.getByRole('textbox'), 'hello{Enter}')
    await waitFor(() => expect(screen.getByText('Hi there')).toBeInTheDocument())

    await userEvent.click(screen.getByRole('button', { name: /new chat/i }))

    expect(screen.getByText(/ask a question/i)).toBeInTheDocument()
  })
})
