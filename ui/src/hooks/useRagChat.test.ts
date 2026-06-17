import { renderHook, act, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { useRagChat } from './useRagChat'
import { sseResponse, jsonResponse, DONE_EVENT, MOCK_MESSAGES } from '../test/sse'

const API_URL = 'http://localhost:8000'
const AUTH_TOKEN = 'test-token'

beforeEach(() => {
  vi.restoreAllMocks()
})

describe('useRagChat — sendMessage', () => {
  it('posts to /v1/chat with query and stream:true', async () => {
    const fetchSpy = vi.spyOn(global, 'fetch').mockResolvedValue(sseResponse([DONE_EVENT]))
    const { result } = renderHook(() => useRagChat(API_URL))

    await act(async () => {
      await result.current.sendMessage('What is consciousness?')
    })

    expect(fetchSpy).toHaveBeenCalledWith(
      `${API_URL}/v1/chat`,
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('"stream":true'),
      })
    )
    expect(JSON.parse((fetchSpy.mock.calls[0][1] as RequestInit).body as string).query).toBe(
      'What is consciousness?'
    )
  })

  it('sends Authorization header when authToken provided', async () => {
    const fetchSpy = vi.spyOn(global, 'fetch').mockResolvedValue(sseResponse([DONE_EVENT]))
    const { result } = renderHook(() => useRagChat(API_URL, AUTH_TOKEN))

    await act(async () => {
      await result.current.sendMessage('hello')
    })

    const headers = (fetchSpy.mock.calls[0][1] as RequestInit).headers as Record<string, string>
    expect(headers['Authorization']).toBe(`Bearer ${AUTH_TOKEN}`)
  })

  it('appends streamed tokens to the assistant message', async () => {
    vi.spyOn(global, 'fetch').mockResolvedValue(
      sseResponse([
        { type: 'token', content: 'Hello' },
        { type: 'token', content: ' world' },
        DONE_EVENT,
      ])
    )
    const { result } = renderHook(() => useRagChat(API_URL))

    await act(async () => {
      await result.current.sendMessage('hi')
    })

    const assistant = result.current.messages.find((m) => m.role === 'assistant')
    expect(assistant?.content).toBe('Hello world')
    expect(assistant?.streaming).toBe(false)
  })

  it('stores conversationId from done event', async () => {
    vi.spyOn(global, 'fetch').mockResolvedValue(sseResponse([DONE_EVENT]))
    const { result } = renderHook(() => useRagChat(API_URL))

    await act(async () => {
      await result.current.sendMessage('hi')
    })

    // Second message should include the conversation_id from the first
    const fetchSpy = vi.spyOn(global, 'fetch').mockResolvedValue(sseResponse([DONE_EVENT]))
    await act(async () => {
      await result.current.sendMessage('follow-up')
    })

    const body = JSON.parse((fetchSpy.mock.calls[0][1] as RequestInit).body as string)
    expect(body.conversation_id).toBe(DONE_EVENT.conversation_id)
  })

  it('sets authError and removes placeholder on 401', async () => {
    vi.spyOn(global, 'fetch').mockResolvedValue(new Response(null, { status: 401 }))
    const { result } = renderHook(() => useRagChat(API_URL, AUTH_TOKEN))

    await act(async () => {
      await result.current.sendMessage('hi')
    })

    expect(result.current.authError).toBe(true)
    expect(result.current.messages).toHaveLength(1) // only user message remains
    expect(result.current.messages[0].role).toBe('user')
  })

  it('shows error content on non-401 API failure', async () => {
    vi.spyOn(global, 'fetch').mockResolvedValue(new Response(null, { status: 500 }))
    const { result } = renderHook(() => useRagChat(API_URL))

    await act(async () => {
      await result.current.sendMessage('hi')
    })

    const assistant = result.current.messages.find((m) => m.role === 'assistant')
    expect(assistant?.content).toMatch(/something went wrong/i)
    expect(assistant?.streaming).toBe(false)
  })
})

describe('useRagChat — reset', () => {
  it('clears messages and resets loading state', async () => {
    vi.spyOn(global, 'fetch').mockResolvedValue(sseResponse([DONE_EVENT]))
    const { result } = renderHook(() => useRagChat(API_URL))

    await act(async () => {
      await result.current.sendMessage('hi')
    })
    expect(result.current.messages).toHaveLength(2)

    act(() => {
      result.current.reset()
    })

    expect(result.current.messages).toHaveLength(0)
    expect(result.current.isLoading).toBe(false)
    expect(result.current.authError).toBe(false)
  })

  it('clears conversationId so next message starts a new conversation', async () => {
    vi.spyOn(global, 'fetch').mockResolvedValue(sseResponse([DONE_EVENT]))
    const { result } = renderHook(() => useRagChat(API_URL))
    await act(async () => {
      await result.current.sendMessage('hi')
    })

    act(() => {
      result.current.reset()
    })

    const fetchSpy = vi.spyOn(global, 'fetch').mockResolvedValue(sseResponse([DONE_EVENT]))
    await act(async () => {
      await result.current.sendMessage('new chat')
    })

    const body = JSON.parse((fetchSpy.mock.calls[0][1] as RequestInit).body as string)
    expect(body.conversation_id).toBeNull()
  })
})

describe('useRagChat — loadConversation', () => {
  it('fetches messages and populates state', async () => {
    vi.spyOn(global, 'fetch').mockResolvedValue(jsonResponse(MOCK_MESSAGES))
    const { result } = renderHook(() => useRagChat(API_URL, AUTH_TOKEN))

    let ok: boolean
    await act(async () => {
      ok = await result.current.loadConversation('conv-1')
    })

    expect(ok!).toBe(true)
    expect(result.current.messages).toHaveLength(2)
    expect(result.current.messages[0].role).toBe('user')
    expect(result.current.messages[1].role).toBe('assistant')
  })

  it('returns false and sets authError on 401', async () => {
    vi.spyOn(global, 'fetch').mockResolvedValue(new Response(null, { status: 401 }))
    const { result } = renderHook(() => useRagChat(API_URL, AUTH_TOKEN))

    let ok: boolean
    await act(async () => {
      ok = await result.current.loadConversation('conv-1')
    })

    expect(ok!).toBe(false)
    expect(result.current.authError).toBe(true)
  })

  it('returns false without authToken', async () => {
    const fetchSpy = vi.spyOn(global, 'fetch')
    const { result } = renderHook(() => useRagChat(API_URL))

    let ok: boolean
    await act(async () => {
      ok = await result.current.loadConversation('conv-1')
    })

    expect(ok!).toBe(false)
    expect(fetchSpy).not.toHaveBeenCalled()
  })
})

describe('useRagChat — stop', () => {
  it('aborts the in-flight request', async () => {
    let abortCalled = false
    vi.spyOn(global, 'fetch').mockImplementation((_url, init) => {
      init?.signal?.addEventListener('abort', () => {
        abortCalled = true
      })
      return new Promise(() => {}) // never resolves
    })

    const { result } = renderHook(() => useRagChat(API_URL))

    act(() => {
      result.current.sendMessage('hi')
    })
    await waitFor(() => expect(result.current.isLoading).toBe(true))

    act(() => {
      result.current.stop()
    })
    await waitFor(() => expect(abortCalled).toBe(true))
  })
})
