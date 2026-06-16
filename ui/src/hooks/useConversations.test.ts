import { renderHook, act, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { useConversations } from './useConversations'
import { jsonResponse, MOCK_CONVERSATIONS } from '../test/sse'

const API_URL = 'http://localhost:8000'
const AUTH_TOKEN = 'test-token'

beforeEach(() => {
  vi.restoreAllMocks()
})

describe('useConversations', () => {
  it('fetches conversations on mount when authToken is provided', async () => {
    const fetchSpy = vi.spyOn(global, 'fetch').mockResolvedValue(
      jsonResponse(MOCK_CONVERSATIONS)
    )
    const { result } = renderHook(() => useConversations(API_URL, AUTH_TOKEN))

    await waitFor(() => expect(result.current.isLoading).toBe(false))

    expect(fetchSpy).toHaveBeenCalledWith(
      `${API_URL}/v1/conversations`,
      expect.objectContaining({
        headers: { Authorization: `Bearer ${AUTH_TOKEN}` },
      })
    )
    expect(result.current.conversations).toHaveLength(2)
    expect(result.current.conversations[0].id).toBe('conv-1')
  })

  it('does not fetch when authToken is absent', async () => {
    const fetchSpy = vi.spyOn(global, 'fetch')
    const { result } = renderHook(() => useConversations(API_URL))

    await waitFor(() => expect(result.current.isLoading).toBe(false))

    expect(fetchSpy).not.toHaveBeenCalled()
    expect(result.current.conversations).toHaveLength(0)
  })

  it('reload triggers a new fetch and updates the list', async () => {
    const fetchSpy = vi.spyOn(global, 'fetch')
      .mockResolvedValueOnce(jsonResponse([MOCK_CONVERSATIONS[0]]))
      .mockResolvedValueOnce(jsonResponse(MOCK_CONVERSATIONS))

    const { result } = renderHook(() => useConversations(API_URL, AUTH_TOKEN))
    await waitFor(() => expect(result.current.conversations).toHaveLength(1))

    await act(async () => { await result.current.reload() })

    expect(fetchSpy).toHaveBeenCalledTimes(2)
    expect(result.current.conversations).toHaveLength(2)
  })

  it('silently handles fetch errors and keeps empty state', async () => {
    vi.spyOn(global, 'fetch').mockRejectedValue(new Error('network error'))
    const { result } = renderHook(() => useConversations(API_URL, AUTH_TOKEN))

    await waitFor(() => expect(result.current.isLoading).toBe(false))

    expect(result.current.conversations).toHaveLength(0)
  })

  it('silently handles non-ok responses', async () => {
    vi.spyOn(global, 'fetch').mockResolvedValue(new Response(null, { status: 401 }))
    const { result } = renderHook(() => useConversations(API_URL, AUTH_TOKEN))

    await waitFor(() => expect(result.current.isLoading).toBe(false))

    expect(result.current.conversations).toHaveLength(0)
  })
})
