import { useState, useRef, useCallback } from 'react'
import type { Message, Source } from '../types'

interface DonePayload {
  topic: string
  sources: Source[]
  conversation_id: string
}

function uid() {
  return Math.random().toString(36).slice(2)
}

export function useRagChat(apiUrl: string, authToken?: string) {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [authError, setAuthError] = useState(false)
  const conversationId = useRef<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  function authHeaders(): HeadersInit {
    return authToken
      ? { 'Content-Type': 'application/json', Authorization: `Bearer ${authToken}` }
      : { 'Content-Type': 'application/json' }
  }

  const sendMessage = useCallback(async (query: string) => {
    if (!query.trim() || isLoading) return

    setAuthError(false)
    const userMsg: Message = { id: uid(), role: 'user', content: query }
    const assistantId = uid()
    const assistantMsg: Message = { id: assistantId, role: 'assistant', content: '', streaming: true }

    setMessages(prev => [...prev, userMsg, assistantMsg])
    setIsLoading(true)

    abortRef.current = new AbortController()

    try {
      const res = await fetch(`${apiUrl}/v1/chat`, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({
          query,
          conversation_id: conversationId.current,
          stream: true,
        }),
        signal: abortRef.current.signal,
      })

      if (res.status === 401) {
        setAuthError(true)
        setMessages(prev => prev.filter(m => m.id !== assistantId))
        return
      }
      if (!res.ok) throw new Error(`API error ${res.status}`)

      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const payload = JSON.parse(line.slice(6))

          if (payload.type === 'token') {
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantId ? { ...m, content: m.content + payload.content } : m
              )
            )
          }

          if (payload.type === 'done') {
            const donePayload = payload as DonePayload
            conversationId.current = donePayload.conversation_id
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantId
                  ? { ...m, streaming: false, topic: donePayload.topic, sources: donePayload.sources }
                  : m
              )
            )
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantId
              ? { ...m, content: 'Something went wrong. Please try again.', streaming: false }
              : m
          )
        )
      }
    } finally {
      setIsLoading(false)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiUrl, authToken, isLoading])

  const stop = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const reset = useCallback(() => {
    abortRef.current?.abort()
    conversationId.current = null
    setMessages([])
    setIsLoading(false)
    setAuthError(false)
  }, [])

  const loadConversation = useCallback(async (id: string): Promise<boolean> => {
    if (!authToken) return false
    try {
      const res = await fetch(`${apiUrl}/v1/conversations/${id}/messages`, {
        headers: { Authorization: `Bearer ${authToken}` },
      })
      if (res.status === 401) { setAuthError(true); return false }
      if (!res.ok) return false
      const data = await res.json()
      const loaded: Message[] = data.messages.map((m: { role: string; content: string }) => ({
        id: uid(),
        role: m.role as 'user' | 'assistant',
        content: m.content,
      }))
      conversationId.current = id
      setMessages(loaded)
      return true
    } catch {
      return false
    }
  }, [apiUrl, authToken])

  return { messages, isLoading, authError, sendMessage, stop, reset, loadConversation }
}
