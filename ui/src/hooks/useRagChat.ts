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

export function useRagChat(apiUrl: string) {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const conversationId = useRef<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const sendMessage = useCallback(async (query: string) => {
    if (!query.trim() || isLoading) return

    const userMsg: Message = { id: uid(), role: 'user', content: query }
    const assistantId = uid()
    const assistantMsg: Message = { id: assistantId, role: 'assistant', content: '', streaming: true }

    setMessages(prev => [...prev, userMsg, assistantMsg])
    setIsLoading(true)

    abortRef.current = new AbortController()

    try {
      const res = await fetch(`${apiUrl}/v1/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          conversation_id: conversationId.current,
          stream: true,
        }),
        signal: abortRef.current.signal,
      })

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
  }, [apiUrl, isLoading])

  const stop = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const reset = useCallback(() => {
    abortRef.current?.abort()
    conversationId.current = null
    setMessages([])
    setIsLoading(false)
  }, [])

  return { messages, isLoading, sendMessage, stop, reset }
}
