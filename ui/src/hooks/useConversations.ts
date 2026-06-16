import { useState, useEffect, useCallback } from 'react'
import type { ConversationSummary } from '../types'

export function useConversations(apiUrl: string, authToken?: string) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [isLoading, setIsLoading] = useState(false)

  const reload = useCallback(async () => {
    if (!authToken) return
    setIsLoading(true)
    try {
      const res = await fetch(`${apiUrl}/v1/conversations`, {
        headers: { Authorization: `Bearer ${authToken}` },
      })
      if (!res.ok) return
      const data: ConversationSummary[] = await res.json()
      setConversations(data)
    } catch {
      // silently ignore — history panel will just stay empty
    } finally {
      setIsLoading(false)
    }
  }, [apiUrl, authToken])

  useEffect(() => {
    reload()
  }, [reload])

  return { conversations, isLoading, reload }
}
