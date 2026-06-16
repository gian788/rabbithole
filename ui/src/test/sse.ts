/**
 * Creates a mock fetch Response that streams SSE events.
 * Matches the format emitted by the FastAPI backend:
 *   data: {"type":"token","content":"..."}\n\n
 *   data: {"type":"done","topic":"...","sources":[...],"conversation_id":"..."}\n\n
 */
export function sseResponse(events: object[], status = 200): Response {
  const encoder = new TextEncoder()
  const stream = new ReadableStream({
    start(controller) {
      for (const event of events) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`))
      }
      controller.close()
    },
  })
  return new Response(stream, {
    status,
    headers: { 'Content-Type': 'text/event-stream' },
  })
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

export const MOCK_CONVERSATIONS = [
  {
    id: 'conv-1',
    title: 'What is consciousness?',
    topic: 'consciousness',
    last_message_at: new Date(Date.now() - 60_000).toISOString(),
    preview: 'Consciousness is the state of being aware...',
  },
  {
    id: 'conv-2',
    title: 'Biohacking basics',
    topic: 'biohacking',
    last_message_at: new Date(Date.now() - 3_600_000).toISOString(),
    preview: 'Biohacking refers to the practice of...',
  },
]

export const MOCK_MESSAGES = {
  conversation_id: 'conv-1',
  messages: [
    { role: 'user', content: 'What is consciousness?' },
    { role: 'assistant', content: 'Consciousness is awareness of oneself and the environment.' },
  ],
}

export const DONE_EVENT = {
  type: 'done',
  topic: 'consciousness',
  sources: [],
  conversation_id: 'new-conv-id',
}
