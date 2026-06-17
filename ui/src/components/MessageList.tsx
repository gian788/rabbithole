import { useEffect, useRef } from 'react'
import { SourceCard } from './SourceCard'
import type { Message } from '../types'

function TypingDots() {
  return (
    <div className="flex items-center gap-1 py-1">
      <span className="w-2 h-2 rounded-full bg-gray-400 animate-bounce [animation-delay:-0.3s]" />
      <span className="w-2 h-2 rounded-full bg-gray-400 animate-bounce [animation-delay:-0.15s]" />
      <span className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" />
    </div>
  )
}

interface Props {
  messages: Message[]
}

export function MessageList({ messages }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-sm text-gray-400">
        Ask a question to get started.
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
      {messages.map((msg) => (
        <div
          key={msg.id}
          className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
        >
          <div className="max-w-[85%] space-y-2">
            <div
              className={
                msg.role === 'user'
                  ? 'rounded-2xl rounded-tr-sm bg-indigo-600 px-4 py-2.5 text-sm text-white'
                  : 'rounded-2xl rounded-tl-sm bg-white border border-gray-200 px-4 py-2.5 text-sm text-gray-800 min-w-[60px]'
              }
            >
              {msg.streaming && !msg.content ? (
                <TypingDots />
              ) : (
                <>
                  {msg.content}
                  {msg.streaming && (
                    <span className="inline-block w-0.5 h-3.5 ml-0.5 bg-gray-500 animate-pulse rounded-sm" />
                  )}
                </>
              )}
            </div>
            {!msg.streaming && msg.sources && msg.sources.length > 0 && (
              <div className="space-y-1.5">
                {msg.sources.map((src) => (
                  <SourceCard key={src.video_id} source={src} />
                ))}
              </div>
            )}
          </div>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
