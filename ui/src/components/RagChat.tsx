import { forwardRef, useImperativeHandle } from 'react'
import { useRagChat } from '../hooks/useRagChat'
import { MessageList } from './MessageList'
import { ChatInput } from './ChatInput'
import type { RagChatProps, RagChatRef } from '../types'

function PencilIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
      <path d="M12 20h9M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
    </svg>
  )
}

export const RagChat = forwardRef<RagChatRef, RagChatProps>(function RagChat(
  { apiUrl, placeholder, className = '', showHeader = true, title = 'AI Chat' },
  ref
) {
  const { messages, isLoading, sendMessage, stop, reset } = useRagChat(apiUrl)

  useImperativeHandle(ref, () => ({ reset }), [reset])

  return (
    <div className={`flex flex-col bg-gray-50 rounded-2xl overflow-hidden border border-gray-200 shadow-sm ${className}`}>
      {showHeader && (
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 bg-white shrink-0">
          <span className="text-sm font-medium text-gray-700">{title}</span>
          <button
            onClick={reset}
            disabled={isLoading || messages.length === 0}
            className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            aria-label="New chat"
          >
            <PencilIcon />
            New chat
          </button>
        </div>
      )}
      <MessageList messages={messages} />
      <ChatInput onSend={sendMessage} onStop={stop} isLoading={isLoading} placeholder={placeholder} />
    </div>
  )
})
