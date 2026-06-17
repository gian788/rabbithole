import { forwardRef, useImperativeHandle, useState } from 'react'
import { useRagChat } from '../hooks/useRagChat'
import { useConversations } from '../hooks/useConversations'
import { MessageList } from './MessageList'
import { ChatInput } from './ChatInput'
import { HistoryPanel } from './HistoryPanel'
import type { RagChatProps, RagChatRef } from '../types'

function PencilIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="w-4 h-4"
    >
      <path d="M12 20h9M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
    </svg>
  )
}

function ClockIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="w-4 h-4"
    >
      <circle cx="12" cy="12" r="10" />
      <path d="M12 6v6l4 2" />
    </svg>
  )
}

function ArrowLeftIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="w-4 h-4"
    >
      <path d="M19 12H5M12 19l-7-7 7-7" />
    </svg>
  )
}

export const RagChat = forwardRef<RagChatRef, RagChatProps>(function RagChat(
  { apiUrl, authToken, placeholder, className = '', showHeader = true, title = 'AI Chat' },
  ref
) {
  const [view, setView] = useState<'chat' | 'history'>('chat')
  const { messages, isLoading, authError, sendMessage, stop, reset, loadConversation } = useRagChat(
    apiUrl,
    authToken
  )
  const { conversations, isLoading: historyLoading, reload } = useConversations(apiUrl, authToken)

  useImperativeHandle(ref, () => ({ reset }), [reset])

  async function handleSelectConversation(id: string) {
    const ok = await loadConversation(id)
    if (ok) setView('chat')
  }

  async function handleSend(query: string) {
    await sendMessage(query)
    reload()
  }

  function handleReset() {
    reset()
    setView('chat')
  }

  const hasHistory = !!authToken

  return (
    <div
      className={`flex flex-col bg-gray-50 rounded-2xl overflow-hidden border border-gray-200 shadow-sm ${className}`}
    >
      {showHeader && (
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 bg-white shrink-0">
          {view === 'history' ? (
            <>
              <button
                onClick={() => setView('chat')}
                className="flex items-center gap-1.5 text-sm text-gray-500 hover:text-gray-800 transition-colors"
                aria-label="Back to chat"
              >
                <ArrowLeftIcon />
                Back
              </button>
              <span className="text-sm font-medium text-gray-700">History</span>
              <div className="w-16" />
            </>
          ) : (
            <>
              <span className="text-sm font-medium text-gray-700">{title}</span>
              <div className="flex items-center gap-2">
                {hasHistory && (
                  <button
                    onClick={() => {
                      reload()
                      setView('history')
                    }}
                    className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-800 transition-colors"
                    aria-label="Conversation history"
                  >
                    <ClockIcon />
                  </button>
                )}
                <button
                  onClick={handleReset}
                  disabled={isLoading || messages.length === 0}
                  className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                  aria-label="New chat"
                >
                  <PencilIcon />
                  New chat
                </button>
              </div>
            </>
          )}
        </div>
      )}

      {view === 'history' ? (
        <HistoryPanel
          conversations={conversations}
          isLoading={historyLoading}
          onSelect={handleSelectConversation}
        />
      ) : authError ? (
        <div className="flex-1 flex items-center justify-center text-sm text-red-500 px-6 text-center">
          Session expired. Please refresh the page.
        </div>
      ) : (
        <MessageList messages={messages} />
      )}

      {view === 'chat' && (
        <ChatInput
          onSend={handleSend}
          onStop={stop}
          isLoading={isLoading}
          placeholder={placeholder}
        />
      )}
    </div>
  )
})
