import type { ConversationSummary } from '../types'

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const mins  = Math.floor(diff / 60_000)
  const hours = Math.floor(diff / 3_600_000)
  const days  = Math.floor(diff / 86_400_000)
  if (mins  <  1) return 'just now'
  if (mins  < 60) return `${mins}m ago`
  if (hours < 24) return `${hours}h ago`
  return `${days}d ago`
}

interface Props {
  conversation: ConversationSummary
  onClick: (id: string) => void
}

export function ConversationCard({ conversation, onClick }: Props) {
  return (
    <button
      onClick={() => onClick(conversation.id)}
      className="w-full text-left px-3 py-3 rounded-xl hover:bg-gray-100 transition-colors group"
    >
      <div className="flex items-start justify-between gap-2">
        <p className="text-sm font-medium text-gray-800 truncate leading-snug">
          {conversation.title}
        </p>
        {conversation.last_message_at && (
          <span className="shrink-0 text-xs text-gray-400 mt-0.5">
            {relativeTime(conversation.last_message_at)}
          </span>
        )}
      </div>
      {conversation.topic && (
        <span className="inline-block mt-1 text-[10px] font-medium uppercase tracking-wide text-indigo-500 bg-indigo-50 rounded px-1.5 py-0.5">
          {conversation.topic.replace('_', ' ')}
        </span>
      )}
      {conversation.preview && (
        <p className="mt-1 text-xs text-gray-500 line-clamp-2 leading-relaxed">
          {conversation.preview}
        </p>
      )}
    </button>
  )
}
