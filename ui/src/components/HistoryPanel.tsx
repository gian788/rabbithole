import { ConversationCard } from './ConversationCard'
import type { ConversationSummary } from '../types'

interface Props {
  conversations: ConversationSummary[]
  isLoading: boolean
  onSelect: (id: string) => void
}

export function HistoryPanel({ conversations, isLoading, onSelect }: Props) {
  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center text-sm text-gray-400">
        Loading history…
      </div>
    )
  }

  if (conversations.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-sm text-gray-400">
        No past conversations yet.
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto px-2 py-2 space-y-0.5">
      {conversations.map((c) => (
        <ConversationCard key={c.id} conversation={c} onClick={onSelect} />
      ))}
    </div>
  )
}
