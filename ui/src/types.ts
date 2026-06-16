export interface Clip {
  chapter: string
  url: string
  start_seconds: number
}

export interface Source {
  video_id: string
  title: string
  channel: string
  speaker: string
  clips: Clip[]
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  topic?: string
  sources?: Source[]
  streaming?: boolean
}

export interface ConversationSummary {
  id: string
  title: string
  topic?: string | null
  last_message_at?: string | null
  preview: string
}

export interface RagChatProps {
  apiUrl: string
  authToken?: string
  placeholder?: string
  className?: string
  showHeader?: boolean
  title?: string
}

export interface RagChatRef {
  reset: () => void
}
