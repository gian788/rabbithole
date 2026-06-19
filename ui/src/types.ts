export interface Clip {
  chapter: string
  url: string
  start_seconds?: number  // undefined for article sections
}

export type SourceType = 'youtube_video' | 'article'

export interface Source {
  source_type: SourceType
  title: string
  clips: Clip[]
  // YouTube-only
  video_id?: string
  channel?: string
  speaker?: string
  // Article-only
  article_id?: string
  author?: string
  website?: string
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
