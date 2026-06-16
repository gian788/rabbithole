import { useState, useRef, type KeyboardEvent } from 'react'

interface Props {
  onSend: (query: string) => void
  onStop: () => void
  isLoading: boolean
  placeholder?: string
}

function ArrowUpIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
      <path d="M12 19V5M5 12l7-7 7 7" />
    </svg>
  )
}

function StopIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
      <rect x="5" y="5" width="14" height="14" rx="2" />
    </svg>
  )
}

export function ChatInput({ onSend, onStop, isLoading, placeholder = 'Ask anything...' }: Props) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  function handleSend() {
    const q = value.trim()
    if (!q || isLoading) return
    setValue('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
    onSend(q)
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  function handleInput() {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 120)}px`
  }

  const canSend = value.trim().length > 0 && !isLoading

  return (
    <div className="flex items-end gap-2 border-t border-gray-200 p-3 bg-white">
      <textarea
        ref={textareaRef}
        rows={1}
        value={value}
        onChange={e => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        onInput={handleInput}
        placeholder={placeholder}
        className="flex-1 resize-none rounded-xl border border-gray-300 px-3 py-2 text-sm leading-5 focus:outline-none focus:ring-2 focus:ring-indigo-500 max-h-[120px] overflow-y-auto"
      />
      {isLoading ? (
        <button
          onClick={onStop}
          className="shrink-0 flex items-center justify-center w-9 h-9 rounded-xl bg-gray-800 text-white hover:bg-gray-700 transition-colors"
          aria-label="Stop"
        >
          <StopIcon />
        </button>
      ) : (
        <button
          onClick={handleSend}
          disabled={!canSend}
          className="shrink-0 flex items-center justify-center w-9 h-9 rounded-xl bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          aria-label="Send"
        >
          <ArrowUpIcon />
        </button>
      )}
    </div>
  )
}
