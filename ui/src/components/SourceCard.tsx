import type { Source } from '../types'

function formatTime(seconds: number) {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

interface Props {
  source: Source
}

export function SourceCard({ source }: Props) {
  const isArticle = source.source_type === 'article'

  return (
    <div className="rounded-lg border border-gray-200 bg-gray-50 p-3 text-xs">
      <p className="font-medium text-gray-800 leading-snug">{source.title}</p>

      {/* YouTube: speaker · channel */}
      {!isArticle && source.speaker && (
        <p className="text-gray-500 mt-0.5">
          {source.speaker} · {source.channel}
        </p>
      )}

      {/* Article: author · website */}
      {isArticle && (source.author || source.website) && (
        <p className="text-gray-500 mt-0.5">
          {[source.author, source.website].filter(Boolean).join(' · ')}
        </p>
      )}

      <div className="mt-2 flex flex-wrap gap-2">
        {source.clips.map((clip, i) => (
          <a
            key={i}
            href={clip.url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 rounded-md bg-white border border-gray-200 px-2 py-1 text-indigo-600 hover:text-indigo-800 hover:border-indigo-300 transition-colors"
          >
            <span>{isArticle ? '§' : '▶'}</span>
            <span>{clip.chapter}</span>
            {!isArticle && clip.start_seconds != null && (
              <>
                <span className="text-gray-400">·</span>
                <span className="text-gray-500">{formatTime(clip.start_seconds)}</span>
              </>
            )}
          </a>
        ))}
      </div>
    </div>
  )
}
