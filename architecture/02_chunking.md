# Stage 2: Chunking

## Responsibility

`core/chunker.py` — converts a raw YouTube transcript (SRT format) into structured, time-stamped paragraph chunks associated with named chapters.

---

## Why Chunking Matters

Embedding a full transcript as one vector loses retrieval precision. Embedding individual SRT lines (1–2 seconds each) creates noisy, context-free vectors. The goal is **semantically coherent paragraphs of 3–8 sentences** that:

- Start at a known second (for deep-link citations)
- Belong to a named chapter (for navigation context)
- Fit within embedding model token limits
- Capture a complete thought

---

## Pre-Processing: SponsorBlock Filtering

Before chapters are detected or paragraphs are segmented, the raw SRT is cleaned using the **SponsorBlock community API** (free, no API key required).

**Endpoint:**
```
GET https://sponsor.ajay.app/api/skipSegments
  ?videoID={video_id}
  &categories=["sponsor","selfpromo","interaction","intro","outro"]
```

**Categories filtered:**

| Category | Description |
| --- | --- |
| `sponsor` | Paid third-party sponsorship reads ("This video is sponsored by…") |
| `selfpromo` | Channel's own product or service plugs |
| `interaction` | Subscribe/like/notification bell requests |
| `intro` | Branded channel intro animations |
| `outro` | End-card sections / channel outros |

**Segment response format:**

```json
[
  {"category": "sponsor", "segment": [120.5, 180.2], ...},
  {"category": "selfpromo", "segment": [542.0, 575.8], ...}
]
```

**Filtering logic:** for each SRT entry, check if `entry["start"]` falls within any sponsor window `[start, end)`. Matching entries are dropped before any chunking occurs.

**Graceful degradation:** a 404 response (no community submissions) or any network error returns an empty segment list and the full SRT is processed unchanged. The sponsorblock call has a 5-second timeout to avoid delaying Lambda execution.

**Implementation:** `core/chunker.fetch_sponsor_segments(video_id)` → `core/chunker.filter_sponsored_srt(srt, segments)`, called from `ingestion/worker_lambda.py` immediately after transcript fetch.

---

## Chapter Detection: Three-Tier Strategy

### Tier 1 — Native Chapters from Video Description

YouTube video owners often embed chapters in the description using timestamp + title format:

```
0:00 Introduction
2:45 The Pineal Gland and DMT
8:30 Ancient Egyptian Correlations
15:00 Practical Biohacking Protocols
```

Regex: `(?:\[)?(\d{1,2}:\d{2}(?::\d{2})?)(?:\])?\s+(.*)`

Matches formats:
- `5:30 Title`
- `[5:30] Title`
- `1:05:30 Title` (hours)

Parse timestamp to seconds:
```python
parts = ts.split(":")
if len(parts) == 2:   # MM:SS
    seconds = int(parts[0]) * 60 + int(parts[1])
elif len(parts) == 3:  # HH:MM:SS
    seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
```

Returns `[{"start_seconds": int, "title": str}]` sorted ascending. Returns `[]` on any exception.

**Trigger:** Use this tier if result has `>= 3` chapters.

### Tier 2 — LLM-Generated Chapters (Claude Haiku 3.5)

When native chapters are absent or sparse, send the transcript text to Claude Haiku with a structured output prompt:

**System prompt:**
```
You are a transcript analyst. Given a YouTube video transcript, identify 5–12 logical
section boundaries where the speaker transitions to a new topic or subtopic.
Return ONLY valid JSON — an array of objects with "title" (string) and "start_seconds" (integer).
Example: [{"title": "Introduction", "start_seconds": 0}, ...]
Output nothing else.
```

**Input:** first 12,000 characters of transcript (safety cap for Haiku context).

The response is parsed with `json.loads`. Each returned `start_seconds` is snapped to the nearest actual SRT segment start to ensure alignment with real subtitle timings.

**Trigger:** Use this tier if Tier 1 returns `< 3` chapters.

### Tier 3 — Fixed Word Chunking (Fallback)

Used when no SRT is available (e.g., auto-captions disabled) or Tier 2 also fails:

- Chunk size: 300 words
- Overlap: 50 words (sliding window)
- Stride: 250 words per step

```python
words = text.split()
for i in range(0, len(words), chunk_size - overlap):
    chunk = " ".join(words[i : i + chunk_size])
```

All word-chunks get `associated_chapter = "General"` and `start_seconds = 0` (no timing available without SRT).

---

## Paragraph Segmentation

Once chapters are known, the SRT transcript is segmented into paragraphs using three boundary conditions evaluated in order:

```
For each SRT segment:
  1. PUNCTUATION END: segment text ends with  .  !  ?
  2. SILENCE GAP:     gap to next segment  >  2.5 seconds
  3. TOKEN CEILING:   accumulated 6 fragments without any break

Any of these → flush current accumulator → new paragraph
```

This mirrors how a human reader would naturally divide speech into sentences and paragraphs.

Each paragraph dict:

```python
{
    "chunk_id": "p_001",
    "associated_chapter": "The Pineal Gland and DMT",
    "start_seconds": 145,
    "deep_link": "https://youtu.be/VIDEO_ID?t=145",
    "text_content": "The pineal gland has been called the seat of the soul..."
}
```

**Chapter assignment:** for each paragraph, find the chapter whose `start_seconds` is `<=` the paragraph's `start_seconds` (binary search). If no chapters, use `"General"`.

---

## S3 JSON Payload

The full structured output stored to S3 for each video:

```json
{
  "video_id": "dQw4w9WgXcQ",
  "video_title": "Advanced Python In Production",
  "channel_id": "UCxxxxxxxxxxxx",
  "video_base_url": "https://youtu.be",
  "topics": ["consciousness", "biohacking"],
  "total_paragraphs": 142,
  "paragraphs": [
    {
      "chunk_id": "p_001",
      "associated_chapter": "Introduction",
      "start_seconds": 12,
      "deep_link": "https://youtu.be/dQw4w9WgXcQ?t=12",
      "text_content": "Welcome back..."
    }
  ]
}
```

S3 key: `transcripts/{primary_topic}/{channel_id}/{video_id}_structured.json`

The S3 payload does **not** contain embedding vectors (those live in Pinecone only).

---

## Design Decisions

| Decision | Rationale |
|---|---|
| 6-fragment ceiling (not word count) | SRT segments vary in length; fragment count is more stable |
| 2.5s silence gap | Typical YouTube editing pause between thoughts |
| 12,000-char LLM cap | ~3,000 tokens — fits Haiku's context and avoids unnecessary spend |
| `>= 3` chapters threshold | 1–2 chapters (e.g., just "Intro" + "Content") adds no navigation value |
| 50-word overlap on word chunks | Preserves sentence context across chunk boundaries |
