# Stage 3: Embedding

## Responsibility

Two workers share the same embedding pipeline:

- `ingestion/worker_lambda.py` — embeds YouTube transcript chunks
- `ingestion/article_worker.py` — embeds article section chunks

Both call `core/gateway.py` for embeddings and `core/topics.py` for topic classification, then upsert to the configured `VectorStore`.

---

## Vector Strategy: Hybrid Dense + Sparse

Relying on dense vectors alone misses exact-term queries. A user asking about *"Schumann resonance"* or *"DMT pineal gland"* needs keyword matching, not just semantic similarity. Hybrid search combines both:

| Vector type | Encoding | Strength |
|---|---|---|
| **Dense** | OpenAI text-embedding-3-small | Semantic meaning, paraphrase matching |
| **Sparse** | BM25 (pinecone-text) | Exact term matching, rare proper nouns |

At query time, scores are blended: `score = alpha × dense + (1 - alpha) × sparse`

Default `alpha = 0.7` (dense-heavy). This can be tuned per topic — more keyword-heavy topics like specific historical figures may benefit from lower alpha.

---

## Dense Embeddings

**Model:** `text-embedding-3-small`
**Provider:** OpenAI
**Dimensions:** 1536
**Cost:** $0.02 per 1 million tokens

For a typical 1-hour video (~15,000 words, ~120 chunks of ~125 words each):
- ~125 words × 1.3 tokens/word ≈ 163 tokens/chunk
- 120 chunks × 163 tokens ≈ 19,500 tokens
- Cost: ~$0.0004 per video

Each chunk is embedded individually via `ModelGateway.get_embedding(text)`. The gateway:
1. Calls `OpenAIProvider.generate_embedding(text, model="text-embedding-3-small")`
2. Measures latency
3. Calculates cost from the pricing ledger
4. Logs a row to `model_telemetry`
5. Returns `ModelResponse(embedding_vector=[...], input_tokens=N, ...)`

---

## Sparse Embeddings (BM25)

**Library:** `pinecone-text` — `BM25Encoder`
**Variant:** `.default()` — pre-trained on MS MARCO (no corpus fitting required)

```python
from pinecone_text.sparse import BM25Encoder
bm25 = BM25Encoder.default()   # module-level singleton, loaded once

sparse_vector = bm25.encode_documents([chunk_text])[0]
# Returns: {"indices": [int, ...], "values": [float, ...]}
```

No API calls, no cost. The BM25 encoder maps text tokens to sparse indices and TF-IDF-weighted float values.

---

## Topic Classification

Before embedding, each video is classified into 1–N topics using Claude Haiku 3.5:

**Input:** video title + first 500 chars of description + channel's `default_topic` hint

**System prompt:**
```
This channel primarily covers {default_topic}.
Classify this video into ALL relevant topics from this list: {available_topics}.
Return ONLY a valid JSON array of strings, e.g. ["consciousness", "spirituality"].
Output nothing else.
```

**Output:** e.g. `["consciousness", "alternative_history"]`

Validation: each returned item must be in `available_topics`. On parse error or empty result, falls back to `[default_topic]`.

The topics list is stored in:
- `videos.topics` column in Neon DB (for analytics and dashboard)
- `metadata.topics` in each Pinecone vector (for filter queries)

---

## Pinecone Upsert

**Index requirements:**
- Metric: `dotproduct` (mandatory for hybrid search — cosine does not support sparse vectors)
- Dimensions: 1536
- Type: Serverless

Each chunk becomes one Pinecone vector:

```python
{
    "id": f"{video_id}_{chunk_id}",   # e.g. "dQw4w9WgXcQ_p_001"
    "values": dense_embedding,         # list[float], len=1536
    "sparse_values": {
        "indices": [...],
        "values": [...]
    },
    "metadata": {
        "video_id":    "dQw4w9WgXcQ",
        "channel_id":  "UCxxxxxxxxx",
        "topics":      ["consciousness", "biohacking"],   # array — enables $in filter
        "chapter":     "The Pineal Gland and DMT",
        "start_seconds": 145,
        "deep_link":   "https://youtu.be/dQw4w9WgXcQ?t=145",
        "text_content": "The pineal gland has been called..."  # truncated to 1000 chars
    }
}
```

Upserts are batched in groups of 100 (Pinecone batch limit).

---

## Cost Telemetry

Every model call — embeddings, topic classification, chapter generation, RAG synthesis — is logged to `model_telemetry`:

```sql
INSERT INTO model_telemetry
    (transaction_type, provider, model, input_tokens, output_tokens,
     latency_ms, cost, associated_id)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
```

`associated_id` is the video ID for ingest operations, or a string label (`'intent_router'`, `'synthesis'`) for retrieval operations. This feeds the dashboard's cost attribution view.

---

## Accumulated Ingest Costs per Video

After all chunks are embedded and upserted, the worker updates the video row:

```sql
UPDATE videos
SET status = 'completed',
    s3_path = %(s3_path)s,
    ingestion_tokens = %(total_tokens)s,
    ingestion_cost   = %(total_cost)s,
    processed_at     = NOW()
WHERE id = %(video_id)s
```

---

## VectorStore Abstraction (`core/vector_store.py`)

Both workers call `store.upsert(ids, embeddings, metadatas, texts)` via the `VectorStore` interface, which dispatches to the correct backend:

**`PineconeStore`** (production, `VECTOR_STORE=pinecone`):
- Hybrid dense + BM25 sparse upsert
- BM25 computed per chunk from `pinecone_text.BM25Encoder.default()`
- Chunks with empty BM25 indices (all stopwords) are skipped
- Batched in groups of 100

**`ChromaStore`** (local dev, `VECTOR_STORE=chroma`):
- Dense-only upsert; Chroma handles indexing
- List metadata values converted to JSON strings (Chroma restriction)
- `None` metadata values dropped before upsert

### Vector Metadata Schema

All vectors carry `source_type` to allow polymorphic retrieval:

```python
# YouTube video chunk
{
    "source_type":   "youtube_video",
    "video_id":      "dQw4w9WgXcQ",
    "channel_id":    "UCxxxxxxxxx",
    "topics":        ["consciousness", "biohacking"],  # Pinecone $in filter
    "primary_topic": "consciousness",                  # Chroma $eq filter
    "chapter":       "The Pineal Gland and DMT",
    "start_seconds": 145,
    "deep_link":     "https://youtu.be/dQw4w9WgXcQ?t=145",
    "text_content":  "..."  # truncated to 1000 chars in PineconeStore
}

# Article section chunk
{
    "source_type":   "article",
    "article_id":    "550e8400-e29b-41d4-a716-446655440000",
    "website_id":    "hubermanlab.com",
    "topics":        ["biohacking"],
    "primary_topic": "biohacking",
    "chapter":       "Cold Exposure Protocols",
    "section_slug":  "cold-exposure-protocols",
    "deep_link":     "https://hubermanlab.com/post#cold-exposure-protocols",
    "text_content":  "..."
}
```

Old YouTube vectors without `source_type` default to `"youtube_video"` via `.get("source_type", "youtube_video")` in retrieval — no migration required.

---

## Article Chunking (`core/article_fetcher.py`)

Articles are chunked by HTML structure rather than transcript timestamps:

1. `fetch_article(url)` — HTTP GET, extracts title/author/published_at via Open Graph and `<time>` tags
2. `extract_sections(html_body, url)` — finds H2/H3 headings, accumulates `<p>` text per section
   - Sections < 50 words are merged into the preceding section
   - Fallback to sliding-window word chunks (300 words, 50 overlap) when no headings exist
   - Each chunk's `deep_link` is `url#slugified-heading`

Topic classification runs identically via `core/topics.py::classify_topics()`, shared with the video worker.

`total_cost` = sum of embedding costs + topic classification cost + chapter generation cost (if LLM was used).
