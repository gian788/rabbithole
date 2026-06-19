# Stage 4: Retrieval

## Responsibility

`retrieval/main.py` — given a natural language query, finds the most relevant chunks across all indexed content (YouTube videos and web articles) and returns a synthesized answer with deep-link citations.

---

## Query Pipeline: Three Steps

```
User query
    │
    ▼ Step A — Intent Classification
    │  gpt-4o-mini classifies query into one of the known topics
    │  "What did the Egyptians know about consciousness?"  →  "alternative_history"
    │
    ▼ Step B — Hybrid Vector Search + Re-ranking
    │  1. Embed query (dense: text-embedding-3-small)
    │  2. Encode query (sparse: BM25)
    │  3. Pinecone query: top-20 candidates, filtered by topic
    │  4. Cross-encoder re-ranks top-20 → selects top-5
    │
    ▼ Step C — Context Synthesis
       gpt-4o-mini answers using retrieved chunks as grounded sources
       Returns: answer + citations[]
```

---

## Step A: Intent Classification

Before searching, the query is classified into one of the configured topics. This narrows the vector search to a semantically relevant partition.

```python
topics = ["alternative_history", "biohacking", "consciousness", "spirituality"]

system_prompt = f"""You are a topic classifier.
Classify the user's question into exactly one topic from this list: {topics}.
Output only the topic name, nothing else."""

response = gateway.get_completion(
    prompt=user_query,
    system_prompt=system_prompt,
    model="gpt-4o-mini",
    provider="openai",
    associated_id="intent_router"
)
predicted_topic = response.text_content.strip().lower()
```

If the response is not in the topics list (hallucination or formatting error), fall back to `topics[0]` — graceful degradation rather than an error.

---

## Step B: Hybrid Search

### Query Encoding

```python
# Dense vector
embed_resp = gateway.get_embedding(user_query, associated_id="query_embed")
dense_vector = embed_resp.embedding_vector

# Sparse vector
from pinecone_text.sparse import BM25Encoder
_bm25 = BM25Encoder.default()   # module-level singleton
sparse_vector = _bm25.encode_queries([user_query])[0]
```

### Pinecone Query

```python
results = pinecone_index.query(
    vector=dense_vector,
    sparse_vector=sparse_vector,
    alpha=0.7,                                      # 70% dense, 30% sparse
    top_k=20,                                       # over-fetch for re-ranking
    filter={"topics": {"$in": [predicted_topic]}},  # restrict to relevant topic
    include_metadata=True
)
```

The `$in` operator matches vectors where the `topics` metadata array contains the predicted topic. A video tagged `["consciousness", "biohacking"]` matches a query for either topic.

### Re-ranking

Initial vector similarity is fast but imprecise — cosine/dotproduct scores can rank a tangentially related chunk above a highly relevant one. A cross-encoder re-scores all 20 candidates by jointly encoding the query + each passage:

```python
from sentence_transformers import CrossEncoder

# Loaded once at module level (stays warm across Lambda invocations)
# SENTENCE_TRANSFORMERS_HOME=/tmp ensures Lambda disk caching
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs  = [(user_query, m.metadata["text_content"]) for m in results.matches]
scores = _reranker.predict(pairs)

top5 = [m for m, _ in sorted(
    zip(results.matches, scores),
    key=lambda x: x[1], reverse=True
)[:5]]
```

**Model details:**
- Size: ~80MB — fits in Lambda package
- Architecture: MiniLM cross-encoder fine-tuned on MS MARCO passage ranking
- Latency: ~50–100ms for 20 passages on CPU
- Cold start: first invocation downloads model to `/tmp`; subsequent invocations reuse

---

## Step C: Context Synthesis

Build the context string from the top-5 re-ranked chunks:

```python
context_blocks = [
    f"[Chapter: {m.metadata['chapter']} | {m.metadata['deep_link']}]\n{m.metadata['text_content']}"
    for m in top5
]
context = "\n\n---\n\n".join(context_blocks)
```

System prompt:
```
You are a knowledgeable research assistant. Answer the user's question using ONLY the
provided source excerpts. Be concise, factual, and cite the relevant sections.
Do not fabricate information not present in the sources.

Sources:
{context}
```

The synthesis completion is logged to `model_telemetry` and its cost is stored in `rag_queries`.

---

## Response Format

```json
{
  "answer": "The ancient Egyptians depicted the Djed pillar as a symbol...",
  "citations": [
    {
      "title": "Graham Hancock — Ancient Egypt and Consciousness",
      "channel": "Gregg Braden Official",
      "chapter": "The Djed Pillar and Spinal Energy",
      "url": "https://youtu.be/dQw4w9WgXcQ?t=1423",
      "start_seconds": 1423
    }
  ]
}
```

---

## Analytics Logging

After every successful query, a row is inserted into `rag_queries`:

```sql
INSERT INTO rag_queries
    (user_query, queried_topic, video_ids, retrieval_cost)
VALUES (%(query)s, %(topic)s, %(video_ids)s, %(cost)s)
```

`video_ids` is an array of the video IDs from the top-5 citations. This feeds the dashboard's value attribution analysis — channels whose videos appear in citations are generating real user value.

---

## Tuning Parameters

| Parameter | Default | Effect of increasing | Effect of decreasing |
|---|---|---|---|
| `alpha` | 0.7 | More semantic, less keyword | More keyword, less semantic |
| `top_k` | 20 | More candidates for re-ranker | Faster but may miss best result |
| Final citations | 5 | More context for LLM | Fewer but higher-precision citations |

For keyword-heavy queries (specific names, places, compounds), try `alpha = 0.5`.
For broad conceptual queries ("what is consciousness"), `alpha = 0.8` works better.

---

## Polymorphic Sources

The retrieval layer handles both YouTube chunks and article chunks returned by a single `store.query()` call. Source type is determined by `metadata.get("source_type", "youtube_video")` — old vectors without this field default to YouTube.

**`_merge_adjacent_chunks`** groups nearby chunks from the same source before building citations. Merging only applies to YouTube chunks (requires `video_id` + `start_seconds`); article chunks are never merged.

**`_format_source_block`** branches on `source_type` to build the LLM context string:

- YouTube: `[Chapter: {chapter} | {deep_link}]\n{text}` — includes timestamp
- Article: `[Section: {chapter} | {deep_link}]\n{text}` — includes anchor URL

**`Source` response model** carries optional fields per type:

```python
class Source(BaseModel):
    source_type: str = "youtube_video"
    title:       str
    clips:       list[Clip]
    # YouTube-only
    video_id:    Optional[str] = None
    channel:     Optional[str] = None
    speaker:     Optional[str] = None
    # Article-only
    article_id:  Optional[str] = None
    author:      Optional[str] = None
    website:     Optional[str] = None

class Clip(BaseModel):
    chapter:       str
    url:           str
    start_seconds: Optional[int] = None  # None for articles
```

Article metadata is fetched from DB via `_fetch_article_meta(db, article_ids)`:

```sql
SELECT a.id, a.title, a.author, w.name AS website_name
FROM articles a JOIN websites w ON w.id = a.website_id
WHERE a.id = ANY(%s)
```

Both `video_ids` and `article_ids` are logged to `rag_queries` for cost attribution.
