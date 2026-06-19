# System Overview: Topic RAG Engine

## Purpose

A fully serverless pipeline that ingests content from curated YouTube channels and websites, stores structured chunks in a vector database, and exposes a REST API for semantic question-answering with deep-link citations back to the exact source.

**Target topics:** human consciousness, alternative history (Egypt, Atlantis), biohacking, spirituality.

---

## Source Types

The system handles two parallel source types with a symmetric ingestion model:

| Concept | YouTube | Articles |
|---|---|---|
| Source registry | `channels` table | `websites` table |
| Content unit | `videos` table | `articles` table |
| Discovery lambda | `fetch_lambda.py` | `article_fetch_lambda.py` |
| Processing worker | `worker_lambda.py` | `article_worker.py` |
| SQS message | `{video_id, channel_id}` | `{article_id, url, website_id}` |
| Chunk deep-link | YouTube timestamp URL | Article section anchor URL |

---

## Topology

```
[ AWS EventBridge Cron (every 6h) ]
              │
    ┌─────────┴──────────┐
    │                    │
    ▼                    ▼
┌──────────────┐  ┌──────────────────────┐
│ fetch_lambda │  │ article_fetch_lambda  │
│  YouTube API │  │  RSS/feed polling     │
│  → videos DB │  │  → articles DB        │
└──────┬───────┘  └──────────┬───────────┘
       │ SQS                 │ SQS
       ▼                     ▼
┌──────────────┐  ┌──────────────────────┐
│ worker_lambda│  │ article_worker        │
│  transcript  │  │  HTML fetch+parse     │
│  + LLM topics│  │  + LLM topics         │
│  + embed     │  │  + embed              │
└──────┬───────┘  └──────────┬───────────┘
       │                     │
       └────────┬────────────┘
                │
                ▼
        [ VectorStore ]
      Pinecone (production)
      Chroma   (local dev)
      dense vectors + metadata
                │
                ▼
  ┌─────────────────────────┐
  │  main.py (retrieval/)   │◄── [ API Gateway HTTP API ]
  │  FastAPI + Mangum        │
  │  1. Classify query topic │
  │  2. Vector search        │
  │  3. Re-rank top-20→5     │
  │  4. Synthesize answer    │
  └─────────────────────────┘
              │
              ▼
  { answer, topic, sources: [{title, clips: [{chapter, url}]}] }

  ┌─────────────────────────┐
  │  dashboard/app.py       │  Streamlit ops portal
  │  Neon PostgreSQL        │  Channels + Websites mgmt, cost attribution
  └─────────────────────────┘
```

---

## Data Paths

### YouTube Ingest Path
`EventBridge` → `fetch_lambda` → `SQS` → `worker_lambda` → `VectorStore` + `S3` + `Neon DB`

### Article Ingest Path
`EventBridge` → `article_fetch_lambda` → `SQS` → `article_worker` → `VectorStore` + `S3` + `Neon DB`

### Query Path
`HTTP POST /v1/chat` → `FastAPI` → `VectorStore query` → `cross-encoder rerank` → `gpt-4o-mini synthesis` → JSON response

---

## VectorStore Abstraction

`core/vector_store.py` provides a unified interface over two backends, switched via `VECTOR_STORE` env var:

| Backend | `VECTOR_STORE` value | Search type | Use case |
|---|---|---|---|
| `PineconeStore` | `pinecone` (default) | Hybrid dense + BM25 sparse | Production |
| `ChromaStore` | `chroma` | Dense-only | Local dev (zero cloud cost) |

Both backends accept the same `where={"primary_topic": topic}` filter, translating internally:
- Pinecone: `{"topics": {"$in": [topic]}}` — works with old and new vectors
- Chroma: `{"primary_topic": {"$eq": topic}}` — requires scalar `primary_topic` field

---

## Stack Decisions

| Layer | Choice | Reason |
|---|---|---|
| Vector DB (prod) | Pinecone Serverless | Hybrid dense+sparse, `$in` array filter, free tier |
| Vector DB (dev) | Chroma | Zero cost, zero cloud deps, local persistence |
| Embeddings | OpenAI text-embedding-3-small | $0.02/M tokens, 1536 dims |
| Sparse encoding | pinecone-text BM25Encoder | Pre-trained on MS MARCO, no corpus fitting |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | ~80MB, runs on Lambda, no API cost |
| RAG synthesis | OpenAI gpt-4o-mini | $0.15/$0.60 per M tokens, fast |
| Topic classification | Anthropic Claude Haiku 3.5 | Reliable JSON output, shared by both workers |
| State DB | Neon PostgreSQL | Serverless, free tier, full SQL for analytics |
| Compute | AWS Lambda | $0 idle, auto-scales |
| Queue | Amazon SQS | Long-polling (20s), DLQ after 3 failures |
| Storage | Amazon S3 | Structured JSON lakehouse |

---

## Topic Model

Topics are assigned **per content unit** (video or article), not per source (channel or website). Each source has an optional `default_topic` used only as a Claude Haiku classification hint. Content is tagged with 1–N topics from `core/topics.py::classify_topics()`, shared by both workers.

This allows a channel or website covering multiple subjects to have all its content correctly indexed by topic.

Both `primary_topic` (scalar, Chroma-compatible) and `topics` (array, Pinecone-compatible) are stored in vector metadata.

---

## Cost Model

**Idle cost: $0.00/month**

| Operation | Cost |
|---|---|
| Embed 1 video (~10k tokens) | ~$0.0002 |
| Embed 1 article (~3k tokens) | ~$0.00006 |
| Classify topics (Claude Haiku) | ~$0.0003 |
| Generate chapters (Claude Haiku, video only) | ~$0.0008 |
| Answer 1 user query | ~$0.002 |
| Store content in Pinecone | Included in free tier |

---

## Phase Roadmap

- **Phase 1:** YouTube channels, transcript ingestion, RAG API, ops dashboard
- **Phase 2 (current):** Article/website ingestion, VectorStore abstraction (Pinecone + Chroma), polymorphic sources in UI
- **Phase 3:** Terraform IaC, channel/website discovery automation, scheduled pruning
