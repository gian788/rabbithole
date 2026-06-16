# System Overview: YouTube Topic RAG Engine

## Purpose

A fully serverless pipeline that ingests YouTube transcripts from a curated list of channels, stores them as structured, time-stamped chunks in a vector database, and exposes a REST API for semantic question-answering with deep-link citations back to the exact video timestamp.

**Target topics:** human consciousness, alternative history (Egypt, Atlantis), biohacking, spirituality.

---

## Topology

```
[ AWS EventBridge Cron (every 6h) ]
              │
              ▼
  ┌─────────────────────────┐
  │  fetch_lambda.py        │  Polls YouTube uploads playlists
  │  (ingestion/fetch)      │  Inserts new video rows → DB
  └───────────┬─────────────┘
              │ send_message_batch
              ▼
  ┌─────────────────────────┐       ┌──────────────────┐
  │  Amazon SQS             │──────►│  Dead-Letter Queue│
  │  (video-process-queue)  │  3x   │  (7-day retention)│
  └───────────┬─────────────┘       └──────────────────┘
              │ triggers (batch)
              ▼
  ┌─────────────────────────┐
  │  worker_lambda.py       │  Downloads transcript
  │  (ingestion/worker)     │  Classifies topics (LLM)
  │                         │  Generates chapters (LLM)
  │                         │  Embeds chunks (OpenAI)
  └────┬────────────────────┘
       │                  │
       ▼                  ▼
  [ Pinecone ]        [ Amazon S3 ]
  dense + sparse      structured JSON
  vectors             transcripts/{topic}/{channel}/{video}.json
       │
       ▼
  ┌─────────────────────────┐
  │  main.py (retrieval/)   │◄── [ API Gateway HTTP API ]
  │  FastAPI + Mangum        │
  │  1. Classify query topic │
  │  2. Hybrid vector search │
  │  3. Re-rank top-20→5     │
  │  4. Synthesize answer    │
  └─────────────────────────┘
              │
              ▼
  { answer, citations: [{title, channel, chapter, url, start_seconds}] }

  ┌─────────────────────────┐
  │  dashboard/app.py       │  Streamlit ops portal (local / EC2)
  │  Neon PostgreSQL        │  Cost attribution, channel mgmt
  └─────────────────────────┘
```

---

## Data Paths

### Ingest Path
`EventBridge` → `fetch_lambda` → `SQS` → `worker_lambda` → `Pinecone` + `S3` + `Neon DB`

### Query Path
`HTTP POST /v1/chat` → `FastAPI` → `Pinecone hybrid query` → `cross-encoder rerank` → `gpt-4o-mini synthesis` → JSON response

---

## Stack Decisions

| Layer | Choice | Reason |
|---|---|---|
| Vector DB | Pinecone Serverless | Hybrid dense+sparse, free tier, `$in` array filter |
| Embeddings | OpenAI text-embedding-3-small | $0.02/M tokens, 1536 dims, best quality/cost |
| Sparse encoding | pinecone-text BM25Encoder | Pre-trained, no corpus fitting, exact term matching |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | ~80MB, runs on Lambda, no API cost |
| RAG synthesis | OpenAI gpt-4o-mini | $0.15/$0.60 per M tokens, fast |
| Chapter gen + topic classification | Anthropic Claude Haiku 3.5 | Reliable JSON output, $0.80/$4.00 per M tokens |
| State DB | Neon PostgreSQL | Serverless, free tier, scales to $0, full SQL for analytics |
| Compute | AWS Lambda | $0 idle, auto-scales, per-invocation billing |
| Queue | Amazon SQS | Long-polling (20s), DLQ after 3 failures |
| Storage | Amazon S3 | Structured JSON lakehouse, Standard-IA pricing |
| IaC | Terraform | Phase 2 |

---

## Topic Model

Topics are assigned **per video**, not per channel. A channel has an optional `default_topic` used only as a classification prompt hint. Each video is tagged with 1–N topics by Claude Haiku at ingest time based on its title and description.

This allows a channel covering multiple subjects (e.g. Joe Rogan: consciousness + biohacking + spirituality) to have all its videos correctly indexed.

Pinecone filter at query time: `{"topics": {"$in": ["consciousness"]}}`

---

## Cost Model

**Idle cost: $0.00/month**

All compute (Lambda, SQS, EventBridge) has no standing charge. Neon and Pinecone free tiers cover initial scale. Costs only accrue when processing or serving requests.

| Operation | Cost |
|---|---|
| Embed 1 video (~10k tokens) | ~$0.0002 |
| Classify topics (Claude Haiku) | ~$0.0003 |
| Generate chapters (Claude Haiku) | ~$0.0008 |
| Answer 1 user query | ~$0.002 |
| Store 1 video in Pinecone | Included in free tier |

---

## Phase Roadmap

- **Phase 1 (current):** Static channel list, transcript ingestion, RAG API, ops dashboard
- **Phase 2:** Terraform IaC, channel discovery automation, scheduled pruning
- **Phase 3:** Multi-language transcript support, YouTube Shorts handling, citation UX
