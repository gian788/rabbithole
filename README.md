# rabbithole

**A multi-source RAG system for going deep on topics you care about — YouTube, blogs, and beyond.**

Ask a question about consciousness, biohacking, spirituality, or whatever rabbit hole you're in. Get a synthesized answer with citations that link back to the exact timestamp in the video (or paragraph in the article) where it came from.

---

## What it does

1. **Ingests** YouTube channels on a schedule — downloads transcripts, generates chapters, classifies topics with an LLM
2. **Indexes** everything into a hybrid vector store (dense + sparse) for semantic + keyword search
3. **Answers** questions via a RAG API — classifies intent, retrieves the best chunks, re-ranks, synthesizes
4. **Cites** the exact source: video title, channel, chapter, and deep link to the timestamp

---

## Architecture

```
[ EventBridge Cron (6h) ]
         │
         ▼
  fetch_lambda          polls YouTube upload playlists → new video rows in DB
         │
         ▼
  Amazon SQS ──(3x retry)──► Dead-Letter Queue
         │
         ▼
  worker_lambda         transcript → chapters (LLM) → topic tags (LLM) → embeddings → Pinecone + S3
         │
         ▼
  FastAPI / Lambda      POST /v1/chat
    1. classify query topic (dot-product against pre-embedded topic vectors)
    2. hybrid Pinecone search  α=0.7 dense + 0.3 BM25, top-20
    3. cross-encoder rerank → top-5
    4. gpt-4o-mini synthesis
         │
         ▼
  { answer, topic, sources: [{ title, channel, clips: [{ chapter, url, start_seconds }] }] }
```

---

## Stack

| Layer | Choice |
|---|---|
| Embeddings | OpenAI `text-embedding-3-small` (1536 dims) |
| Sparse encoding | BM25Encoder (pinecone-text) |
| Vector DB | Pinecone Serverless — dotproduct metric, hybrid search |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` (runs on Lambda CPU) |
| RAG synthesis | OpenAI `gpt-4o-mini` |
| Chapter gen + topic tags | Anthropic `claude-haiku-4-5` |
| State DB | Neon Serverless PostgreSQL |
| Compute | AWS Lambda + API Gateway |
| Queue | Amazon SQS + DLQ |
| Storage | Amazon S3 (structured transcript JSON) |
| Dashboard | Streamlit |

**Idle cost: $0.00/month.** All compute is pay-per-invocation.

---

## Project structure

```
core/
  gateway.py       model abstraction (OpenAI + Anthropic), token tracking, cost logging
  db.py            Neon connection + query helpers
  chunker.py       transcript → timed chunks
ingestion/
  fetch_lambda.py  YouTube playlist polling → DB
  worker_lambda.py transcript download → chapter gen → embed → upsert
retrieval/
  main.py          FastAPI app (chat endpoint, conversation history, streaming)
dashboard/
  app.py           Streamlit ops portal — channel management, cost attribution
schema.sql         PostgreSQL schema
seed.csv           starter set of channels
run_local.py       local dev runner (no AWS needed for ingestion)
```

---

## Running locally

**Prerequisites:** Python 3.12, [`uv`](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/yourusername/rabbithole
cd rabbithole
cp .env.example .env   # fill in your keys
uv sync
```

**Start the API:**
```bash
uv run uvicorn retrieval.main:app --port 8000 --env-file .env
```

**Query it:**
```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the relationship between consciousness and the body?"}'
```

**Ops dashboard:**
```bash
uv run streamlit run dashboard/app.py
```

**Ingest videos locally** (no AWS required — uses local disk instead of S3):
```bash
uv run python run_local.py --fetch              # discover new videos from registered channels
uv run python run_local.py --process-pending    # transcribe + embed all discovered videos
```

---

## Environment variables

| Variable | Description |
|---|---|
| `DATABASE_URL` | Neon PostgreSQL connection string |
| `OPENAI_API_KEY` | For embeddings + synthesis |
| `ANTHROPIC_API_KEY` | For chapter generation + topic classification |
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_INDEX_NAME` | Index name (dotproduct metric, 1536 dims) |
| `YOUTUBE_API_KEY` | YouTube Data API v3 |
| `AWS_ACCESS_KEY_ID` | For local dev only (Lambda uses IAM role in prod) |
| `AWS_SECRET_ACCESS_KEY` | For local dev only |
| `S3_BUCKET_NAME` | Transcript storage bucket |
| `S3_LOCAL_PATH` | Set to `./local_data` to skip S3 and use local disk |

---

## Topics

Topics are assigned **per video**, not per channel — so a channel that covers multiple subjects gets every video correctly tagged. Classification runs at ingest time via Claude Haiku against the video title and description.

Default topics: `consciousness`, `biohacking`, `spirituality`, `alternative_history`

Add your own in the DB:
```sql
INSERT INTO topics (name, description) VALUES ('finance', 'Personal finance, investing, and economics');
```

---

## Roadmap

- [x] YouTube transcript ingestion pipeline
- [x] Hybrid vector search (dense + sparse)
- [x] Cross-encoder re-ranking
- [x] Multi-turn conversation history
- [x] Streaming responses
- [ ] Blog / article ingestion
- [ ] GraphRAG — topic knowledge graph
- [ ] Terraform IaC
- [ ] Frontend (chat UI)
