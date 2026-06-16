# Plan: YouTube Transcript RAG Backend

## Context

Build a production-ready, fully serverless RAG pipeline over YouTube transcripts.
All existing Python and SQL files are discarded — writing everything fresh.
Phase 1: ingest from a static list of channels. Channel discovery is Phase 2.

**Stack:**
- Vector DB: Pinecone Serverless (hybrid search — dotproduct metric, dense + sparse BM25)
- Embeddings: OpenAI `text-embedding-3-small` ($0.02/M tokens)
- Sparse encoding: `pinecone-text` BM25Encoder (pre-trained default, no fitting needed)
- Re-ranking: `cross-encoder/ms-marco-MiniLM-L-6-v2` on Lambda (~80MB, loaded at init)
- Completions / RAG synthesis: OpenAI `gpt-4o-mini`
- Chapter generation + topic classification: Anthropic Claude Haiku 3.5
- State DB: Neon PostgreSQL (serverless, free tier, scales to $0)
- Infrastructure: Terraform (Phase 2 — Python only for now)

**Topic model:** topics are assigned **per video** (not per channel). A channel has an optional `default_topic` used only as a classification hint. Each video is tagged with 1–N topics by LLM at ingest time.

---

## Project Layout

```
youtube-topic-rag/
├── architecture/
│   ├── overview.md             # High-level system topology + data flow
│   ├── 01_fetching.md          # YouTube polling, quota management, SQS dispatch
│   ├── 02_chunking.md          # Transcript segmentation, chapter extraction, LLM fallback
│   ├── 03_embedding.md         # Dense (text-embedding-3-small) + sparse (BM25) vector generation
│   ├── 04_retrieval.md         # Hybrid search, re-ranking, RAG synthesis
│   ├── 05_serving.md           # FastAPI + Mangum, API Gateway, Lambda packaging
│   └── 06_dashboard.md         # Streamlit ops portal, cost attribution
├── core/
│   ├── gateway.py              # Model-agnostic AI wrapper (OpenAI + Anthropic)
│   ├── db.py                   # DB connection + shared query helpers
│   └── chunker.py              # Transcript segmentation + chapter logic
├── ingestion/
│   ├── fetch_lambda.py         # EventBridge → polls YouTube, queues new videos to SQS
│   └── worker_lambda.py        # SQS → processes transcripts, embeds, uploads to S3/Pinecone
├── retrieval/
│   └── main.py                 # FastAPI + Mangum RAG API
├── dashboard/
│   └── app.py                  # Streamlit ops portal
├── schema.sql                  # PostgreSQL schema
├── plan.md                     # Implementation plan (copy of this document)
├── run_local.py                # Dev script: run ingestion locally without AWS
├── pyproject.toml              # uv-managed Python project config + deps
└── .env.example
```

**Python environment:** managed with `uv` (modern replacement for pip + virtualenv + pip-tools).
No `requirements.txt` — deps live in `pyproject.toml`, locked in `uv.lock`.

---

## `schema.sql` (fresh)

```sql
CREATE TABLE topics (
    id     SERIAL PRIMARY KEY,
    name   VARCHAR(100) NOT NULL UNIQUE,
    description TEXT
);

CREATE TABLE channels (
    id                   VARCHAR(50) PRIMARY KEY,   -- YouTube Channel ID (UC...)
    name                 VARCHAR(255) NOT NULL,
    handle               VARCHAR(100),
    uploads_playlist_id  VARCHAR(50) NOT NULL,       -- UU... (derived from UC..., 1 quota/50 videos)
    default_topic_id     INT REFERENCES topics(id) ON DELETE SET NULL,  -- hint only, not exclusive
    videos_to_fetch      INT DEFAULT 10,
    is_active            BOOLEAN DEFAULT TRUE,
    last_checked_at      TIMESTAMP WITH TIME ZONE,
    created_at           TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE videos (
    id              VARCHAR(50) PRIMARY KEY,
    channel_id      VARCHAR(50) REFERENCES channels(id) ON DELETE CASCADE,
    title           VARCHAR(255) NOT NULL,
    description     TEXT,
    view_count      BIGINT DEFAULT 0,
    like_count      BIGINT DEFAULT 0,
    published_at    TIMESTAMP WITH TIME ZONE,
    topics          TEXT[] DEFAULT '{}',             -- ['consciousness','biohacking'] — multi-topic
    status          VARCHAR(20) DEFAULT 'discovered', -- discovered|processing|completed|failed
    error_message   TEXT,
    s3_path         VARCHAR(512),
    ingestion_tokens    INT DEFAULT 0,
    ingestion_cost      NUMERIC(10, 6) DEFAULT 0,
    processed_at    TIMESTAMP WITH TIME ZONE,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE rag_queries (
    id            BIGSERIAL PRIMARY KEY,
    user_query    TEXT NOT NULL,
    queried_topic VARCHAR(100),
    video_ids     TEXT[],                             -- which videos were cited
    retrieval_cost NUMERIC(10, 6) DEFAULT 0,
    queried_at    TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE model_telemetry (
    id               BIGSERIAL PRIMARY KEY,
    transaction_type VARCHAR(20) NOT NULL,            -- 'embedding' | 'completion'
    provider         VARCHAR(50) NOT NULL,            -- 'openai' | 'anthropic'
    model            VARCHAR(100) NOT NULL,
    input_tokens     INT DEFAULT 0,
    output_tokens    INT DEFAULT 0,
    latency_ms       INT NOT NULL,
    cost             NUMERIC(10, 7) DEFAULT 0,
    associated_id    VARCHAR(100),                    -- video_id or 'intent_router' etc.
    created_at       TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_channels_topic    ON channels(default_topic_id);
CREATE INDEX idx_videos_channel    ON videos(channel_id);
CREATE INDEX idx_videos_status     ON videos(status);
CREATE INDEX idx_videos_topics     ON videos USING GIN(topics);  -- array index
CREATE INDEX idx_telemetry_model   ON model_telemetry(model);

INSERT INTO topics (name, description) VALUES
  ('consciousness',      'Human consciousness, mind, awareness, and perception'),
  ('alternative_history','Alternative history, ancient Egypt, Atlantis, lost civilizations'),
  ('biohacking',         'Biohacking, longevity, nootropics, and performance optimization'),
  ('spirituality',       'Spirituality, metaphysics, meditation, and esoteric knowledge');
```

---

## S3 JSON Payload (exact user-specified structure)

Key: `transcripts/{topic_primary}/{channel_id}/{video_id}_structured.json`

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
      "text_content": "..."
    }
  ]
}
```

---

## `core/gateway.py`

### Pricing ledger
```python
MODEL_PRICING_LEDGER = {
    "openai": {
        "text-embedding-3-small": {"input": 0.00002 / 1000, "output": 0.0},
        "gpt-4o-mini":            {"input": 0.00015 / 1000, "output": 0.00060 / 1000},
    },
    "anthropic": {
        "claude-haiku-4-5-20251001": {"input": 0.00080 / 1000, "output": 0.00400 / 1000},
    }
}
```

### `ModelResponse` dataclass
Fields: `text_content`, `embedding_vector`, `input_tokens`, `output_tokens`, `latency_ms`, `cost`, `model`, `provider`

### `BaseAIProvider` (ABC)
Abstract methods: `generate_completion(prompt, system_prompt, model) -> ModelResponse`, `generate_embedding(text, model) -> ModelResponse`

### `OpenAIProvider(BaseAIProvider)`
- `generate_completion`: `client.chat.completions.create` → returns `ModelResponse` with tokens + text
- `generate_embedding`: `client.embeddings.create(model="text-embedding-3-small")` → returns `ModelResponse` with `embedding_vector`
- Does NOT compute cost or latency (that's `ModelGateway`'s job)

### `AnthropicProvider(BaseAIProvider)`
- `generate_completion`: `anthropic.Anthropic().messages.create` → `ModelResponse` with tokens + text
- `generate_embedding`: raises `NotImplementedError`

### `ModelGateway`
```python
class ModelGateway:
    def __init__(self, db_conn=None):
        self._providers = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider()
        }
        self.db_conn = db_conn  # optional — if None, telemetry logging is skipped

    def get_completion(self, prompt, system_prompt, model="gpt-4o-mini",
                       provider="openai", associated_id="") -> ModelResponse:
        # measures latency, calculates cost, logs to model_telemetry, returns response

    def get_embedding(self, text, model="text-embedding-3-small",
                      provider="openai", associated_id="") -> ModelResponse:
        # same pattern

    def _log_telemetry(self, tx_type, response, associated_id):
        # INSERT INTO model_telemetry — never raises, wraps in try/except
        # INSERT INTO model_telemetry
        # (transaction_type, provider, model, input_tokens, output_tokens,
        #  latency_ms, cost, associated_id)
        # VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
```

---

## `core/db.py`

```python
import os, psycopg2, psycopg2.extras

def get_connection():
    return psycopg2.connect(os.environ["DATABASE_URL"])

# Helper: fetch all active topic names
def get_topic_names(conn) -> list[str]:
    # SELECT name FROM topics ORDER BY name

# Helper: fetch channel's default_topic name
def get_channel_default_topic(conn, channel_id: str) -> str | None:
    # SELECT t.name FROM channels c JOIN topics t ON t.id = c.default_topic_id WHERE c.id = %s
```

---

## `core/chunker.py`

### `extract_chapters_from_description(description: str) -> list[dict]`
- Regex: `(?:\[)?(\d{1,2}:\d{2}(?::\d{2})?)(?:\])?\s+(.*)`
- Parse timestamp → total seconds (handle `MM:SS` and `HH:MM:SS`)
- Return `[{"start_seconds": int, "title": str}]` sorted ascending; `[]` on any exception

### `generate_chapters_with_llm(transcript_text: str, gateway) -> list[dict]`
- Provider: anthropic / claude-haiku-4-5-20251001
- System prompt: instructs to return JSON array `[{"title": str, "start_seconds": int}]` only
- Truncate input to first 12,000 chars; parse with `json.loads`; return `[]` on parse error
- `associated_id="chapter_gen"`

### `segment_into_paragraphs(srt: list, chapters: list, video_id: str) -> list[dict]`
- Existing proven logic: break on `.!?` punctuation, >2.5s silence gap, or 6-fragment ceiling
- Each chunk: `{chunk_id, associated_chapter, start_seconds, deep_link, text_content}`
- `associated_chapter`: find which chapter the `start_seconds` falls under (binary search)
- If chapters empty: `associated_chapter = "General"`

### `fixed_word_chunking(text: str, video_id: str, chunk_size=300, overlap=50) -> list[dict]`
- Fallback when SRT not available or yields < 3 chunks
- Stride = `chunk_size - overlap` (250 words)
- `associated_chapter = "General"`, `start_seconds = 0`, `deep_link = f"https://youtu.be/{video_id}"`

---

## `ingestion/fetch_lambda.py`

Triggered by EventBridge cron (`cron(0 */6 * * ? *)`).

**Flow:**
1. `SELECT id, name, uploads_playlist_id, default_topic_id, videos_to_fetch FROM channels WHERE is_active = TRUE`
2. For each channel, call YouTube Data API:
   ```python
   youtube.playlistItems().list(
       part="contentDetails",
       playlistId=uploads_playlist_id,  # UU... — 1 quota point per 50 videos
       maxResults=channel["videos_to_fetch"]
   ).execute()
   ```
3. `SELECT id FROM videos WHERE id = ANY(%s)` → `new_ids = api_set - known_set`
4. Fetch metadata for new videos (title, description, viewCount, likeCount, publishedAt):
   `youtube.videos().list(part="snippet,statistics", id=",".join(new_ids))`
5. `INSERT INTO videos (id, channel_id, title, description, view_count, like_count, published_at, status) VALUES ... ON CONFLICT (id) DO NOTHING`
6. `SELECT name FROM topics WHERE id = (SELECT default_topic_id FROM channels WHERE id = %s)`
7. SQS `send_message_batch` in chunks of 10:
   ```json
   {"video_id": "...", "channel_id": "..."}
   ```
   (topic classification happens in worker, not here)
8. `UPDATE channels SET last_checked_at = NOW() WHERE id = %s`

**Error handling per channel:**
- YouTube 404 on playlist → `UPDATE channels SET is_active=FALSE WHERE id=%s`
- Quota 403 → log, stop processing remaining channels, return partial result
- DB errors → fatal, re-raise

---

## `ingestion/worker_lambda.py`

Triggered by SQS. Uses `batchItemFailures` partial failure pattern.

**`process_video(video_id, channel_id, db_conn, s3_client, pinecone_index, gateway)`:**

1. Guard: skip if `status = 'completed'` (idempotency)
2. `UPDATE videos SET status = 'processing' WHERE id = %s`
3. Wrap steps 4–12 in try/except → on failure: `UPDATE videos SET status='failed', error_message=%s WHERE id=%s`, re-raise
4. Fetch transcript: `YouTubeTranscriptApi.get_transcript(video_id)`
   - On `TranscriptsDisabled`/`NoTranscriptFound`: set `status='failed'`, return without re-raising (no retry benefit)
5. Get video title + description from DB: `SELECT title, description FROM videos WHERE id = %s`
6. **Chapter strategy:**
   - Try `extract_chapters_from_description(description)` from `core/chunker.py`
   - If `len(chapters) < 3`: call `generate_chapters_with_llm(full_transcript_text, gateway)`
   - If still `< 3`: use `fixed_word_chunking` fallback
7. `chunks = segment_into_paragraphs(srt, chapters, video_id)` (or fixed chunks)
8. **Topic classification:**
   - `default_hint = get_channel_default_topic(db_conn, channel_id)` from `core/db.py`
   - `available_topics = get_topic_names(db_conn)`
   - Call Claude Haiku: classify video title + description into 1–N topics
   - System prompt: `"This channel primarily covers {default_hint}. Classify this video into ALL relevant topics from: {available_topics}. Return only a JSON array, e.g. [\"consciousness\",\"spirituality\"]. Nothing else."`
   - Parse response; validate items are in `available_topics`; fallback to `[default_hint]`
   - `UPDATE videos SET topics = %s WHERE id = %s`
9. Embed each chunk: `gateway.get_embedding(chunk["text_content"], associated_id=video_id)`; accumulate `total_tokens`, `total_cost`
10. **Pinecone upsert** (hybrid: dense + sparse) in batches of 100:
    ```python
    from pinecone_text.sparse import BM25Encoder
    bm25 = BM25Encoder.default()   # pre-trained on MS MARCO, no corpus fitting needed

    vectors = [{
      "id": f"{video_id}_{chunk['chunk_id']}",
      "values": dense_embedding_vector,
      "sparse_values": bm25.encode_documents([chunk["text_content"]])[0],
      "metadata": {
        "video_id": video_id,
        "channel_id": channel_id,
        "topics": video_topics,           # list — filter: {"topics": {"$in": ["consciousness"]}}
        "chapter": chunk["associated_chapter"],
        "start_seconds": chunk["start_seconds"],
        "deep_link": chunk["deep_link"],
        "text_content": chunk["text_content"][:1000]
      }
    } for chunk, dense_embedding_vector in zip(chunks, embeddings)]
    index.upsert(vectors=vectors)
    ```
    **Note:** Pinecone index must use `dotproduct` metric (required for hybrid search).
11. Build S3 payload (exact user schema) and upload:
    Key: `transcripts/{video_topics[0]}/{channel_id}/{video_id}_structured.json`
12. `UPDATE videos SET status='completed', s3_path=%s, ingestion_tokens=%s, ingestion_cost=%s, processed_at=NOW() WHERE id=%s`

**`lambda_handler(event, context)`:**
- Init `db_conn`, `s3_client`, `Pinecone().Index(...)`, `ModelGateway(db_conn=db_conn)`
- Iterate `event["Records"]`, call `process_video` per record
- Return `{"batchItemFailures": [{"itemIdentifier": msg_id}]}` for failed records

---

## `retrieval/main.py`

```python
app = FastAPI(lifespan=lifespan)
handler = Mangum(app)   # Lambda entrypoint
```

**Lifespan:** opens `psycopg2` connection, Pinecone index, `ModelGateway`.

**`POST /v1/chat`** — `{"query": str}` → `{"answer": str, "citations": [...]}`

Step A — Intent classification:
```python
topics = get_topic_names(db_conn)
# cheap gpt-4o-mini call: classify query into one topic from the list
# fallback to topics[0] if result not in list
```

Step B — Hybrid vector search (top-20, then re-rank to top-5):
```python
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import CrossEncoder

# Module-level singletons (stay warm between Lambda invocations)
# Set SENTENCE_TRANSFORMERS_HOME=/tmp for Lambda disk cache
_bm25 = BM25Encoder.default()
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Hybrid query: alpha=0.7 weights dense higher; adjust per use case
results = index.query(
    vector=dense_query_embedding,
    sparse_vector=_bm25.encode_queries([query])[0],
    alpha=0.7,
    top_k=20,                                        # fetch 20, re-rank to 5
    filter={"topics": {"$in": [predicted_topic]}},
    include_metadata=True
)
# if 0 matches: return {"answer": "No relevant content found.", "citations": []}

# Re-rank: cross-encoder scores query vs each retrieved chunk text
pairs = [(query, m.metadata["text_content"]) for m in results.matches]
scores = _reranker.predict(pairs)
top5_matches = [m for m, _ in sorted(
    zip(results.matches, scores), key=lambda x: x[1], reverse=True
)[:5]]
```

Step C — Synthesis (using `top5_matches`):
```python
# Fetch video titles + channel names in one DB query: SELECT v.id, v.title, c.name FROM videos v JOIN channels c ON ...
# Build context from match metadata (text_content + chapter + deep_link)
# gpt-4o-mini completion with sources in system prompt
# Log to rag_queries: INSERT INTO rag_queries (user_query, queried_topic, video_ids, retrieval_cost)
```

Citations format:
```python
{"title": str, "channel": str, "chapter": str, "url": deep_link, "start_seconds": int}
```

**`GET /v1/health`** → `{"status": "ok"}`

---

## `dashboard/app.py`

**Connections:** `@st.cache_resource` for psycopg2 + boto3 SQS client.

**Metrics row (3 cols):**
- `SELECT COUNT(*) FROM videos WHERE status = 'completed'`
- SQS `ApproximateNumberOfMessages` for main queue and DLQ (try/except → "N/A" if no IAM)

**Sidebar — Add Channel:**
```python
# Input: channel_id (UC...), name, default_topic, videos_to_fetch
# Derive uploads_playlist_id: "UU" + channel_id[2:]
# INSERT INTO channels (id, name, uploads_playlist_id, default_topic_id, videos_to_fetch)
# SELECT %s,%s,%s, t.id, %s FROM topics t WHERE t.name = %s
# ON CONFLICT (id) DO NOTHING
```

**Value Attribution Table:**
```sql
SELECT
    c.id, c.name,
    t.name                                AS default_topic,
    c.is_active,
    COALESCE(SUM(v.ingestion_cost), 0)    AS total_cost,
    COUNT(DISTINCT q.id)                  AS search_count,
    CASE WHEN COUNT(DISTINCT q.id) = 0 THEN NULL
         ELSE SUM(v.ingestion_cost) / COUNT(DISTINCT q.id)
    END                                   AS cost_per_search
FROM channels c
LEFT JOIN topics t ON t.id = c.default_topic_id
LEFT JOIN videos v ON v.channel_id = c.id AND v.status = 'completed'
LEFT JOIN rag_queries q ON c.id = ANY(q.video_ids::varchar[])
GROUP BY c.id, c.name, t.name, c.is_active
ORDER BY cost_per_search DESC NULLS LAST
```

Row coloring: green = high value (`search_count > 10 AND cost_per_search < 0.01`), red = low value (`search_count < 2 AND total_cost > 1.0`).

**Per-row active toggle:**
```python
is_active = st.toggle("Active", value=row.is_active, key=row.channel_id)
if is_active != row.is_active:
    UPDATE channels SET is_active = %s WHERE id = %s
    st.rerun()
```

---

## `pyproject.toml` (uv-managed)

```toml
[project]
name = "youtube-topic-rag"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.111.0",
    "mangum>=0.17.0",
    "uvicorn>=0.29.0",
    "psycopg2-binary>=2.9.9",
    "openai>=1.35.0",
    "anthropic>=0.28.0",
    "pinecone-client>=4.1.0",
    "pinecone-text>=0.9.0",           # BM25Encoder — hybrid search sparse vectors
    "sentence-transformers>=3.1.0",   # cross-encoder re-ranker
    "youtube-transcript-api>=0.6.2",
    "google-api-python-client>=2.131.0",
    "boto3>=1.34.131",
    "pydantic>=2.7.1",
    "streamlit>=1.35.0",
]
```

---

## `.env.example`

```
DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=youtube-rag-index
YOUTUBE_API_KEY=AIza...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
SQS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/ACCOUNT/youtube-video-process-queue
SQS_DLQ_URL=https://sqs.us-east-1.amazonaws.com/ACCOUNT/youtube-video-process-dlq
S3_BUCKET_NAME=startup-thematic-transcripts-production

# Local dev only — if set, saves transcripts to disk instead of S3
S3_LOCAL_PATH=./local_data

# cross-encoder model cache dir (set to /tmp for Lambda)
SENTENCE_TRANSFORMERS_HOME=./local_data/models
```

---

## Architecture Document Outlines

**`architecture/overview.md`**
- System purpose + topic domains (consciousness, alternative_history, biohacking, spirituality)
- Full topology diagram (EventBridge → fetch Lambda → SQS → worker Lambda → S3 + Pinecone → API Gateway → retrieval Lambda)
- Data flow: ingest path vs query path
- Stack decisions table (vector DB, embedding, LLM providers, state DB)
- Cost model: $0 idle, per-unit spend breakdown

**`architecture/01_fetching.md`**
- YouTube quota model: 1 API unit = 50 videos via uploads playlist (`UC→UU` trick)
- Channel config: `videos_to_fetch`, `default_topic_id`, `is_active` toggle
- State machine: `discovered → processing → completed | failed`
- SQS message structure + DLQ policy (3 retries, 7-day retention)
- EventBridge cron schedule

**`architecture/02_chunking.md`**
- Chapter extraction: regex on video description (`[HH:]MM:SS Title` pattern)
- LLM chapter generation fallback: Claude Haiku 3.5 when `< 3` native chapters found
- Paragraph segmentation rules: punctuation break, 2.5s silence gap, 6-fragment ceiling
- Word-chunk fallback: 300 words / 50-word overlap for videos without SRT
- `associated_chapter` assignment: binary search by `start_seconds`
- S3 JSON payload structure

**`architecture/03_embedding.md`**
- Dense vectors: OpenAI `text-embedding-3-small`, 1536 dims, $0.02/M tokens
- Sparse vectors: `BM25Encoder.default()` from `pinecone-text` (pre-trained on MS MARCO, no fitting)
- Pinecone upsert: `dotproduct` metric index (required for hybrid), batches of 100
- Metadata stored per vector: `topics[]`, `chapter`, `start_seconds`, `deep_link`, `text_content[:1000]`
- Topic classification: Claude Haiku 3.5 classifies each video into 1–N topics using title + description
- Cost telemetry: every model call logged to `model_telemetry` table

**`architecture/04_retrieval.md`**
- Hybrid query: `alpha=0.7` (dense-heavy blend), `top_k=20`
- Pinecone metadata filter: `{"topics": {"$in": ["consciousness"]}}`
- Re-ranking: `cross-encoder/ms-marco-MiniLM-L-6-v2` scores top-20 → selects top-5
- Lambda cold start: model cached at `/tmp` via `SENTENCE_TRANSFORMERS_HOME`
- Intent classification: gpt-4o-mini classifies query into topic before vector search
- Citation format: `{title, channel, chapter, url, start_seconds}`

**`architecture/05_serving.md`**
- FastAPI app wrapped in Mangum ASGI adapter for Lambda
- Endpoints: `POST /v1/chat`, `GET /v1/health`
- Lifespan: DB connection + Pinecone index + ModelGateway initialized on Lambda container start
- Local dev: `uvicorn retrieval.main:app --reload`
- API Gateway: HTTP API (not REST API) — lower cost, lower latency

**`architecture/06_dashboard.md`**
- Streamlit app connecting to Neon PostgreSQL via psycopg2
- Live metrics: indexed videos, SQS queue depth, DLQ count
- Channel registration: derive `uploads_playlist_id` from channel ID
- Value attribution table: cost vs search volume per channel, red/green highlighting
- Deactivate toggle: single click sets `is_active = FALSE`

---

## Pinecone Index Setup (one-time prerequisite)

Create via Pinecone console before first ingest run:
- Dimension: `1536` (text-embedding-3-small)
- Metric: **`dotproduct`** (required for hybrid dense+sparse search — NOT cosine)
- Type: Serverless / AWS us-east-1

---

## Local Testing (no AWS infra needed)

The only AWS runtime dependency is S3. Everything else hits external free-tier APIs directly.

**Env var switch for S3:**
```
S3_LOCAL_PATH=./local_data   # if set, saves JSON to disk instead of uploading to S3
                              # if unset, uses boto3 S3 client (AWS)
```

In `worker_lambda.py`, the upload step checks:
```python
local_path = os.environ.get("S3_LOCAL_PATH")
if local_path:
    os.makedirs(os.path.join(local_path, "transcripts", ...), exist_ok=True)
    with open(os.path.join(local_path, key), "w") as f:
        json.dump(payload, f)
else:
    s3_client.put_object(Bucket=..., Key=key, Body=json.dumps(payload))
```

**`run_local.py`** (root-level dev script):
```python
# Test the full ingestion pipeline on a specific video
from ingestion.worker_lambda import process_video
from core.gateway import ModelGateway
from core.db import get_connection
from pinecone import Pinecone
import boto3, os

db = get_connection()
gateway = ModelGateway(db_conn=db)
index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(os.environ["PINECONE_INDEX_NAME"])

process_video(
    video_id="YOUR_VIDEO_ID",
    channel_id="UCxxxxxx",
    db_conn=db,
    s3_client=None,          # None = local mode (S3_LOCAL_PATH must be set)
    pinecone_index=index,
    gateway=gateway
)
```

**Running each service locally:**
```bash
# Retrieval API
uvicorn retrieval.main:app --reload --port 8000

# Dashboard
streamlit run dashboard/app.py

# Fetch new videos from YouTube (runs fetch_lambda handler directly)
python -c "from ingestion.fetch_lambda import lambda_handler; lambda_handler({}, None)"
```

---

## Implementation Order

**Phase 0 — Environment + Docs (before any code)**
1. Install `uv` (if not present): `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Init project: `uv init youtube-topic-rag --python 3.12` (or `uv python pin 3.12` in existing dir)
3. Add all deps: `uv add fastapi mangum uvicorn psycopg2-binary openai anthropic ...`
4. Create `architecture/` folder + all 6 architecture markdown files + `plan.md`
5. Create `schema.sql` + `.env.example`

**Phase 1 — Core**
6. `core/gateway.py`
7. `core/db.py` + `core/chunker.py`

**Phase 2 — Ingestion**
8. `ingestion/fetch_lambda.py`
9. `ingestion/worker_lambda.py` + `run_local.py`

**Phase 3 — Retrieval**
10. `retrieval/main.py`

**Phase 4 — Dashboard**
11. `dashboard/app.py`

---

## Verification

1. `gateway.py`: `ModelGateway().get_embedding("test")` → 1536-dim vector; `get_completion("hello", provider="anthropic", model="claude-haiku-4-5-20251001")` → text response
2. `chunker.py`: `extract_chapters_from_description("0:00 Intro\n5:30 Deep Dive")` → 2 chapters; `fixed_word_chunking(400_word_text, "vid1")` → 2 chunks
3. `worker_lambda.py`: call `process_video` with a real public video ID → DB `status='completed'`, S3 file exists, Pinecone vectors present
4. `main.py`: `uvicorn retrieval.main:app --reload` → `POST /v1/chat {"query": "what is consciousness"}` → structured JSON with citations
5. `dashboard/app.py`: `streamlit run dashboard/app.py` → live metrics load; add channel via sidebar → DB row created
