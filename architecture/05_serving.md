# Stage 5: Serving

## Responsibility

`retrieval/main.py` — the user-facing API. A FastAPI application wrapped in a Mangum ASGI adapter for deployment on AWS Lambda + API Gateway.

---

## Architecture

```
Internet → API Gateway (HTTP API) → Lambda (retrieval) → FastAPI app
                                                              │
                                           ┌──────────────────┤
                                           │                  │
                                     Neon Postgres      Pinecone Index
                                     (topic names,      (vector search)
                                      video metadata)
                                           │
                                     OpenAI API + Anthropic API
                                     (via ModelGateway)
```

---

## FastAPI + Mangum

`Mangum` is an ASGI adapter that translates AWS Lambda event/context objects into standard ASGI `scope`/`receive`/`send` calls, allowing any ASGI framework (FastAPI, Starlette) to run on Lambda without modification.

```python
from fastapi import FastAPI
from mangum import Mangum

app = FastAPI(lifespan=lifespan)
handler = Mangum(app)   # This is the Lambda `handler` entry point
```

When Lambda invokes the function, `handler(event, context)` is called, which internally runs the FastAPI application through Mangum's ASGI bridge.

---

## Lifespan: Connection Init

FastAPI's `lifespan` context manager runs startup/shutdown code once per Lambda container lifecycle — not per request. This means DB connections and the Pinecone index handle are created once and reused across warm invocations:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — runs once when Lambda container initialises
    app.state.db      = psycopg2.connect(os.environ["DATABASE_URL"])
    app.state.index   = Pinecone(api_key=...).Index(os.environ["PINECONE_INDEX_NAME"])
    app.state.gateway = ModelGateway(db_conn=app.state.db)
    yield
    # Shutdown — runs when Lambda container is recycled
    app.state.db.close()
```

---

## Endpoints

### `POST /v1/chat`

**Request:**
```json
{ "query": "What did ancient Egyptians know about consciousness?" }
```

**Response:**
```json
{
  "answer": "According to researcher...",
  "citations": [
    {
      "title": "Graham Hancock — The Egypt Code",
      "channel": "Gaia",
      "chapter": "The Hall of Records",
      "url": "https://youtu.be/VIDEO_ID?t=743",
      "start_seconds": 743
    }
  ]
}
```

Full retrieval flow: intent classification → hybrid search → re-rank → synthesis.
See [04_retrieval.md](04_retrieval.md) for details.

### `GET /v1/health`

```json
{ "status": "ok", "timestamp": 1749600000.0 }
```

Used by API Gateway health checks and monitoring.

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| DB connection closed (Lambda idle timeout) | Detect `psycopg2.InterfaceError`, reconnect once, retry; raise `HTTP 503` if still failing |
| Pinecone 0 matches | Return `HTTP 200` with `{"answer": "No relevant content found.", "citations": []}` |
| Intent classification returns unknown topic | Log raw response, fall back to `topics[0]` |
| OpenAI / Anthropic API error | Log, return `HTTP 502` with `{"detail": "upstream model error"}` |
| Unhandled exception | Log full traceback to CloudWatch, return `HTTP 500` |

---

## API Gateway: HTTP API vs REST API

We use **HTTP API** (not REST API):

| | HTTP API | REST API |
|---|---|---|
| Cost | $1.00 per million requests | $3.50 per million requests |
| Latency | ~1ms overhead | ~6ms overhead |
| Features needed | Routes, Lambda proxy | ✓ |

HTTP API is the right choice for a simple Lambda proxy with no custom authorizers or caching.

---

## Local Development

Run the FastAPI app directly with `uvicorn` — no Lambda or API Gateway needed:

```bash
# Load env vars from .env
export $(cat .env | grep -v '^#' | xargs)

# Start dev server with hot reload
uvicorn retrieval.main:app --reload --port 8000
```

Test with curl or any HTTP client:
```bash
curl -X POST http://localhost:8000/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "what is the pineal gland?"}'
```

---

## Lambda Packaging Notes (Phase 2)

When deploying to Lambda:

1. `sentence-transformers` and `torch` add ~600MB to the package. Use a Lambda container image (up to 10GB) rather than a zip deployment package (50MB limit).
2. Set `SENTENCE_TRANSFORMERS_HOME=/tmp` so the cross-encoder model caches to Lambda's ephemeral storage.
3. Set Lambda memory to at least 1024MB for the cross-encoder inference.
4. Provisioned concurrency can eliminate cold starts if latency SLA is strict.
