# Entity Metadata for RAG Retrieval

**Date:** 2026-06-20
**Status:** Approved

## Goal

Improve retrieval quality by extracting key concepts and people per chunk at ingestion time, storing them as vector metadata, and using entity overlap at query time to boost ranking — getting ~60% of entity-retrieval benefit at near-zero cost and zero new infrastructure.

---

## Decisions

| Question | Decision |
| --- | --- |
| Entity types | Concepts/frameworks (primary), people/speakers (secondary); compounds fold into concepts |
| Extraction granularity | Per-chunk (one Haiku call each) — maximises precision |
| Speaker extraction | Extended per-video Haiku call (folded into existing `classify_topics` call) |
| Query-time strategy | Soft boost via entity overlap score combined with cross-encoder; hard filter deferred |
| Backfill | None — starting fresh (Chroma for dev, clean Pinecone for go-live) |

---

## Section 1: Ingestion Changes

Both workers (`ingestion/worker_lambda.py` and `ingestion/article_worker.py`) get the same two additions.

### 1a — Speaker extraction (per-video, zero extra cost)

Extend the existing `classify_topics` Haiku call in `core/topics.py` to also extract the guest/speaker name from the video title.

- **Return type change:** `list[str]` → `VideoMeta(topics: list[str], speaker: str | None)`, defined in `core/topics.py`
- Speaker is stored as a scalar field on every chunk from that video: `"speaker": "Graham Hancock"`
- `worker_lambda.py` unpacks the new dataclass; `article_worker.py` calls `classify_topics` the same way but ignores `speaker` (articles have no speaker concept)

### 1b — Entity extraction (per-chunk, one new Haiku call per chunk)

New module `core/entities.py` with a single function:

```python
def extract_chunk_entities(text: str, gateway: ModelGateway) -> list[str]:
    ...
```

- Calls Haiku with a compact, domain-agnostic prompt
- Asks for 5–8 key concepts and people *mentioned in this chunk*
- Returns a flat list of lowercase strings: `["non-duality", "consciousness", "rupert spira"]`
- On Haiku JSON parse failure: returns `[]` (silent fallback, never blocks ingestion)
- Stored as `"entities": [...]` in chunk metadata

**Storage compatibility:**

- Chroma: existing `_safe_chroma_meta` already JSON-serialises lists — no changes needed
- Pinecone: accepts list metadata natively, supports `$in` filter

**Cost estimate:** ~20 chunks × ~300 tokens input × $0.0008/1K ≈ **$0.005 per video** — negligible.

---

## Section 2: Retrieval Changes (`retrieval/main.py`)

Entities slot between vector retrieval and cross-encoder rerank. The response shape and API contract are unchanged.

### 2a — Entity overlap scoring

After top-20 chunks are returned from the vector store, compute a lightweight overlap score per chunk:

- Lowercase the query string
- Check which of the chunk's `entities` appear as substrings in the query
- `entity_score = matched_count / len(entities)` if entities is non-empty, else `0.0`
- Zero latency — pure Python string ops, no API calls

### 2b — Combined scoring in `_rerank`

```python
final_score = cross_encoder_score + ENTITY_WEIGHT * entity_overlap_score
```

- `ENTITY_WEIGHT = 0.3` (default; tunable via `ENTITY_WEIGHT` env var)
- Entity matches can lift a chunk but cannot override a strongly semantically-relevant one
- Hard filter (Pinecone `$in` on `entities`) is intentionally deferred — adds query-time Haiku latency; add once soft boost is validated

### 2c — Speaker bonus

If the query string contains the chunk's `speaker` field value (case-insensitive substring match), apply a flat `+0.15` bonus on top of the entity score. Handles "what did Graham Hancock say about X" queries without requiring the speaker to appear in every chunk's entity list.

---

## Section 3: Testing

### Unit tests

| File | Coverage added |
| --- | --- |
| `tests/unit/test_topics.py` | Updated `classify_topics` return type (`VideoMeta`); mock Haiku response |
| `tests/unit/test_entities.py` (new) | `extract_chunk_entities`: mock Haiku response, flat lowercase list, graceful fallback on bad JSON |
| `tests/unit/test_retrieval_helpers.py` | Entity overlap scoring function; updated `_rerank` combining cross-encoder + entity score |

### Integration tests

| File | Change |
| --- | --- |
| `tests/integration/test_worker_process_video.py` | Assert upserted chunk metadata contains `"entities"` (non-empty list) and `"speaker"` (str or None) |
| `tests/integration/test_chat_*` | No changes — response shape unchanged |

### Quality (manual)

After ingesting 2–3 test videos into local Chroma, run entity-centric queries (e.g. "what did Rupert Spira say about awareness") and verify improved chunk relevance vs. baseline. No automated assertion — sanity check before go-live.

---

## Files Affected

| File | Change |
| --- | --- |
| `core/topics.py` | Extend Haiku prompt + return `VideoMeta(topics, speaker)` |
| `core/entities.py` | New — `extract_chunk_entities` |
| `ingestion/worker_lambda.py` | Unpack `VideoMeta`; call `extract_chunk_entities` per chunk; add `entities` + `speaker` to metadata |
| `ingestion/article_worker.py` | Call `extract_chunk_entities` per chunk; add `entities` to metadata (no speaker) |
| `retrieval/main.py` | Entity overlap scoring; updated `_rerank` with combined score + speaker bonus |
| `tests/unit/test_topics.py` | Updated return type tests |
| `tests/unit/test_entities.py` | New |
| `tests/unit/test_retrieval_helpers.py` | Entity scoring tests |
| `tests/integration/test_worker_process_video.py` | Metadata assertions |

---

## Out of Scope

- Hard filter at query time (defer until soft boost is validated)
- Backfill of existing Pinecone vectors
- Entity deduplication / normalisation (e.g. "DMT" vs "dimethyltryptamine") — accept fuzziness for now
- Per-article speaker extraction (articles have no speaker concept)
