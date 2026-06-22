# Entity Metadata Ingestion + Retrieval Boost — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract key concepts and people per chunk at ingestion time via Haiku, store as vector metadata, and apply an entity-overlap soft boost during reranking to improve retrieval quality.

**Architecture:** Two new ingestion steps run per-video/article: `classify_video_meta` replaces `classify_topics` in the YouTube worker and returns host + guest list alongside topics from a single Haiku call; `extract_chunk_entities` makes one Haiku call per chunk to extract 5–8 key concepts and people. At query time, `_rerank` combines cross-encoder scores with a lightweight entity overlap score and a people-name bonus — zero added latency, no new infrastructure.

**Tech Stack:** Python 3.12, Claude Haiku (`claude-haiku-4-5-20251001`) via existing `ModelGateway`, pytest, existing `PineconeStore` / `ChromaStore`.

## Global Constraints

- Never modify `classify_topics` signature or return type — it is still used by `article_worker.py` unchanged.
- `extract_chunk_entities` must return `[]` on any failure — never raise.
- `classify_video_meta` must return a valid `VideoMeta` on any failure — never raise.
- Run tests with: `uv run pytest tests/unit -v` and `uv run pytest tests/integration -v`
- Haiku model ID: `claude-haiku-4-5-20251001` — copy verbatim, never abbreviate.
- `ENTITY_WEIGHT` defaults to `0.3`; `_people_bonus` caps at `0.15` regardless of how many people fields match.

---

## File Map

| File | Action | Responsibility |
| --- | --- | --- |
| `core/topics.py` | Modify | Add `VideoMeta` dataclass + `classify_video_meta` function |
| `core/entities.py` | Create | `extract_chunk_entities` — per-chunk Haiku entity extraction |
| `ingestion/worker_lambda.py` | Modify | Switch to `classify_video_meta`; call `extract_chunk_entities` per chunk; add `entities`, `host`, `guests` to metadata |
| `ingestion/article_worker.py` | Modify | Call `extract_chunk_entities` per chunk; add `entities`, `author` to metadata |
| `retrieval/main.py` | Modify | Add `_parse_list_meta`, `_entity_overlap_score`, `_people_bonus`; update `_rerank` |
| `tests/unit/test_topics.py` | Modify | Add tests for `classify_video_meta` and `VideoMeta` |
| `tests/unit/test_entities.py` | Create | Full unit coverage for `extract_chunk_entities` |
| `tests/unit/test_retrieval_helpers.py` | Modify | Add tests for `_entity_overlap_score`, `_people_bonus`, updated `_rerank` |
| `tests/integration/test_worker_process_video.py` | Modify | Update gateway mock; assert `entities`, `host`, `guests` in upserted metadata |
| `tests/integration/test_article_worker.py` | Modify | Update gateway mock; assert `entities`, `author` in upserted metadata |

---

### Task 1: `VideoMeta` dataclass + `classify_video_meta` in `core/topics.py`

**Files:**
- Modify: `core/topics.py`
- Modify: `tests/unit/test_topics.py`

**Interfaces:**
- Produces:
  - `VideoMeta` dataclass with fields `topics: list[str]`, `host: str | None`, `guests: list[str]`
  - `classify_video_meta(title, channel_name, text_excerpt, available_topics, default_hint, gateway) -> VideoMeta`
- `classify_topics` remains unchanged.

---

- [ ] **Step 1: Write failing tests for `VideoMeta` and `classify_video_meta`**

Add to `tests/unit/test_topics.py` (below existing imports):

```python
from core.topics import VideoMeta, classify_video_meta


def _make_video_gateway(response_dict: dict):
    gw = MagicMock()
    gw.get_completion.return_value = MagicMock(
        text_content=json.dumps(response_dict), cost=0.001
    )
    return gw


def test_video_meta_is_dataclass():
    vm = VideoMeta(topics=["consciousness"], host="Joe Rogan", guests=["Graham Hancock"])
    assert vm.topics == ["consciousness"]
    assert vm.host == "Joe Rogan"
    assert vm.guests == ["Graham Hancock"]


def test_classify_video_meta_happy_path():
    gw = _make_video_gateway({
        "topics": ["consciousness"], "host": "Joe Rogan", "guests": ["Graham Hancock"]
    })
    result = classify_video_meta(
        "Ep 1 | Graham Hancock", "Joe Rogan Experience",
        "some text", AVAILABLE, "consciousness", gw,
    )
    assert isinstance(result, VideoMeta)
    assert result.topics == ["consciousness"]
    assert result.host == "Joe Rogan"
    assert result.guests == ["Graham Hancock"]


def test_classify_video_meta_multiple_guests():
    gw = _make_video_gateway({
        "topics": ["consciousness"], "host": "Host",
        "guests": ["Guest A", "Guest B"],
    })
    result = classify_video_meta("Ep | A & B", "Show", "text", AVAILABLE, "consciousness", gw)
    assert result.guests == ["Guest A", "Guest B"]


def test_classify_video_meta_solo_episode():
    gw = _make_video_gateway({
        "topics": ["biohacking"], "host": "Andrew Huberman", "guests": []
    })
    result = classify_video_meta(
        "How to Sleep Better", "Huberman Lab", "text", AVAILABLE, "biohacking", gw
    )
    assert result.host == "Andrew Huberman"
    assert result.guests == []


def test_classify_video_meta_null_host():
    gw = _make_video_gateway({"topics": ["consciousness"], "host": None, "guests": ["Guest"]})
    result = classify_video_meta("T", "C", "t", AVAILABLE, "consciousness", gw)
    assert result.host is None
    assert result.guests == ["Guest"]


def test_classify_video_meta_bad_json_fallback():
    gw = MagicMock()
    gw.get_completion.return_value = MagicMock(text_content="not json", cost=0.001)
    result = classify_video_meta("Title", "Channel", "text", AVAILABLE, "consciousness", gw)
    assert isinstance(result, VideoMeta)
    assert result.topics == ["consciousness"]
    assert result.host is None
    assert result.guests == []


def test_classify_video_meta_invalid_topics_fall_back_to_hint():
    gw = _make_video_gateway({"topics": ["bogus"], "host": "H", "guests": []})
    result = classify_video_meta("T", "C", "t", AVAILABLE, "biohacking", gw)
    assert result.topics == ["biohacking"]


def test_classify_video_meta_strips_code_fence():
    gw = MagicMock()
    gw.get_completion.return_value = MagicMock(
        text_content='```json\n{"topics": ["biohacking"], "host": "H", "guests": []}\n```',
        cost=0.001,
    )
    result = classify_video_meta("T", "C", "t", AVAILABLE, "biohacking", gw)
    assert result.topics == ["biohacking"]


def test_classify_video_meta_api_exception_fallback():
    gw = MagicMock()
    gw.get_completion.side_effect = RuntimeError("API down")
    result = classify_video_meta("T", "C", "t", AVAILABLE, "consciousness", gw)
    assert result.topics == ["consciousness"]
    assert result.host is None
    assert result.guests == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_topics.py -k "video_meta" -v
```

Expected: `ImportError` or `AttributeError` — `VideoMeta` and `classify_video_meta` do not exist yet.

- [ ] **Step 3: Implement `VideoMeta` and `classify_video_meta` in `core/topics.py`**

Add after the existing imports (add `from dataclasses import dataclass, field`):

```python
from dataclasses import dataclass, field
```

Add after the existing `classify_topics` function:

```python
@dataclass
class VideoMeta:
    topics: list[str]
    host: str | None
    guests: list[str] = field(default_factory=list)


def classify_video_meta(
    title: str,
    channel_name: str,
    text_excerpt: str,
    available_topics: list[str],
    default_hint: str | None,
    gateway: ModelGateway,
) -> VideoMeta:
    """Classify a YouTube video into topics and extract host + guests via a single Haiku call.

    Returns a VideoMeta with fallback topics on any failure — never raises.
    """
    hint = default_hint or (available_topics[0] if available_topics else "")
    fallback = VideoMeta(
        topics=[hint] if hint else available_topics[:1],
        host=None,
        guests=[],
    )
    system_prompt = (
        "Classify this YouTube video and extract people. "
        "Return ONLY a valid JSON object with exactly three keys: "
        f'"topics" (array of strings from this list only: {available_topics}; '
        f"include ALL relevant ones; hint: {hint}), "
        '"host" (string or null — the channel\'s regular host inferred from the channel name), '
        '"guests" (array of strings — guest names from the title; empty array if none). '
        'Example: {"topics": ["consciousness"], "host": "Joe Rogan", "guests": ["Graham Hancock"]}. '
        "Output nothing else."
    )
    try:
        resp = gateway.get_completion(
            prompt=(
                f"Title: {title}\n"
                f"Channel: {channel_name}\n"
                f"Excerpt (first 500 chars): {text_excerpt[:500]}"
            ),
            system_prompt=system_prompt,
            model="claude-haiku-4-5-20251001",
            provider="anthropic",
            associated_id="video_meta_classify",
        )
        raw = resp.text_content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[^\n]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())
        data = json.loads(raw)
        validated = [t for t in (data.get("topics") or []) if t in available_topics]
        topics = validated if validated else ([hint] if hint else available_topics[:1])
        host = data.get("host") or None
        guests = [
            g for g in (data.get("guests") or [])
            if isinstance(g, str) and g.strip()
        ]
        return VideoMeta(topics=topics, host=host, guests=guests)
    except Exception:
        return fallback
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_topics.py -v
```

Expected: all tests pass (existing tests for `classify_topics` unchanged, new tests pass).

- [ ] **Step 5: Commit**

```bash
git add core/topics.py tests/unit/test_topics.py
git commit -m "feat: add VideoMeta dataclass and classify_video_meta to core/topics"
```

---

### Task 2: `core/entities.py` — per-chunk entity extraction

**Files:**
- Create: `core/entities.py`
- Create: `tests/unit/test_entities.py`

**Interfaces:**
- Consumes: `ModelGateway` from `core.gateway`
- Produces: `extract_chunk_entities(text: str, gateway: ModelGateway) -> list[str]`

---

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_entities.py`:

```python
"""Unit tests for core/entities.py — per-chunk entity extraction."""
from unittest.mock import MagicMock

import pytest

from core.entities import extract_chunk_entities


def _make_gateway(response_text: str):
    gw = MagicMock()
    gw.get_completion.return_value = MagicMock(text_content=response_text, cost=0.0001)
    return gw


def test_happy_path_returns_lowercase_list():
    gw = _make_gateway('["Non-Duality", "Consciousness", "Rupert Spira"]')
    result = extract_chunk_entities("some text about consciousness", gw)
    assert result == ["non-duality", "consciousness", "rupert spira"]


def test_bad_json_returns_empty_list():
    gw = _make_gateway("not valid json at all")
    result = extract_chunk_entities("text", gw)
    assert result == []


def test_api_error_returns_empty_list():
    gw = MagicMock()
    gw.get_completion.side_effect = RuntimeError("API down")
    result = extract_chunk_entities("text", gw)
    assert result == []


def test_filters_non_string_items():
    gw = _make_gateway('["concept", 42, null, "person"]')
    result = extract_chunk_entities("text", gw)
    assert result == ["concept", "person"]


def test_strips_code_fence():
    gw = _make_gateway('```json\n["concept"]\n```')
    result = extract_chunk_entities("text", gw)
    assert result == ["concept"]


def test_empty_array_returns_empty_list():
    gw = _make_gateway("[]")
    result = extract_chunk_entities("text", gw)
    assert result == []


def test_text_truncated_to_1000_chars():
    gw = _make_gateway('["concept"]')
    long_text = "a" * 2000
    extract_chunk_entities(long_text, gw)
    call_kwargs = gw.get_completion.call_args.kwargs
    assert len(call_kwargs["prompt"]) <= 1000


def test_whitespace_only_strings_filtered():
    gw = _make_gateway('["concept", "   ", "person"]')
    result = extract_chunk_entities("text", gw)
    assert result == ["concept", "person"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_entities.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.entities'`

- [ ] **Step 3: Implement `core/entities.py`**

Create `core/entities.py`:

```python
"""core/entities.py — Per-chunk entity extraction via Claude Haiku."""
import json
import re

from core.gateway import ModelGateway

_SYSTEM_PROMPT = (
    "Extract the 5 to 8 most important concepts, frameworks, and people explicitly "
    "mentioned in this text. Return ONLY a valid JSON array of lowercase strings. "
    'Example: ["non-duality", "consciousness", "rupert spira", "advaita vedanta"]. '
    "Output nothing else."
)


def extract_chunk_entities(text: str, gateway: ModelGateway) -> list[str]:
    """Return a flat list of lowercase entity strings for a chunk of text.

    Returns [] on any failure — never raises.
    """
    try:
        resp = gateway.get_completion(
            prompt=text[:1000],
            system_prompt=_SYSTEM_PROMPT,
            model="claude-haiku-4-5-20251001",
            provider="anthropic",
            associated_id="entity_extract",
        )
        raw = resp.text_content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[^\n]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())
        result = json.loads(raw)
        return [e.lower() for e in result if isinstance(e, str) and e.strip()]
    except Exception:
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_entities.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add core/entities.py tests/unit/test_entities.py
git commit -m "feat: add extract_chunk_entities to core/entities"
```

---

### Task 3: `ingestion/worker_lambda.py` — wire up VideoMeta + entity extraction

**Files:**
- Modify: `ingestion/worker_lambda.py`
- Modify: `tests/integration/test_worker_process_video.py`

**Interfaces:**
- Consumes:
  - `VideoMeta, classify_video_meta` from `core.topics`
  - `extract_chunk_entities` from `core.entities`
- Produces: chunk metadata now includes `"entities": list[str]`, `"host": str | None`, `"guests": list[str]`

---

- [ ] **Step 1: Update the integration test**

In `tests/integration/test_worker_process_video.py`, replace the existing `_make_gateway` function:

```python
def _make_gateway(topics=None, host="Test Host", guests=None):
    gw = MagicMock()
    gw.get_embedding.return_value = MagicMock(
        embedding_vector=[0.0] * 1536, cost=0.0001, input_tokens=10
    )
    video_meta_resp = MagicMock(
        text_content=json.dumps({
            "topics": topics or ["consciousness"],
            "host": host,
            "guests": guests if guests is not None else [],
        }),
        cost=0.001,
    )
    entity_resp = MagicMock(
        text_content='["concept1", "concept2"]',
        cost=0.0001,
    )
    # First call: classify_video_meta; remaining calls: extract_chunk_entities (one per chunk)
    gw.get_completion.side_effect = [video_meta_resp] + [entity_resp] * 50
    return gw
```

Add a new test at the bottom of the file:

```python
@patch("ingestion.worker_lambda.fetch_sponsor_segments", return_value=[])
@patch("ingestion.worker_lambda.YouTubeTranscriptApi")
def test_upserted_metadata_includes_entities_and_people(mock_api_cls, mock_sponsor):
    conn, cur = _make_db()
    mock_store = MagicMock()
    gw = _make_gateway(topics=["consciousness"], host="The Host", guests=["The Guest"])

    mock_api_cls.return_value.fetch.return_value = _fake_transcript()

    from ingestion.worker_lambda import process_video
    process_video("vid1", "ch1", conn, None, mock_store, gw)

    mock_store.upsert.assert_called_once()
    call_kwargs = mock_store.upsert.call_args.kwargs
    metadatas = call_kwargs["metadatas"]
    assert len(metadatas) > 0
    first = metadatas[0]
    assert "entities" in first
    assert isinstance(first["entities"], list)
    assert first["host"] == "The Host"
    assert first["guests"] == ["The Guest"]
```

- [ ] **Step 2: Run the new test to verify it fails**

```bash
uv run pytest tests/integration/test_worker_process_video.py::test_upserted_metadata_includes_entities_and_people -v
```

Expected: FAIL — `entities` / `host` / `guests` not in metadata yet.

- [ ] **Step 3: Update `ingestion/worker_lambda.py`**

At the top, replace:

```python
from core.topics import classify_topics
```

with:

```python
from core.entities import extract_chunk_entities
from core.topics import classify_video_meta
```

Inside `process_video`, after fetching `title` and `description`, add a channel name lookup (insert after the `meta` fetch block):

```python
        with db_conn.cursor() as cur:
            cur.execute("SELECT name FROM channels WHERE id = %s", (channel_id,))
            ch_row = cur.fetchone()
        channel_name = ch_row[0] if ch_row else ""
```

Replace the topic classification block:

```python
        # -- Topic classification ---------------------------------------------
        available_topics = get_topic_names(db_conn)
        default_hint     = get_channel_default_topic(db_conn, channel_id) or available_topics[0]
        # Use transcript excerpt as fallback when description is missing
        classify_text = description.strip() if description and description.strip() else (
            " ".join(seg["text"] for seg in srt[:120])
        )
        video_topics = classify_topics(
            title=title,
            text_excerpt=classify_text,
            available_topics=available_topics,
            default_hint=default_hint,
            gateway=gateway,
        )
        primary_topic = video_topics[0] if video_topics else available_topics[0]
```

with:

```python
        # -- Topic classification + people extraction -------------------------
        available_topics = get_topic_names(db_conn)
        default_hint     = get_channel_default_topic(db_conn, channel_id) or available_topics[0]
        classify_text = description.strip() if description and description.strip() else (
            " ".join(seg["text"] for seg in srt[:120])
        )
        video_meta = classify_video_meta(
            title=title,
            channel_name=channel_name,
            text_excerpt=classify_text,
            available_topics=available_topics,
            default_hint=default_hint,
            gateway=gateway,
        )
        video_topics  = video_meta.topics
        primary_topic = video_topics[0] if video_topics else available_topics[0]
```

Update the `metadatas.append` block inside the chunk loop. Replace:

```python
            metadatas.append({
                "source_type":   "youtube_video",
                "video_id":      video_id,
                "channel_id":    channel_id,
                "topics":        video_topics,       # list — Pinecone $in filter
                "primary_topic": primary_topic,      # scalar — Chroma $eq filter
                "chapter":       chunk["associated_chapter"],
                "start_seconds": chunk["start_seconds"],
                "deep_link":     chunk["deep_link"],
            })
```

with:

```python
            entities = extract_chunk_entities(chunk["text_content"], gateway)
            metadatas.append({
                "source_type":   "youtube_video",
                "video_id":      video_id,
                "channel_id":    channel_id,
                "topics":        video_topics,       # list — Pinecone $in filter
                "primary_topic": primary_topic,      # scalar — Chroma $eq filter
                "chapter":       chunk["associated_chapter"],
                "start_seconds": chunk["start_seconds"],
                "deep_link":     chunk["deep_link"],
                "entities":      entities,
                "host":          video_meta.host,
                "guests":        video_meta.guests,
            })
```

- [ ] **Step 4: Run the full integration test suite**

```bash
uv run pytest tests/integration/test_worker_process_video.py -v
```

Expected: all tests pass, including the new metadata assertion test.

- [ ] **Step 5: Commit**

```bash
git add ingestion/worker_lambda.py tests/integration/test_worker_process_video.py
git commit -m "feat: add entity/host/guests metadata to YouTube ingestion worker"
```

---

### Task 4: `ingestion/article_worker.py` — entity extraction + author metadata

**Files:**
- Modify: `ingestion/article_worker.py`
- Modify: `tests/integration/test_article_worker.py`

**Interfaces:**
- Consumes: `extract_chunk_entities` from `core.entities`
- Produces: chunk metadata now includes `"entities": list[str]`, `"author": str`

---

- [ ] **Step 1: Update the article integration test**

In `tests/integration/test_article_worker.py`, replace the existing `_make_gateway` function:

```python
def _make_gateway(topics=None):
    gw = MagicMock()
    gw.get_embedding.return_value = MagicMock(
        embedding_vector=[0.0] * 1536, cost=0.0001, input_tokens=10
    )
    topic_resp = MagicMock(
        text_content=json.dumps(topics or ["consciousness"]), cost=0.001
    )
    entity_resp = MagicMock(
        text_content='["consciousness", "awareness"]', cost=0.0001
    )
    # First call: classify_topics; remaining calls: extract_chunk_entities (one per chunk)
    gw.get_completion.side_effect = [topic_resp] + [entity_resp] * 50
    return gw
```

Add a new test at the bottom of the file:

```python
@patch("ingestion.article_worker.get_topic_names", return_value=["consciousness", "biohacking"])
@patch("ingestion.article_worker.extract_sections", return_value=_FAKE_SECTIONS)
@patch("ingestion.article_worker.fetch_article", return_value=_FAKE_ARTICLE)
@patch("ingestion.article_worker._save_payload", return_value="local/path")
def test_upserted_metadata_includes_entities_and_author(mock_save, mock_fetch, mock_sections, mock_topics):
    conn, cur = _make_db()
    store = MagicMock()
    gw = _make_gateway()

    from ingestion.article_worker import process_article
    process_article("art-uuid-1", "https://example.com/post", "example.com", conn, store, gw)

    store.upsert.assert_called_once()
    call_kwargs = store.upsert.call_args.kwargs
    metadatas = call_kwargs["metadatas"]
    assert len(metadatas) > 0
    first = metadatas[0]
    assert "entities" in first
    assert isinstance(first["entities"], list)
    assert first["author"] == "Dr. Smith"  # from _FAKE_ARTICLE["author"]
```

- [ ] **Step 2: Run the new test to verify it fails**

```bash
uv run pytest tests/integration/test_article_worker.py::test_upserted_metadata_includes_entities_and_author -v
```

Expected: FAIL — `entities` / `author` not in metadata yet.

- [ ] **Step 3: Update `ingestion/article_worker.py`**

Add import at the top (after existing imports from `core`):

```python
from core.entities import extract_chunk_entities
```

Before the chunk loop, extract author from the already-parsed article dict (it was written to DB but not held in a variable for metadata). Insert after `primary_topic` is set:

```python
        author = (article.get("author") or "")[:255]
```

Inside the chunk loop, replace the `metadatas.append` block:

```python
            metadatas.append({
                "source_type":   "article",
                "article_id":    article_id,
                "website_id":    website_id,
                "topics":        article_topics,        # list — Pinecone $in filter
                "primary_topic": primary_topic,         # scalar — Chroma $eq filter
                "chapter":       chunk["associated_chapter"],
                "section_slug":  chunk.get("section_slug", ""),
                "deep_link":     chunk["deep_link"],
            })
```

with:

```python
            entities = extract_chunk_entities(chunk["text_content"], gateway)
            metadatas.append({
                "source_type":   "article",
                "article_id":    article_id,
                "website_id":    website_id,
                "topics":        article_topics,        # list — Pinecone $in filter
                "primary_topic": primary_topic,         # scalar — Chroma $eq filter
                "chapter":       chunk["associated_chapter"],
                "section_slug":  chunk.get("section_slug", ""),
                "deep_link":     chunk["deep_link"],
                "entities":      entities,
                "author":        author,
            })
```

- [ ] **Step 4: Run the full article integration test suite**

```bash
uv run pytest tests/integration/test_article_worker.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add ingestion/article_worker.py tests/integration/test_article_worker.py
git commit -m "feat: add entity/author metadata to article ingestion worker"
```

---

### Task 5: `retrieval/main.py` — entity overlap + people bonus in `_rerank`

**Files:**
- Modify: `retrieval/main.py`
- Modify: `tests/unit/test_retrieval_helpers.py`

**Interfaces:**
- Consumes: chunk metadata fields `entities` (list or JSON string), `host` (str or None), `guests` (list or JSON string), `author` (str or None)
- Produces: updated `_rerank` that incorporates entity and people scoring; new helpers `_parse_list_meta`, `_entity_overlap_score`, `_people_bonus`

---

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_retrieval_helpers.py` (at the bottom, after existing tests):

```python
import pytest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# _parse_list_meta
# ---------------------------------------------------------------------------

def test_parse_list_meta_native_list(main_mod):
    assert main_mod._parse_list_meta(["a", "b"]) == ["a", "b"]


def test_parse_list_meta_json_string(main_mod):
    assert main_mod._parse_list_meta('["a", "b"]') == ["a", "b"]


def test_parse_list_meta_none(main_mod):
    assert main_mod._parse_list_meta(None) == []


def test_parse_list_meta_bad_json(main_mod):
    assert main_mod._parse_list_meta("not json") == []


# ---------------------------------------------------------------------------
# _entity_overlap_score
# ---------------------------------------------------------------------------

def test_entity_overlap_partial_match(main_mod):
    meta = {"entities": ["consciousness", "non-duality"]}
    score = main_mod._entity_overlap_score("tell me about consciousness", meta)
    assert score == pytest.approx(0.5)


def test_entity_overlap_all_match(main_mod):
    meta = {"entities": ["concept"]}
    assert main_mod._entity_overlap_score("concept here", meta) == pytest.approx(1.0)


def test_entity_overlap_no_entities(main_mod):
    assert main_mod._entity_overlap_score("query", {"entities": []}) == 0.0


def test_entity_overlap_missing_field(main_mod):
    assert main_mod._entity_overlap_score("query", {}) == 0.0


def test_entity_overlap_chroma_json_string(main_mod):
    meta = {"entities": '["consciousness", "non-duality"]'}
    score = main_mod._entity_overlap_score("consciousness", meta)
    assert score == pytest.approx(0.5)


def test_entity_overlap_no_match(main_mod):
    meta = {"entities": ["zen", "taoism"]}
    assert main_mod._entity_overlap_score("biohacking protocols", meta) == 0.0


# ---------------------------------------------------------------------------
# _people_bonus
# ---------------------------------------------------------------------------

def test_people_bonus_host_match(main_mod):
    meta = {"host": "Joe Rogan", "guests": [], "author": None}
    assert main_mod._people_bonus("what did joe rogan say", meta) == pytest.approx(0.15)


def test_people_bonus_guest_match(main_mod):
    meta = {"host": "Joe Rogan", "guests": ["Graham Hancock"], "author": None}
    assert main_mod._people_bonus("graham hancock on consciousness", meta) == pytest.approx(0.15)


def test_people_bonus_multiple_guests_capped(main_mod):
    meta = {"host": "Host", "guests": ["Guest A", "Guest B"], "author": None}
    assert main_mod._people_bonus("guest a and guest b discussed", meta) == pytest.approx(0.15)


def test_people_bonus_no_match(main_mod):
    meta = {"host": "Joe Rogan", "guests": ["Guest"], "author": None}
    assert main_mod._people_bonus("something completely unrelated", meta) == 0.0


def test_people_bonus_author_match(main_mod):
    meta = {"host": None, "guests": [], "author": "Mark Manson"}
    assert main_mod._people_bonus("mark manson on meaning", meta) == pytest.approx(0.15)


def test_people_bonus_chroma_json_guests(main_mod):
    meta = {"host": None, "guests": '["Graham Hancock"]', "author": None}
    assert main_mod._people_bonus("graham hancock", meta) == pytest.approx(0.15)


def test_people_bonus_empty_meta(main_mod):
    assert main_mod._people_bonus("query", {}) == 0.0


# ---------------------------------------------------------------------------
# _rerank — entity boost changes ordering
# ---------------------------------------------------------------------------

def _make_yt_chunk(video_id: str, start: int, entities: list, text: str = "text") -> dict:
    return {"metadata": {
        "source_type":   "youtube_video",
        "video_id":      video_id,
        "start_seconds": start,
        "text_content":  text,
        "chapter":       "Intro",
        "deep_link":     f"https://youtu.be/{video_id}?t={start}",
        "entities":      entities,
        "host":          None,
        "guests":        [],
        "author":        None,
    }}


def test_rerank_entity_boost_changes_order(main_mod, monkeypatch):
    # chunk_a: CE=0.4, no entity match → final=0.4
    # chunk_b: CE=0.2, both entities match → final=0.2 + 0.3*(2/2)=0.5
    chunk_a = _make_yt_chunk("v1", 0, entities=[])
    chunk_b = _make_yt_chunk("v2", 0, entities=["consciousness", "awareness"])

    monkeypatch.setattr(
        main_mod, "_reranker",
        MagicMock(predict=lambda pairs: [0.4, 0.2])
    )

    result = main_mod._rerank("consciousness and awareness", [chunk_a, chunk_b], top_n=2)
    assert result[0]["metadata"]["video_id"] == "v2"


def test_rerank_people_bonus_changes_order(main_mod, monkeypatch):
    # chunk_a: CE=0.4, no people match → final=0.4
    # chunk_b: CE=0.2, host match → final=0.2 + 0.15=0.35  → chunk_a still wins
    # Adjust: chunk_b CE=0.3, host match → final=0.3 + 0.15=0.45 → chunk_b wins
    chunk_a = _make_yt_chunk("v1", 0, entities=[])
    chunk_b = _make_yt_chunk("v2", 0, entities=[])
    chunk_b["metadata"]["host"] = "Graham Hancock"

    monkeypatch.setattr(
        main_mod, "_reranker",
        MagicMock(predict=lambda pairs: [0.4, 0.3])
    )

    result = main_mod._rerank("what did graham hancock say", [chunk_a, chunk_b], top_n=2)
    assert result[0]["metadata"]["video_id"] == "v2"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_retrieval_helpers.py -k "parse_list_meta or entity_overlap or people_bonus or entity_boost or people_bonus_changes" -v
```

Expected: `AttributeError` — `_parse_list_meta`, `_entity_overlap_score`, `_people_bonus` not defined.

- [ ] **Step 3: Implement helpers and update `_rerank` in `retrieval/main.py`**

After the existing imports, add `import json as _json` (use alias to avoid shadowing any local var):

```python
import json as _json
```

Add `ENTITY_WEIGHT` after `_WIDGET_SECRET`:

```python
ENTITY_WEIGHT = float(os.environ.get("ENTITY_WEIGHT", "0.3"))
```

Add three new helper functions immediately before the existing `_rerank` function:

```python
def _parse_list_meta(value) -> list:
    """Coerce a metadata field that is either a native list (Pinecone) or a JSON string (Chroma)."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            result = _json.loads(value)
            return result if isinstance(result, list) else []
        except Exception:
            return []
    return []


def _entity_overlap_score(query_lower: str, meta: dict) -> float:
    """Fraction of a chunk's entities that appear as substrings in the query."""
    entities = _parse_list_meta(meta.get("entities"))
    if not entities:
        return 0.0
    matched = sum(1 for e in entities if str(e).lower() in query_lower)
    return matched / len(entities)


def _people_bonus(query_lower: str, meta: dict) -> float:
    """Return 0.15 if any of host/guests/author appears in the query, else 0.0."""
    people: list[str] = []
    if meta.get("host"):
        people.append(str(meta["host"]))
    people.extend(str(g) for g in _parse_list_meta(meta.get("guests")) if g)
    if meta.get("author"):
        people.append(str(meta["author"]))
    return 0.15 if any(p.lower() in query_lower for p in people if p) else 0.0
```

Replace the existing `_rerank` function:

```python
def _rerank(query: str, matches: list[dict], top_n: int = 5) -> list[dict]:
    if not matches:
        return []
    merged    = _merge_adjacent_chunks(matches)
    pairs     = [(query, m["metadata"].get("text_content", "")) for m in merged]
    ce_scores = _reranker.predict(pairs)
    query_lower = query.lower()
    final_scores = [
        ce
        + ENTITY_WEIGHT * _entity_overlap_score(query_lower, m["metadata"])
        + _people_bonus(query_lower, m["metadata"])
        for m, ce in zip(merged, ce_scores)
    ]
    ranked = sorted(zip(merged, final_scores), key=lambda x: x[1], reverse=True)
    return [m for m, _ in ranked[:top_n]]
```

- [ ] **Step 4: Run all unit tests**

```bash
uv run pytest tests/unit/ -v
```

Expected: all tests pass, including the new retrieval helper tests.

- [ ] **Step 5: Run the full test suite**

```bash
uv run pytest tests/unit tests/integration -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add retrieval/main.py tests/unit/test_retrieval_helpers.py
git commit -m "feat: add entity overlap + people bonus scoring to _rerank"
```
