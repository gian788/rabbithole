"""Integration tests for ingestion/article_worker.py — happy path, idempotency, failures."""
import json
from unittest.mock import MagicMock, patch

import pytest


def _make_db(status="discovered"):
    """Return a mock DB that reports the given article status."""
    conn = MagicMock()
    ctx = MagicMock()
    cur = MagicMock()

    call_count = [0]

    def fetchone_side_effect():
        call_count[0] += 1
        if call_count[0] == 1:
            return (status,) if status else None  # status check
        return None  # website default topic lookup

    cur.fetchone.side_effect = fetchone_side_effect
    ctx.__enter__ = MagicMock(return_value=cur)
    ctx.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = ctx
    return conn, cur


def _make_gateway(topics=None):
    gw = MagicMock()
    gw.get_embedding.return_value = MagicMock(
        embedding_vector=[0.0] * 1536, cost=0.0001, input_tokens=10
    )
    gw.get_completion.return_value = MagicMock(
        text_content=json.dumps(topics or ["consciousness"]), cost=0.001
    )
    return gw


_FAKE_ARTICLE = {
    "url": "https://example.com/post",
    "title": "Understanding Consciousness",
    "author": "Dr. Smith",
    "publisher": "Example Blog",
    "published_at": None,
    "html_body": "<article><p>text</p></article>",
}

_FAKE_SECTIONS = [
    {
        "chunk_id": "s_001",
        "associated_chapter": "Introduction",
        "section_slug": "introduction",
        "deep_link": "https://example.com/post#introduction",
        "text_content": "A detailed introduction to consciousness. " * 20,
    }
]


@patch("ingestion.article_worker.get_topic_names", return_value=["consciousness", "biohacking"])
@patch("ingestion.article_worker.extract_sections", return_value=_FAKE_SECTIONS)
@patch("ingestion.article_worker.fetch_article", return_value=_FAKE_ARTICLE)
@patch("ingestion.article_worker._save_payload", return_value="local/path")
def test_happy_path_marks_completed(mock_save, mock_fetch, mock_sections, mock_topics):
    conn, cur = _make_db()
    store = MagicMock()
    gw = _make_gateway()

    from ingestion.article_worker import process_article
    process_article("art-uuid-1", "https://example.com/post", "example.com", conn, store, gw)

    store.upsert.assert_called_once()
    execute_calls = [str(c) for c in cur.execute.call_args_list]
    assert any("completed" in c for c in execute_calls)


@patch("ingestion.article_worker.get_topic_names", return_value=["consciousness"])
@patch("ingestion.article_worker.extract_sections", return_value=_FAKE_SECTIONS)
@patch("ingestion.article_worker.fetch_article", return_value=_FAKE_ARTICLE)
@patch("ingestion.article_worker._save_payload", return_value="local/path")
def test_already_completed_skips(mock_save, mock_fetch, mock_sections, mock_topics):
    conn, cur = _make_db(status="completed")
    store = MagicMock()
    gw = _make_gateway()

    from ingestion.article_worker import process_article
    process_article("art-uuid-2", "https://example.com/post", "example.com", conn, store, gw)

    mock_fetch.assert_not_called()
    store.upsert.assert_not_called()


@patch("ingestion.article_worker.get_topic_names", return_value=["consciousness"])
@patch("ingestion.article_worker.extract_sections", return_value=[])
@patch("ingestion.article_worker.fetch_article", return_value=_FAKE_ARTICLE)
def test_empty_sections_marks_failed(mock_fetch, mock_sections, mock_topics):
    conn, cur = _make_db()
    store = MagicMock()
    gw = _make_gateway()

    from ingestion.article_worker import process_article
    process_article("art-uuid-3", "https://example.com/post", "example.com", conn, store, gw)

    store.upsert.assert_not_called()
    execute_calls = [str(c) for c in cur.execute.call_args_list]
    assert any("failed" in c for c in execute_calls)


@patch("ingestion.article_worker.get_topic_names", return_value=["consciousness"])
@patch("ingestion.article_worker.fetch_article", side_effect=RuntimeError("Connection refused"))
def test_fetch_exception_reraises_and_marks_failed(mock_fetch, mock_topics):
    conn, cur = _make_db()
    store = MagicMock()
    gw = _make_gateway()

    from ingestion.article_worker import process_article
    with pytest.raises(RuntimeError, match="Connection refused"):
        process_article("art-uuid-4", "https://example.com/post", "example.com", conn, store, gw)

    execute_calls = [str(c) for c in cur.execute.call_args_list]
    assert any("failed" in c for c in execute_calls)


@patch("ingestion.article_worker.get_topic_names", return_value=["consciousness", "biohacking"])
@patch("ingestion.article_worker.extract_sections", return_value=_FAKE_SECTIONS)
@patch("ingestion.article_worker.fetch_article", return_value=_FAKE_ARTICLE)
@patch("ingestion.article_worker._save_payload", return_value="local/path")
def test_upsert_receives_correct_metadata(mock_save, mock_fetch, mock_sections, mock_topics):
    conn, cur = _make_db()
    store = MagicMock()
    gw = _make_gateway(topics=["consciousness"])

    from ingestion.article_worker import process_article
    process_article("art-uuid-5", "https://example.com/post", "example.com", conn, store, gw)

    call_kwargs = store.upsert.call_args.kwargs
    meta = call_kwargs["metadatas"][0]
    assert meta["source_type"] == "article"
    assert meta["article_id"] == "art-uuid-5"
    assert meta["website_id"] == "example.com"
    assert "primary_topic" in meta
    assert "topics" in meta


@patch("ingestion.article_worker.get_topic_names", return_value=["consciousness"])
@patch("ingestion.article_worker.extract_sections", return_value=_FAKE_SECTIONS)
@patch("ingestion.article_worker.fetch_article", return_value=_FAKE_ARTICLE)
@patch("ingestion.article_worker._save_payload", return_value="local/path")
def test_embed_exception_reraises_and_marks_failed(mock_save, mock_fetch, mock_sections, mock_topics):
    conn, cur = _make_db()
    store = MagicMock()
    gw = _make_gateway()
    gw.get_embedding.side_effect = RuntimeError("OpenAI down")

    from ingestion.article_worker import process_article
    with pytest.raises(RuntimeError, match="OpenAI down"):
        process_article("art-uuid-6", "https://example.com/post", "example.com", conn, store, gw)

    execute_calls = [str(c) for c in cur.execute.call_args_list]
    assert any("failed" in c for c in execute_calls)
