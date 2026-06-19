"""Unit tests for core/vector_store.py — ChromaStore and _safe_chroma_meta."""
import json
from unittest.mock import MagicMock, patch

from core.vector_store import _safe_chroma_meta


# ---------------------------------------------------------------------------
# _safe_chroma_meta
# ---------------------------------------------------------------------------

def test_safe_chroma_meta_passes_scalars():
    meta = {"source_type": "article", "start_seconds": 30, "score": 0.95, "active": True}
    result = _safe_chroma_meta(meta)
    assert result == meta


def test_safe_chroma_meta_converts_list_to_json_string():
    meta = {"topics": ["consciousness", "biohacking"]}
    result = _safe_chroma_meta(meta)
    assert isinstance(result["topics"], str)
    assert json.loads(result["topics"]) == ["consciousness", "biohacking"]


def test_safe_chroma_meta_drops_none_values():
    meta = {"title": "Test", "author": None, "score": 0.9}
    result = _safe_chroma_meta(meta)
    assert "author" not in result
    assert result["title"] == "Test"


def test_safe_chroma_meta_converts_dict_to_json_string():
    meta = {"nested": {"key": "value"}}
    result = _safe_chroma_meta(meta)
    assert isinstance(result["nested"], str)
    assert json.loads(result["nested"]) == {"key": "value"}


def test_safe_chroma_meta_preserves_string_values():
    meta = {"primary_topic": "consciousness", "chapter": "Intro"}
    result = _safe_chroma_meta(meta)
    assert result["primary_topic"] == "consciousness"
    assert result["chapter"] == "Intro"


# ---------------------------------------------------------------------------
# ChromaStore — upsert + query with mocked chromadb
# ---------------------------------------------------------------------------

def _make_chroma_store():
    """Build a ChromaStore with chromadb fully mocked at sys.modules level.

    chromadb has a protobuf version conflict in this environment, so we prevent
    the real import entirely by injecting a mock into sys.modules before
    ChromaStore.__init__ runs its `import chromadb` statement.
    """
    import sys

    mock_chroma = MagicMock()
    mock_col = MagicMock()
    mock_col.count.return_value = 5
    mock_chroma.PersistentClient.return_value.get_or_create_collection.return_value = mock_col

    with patch.dict(sys.modules, {"chromadb": mock_chroma}):
        from core.vector_store import ChromaStore
        store = ChromaStore(path="/tmp/test-chroma")

    # mock_col is still accessible via store._col after the patch context exits
    return store, mock_col


def test_chroma_upsert_calls_collection():
    store, mock_col = _make_chroma_store()
    store.upsert(
        ids=["a1"],
        embeddings=[[0.1] * 10],
        metadatas=[{"source_type": "article", "topics": ["consciousness"]}],
        texts=["some text"],
    )
    mock_col.upsert.assert_called_once()


def test_chroma_upsert_converts_list_metadata():
    store, mock_col = _make_chroma_store()
    store.upsert(
        ids=["a1"],
        embeddings=[[0.1] * 10],
        metadatas=[{"topics": ["consciousness", "biohacking"]}],
        texts=["text"],
    )
    call_kwargs = mock_col.upsert.call_args.kwargs
    meta = call_kwargs["metadatas"][0]
    assert isinstance(meta["topics"], str)
    assert json.loads(meta["topics"]) == ["consciousness", "biohacking"]


def test_chroma_upsert_empty_is_noop():
    store, mock_col = _make_chroma_store()
    store.upsert(ids=[], embeddings=[], metadatas=[], texts=[])
    mock_col.upsert.assert_not_called()


def test_chroma_query_returns_normalised_list():
    store, mock_col = _make_chroma_store()
    mock_col.query.return_value = {
        "metadatas": [[{"source_type": "article", "article_id": "abc"}]],
        "documents": [["chunk text"]],
    }
    results = store.query(embedding=[0.1] * 10, n_results=5)
    assert len(results) == 1
    assert results[0]["metadata"]["text_content"] == "chunk text"
    assert results[0]["metadata"]["source_type"] == "article"


def test_chroma_query_empty_collection_returns_empty():
    store, mock_col = _make_chroma_store()
    mock_col.count.return_value = 0
    results = store.query(embedding=[0.1] * 10, n_results=5)
    assert results == []
    mock_col.query.assert_not_called()


def test_chroma_query_translates_primary_topic_to_eq_filter():
    store, mock_col = _make_chroma_store()
    mock_col.query.return_value = {"metadatas": [[]], "documents": [[]]}
    store.query(embedding=[0.1] * 10, where={"primary_topic": "consciousness"})
    call_kwargs = mock_col.query.call_args.kwargs
    assert call_kwargs["where"] == {"primary_topic": {"$eq": "consciousness"}}


def test_chroma_query_no_filter_when_no_where():
    store, mock_col = _make_chroma_store()
    mock_col.query.return_value = {"metadatas": [[]], "documents": [[]]}
    store.query(embedding=[0.1] * 10)
    call_kwargs = mock_col.query.call_args.kwargs
    assert "where" not in call_kwargs


def test_chroma_query_caps_n_results_at_collection_size():
    store, mock_col = _make_chroma_store()
    mock_col.count.return_value = 3
    mock_col.query.return_value = {"metadatas": [[]], "documents": [[]]}
    store.query(embedding=[0.1] * 10, n_results=20)
    call_kwargs = mock_col.query.call_args.kwargs
    assert call_kwargs["n_results"] == 3
