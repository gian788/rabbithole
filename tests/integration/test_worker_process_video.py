import json
from unittest.mock import MagicMock, patch

import pytest


def _make_db(status="discovered", title="Ep 1 | Guest", description="0:00 Intro\n2:00 Main\n10:00 Outro"):
    """Return a mock DB that reports the given video status and metadata."""
    conn = MagicMock()
    ctx = MagicMock()

    call_count = [0]

    def fetchone_side_effect():
        call_count[0] += 1
        if call_count[0] == 1:
            return (status,) if status else None  # status check
        return (title, description)               # metadata fetch

    cur = MagicMock()
    cur.fetchone.side_effect = fetchone_side_effect
    ctx.__enter__ = MagicMock(return_value=cur)
    ctx.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = ctx
    return conn, cur


def _fake_transcript():
    """Return a mock YouTubeTranscriptApi fetch result with enough words to chunk."""
    words = " ".join([f"word{i}" for i in range(300)])
    snippet = MagicMock()
    snippet.text = words
    snippet.start = 0.0
    snippet.duration = 300.0
    result = MagicMock()
    result.snippets = [snippet]
    return result


def _make_gateway(topics=None):
    gw = MagicMock()
    gw.get_embedding.return_value = MagicMock(
        embedding_vector=[0.0] * 1536, cost=0.0001, input_tokens=10
    )
    gw.get_completion.return_value = MagicMock(
        text_content=json.dumps(topics or ["consciousness"]), cost=0.001
    )
    return gw


@patch("ingestion.worker_lambda.fetch_sponsor_segments", return_value=[])
@patch("ingestion.worker_lambda.YouTubeTranscriptApi")
def test_already_completed_skips(mock_api_cls, mock_sponsor):
    conn, cur = _make_db(status="completed")
    mock_store = MagicMock()
    gw = _make_gateway()

    from ingestion.worker_lambda import process_video
    process_video("vid1", "ch1", conn, None, mock_store, gw)

    mock_api_cls.return_value.fetch.assert_not_called()
    mock_store.upsert.assert_not_called()


@patch("ingestion.worker_lambda.fetch_sponsor_segments", return_value=[])
@patch("ingestion.worker_lambda.YouTubeTranscriptApi")
@patch("ingestion.worker_lambda.get_topic_names", return_value=["consciousness", "biohacking"])
@patch("ingestion.worker_lambda.get_channel_default_topic", return_value="consciousness")
@patch("ingestion.worker_lambda._save_payload", return_value="local/path")
def test_happy_path_marks_completed(mock_save, mock_topic, mock_names, mock_api_cls, mock_sponsor):
    mock_api_cls.return_value.fetch.return_value = _fake_transcript()
    conn, cur = _make_db(description="0:00 Intro\n2:00 Main\n10:00 Outro\n20:00 Wrap")
    mock_store = MagicMock()
    gw = _make_gateway()

    from ingestion.worker_lambda import process_video
    process_video("vid1", "ch1", conn, None, mock_store, gw)

    mock_store.upsert.assert_called()
    execute_calls = [str(c) for c in cur.execute.call_args_list]
    assert any("completed" in c for c in execute_calls)


@patch("ingestion.worker_lambda.fetch_sponsor_segments", return_value=[])
@patch("ingestion.worker_lambda.YouTubeTranscriptApi")
@patch("ingestion.worker_lambda.get_topic_names", return_value=["consciousness"])
@patch("ingestion.worker_lambda.get_channel_default_topic", return_value="consciousness")
@patch("ingestion.worker_lambda._save_payload", return_value="local/path")
def test_description_without_chapters_calls_llm(mock_save, mock_topic, mock_names, mock_api_cls, mock_sponsor):
    mock_api_cls.return_value.fetch.return_value = _fake_transcript()
    # No chapter timestamps in description
    conn, cur = _make_db(description="Great episode about consciousness.")
    mock_store = MagicMock()
    gw = _make_gateway()
    # LLM chapter generation: first call returns 4 chapters, second returns topics
    chapters_json = json.dumps([
        {"title": "Intro", "start_seconds": 0},
        {"title": "Main",  "start_seconds": 60},
        {"title": "Deep",  "start_seconds": 120},
        {"title": "End",   "start_seconds": 200},
    ])
    gw.get_completion.side_effect = [
        MagicMock(text_content=chapters_json, cost=0.001),  # chapter gen
        MagicMock(text_content='["consciousness"]', cost=0.001),  # topic classify
    ]

    from ingestion.worker_lambda import process_video
    process_video("vid1", "ch1", conn, None, mock_store, gw)

    # gateway.get_completion called at least twice: chapter gen + topic classification
    assert gw.get_completion.call_count >= 2


@patch("ingestion.worker_lambda.fetch_sponsor_segments", return_value=[])
@patch("ingestion.worker_lambda.YouTubeTranscriptApi")
@patch("ingestion.worker_lambda.get_topic_names", return_value=["consciousness"])
@patch("ingestion.worker_lambda.get_channel_default_topic", return_value="consciousness")
@patch("ingestion.worker_lambda._save_payload", return_value="local/path")
def test_llm_returns_few_chapters_uses_fixed_chunking(mock_save, mock_topic, mock_names, mock_api_cls, mock_sponsor):
    mock_api_cls.return_value.fetch.return_value = _fake_transcript()
    conn, cur = _make_db(description="No timestamps here.")
    mock_store = MagicMock()
    gw = _make_gateway()
    # Chapter gen returns only 1 chapter → fallback to fixed_word_chunking
    gw.get_completion.side_effect = [
        MagicMock(text_content='[{"title":"Intro","start_seconds":0}]', cost=0.001),
        MagicMock(text_content='["consciousness"]', cost=0.001),
    ]

    from ingestion.worker_lambda import process_video

    with patch("ingestion.worker_lambda.fixed_word_chunking", wraps=__import__("core.chunker", fromlist=["fixed_word_chunking"]).fixed_word_chunking) as mock_fixed:
        process_video("vid1", "ch1", conn, None, mock_store, gw)
        mock_fixed.assert_called_once()


@patch("ingestion.worker_lambda.fetch_sponsor_segments", return_value=[])
@patch("ingestion.worker_lambda.YouTubeTranscriptApi")
def test_transcripts_disabled_sets_failed(mock_api_cls, mock_sponsor):
    from youtube_transcript_api import TranscriptsDisabled
    mock_api_cls.return_value.fetch.side_effect = TranscriptsDisabled("vid1")
    conn, cur = _make_db()
    mock_store = MagicMock()
    gw = _make_gateway()

    from ingestion.worker_lambda import process_video
    process_video("vid1", "ch1", conn, None, mock_store, gw)  # must NOT raise

    execute_calls = [str(c) for c in cur.execute.call_args_list]
    assert any("failed" in c for c in execute_calls)
    mock_store.upsert.assert_not_called()


@patch("ingestion.worker_lambda.fetch_sponsor_segments", return_value=[])
@patch("ingestion.worker_lambda.YouTubeTranscriptApi")
@patch("ingestion.worker_lambda.get_topic_names", return_value=["consciousness"])
@patch("ingestion.worker_lambda.get_channel_default_topic", return_value="consciousness")
def test_embed_exception_reraises(mock_topic, mock_names, mock_api_cls, mock_sponsor):
    mock_api_cls.return_value.fetch.return_value = _fake_transcript()
    conn, cur = _make_db(description="0:00 Intro\n2:00 Main\n10:00 Outro\n20:00 End")
    mock_store = MagicMock()
    gw = _make_gateway()
    gw.get_completion.return_value = MagicMock(text_content='["consciousness"]', cost=0.001)
    gw.get_embedding.side_effect = RuntimeError("OpenAI down")

    from ingestion.worker_lambda import process_video
    with pytest.raises(RuntimeError, match="OpenAI down"):
        process_video("vid1", "ch1", conn, None, mock_store, gw)

    execute_calls = [str(c) for c in cur.execute.call_args_list]
    assert any("failed" in c for c in execute_calls)


@patch("ingestion.worker_lambda.fetch_sponsor_segments", return_value=[])
@patch("ingestion.worker_lambda.YouTubeTranscriptApi")
@patch("ingestion.worker_lambda.get_topic_names", return_value=["consciousness"])
@patch("ingestion.worker_lambda.get_channel_default_topic", return_value="consciousness")
@patch("ingestion.worker_lambda._save_payload", return_value="local/path")
def test_store_upsert_called_with_correct_metadata(mock_save, mock_topic, mock_names, mock_api_cls, mock_sponsor):
    """store.upsert receives ids/embeddings/metadatas/texts with correct source_type and primary_topic."""
    mock_api_cls.return_value.fetch.return_value = _fake_transcript()
    conn, cur = _make_db(description="0:00 Intro\n2:00 Main\n10:00 Outro\n20:00 End")
    mock_store = MagicMock()
    gw = _make_gateway()

    from ingestion.worker_lambda import process_video
    process_video("vid1", "ch1", conn, None, mock_store, gw)

    mock_store.upsert.assert_called()
    call_kwargs = mock_store.upsert.call_args.kwargs
    assert len(call_kwargs["ids"]) > 0
    assert all(m.get("source_type") == "youtube_video" for m in call_kwargs["metadatas"])
    assert all("primary_topic" in m for m in call_kwargs["metadatas"])


def test_local_s3_path_writes_to_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("S3_LOCAL_PATH", str(tmp_path))

    import importlib

    from ingestion import worker_lambda
    importlib.reload(worker_lambda)

    conn, cur = _make_db(description="0:00 Intro\n2:00 Main\n10:00 Outro\n20:00 End")
    mock_store = MagicMock()
    gw = _make_gateway()
    fake_transcript = _fake_transcript()

    with patch.object(worker_lambda, "YouTubeTranscriptApi") as mock_api_cls, \
         patch.object(worker_lambda, "fetch_sponsor_segments", return_value=[]), \
         patch.object(worker_lambda, "get_topic_names", return_value=["consciousness"]), \
         patch.object(worker_lambda, "get_channel_default_topic", return_value="consciousness"):
        mock_api_cls.return_value.fetch.return_value = fake_transcript
        worker_lambda.process_video("vid1", "ch1", conn, None, mock_store, gw)

    saved_files = list(tmp_path.rglob("*.json"))
    assert len(saved_files) >= 1
