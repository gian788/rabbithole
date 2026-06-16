import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from unittest.mock import MagicMock
from tests.conftest import parse_sse_events
from .conftest import FAKE_MATCHES, FAKE_VIDEO_META


def _setup(app_client, matches=FAKE_MATCHES, video_meta=FAKE_VIDEO_META):
    client, mock_db, mock_index, mock_gateway, main_mod = app_client
    mock_index.query.return_value = {"matches": matches}
    main_mod._fetch_video_meta = MagicMock(return_value=video_meta)
    main_mod._reranker.predict = MagicMock(return_value=[1.0] * len(matches))

    ctx = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = ("conv-stream-id",)
    cur.fetchall.return_value = []
    ctx.__enter__ = MagicMock(return_value=cur)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_db.cursor.return_value = ctx

    mock_gateway.stream_completion.return_value = iter(["According ", "to Joe Dispenza, ", "consciousness is..."])
    return client, mock_db, mock_gateway


def test_stream_content_type(app_client):
    client, *_ = _setup(app_client)
    resp = client.post("/v1/chat", json={"query": "What is consciousness?", "stream": True})
    assert "text/event-stream" in resp.headers["content-type"]


def test_stream_all_events_have_type(app_client):
    client, *_ = _setup(app_client)
    resp = client.post("/v1/chat", json={"query": "What is consciousness?", "stream": True})
    events = parse_sse_events(resp.text)
    assert len(events) > 0
    for event in events:
        assert "type" in event


def test_stream_has_token_events(app_client):
    client, *_ = _setup(app_client)
    resp = client.post("/v1/chat", json={"query": "What is DMT?", "stream": True})
    events = parse_sse_events(resp.text)
    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) > 0
    for te in token_events:
        assert "content" in te


def test_stream_last_event_is_done(app_client):
    client, *_ = _setup(app_client)
    resp = client.post("/v1/chat", json={"query": "Tell me about awareness", "stream": True})
    events = parse_sse_events(resp.text)
    assert events[-1]["type"] == "done"


def test_stream_done_event_has_required_fields(app_client):
    client, *_ = _setup(app_client)
    resp = client.post("/v1/chat", json={"query": "What is consciousness?", "stream": True})
    events = parse_sse_events(resp.text)
    done = events[-1]
    assert done["type"] == "done"
    assert "answer" in done
    assert "topic" in done
    assert "sources" in done
    assert "conversation_id" in done


def test_stream_done_answer_matches_tokens(app_client):
    client, *_ = _setup(app_client)
    resp = client.post("/v1/chat", json={"query": "What is consciousness?", "stream": True})
    events = parse_sse_events(resp.text)
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    done_answer = next(e for e in events if e["type"] == "done")["answer"]
    assert tokens == done_answer


def test_stream_no_matches_single_done_event(app_client):
    client, mock_db, mock_index, mock_gateway, main_mod = app_client
    mock_index.query.return_value = {"matches": []}

    ctx = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = ("conv-empty",)
    cur.fetchall.return_value = []
    ctx.__enter__ = MagicMock(return_value=cur)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_db.cursor.return_value = ctx

    resp = client.post("/v1/chat", json={"query": "nonsense xyzzy", "stream": True})
    events = parse_sse_events(resp.text)
    assert len(events) == 1
    assert events[0]["type"] == "done"
    assert events[0]["sources"] == []


def test_stream_save_message_called_after_stream(app_client):
    client, mock_db, mock_gateway = _setup(app_client)
    mock_db.reset_mock()
    client.post("/v1/chat", json={"query": "What is consciousness?", "stream": True})
    # save_message is implemented as cursor().execute() calls — verify commit was called
    assert mock_db.commit.called
