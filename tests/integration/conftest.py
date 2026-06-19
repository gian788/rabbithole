import importlib
from unittest.mock import MagicMock, patch

import pytest

FAKE_MATCHES = [
    {"metadata": {
        "source_type": "youtube_video",
        "video_id": "vid1",
        "chapter": "Intro",
        "start_seconds": 30,
        "deep_link": "https://youtu.be/vid1?t=30",
        "text_content": "Joe Dispenza on consciousness and the mind-body connection.",
        "topics": ["consciousness"],
    }},
    {"metadata": {
        "source_type": "youtube_video",
        "video_id": "vid2",
        "chapter": "Main",
        "start_seconds": 120,
        "deep_link": "https://youtu.be/vid2?t=120",
        "text_content": "Another perspective on awareness from the same tradition.",
        "topics": ["consciousness"],
    }},
]

FAKE_VIDEO_META = {
    "vid1": {"id": "vid1", "title": "Mind over Matter | Joe Dispenza", "channel_name": "AMP"},
    "vid2": {"id": "vid2", "title": "Awareness Unpacked | Sean Carroll",  "channel_name": "AMP"},
}


@pytest.fixture
def mock_db():
    conn = MagicMock()
    ctx = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = None
    cur.fetchall.return_value = []
    ctx.__enter__ = MagicMock(return_value=cur)
    ctx.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = ctx
    return conn


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.query.return_value = []
    return store


@pytest.fixture
def mock_gateway():
    gw = MagicMock()
    gw.get_embedding.return_value = MagicMock(
        embedding_vector=[0.1] * 1536, cost=0.0001, input_tokens=10
    )
    gw.get_completion.return_value = MagicMock(
        text_content="Joe Dispenza describes consciousness as...", cost=0.0005
    )
    gw.stream_completion.return_value = iter(["According ", "to Joe Dispenza..."])
    return gw


@pytest.fixture
def app_client(mock_db, mock_store, mock_gateway, monkeypatch):
    """
    Build a TestClient with all external dependencies patched.

    Strategy:
    1. Set dummy env vars so the lifespan can read them without error.
    2. Patch module-level init (CrossEncoder) at source so it survives reload.
    3. After reload, monkeypatch service references on the reloaded module object
       (get_connection, get_vector_store, ModelGateway) — set BEFORE TestClient
       enters and fires the lifespan.
    4. After lifespan runs, set _topic_vectors so _nearest_topic works.
    """
    monkeypatch.setenv("DATABASE_URL",       "postgresql://test:test@localhost/test")
    monkeypatch.setenv("OPENAI_API_KEY",     "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY",  "test-anthropic-key")

    with patch("sentence_transformers.CrossEncoder", return_value=MagicMock()):
        import retrieval.main as m
        importlib.reload(m)

        monkeypatch.setattr(m, "get_connection",   lambda: mock_db)
        monkeypatch.setattr(m, "get_vector_store", lambda: mock_store)
        monkeypatch.setattr(m, "ModelGateway",     MagicMock(return_value=mock_gateway))

        # Override auth dependencies so tests don't need real JWTs
        m.app.dependency_overrides[m.get_current_user] = lambda: "test-user-id"
        m.app.dependency_overrides[m.optional_current_user] = lambda: "test-user-id"

        from fastapi.testclient import TestClient
        with TestClient(m.app) as client:
            monkeypatch.setattr(m, "_topic_vectors", {
                "consciousness": [1.0] + [0.0] * 1535,
                "biohacking":    [0.0, 1.0] + [0.0] * 1534,
            })
            yield client, mock_db, mock_store, mock_gateway, m

        m.app.dependency_overrides.clear()
