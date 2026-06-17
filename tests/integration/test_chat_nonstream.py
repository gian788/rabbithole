from unittest.mock import MagicMock

from .conftest import FAKE_MATCHES, FAKE_VIDEO_META

KNOWN_TOPICS = {"consciousness", "biohacking", "spirituality", "alternative_history"}


def _setup_matches(app_client, matches=FAKE_MATCHES, video_meta=FAKE_VIDEO_META):
    """Configure mock index and DB to return the provided matches and video metadata."""
    client, mock_db, mock_index, mock_gateway, main_mod = app_client
    mock_index.query.return_value = {"matches": matches}

    # Patch _fetch_video_meta to return our fake lookup
    main_mod._fetch_video_meta = MagicMock(return_value=video_meta)

    # Mock DB cursor for create_conversation / save_message / update_conversation
    ctx = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = ("conv-test-id",)
    cur.fetchall.return_value = []
    ctx.__enter__ = MagicMock(return_value=cur)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_db.cursor.return_value = ctx

    # Mock reranker to return matches in order
    import retrieval.main as m
    m._reranker.predict = MagicMock(return_value=[1.0] * len(matches))

    return client


def test_empty_query_returns_400(app_client):
    client, *_ = app_client
    resp = client.post("/v1/chat", json={"query": ""})
    assert resp.status_code == 400


def test_no_matches_returns_graceful_message(app_client):
    client, mock_db, mock_index, mock_gateway, main_mod = app_client
    mock_index.query.return_value = {"matches": []}

    ctx = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = ("conv-id",)
    cur.fetchall.return_value = []
    ctx.__enter__ = MagicMock(return_value=cur)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_db.cursor.return_value = ctx

    resp = client.post("/v1/chat", json={"query": "What is DMT?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "could not find" in body["answer"].lower()
    assert body["sources"] == []


def test_with_matches_returns_valid_structure(app_client):
    client = _setup_matches(app_client)
    resp = client.post("/v1/chat", json={"query": "What is consciousness?"})
    assert resp.status_code == 200
    body = resp.json()

    assert "answer" in body
    assert "topic" in body
    assert "sources" in body
    assert "conversation_id" in body


def test_topic_is_known(app_client):
    client = _setup_matches(app_client)
    resp = client.post("/v1/chat", json={"query": "What is consciousness?"})
    assert resp.json()["topic"] in KNOWN_TOPICS


def test_sources_structure(app_client):
    client = _setup_matches(app_client)
    resp = client.post("/v1/chat", json={"query": "What is consciousness?"})
    sources = resp.json()["sources"]
    assert len(sources) > 0
    for src in sources:
        assert "video_id" in src
        assert "title" in src
        assert "channel" in src
        assert "speaker" in src
        assert "clips" in src
        assert len(src["clips"]) > 0
        for clip in src["clips"]:
            assert "chapter" in clip
            assert clip["url"].startswith("https://youtu.be/")
            assert isinstance(clip["start_seconds"], int)
            assert clip["start_seconds"] >= 0


def test_max_two_clips_per_source(app_client):
    # Three matches from the same video — clips should be capped at 2
    three_same = [
        {"metadata": {
            "video_id": "vid1", "chapter": f"Ch{i}", "start_seconds": i * 30,
            "deep_link": f"https://youtu.be/vid1?t={i*30}",
            "text_content": f"content chunk {i}", "topics": ["consciousness"],
        }}
        for i in range(3)
    ]
    client = _setup_matches(app_client, matches=three_same, video_meta={
        "vid1": {"id": "vid1", "title": "Show | Speaker", "channel_name": "Pod"},
    })
    # Reranker needs to return correct number of scores
    import retrieval.main as m
    m._reranker.predict = MagicMock(return_value=[1.0, 0.9, 0.8])

    resp = client.post("/v1/chat", json={"query": "What is consciousness?"})
    sources = resp.json()["sources"]
    for src in sources:
        if src["video_id"] == "vid1":
            assert len(src["clips"]) <= 2


def test_conversation_id_in_response(app_client):
    client = _setup_matches(app_client)
    resp = client.post("/v1/chat", json={"query": "Tell me about awareness"})
    assert resp.json()["conversation_id"] == "conv-test-id"


def test_stream_false_returns_json(app_client):
    client = _setup_matches(app_client)
    resp = client.post("/v1/chat", json={"query": "Consciousness?", "stream": False})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
