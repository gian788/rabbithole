import psycopg2.extras
from unittest.mock import MagicMock


def test_health(app_client):
    client, *_ = app_client
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["timestamp"], float)


def test_create_conversation(app_client):
    client, mock_db, *_ = app_client
    # Mock DB returns a UUID string for the new conversation id
    ctx = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = ("conv-uuid-123",)
    ctx.__enter__ = MagicMock(return_value=cur)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_db.cursor.return_value = ctx

    resp = client.post("/v1/conversations", json={"session_id": "s1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["conversation_id"] == "conv-uuid-123"
    assert body["messages"] == []


def test_create_conversation_default_session(app_client):
    client, mock_db, *_ = app_client
    ctx = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = ("conv-anon",)
    ctx.__enter__ = MagicMock(return_value=cur)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_db.cursor.return_value = ctx

    resp = client.post("/v1/conversations", json={})
    assert resp.status_code == 200


def test_get_messages(app_client):
    from datetime import datetime
    client, mock_db, *_ = app_client

    ctx = MagicMock()
    cur = MagicMock()
    fake_rows = [
        {"role": "user",      "content": "Hello",   "citations": None, "created_at": datetime(2024, 1, 1, 12, 0)},
        {"role": "assistant", "content": "Hi there", "citations": None, "created_at": datetime(2024, 1, 1, 12, 1)},
    ]
    cur.fetchall.return_value = list(reversed(fake_rows))  # DB returns DESC, helper reverses
    ctx.__enter__ = MagicMock(return_value=cur)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_db.cursor.return_value = ctx

    resp = client.get("/v1/conversations/conv-123/messages")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["messages"]) == 2
    # DB returns DESC (assistant at 12:01, user at 12:00); helper reverses → chronological
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][1]["role"] == "assistant"
