"""Unit tests for core/channel_discovery.py."""
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _search_channels
# ---------------------------------------------------------------------------

def test_search_channels_returns_channel_ids():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "items": [
            {"snippet": {"channelId": "UC111"}},
            {"snippet": {"channelId": "UC222"}},
        ]
    }
    with patch("core.channel_discovery.httpx.get", return_value=mock_resp):
        from core.channel_discovery import _search_channels
        result = _search_channels("Graham Hancock", "fake_key")
    assert result == ["UC111", "UC222"]


def test_search_channels_returns_empty_on_no_results():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"items": []}
    with patch("core.channel_discovery.httpx.get", return_value=mock_resp):
        from core.channel_discovery import _search_channels
        result = _search_channels("Nobody Famous", "fake_key")
    assert result == []


# ---------------------------------------------------------------------------
# _get_channel_details
# ---------------------------------------------------------------------------

def test_get_channel_details_sorted_by_subscriber_count():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "items": [
            {
                "id": "UC111",
                "snippet": {"title": "Small Ch", "customUrl": "@small"},
                "statistics": {"subscriberCount": "1000"},
            },
            {
                "id": "UC222",
                "snippet": {"title": "Big Ch", "customUrl": "@big"},
                "statistics": {"subscriberCount": "5000000"},
            },
        ]
    }
    with patch("core.channel_discovery.httpx.get", return_value=mock_resp):
        from core.channel_discovery import _get_channel_details
        result = _get_channel_details(["UC111", "UC222"], "fake_key")
    assert result[0]["id"] == "UC222"
    assert result[0]["subscriber_count"] == 5_000_000
    assert result[1]["id"] == "UC111"


def test_get_channel_details_caps_at_three():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "items": [
            {
                "id": f"UC{i:03}",
                "snippet": {"title": f"Ch{i}", "customUrl": f"@ch{i}"},
                "statistics": {"subscriberCount": str(i * 100)},
            }
            for i in range(1, 6)  # 5 items
        ]
    }
    with patch("core.channel_discovery.httpx.get", return_value=mock_resp):
        from core.channel_discovery import _get_channel_details
        result = _get_channel_details([f"UC{i:03}" for i in range(1, 6)], "fake_key")
    assert len(result) == 3


def test_get_channel_details_empty_input_skips_api():
    with patch("core.channel_discovery.httpx.get") as mock_get:
        from core.channel_discovery import _get_channel_details
        result = _get_channel_details([], "fake_key")
    assert result == []
    mock_get.assert_not_called()


def test_get_channel_details_missing_subscriber_count_defaults_to_zero():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "items": [{"id": "UC999", "snippet": {"title": "Hidden", "customUrl": "@h"}, "statistics": {}}]
    }
    with patch("core.channel_discovery.httpx.get", return_value=mock_resp):
        from core.channel_discovery import _get_channel_details
        result = _get_channel_details(["UC999"], "fake_key")
    assert result[0]["subscriber_count"] == 0


# ---------------------------------------------------------------------------
# _get_existing_channel_ids
# ---------------------------------------------------------------------------

def test_get_existing_channel_ids_returns_matching_ids():
    db_conn = MagicMock()
    cur = db_conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [("UC111",), ("UC333",)]

    from core.channel_discovery import _get_existing_channel_ids
    result = _get_existing_channel_ids(db_conn, ["UC111", "UC222", "UC333"])
    assert result == {"UC111", "UC333"}


def test_get_existing_channel_ids_empty_input_skips_query():
    db_conn = MagicMock()

    from core.channel_discovery import _get_existing_channel_ids
    result = _get_existing_channel_ids(db_conn, [])
    assert result == set()
    db_conn.cursor.assert_not_called()


# ---------------------------------------------------------------------------
# discover_guest_channels — top-level integration
# ---------------------------------------------------------------------------

def test_discover_returns_zero_when_no_api_key():
    db_conn = MagicMock()

    from core.channel_discovery import discover_guest_channels
    count = discover_guest_channels(["Guest A"], "vid123", db_conn, "")
    assert count == 0
    db_conn.cursor.assert_not_called()


def test_discover_returns_zero_when_no_guests():
    db_conn = MagicMock()

    from core.channel_discovery import discover_guest_channels
    count = discover_guest_channels([], "vid123", db_conn, "key123")
    assert count == 0
    db_conn.cursor.assert_not_called()


def test_discover_skips_when_all_results_already_in_db():
    db_conn = MagicMock()

    search_resp = MagicMock()
    search_resp.json.return_value = {"items": [{"snippet": {"channelId": "UC111"}}]}

    existing_cur = db_conn.cursor.return_value.__enter__.return_value
    existing_cur.fetchall.return_value = [("UC111",)]  # already in DB

    with patch("core.channel_discovery.httpx.get", return_value=search_resp):
        from core.channel_discovery import discover_guest_channels
        count = discover_guest_channels(["Graham Hancock"], "vid123", db_conn, "key123")

    assert count == 0
    db_conn.commit.assert_not_called()


def test_discover_inserts_new_candidate_and_commits():
    db_conn = MagicMock()

    search_resp = MagicMock()
    search_resp.json.return_value = {"items": [{"snippet": {"channelId": "UCnew1"}}]}

    details_resp = MagicMock()
    details_resp.json.return_value = {
        "items": [{
            "id": "UCnew1",
            "snippet": {"title": "New Guest Channel", "customUrl": "@newguest"},
            "statistics": {"subscriberCount": "250000"},
        }]
    }

    existing_cur = db_conn.cursor.return_value.__enter__.return_value
    existing_cur.fetchall.return_value = []  # not in DB
    existing_cur.rowcount = 1  # simulate a successful INSERT (not skipped by ON CONFLICT)

    with patch("core.channel_discovery.httpx.get", side_effect=[search_resp, details_resp]):
        from core.channel_discovery import discover_guest_channels
        count = discover_guest_channels(["New Guest"], "vid123", db_conn, "key123")

    assert count == 1
    db_conn.commit.assert_called_once()


def test_discover_logs_and_continues_on_api_error():
    db_conn = MagicMock()

    with patch("core.channel_discovery.httpx.get", side_effect=Exception("API down")):
        from core.channel_discovery import discover_guest_channels
        # Must not raise — errors are swallowed
        count = discover_guest_channels(["Guest A", "Guest B"], "vid123", db_conn, "key123")

    assert count == 0


# ---------------------------------------------------------------------------
# enqueue_guests
# ---------------------------------------------------------------------------

def test_enqueue_guests_inserts_rows():
    db_conn = MagicMock()
    cur = db_conn.cursor.return_value.__enter__.return_value
    cur.rowcount = 1

    from core.channel_discovery import enqueue_guests
    count = enqueue_guests(["Darius J Wright", "Joe Rogan"], "vid1", "ch1", db_conn)

    assert count == 2
    db_conn.commit.assert_called_once()


def test_enqueue_guests_returns_zero_for_empty_list():
    db_conn = MagicMock()

    from core.channel_discovery import enqueue_guests
    count = enqueue_guests([], "vid1", "ch1", db_conn)

    assert count == 0
    db_conn.cursor.assert_not_called()


def test_enqueue_guests_swallows_db_errors():
    db_conn = MagicMock()
    db_conn.cursor.side_effect = Exception("DB down")

    from core.channel_discovery import enqueue_guests
    count = enqueue_guests(["Guest A"], "vid1", "ch1", db_conn)

    assert count == 0


# ---------------------------------------------------------------------------
# process_discovery_queue
# ---------------------------------------------------------------------------

def test_process_discovery_queue_returns_zero_without_api_key():
    db_conn = MagicMock()

    from core.channel_discovery import process_discovery_queue
    count = process_discovery_queue(db_conn, "", max_guests=5)

    assert count == 0
    db_conn.cursor.assert_not_called()


def test_process_discovery_queue_discovers_and_updates_status():
    db_conn = MagicMock()

    # fetchall returns one pending guest row
    pending_cur = MagicMock()
    pending_cur.fetchall.return_value = [
        (1, "Darius J Wright", "vid1", "ch1"),
    ]

    # subsequent cursors handle _get_existing_channel_ids and _insert_candidate
    existing_cur = MagicMock()
    existing_cur.fetchall.return_value = []  # not in DB
    existing_cur.rowcount = 1

    call_count = [0]
    def cursor_factory(*args, **kwargs):
        ctx = MagicMock()
        call_count[0] += 1
        if call_count[0] == 1:
            ctx.__enter__ = MagicMock(return_value=pending_cur)
        else:
            ctx.__enter__ = MagicMock(return_value=existing_cur)
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    db_conn.cursor.side_effect = cursor_factory

    search_resp = MagicMock()
    search_resp.json.return_value = {"items": [{"snippet": {"channelId": "UCnew1"}}]}

    details_resp = MagicMock()
    details_resp.json.return_value = {
        "items": [{
            "id": "UCnew1",
            "snippet": {"title": "Darius Channel", "customUrl": "@darius"},
            "statistics": {"subscriberCount": "100000"},
        }]
    }

    with patch("core.channel_discovery.httpx.get", side_effect=[search_resp, details_resp]):
        from core.channel_discovery import process_discovery_queue
        count = process_discovery_queue(db_conn, "fake_key", max_guests=5)

    assert count == 1


def test_process_discovery_queue_stops_on_quota_error():
    db_conn = MagicMock()

    pending_cur = MagicMock()
    pending_cur.fetchall.return_value = [
        (1, "Guest A", "vid1", "ch1"),
        (2, "Guest B", "vid1", "ch1"),
    ]

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=pending_cur)
    ctx.__exit__ = MagicMock(return_value=False)
    db_conn.cursor.return_value = ctx

    quota_err = Exception("quotaExceeded: API quota exceeded")
    with patch("core.channel_discovery.httpx.get", side_effect=quota_err):
        from core.channel_discovery import process_discovery_queue
        count = process_discovery_queue(db_conn, "fake_key", max_guests=5)

    # Stopped after first guest; no channels inserted
    assert count == 0


def test_discover_processes_multiple_guests_independently():
    db_conn = MagicMock()

    def make_search_resp(channel_id):
        r = MagicMock()
        r.json.return_value = {"items": [{"snippet": {"channelId": channel_id}}]}
        return r

    def make_details_resp(channel_id, name):
        r = MagicMock()
        r.json.return_value = {
            "items": [{
                "id": channel_id,
                "snippet": {"title": name, "customUrl": f"@{name.lower()}"},
                "statistics": {"subscriberCount": "100000"},
            }]
        }
        return r

    existing_cur = db_conn.cursor.return_value.__enter__.return_value
    existing_cur.fetchall.return_value = []  # nothing in DB
    existing_cur.rowcount = 1  # simulate successful INSERT for each candidate

    with patch("core.channel_discovery.httpx.get", side_effect=[
        make_search_resp("UCa"), make_details_resp("UCa", "Guest A"),
        make_search_resp("UCb"), make_details_resp("UCb", "Guest B"),
    ]):
        from core.channel_discovery import discover_guest_channels
        count = discover_guest_channels(["Guest A", "Guest B"], "vid123", db_conn, "key123")

    assert count == 2
    assert db_conn.commit.call_count == 2
