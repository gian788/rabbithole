# Adjacent Channel Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically discover YouTube channels from podcast guest names during video ingestion and surface them in the dashboard for operator approval before they enter the scraping queue.

**Architecture:** A new `core/channel_discovery.py` module searches the YouTube Data API for guest names extracted by the existing `classify_video_meta` call, inserting up to 3 candidates per guest (ranked by subscriber count) as unapproved rows in the `channels` table. Five new columns on `channels` (and two on `websites` for future parity) track approval state and discovery provenance. The dashboard gains a "Pending Approvals" section where the operator approves (immediately activates) or rejects (kept to prevent re-discovery) each candidate.

**Tech Stack:** Python 3.12, psycopg2, httpx (YouTube Data API v3 over raw HTTP), Streamlit

## Global Constraints

- Python ≥ 3.12 (project standard)
- All SQL changes use `IF NOT EXISTS` / `ON CONFLICT (id) DO NOTHING` for idempotency
- `YOUTUBE_API_KEY` env var: discovery is skipped silently when absent — never raises
- Per-guest discovery failures are logged and swallowed; they never affect main video processing
- Backfill: all pre-existing `channels` rows → `is_approved=TRUE, source='manual'`; same for `websites`
- Existing channels already in the DB (any state) are never re-inserted — log and skip
- Up to 3 candidates per guest, ordered by subscriber count descending
- Approved channels: `is_approved=TRUE, is_active=TRUE` (both set at approval time)
- Rejected channels: `is_rejected=TRUE` — row kept so future discoveries skip the same channel ID
- Manually registered channels (sidebar form) must also set `is_approved=TRUE, source='manual'`

---

### Task 1: Schema Migration

**Files:**
- Create: `migrations/001_channel_discovery.sql`
- Modify: `schema.sql`

**Interfaces:**
- Produces: `channels` table with `is_approved BOOLEAN DEFAULT FALSE`, `is_rejected BOOLEAN DEFAULT FALSE`, `source VARCHAR(20) DEFAULT 'manual'`, `discovered_from_video_id VARCHAR(50)`, `discovered_guest_name VARCHAR(255)`, `subscriber_count BIGINT`
- Produces: `websites` table with `is_approved BOOLEAN DEFAULT FALSE`, `source VARCHAR(20) DEFAULT 'manual'`

- [ ] **Step 1: Create the migration file**

Create `migrations/001_channel_discovery.sql`:

```sql
-- Migration 001: Add channel discovery support
-- Safe to re-run: all statements use IF NOT EXISTS / WHERE guards.

ALTER TABLE channels
  ADD COLUMN IF NOT EXISTS is_approved              BOOLEAN     DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS is_rejected              BOOLEAN     DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS source                   VARCHAR(20) DEFAULT 'manual',
  ADD COLUMN IF NOT EXISTS discovered_from_video_id VARCHAR(50),
  ADD COLUMN IF NOT EXISTS discovered_guest_name    VARCHAR(255),
  ADD COLUMN IF NOT EXISTS subscriber_count         BIGINT;

-- All rows that existed before this migration were manually registered.
UPDATE channels SET is_approved = TRUE WHERE is_approved = FALSE AND source = 'manual';

ALTER TABLE websites
  ADD COLUMN IF NOT EXISTS is_approved BOOLEAN     DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS source      VARCHAR(20) DEFAULT 'manual';

UPDATE websites SET is_approved = TRUE WHERE is_approved = FALSE AND source = 'manual';
```

- [ ] **Step 2: Apply migration to the local / dev database**

```bash
psql $DATABASE_URL -f migrations/001_channel_discovery.sql
```

Expected output (no errors):
```
ALTER TABLE
UPDATE N
ALTER TABLE
UPDATE M
```

- [ ] **Step 3: Update `schema.sql` to reflect the final table shape**

Replace the `CREATE TABLE channels` block with:

```sql
CREATE TABLE channels (
    id                       VARCHAR(50) PRIMARY KEY,
    name                     VARCHAR(255) NOT NULL,
    handle                   VARCHAR(100),
    uploads_playlist_id      VARCHAR(50)  NOT NULL,
    default_topic_id         INT REFERENCES topics(id) ON DELETE SET NULL,
    videos_to_fetch          INT DEFAULT 10,
    max_videos               INT DEFAULT 100,
    is_active                BOOLEAN DEFAULT TRUE,
    is_approved              BOOLEAN DEFAULT FALSE,
    is_rejected              BOOLEAN DEFAULT FALSE,
    source                   VARCHAR(20) DEFAULT 'manual',
    discovered_from_video_id VARCHAR(50),
    discovered_guest_name    VARCHAR(255),
    subscriber_count         BIGINT,
    last_checked_at          TIMESTAMP WITH TIME ZONE,
    created_at               TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

Replace the `CREATE TABLE websites` block with:

```sql
CREATE TABLE websites (
    id                VARCHAR(100) PRIMARY KEY,
    name              VARCHAR(255) NOT NULL,
    base_url          TEXT NOT NULL,
    rss_url           TEXT,
    default_topic_id  INT REFERENCES topics(id) ON DELETE SET NULL,
    articles_to_fetch INT DEFAULT 10,
    max_articles      INT DEFAULT 100,
    is_active         BOOLEAN DEFAULT TRUE,
    is_approved       BOOLEAN DEFAULT FALSE,
    source            VARCHAR(20) DEFAULT 'manual',
    last_checked_at   TIMESTAMP WITH TIME ZONE,
    created_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

- [ ] **Step 4: Commit**

```bash
git add migrations/001_channel_discovery.sql schema.sql
git commit -m "feat: schema migration — add is_approved/is_rejected/source columns for channel discovery"
```

---

### Task 2: Channel Discovery Module

**Files:**
- Create: `core/channel_discovery.py`
- Create: `tests/unit/test_channel_discovery.py`

**Interfaces:**
- Consumes: `db_conn` (psycopg2 connection), `youtube_api_key: str`, `guest_names: list[str]`, `source_video_id: str`
- Produces: `discover_guest_channels(guest_names, source_video_id, db_conn, youtube_api_key) -> int` — returns count of newly inserted candidates

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_channel_discovery.py`:

```python
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

    with patch("core.channel_discovery.httpx.get", side_effect=[
        make_search_resp("UCa"), make_details_resp("UCa", "Guest A"),
        make_search_resp("UCb"), make_details_resp("UCb", "Guest B"),
    ]):
        from core.channel_discovery import discover_guest_channels
        count = discover_guest_channels(["Guest A", "Guest B"], "vid123", db_conn, "key123")

    assert count == 2
    assert db_conn.commit.call_count == 2
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/unit/test_channel_discovery.py -v 2>&1 | head -20
```

Expected: `ImportError` — `No module named 'core.channel_discovery'`

- [ ] **Step 3: Implement `core/channel_discovery.py`**

```python
"""core/channel_discovery.py — Discover YouTube channels from podcast guest names."""
import logging

import httpx

logger = logging.getLogger(__name__)

_SEARCH_URL   = "https://www.googleapis.com/youtube/v3/search"
_CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
_MAX_CANDIDATES = 3


def _search_channels(name: str, api_key: str) -> list[str]:
    """Return up to 5 YouTube channel IDs matching name (YouTube relevance order)."""
    resp = httpx.get(
        _SEARCH_URL,
        params={"part": "snippet", "type": "channel", "q": name, "maxResults": 5, "key": api_key},
        timeout=10,
    )
    resp.raise_for_status()
    return [item["snippet"]["channelId"] for item in resp.json().get("items", [])]


def _get_channel_details(channel_ids: list[str], api_key: str) -> list[dict]:
    """Fetch snippet + statistics for the given channel IDs.

    Returns up to _MAX_CANDIDATES results sorted by subscriber count descending.
    """
    if not channel_ids:
        return []
    resp = httpx.get(
        _CHANNELS_URL,
        params={"part": "snippet,statistics", "id": ",".join(channel_ids), "key": api_key},
        timeout=10,
    )
    resp.raise_for_status()
    results = []
    for item in resp.json().get("items", []):
        results.append({
            "id":               item["id"],
            "name":             item["snippet"].get("title", ""),
            "handle":           item["snippet"].get("customUrl") or None,
            "subscriber_count": int(item.get("statistics", {}).get("subscriberCount", 0)),
        })
    results.sort(key=lambda x: x["subscriber_count"], reverse=True)
    return results[:_MAX_CANDIDATES]


def _get_existing_channel_ids(db_conn, channel_ids: list[str]) -> set[str]:
    """Return the subset of channel_ids already in channels table (any state)."""
    if not channel_ids:
        return set()
    with db_conn.cursor() as cur:
        cur.execute("SELECT id FROM channels WHERE id = ANY(%s)", (channel_ids,))
        return {row[0] for row in cur.fetchall()}


def _insert_candidate(
    db_conn, channel: dict, source_video_id: str, guest_name: str
) -> None:
    uploads_playlist_id = "UU" + channel["id"][2:]
    with db_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO channels
                (id, name, handle, uploads_playlist_id,
                 is_active, is_approved, is_rejected,
                 source, discovered_from_video_id, discovered_guest_name, subscriber_count)
            VALUES (%s, %s, %s, %s, FALSE, FALSE, FALSE, 'discovered', %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (
                channel["id"],
                channel["name"],
                channel["handle"],
                uploads_playlist_id,
                source_video_id,
                guest_name,
                channel["subscriber_count"],
            ),
        )


def discover_guest_channels(
    guest_names: list[str],
    source_video_id: str,
    db_conn,
    youtube_api_key: str,
) -> int:
    """Search YouTube for channels matching each guest name; insert unapproved candidates.

    Skips guests whose search results are all already in the channels table.
    Swallows all errors — never raises.
    Returns count of newly inserted candidates.
    """
    if not guest_names or not youtube_api_key:
        return 0

    inserted = 0
    for name in guest_names:
        try:
            channel_ids = _search_channels(name, youtube_api_key)
            if not channel_ids:
                logger.info("channel_discovery: no YouTube results for guest %r", name)
                continue

            existing = _get_existing_channel_ids(db_conn, channel_ids)
            new_ids = [cid for cid in channel_ids if cid not in existing]

            if not new_ids:
                logger.info(
                    "channel_discovery: all %d results for %r already in channels table",
                    len(channel_ids), name,
                )
                continue

            candidates = _get_channel_details(new_ids, youtube_api_key)
            for channel in candidates:
                _insert_candidate(db_conn, channel, source_video_id, name)
                inserted += 1
            db_conn.commit()

        except Exception as exc:
            logger.warning("channel_discovery: failed for guest %r: %s", name, exc)
            try:
                db_conn.rollback()
            except Exception:
                pass

    return inserted
```

- [ ] **Step 4: Run tests and confirm they all pass**

```bash
uv run pytest tests/unit/test_channel_discovery.py -v
```

Expected: all 14 tests pass.

- [ ] **Step 5: Commit**

```bash
git add core/channel_discovery.py tests/unit/test_channel_discovery.py
git commit -m "feat: add channel_discovery module — YouTube guest channel lookup via Data API"
```

---

### Task 3: Worker Integration

**Files:**
- Modify: `ingestion/worker_lambda.py` — add import + 4-line discovery call
- Modify: `tests/integration/test_worker_process_video.py` — add two tests

**Interfaces:**
- Consumes: `discover_guest_channels` from `core.channel_discovery` (Task 2)
- Consumes: `video_meta.guests` (list[str]) already returned by `classify_video_meta`
- Consumes: `YOUTUBE_API_KEY` env var

- [ ] **Step 1: Write the failing tests**

Append to `tests/integration/test_worker_process_video.py`:

```python
@patch("ingestion.worker_lambda.fetch_sponsor_segments", return_value=[])
@patch("ingestion.worker_lambda.YouTubeTranscriptApi")
@patch("ingestion.worker_lambda.get_topic_names", return_value=["consciousness"])
@patch("ingestion.worker_lambda.get_channel_default_topic", return_value="consciousness")
@patch("ingestion.worker_lambda._save_payload", return_value="local/path")
@patch("ingestion.worker_lambda.discover_guest_channels")
def test_process_video_calls_discovery_when_api_key_set(
    mock_discover, mock_save, mock_topic, mock_names, mock_api_cls, mock_sponsor,
    monkeypatch,
):
    monkeypatch.setenv("YOUTUBE_API_KEY", "test_key")
    mock_api_cls.return_value.fetch.return_value = _fake_transcript()
    conn, cur = _make_db()
    gw = _make_gateway(guests=["Graham Hancock"])

    from ingestion.worker_lambda import process_video
    process_video("vid1", "ch1", conn, None, MagicMock(), gw)

    mock_discover.assert_called_once_with(
        guest_names=["Graham Hancock"],
        source_video_id="vid1",
        db_conn=conn,
        youtube_api_key="test_key",
    )


@patch("ingestion.worker_lambda.fetch_sponsor_segments", return_value=[])
@patch("ingestion.worker_lambda.YouTubeTranscriptApi")
@patch("ingestion.worker_lambda.get_topic_names", return_value=["consciousness"])
@patch("ingestion.worker_lambda.get_channel_default_topic", return_value="consciousness")
@patch("ingestion.worker_lambda._save_payload", return_value="local/path")
@patch("ingestion.worker_lambda.discover_guest_channels")
def test_process_video_skips_discovery_when_no_api_key(
    mock_discover, mock_save, mock_topic, mock_names, mock_api_cls, mock_sponsor,
    monkeypatch,
):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    mock_api_cls.return_value.fetch.return_value = _fake_transcript()
    conn, cur = _make_db()
    gw = _make_gateway(guests=["Graham Hancock"])

    from ingestion.worker_lambda import process_video
    process_video("vid1", "ch1", conn, None, MagicMock(), gw)

    mock_discover.assert_not_called()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/integration/test_worker_process_video.py::test_process_video_calls_discovery_when_api_key_set -v 2>&1 | tail -10
```

Expected: `FAILED` — `AttributeError` or `assert mock_discover.called` fails because `discover_guest_channels` is not yet imported in `worker_lambda`.

- [ ] **Step 3: Add the import and discovery call to `worker_lambda.py`**

At the top of `ingestion/worker_lambda.py`, add to the imports block (after the `core.*` imports):

```python
from core.channel_discovery import discover_guest_channels
```

In `process_video`, immediately after the block that sets `video_meta = classify_video_meta(...)` and before the `with db_conn.cursor() as cur: cur.execute("UPDATE videos SET topics...")` block, add:

```python
        # -- Guest channel discovery ------------------------------------------
        api_key = os.environ.get("YOUTUBE_API_KEY", "")
        if api_key and video_meta.guests:
            discover_guest_channels(
                guest_names=video_meta.guests,
                source_video_id=video_id,
                db_conn=db_conn,
                youtube_api_key=api_key,
            )
```

- [ ] **Step 4: Run all worker tests**

```bash
uv run pytest tests/integration/test_worker_process_video.py -v
```

Expected: all tests pass (including the two new ones).

- [ ] **Step 5: Commit**

```bash
git add ingestion/worker_lambda.py tests/integration/test_worker_process_video.py
git commit -m "feat: call discover_guest_channels from worker after video meta extraction"
```

---

### Task 4: Dashboard Approval UI

**Files:**
- Modify: `dashboard/app.py`

Three changes: (a) update the sidebar channel registration INSERT to set `is_approved=TRUE, source='manual'`; (b) add `AND c.is_approved = TRUE` to the main Channels tab query; (c) add a Pending Approvals section at the top of the Channels tab.

- [ ] **Step 1: Fix the sidebar channel registration INSERT**

In `dashboard/app.py`, find the `INSERT INTO channels` statement inside the `with st.form("add_channel"):` block and replace it with:

```python
                    cur.execute(
                        """
                        INSERT INTO channels
                            (id, name, uploads_playlist_id, default_topic_id,
                             videos_to_fetch, max_videos, is_approved, source)
                        SELECT %s, %s, %s, t.id, %s, %s, TRUE, 'manual'
                        FROM topics t WHERE t.name = %s
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (
                            channel_id,
                            channel_name.strip(),
                            uploads_playlist_id,
                            int(videos_to_fetch),
                            int(max_videos),
                            default_topic,
                        ),
                    )
```

- [ ] **Step 2: Add `AND c.is_approved = TRUE` to the main Channels tab query**

In `dashboard/app.py`, find the `SELECT ... FROM channels c LEFT JOIN topics t ...` query in the `with tab_channels:` block. Add the WHERE clause so only approved channels appear in the value-attribution table:

```python
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                c.id,
                c.name,
                t.name                                                              AS default_topic,
                c.is_active,
                COUNT(DISTINCT v.id) FILTER (WHERE v.status = 'completed')         AS indexed_videos,
                COALESCE(SUM(v.ingestion_cost), 0)                                  AS total_cost,
                COUNT(DISTINCT q.id)                                                AS search_count,
                CASE WHEN COUNT(DISTINCT q.id) = 0 THEN NULL
                     ELSE COALESCE(SUM(v.ingestion_cost), 0) / COUNT(DISTINCT q.id)
                END                                                                 AS cost_per_search
            FROM channels c
            LEFT JOIN topics t      ON t.id = c.default_topic_id
            LEFT JOIN videos v      ON v.channel_id = c.id
            LEFT JOIN rag_queries q ON c.id = ANY(q.video_ids)
            WHERE c.is_approved = TRUE
            GROUP BY c.id, c.name, t.name, c.is_active
            ORDER BY total_cost DESC
            """
        )
        rows = cur.fetchall()
```

- [ ] **Step 3: Add the Pending Approvals section**

In `dashboard/app.py`, at the very start of the `with tab_channels:` block (before the `st.subheader("Channel Value Attribution")` line), add:

```python
    # ── Pending Channel Approvals ──────────────────────────────────────────
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT c.id, c.name, c.discovered_guest_name,
                   c.discovered_from_video_id, c.subscriber_count,
                   v.title AS source_video_title
            FROM channels c
            LEFT JOIN videos v ON v.id = c.discovered_from_video_id
            WHERE c.is_approved = FALSE AND c.is_rejected = FALSE
            ORDER BY c.created_at DESC
            """
        )
        pending = cur.fetchall()

    if pending:
        st.subheader(f"Pending Channel Approvals ({len(pending)})")
        hdr = st.columns([3, 2, 3, 2, 1, 1])
        for col, label in zip(hdr, ["Channel", "Guest Name", "Source Video", "Subscribers", "", ""]):
            col.markdown(f"**{label}**")
        st.divider()

        for row in pending:
            channel_url = f"https://www.youtube.com/channel/{row['id']}"
            cols = st.columns([3, 2, 3, 2, 1, 1])
            cols[0].markdown(f"[{row['name']}]({channel_url})")
            cols[1].write(row["discovered_guest_name"] or "—")
            if row["discovered_from_video_id"]:
                video_url = f"https://youtu.be/{row['discovered_from_video_id']}"
                label = row["source_video_title"] or row["discovered_from_video_id"]
                cols[2].markdown(f"[{label}]({video_url})")
            else:
                cols[2].write("—")
            subs = row["subscriber_count"]
            cols[3].write(f"{subs:,}" if subs else "—")

            if cols[4].button("Approve", key=f"approve_{row['id']}"):
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE channels SET is_approved = TRUE, is_active = TRUE WHERE id = %s",
                        (row["id"],),
                    )
                conn.commit()
                st.rerun()

            if cols[5].button("Reject", key=f"reject_{row['id']}"):
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE channels SET is_rejected = TRUE WHERE id = %s",
                        (row["id"],),
                    )
                conn.commit()
                st.rerun()

        st.divider()
```

- [ ] **Step 4: Verify the dashboard manually**

Start the dashboard:

```bash
uv run streamlit run dashboard/app.py
```

Check:
1. The Channels tab shows only approved channels in the value-attribution table.
2. The "Register Channel" sidebar form still works — registered channel appears immediately in the main table (not in Pending).
3. With a pending row in the DB (insert one manually to test): it appears in the Pending section; clicking Approve moves it to the main table and it vanishes from Pending; clicking Reject removes it from Pending and re-running the discovery for the same channel ID does not re-insert it.

To insert a test pending row manually:

```sql
INSERT INTO channels (id, name, uploads_playlist_id, is_active, is_approved, is_rejected, source, discovered_guest_name, subscriber_count)
VALUES ('UCtest123', 'Test Pending Channel', 'UUtest123', FALSE, FALSE, FALSE, 'discovered', 'Test Guest', 150000);
```

- [ ] **Step 5: Run full unit test suite to catch any regressions**

```bash
uv run pytest tests/unit/ tests/integration/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: dashboard pending approvals section for discovered channels"
```

---

## Self-Review

**Spec coverage:**
- Schema columns (`is_approved`, `is_rejected`, `source`, `discovered_from_video_id`, `discovered_guest_name`, `subscriber_count`) → Task 1 ✓
- Website parity columns → Task 1 ✓
- Backfill existing rows → Task 1 ✓
- YouTube Data API search + subscriber ranking → Task 2 (`_search_channels`, `_get_channel_details`) ✓
- Top-3 cap → Task 2 (`_MAX_CANDIDATES = 3`) ✓
- Skip existing channel IDs (any state) → Task 2 (`_get_existing_channel_ids`) ✓
- Log-but-skip behavior for existing channels → Task 2 (`logger.info` in `discover_guest_channels`) ✓
- Inline in worker after `classify_video_meta` → Task 3 ✓
- Skip when `YOUTUBE_API_KEY` not set → Task 2 + Task 3 (`if not youtube_api_key: return 0`) ✓
- Discovery failures swallowed → Task 2 (`except Exception` block) ✓
- Sidebar INSERT sets `is_approved=TRUE, source='manual'` → Task 4 Step 1 ✓
- Main channel list filtered to approved only → Task 4 Step 2 ✓
- Pending Approvals section in dashboard → Task 4 Step 3 ✓
- Approve sets `is_approved=TRUE, is_active=TRUE` → Task 4 Step 3 ✓
- Reject sets `is_rejected=TRUE` (row kept) → Task 4 Step 3 ✓

**Placeholder scan:** None found.

**Type consistency:**
- `discover_guest_channels(guest_names, source_video_id, db_conn, youtube_api_key)` — same signature used in Task 2, Task 3 worker code, and Task 3 test assertion ✓
- `_MAX_CANDIDATES = 3` — used in Task 2 implementation and verified in test ✓
- `uploads_playlist_id = "UU" + channel["id"][2:]` — same derivation as the existing sidebar manual registration ✓
