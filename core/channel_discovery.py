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
) -> int:
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
        return cur.rowcount


# ---------------------------------------------------------------------------
# Queue-based discovery (preferred path)
# ---------------------------------------------------------------------------

def enqueue_guests(
    guest_names: list[str],
    source_video_id: str,
    source_channel_id: str | None,
    db_conn,
) -> int:
    """Add extracted guest names to the discovery queue; no API calls made.

    Uses ON CONFLICT DO NOTHING so re-processing the same video is safe.
    Returns count of newly queued rows.
    """
    if not guest_names:
        return 0
    queued = 0
    try:
        with db_conn.cursor() as cur:
            for name in guest_names:
                cur.execute(
                    """
                    INSERT INTO pending_guest_discovery
                        (guest_name, source_video_id, source_channel_id)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (guest_name, source_video_id) DO NOTHING
                    """,
                    (name, source_video_id, source_channel_id),
                )
                queued += cur.rowcount
        db_conn.commit()
    except Exception as exc:
        logger.warning("channel_discovery: failed to enqueue guests: %s", exc)
        try:
            db_conn.rollback()
        except Exception:
            pass
    return queued


def process_discovery_queue(
    db_conn,
    youtube_api_key: str,
    max_guests: int = 10,
) -> int:
    """Drain the pending_guest_discovery queue, calling YouTube API for each guest.

    On quota error (403): logs a warning and stops the batch, leaving remaining
    rows as 'pending' so they are retried on the next run.
    Returns count of channels inserted into the channels table.
    """
    if not youtube_api_key:
        return 0

    with db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, guest_name, source_video_id, source_channel_id
            FROM pending_guest_discovery
            WHERE status = 'pending'
            ORDER BY created_at
            LIMIT %s
            """,
            (max_guests,),
        )
        rows = cur.fetchall()

    total_inserted = 0
    for row_id, guest_name, source_video_id, source_channel_id in rows:
        try:
            channel_ids = _search_channels(guest_name, youtube_api_key)
            if not channel_ids:
                with db_conn.cursor() as cur:
                    cur.execute(
                        """UPDATE pending_guest_discovery
                           SET status = 'not_found', attempts = attempts + 1,
                               last_attempted_at = NOW()
                           WHERE id = %s""",
                        (row_id,),
                    )
                db_conn.commit()
                logger.info("channel_discovery: no YouTube results for guest %r", guest_name)
                continue

            existing = _get_existing_channel_ids(db_conn, channel_ids)
            new_ids = [cid for cid in channel_ids if cid not in existing]

            candidates = _get_channel_details(new_ids, youtube_api_key) if new_ids else []
            inserted_ids = []
            for channel in candidates:
                n = _insert_candidate(db_conn, channel, source_video_id, guest_name)
                if n > 0:
                    total_inserted += n
                    inserted_ids.append(channel["id"])

            # Link to the top candidate (or first search result if all pre-existed)
            linked_id = inserted_ids[0] if inserted_ids else (channel_ids[0] if channel_ids else None)
            with db_conn.cursor() as cur:
                cur.execute(
                    """UPDATE pending_guest_discovery
                       SET status = 'discovered', attempts = attempts + 1,
                           last_attempted_at = NOW(), linked_channel_id = %s
                       WHERE id = %s""",
                    (linked_id, row_id),
                )
            db_conn.commit()

        except Exception as exc:
            logger.warning("channel_discovery: failed for guest %r: %s", guest_name, exc)
            try:
                db_conn.rollback()
            except Exception:
                pass
            if "quotaExceeded" in str(exc) or "403" in str(exc):
                logger.warning("channel_discovery: quota exceeded, stopping batch early")
                break

    return total_inserted


# ---------------------------------------------------------------------------
# Legacy inline discovery (kept for backward compatibility with existing tests)
# ---------------------------------------------------------------------------

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
                inserted += _insert_candidate(db_conn, channel, source_video_id, name)
            db_conn.commit()

        except Exception as exc:
            logger.warning("channel_discovery: failed for guest %r: %s", name, exc)
            try:
                db_conn.rollback()
            except Exception:
                pass

    return inserted
