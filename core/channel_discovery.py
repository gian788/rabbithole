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
