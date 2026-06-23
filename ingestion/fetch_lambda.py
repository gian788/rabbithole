"""
fetch_lambda.py
Triggered by EventBridge cron (every 6 hours).
Polls the uploads playlist of each active channel, detects new videos,
inserts them into the DB, and dispatches processing messages to SQS.
"""
import json
import os
import sys

import boto3
import psycopg2.extras
from googleapiclient.discovery import build

from core.db import get_connection


def _build_youtube():
    return build("youtube", "v3", developerKey=os.environ["YOUTUBE_API_KEY"])


def _fetch_playlist_video_ids(youtube, playlist_id: str, max_results: int) -> list[str]:
    resp = youtube.playlistItems().list(
        part="contentDetails",
        playlistId=playlist_id,
        maxResults=max_results,
    ).execute()
    return [item["contentDetails"]["videoId"] for item in resp.get("items", [])]


def _fetch_video_metadata(youtube, video_ids: list[str]) -> list[dict]:
    if not video_ids:
        return []
    resp = youtube.videos().list(
        part="snippet,statistics,contentDetails",
        id=",".join(video_ids),
    ).execute()
    return resp.get("items", [])


def _is_short(item: dict) -> bool:
    """Return True if this video is a YouTube Short (≤ 60 s or #shorts tag)."""
    title = item.get("snippet", {}).get("title", "")
    description = item.get("snippet", {}).get("description", "")
    if "#shorts" in title.lower() or "#shorts" in description.lower():
        return True
    duration = item.get("contentDetails", {}).get("duration", "")
    # ISO 8601: PT1M30S — shorts are PT60S or less (no minutes/hours component, or M=0)
    import re
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
    if m:
        hours = int(m.group(1) or 0)
        minutes = int(m.group(2) or 0)
        seconds = int(m.group(3) or 0)
        total = hours * 3600 + minutes * 60 + seconds
        if total <= 60:
            return True
    return False


def lambda_handler(event: dict, context) -> dict:
    conn = get_connection()
    sqs = boto3.client("sqs", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    queue_url = os.environ["SQS_QUEUE_URL"]
    youtube = _build_youtube()

    processed_channels = 0
    new_videos_queued = 0

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, name, uploads_playlist_id, videos_to_fetch, max_videos
            FROM channels
            WHERE is_active = TRUE
              AND is_approved = TRUE
            ORDER BY last_checked_at ASC NULLS FIRST
            """
        )
        channels = cur.fetchall()

    for channel in channels:
        channel_id = channel["id"]
        try:
            api_ids = _fetch_playlist_video_ids(
                youtube,
                channel["uploads_playlist_id"],
                channel["videos_to_fetch"] or 10,
            )
            if not api_ids:
                _mark_checked(conn, channel_id)
                processed_channels += 1
                continue

            # Identify which video IDs are already in the DB
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM videos WHERE id = ANY(%s)", (api_ids,))
                known_ids = {row[0] for row in cur.fetchall()}

            new_ids = [vid for vid in api_ids if vid not in known_ids]

            # Respect max_videos cap: only queue as many as the channel has room for
            max_videos = channel["max_videos"] or 100
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM videos WHERE channel_id = %s", (channel_id,)
                )
                current_count = cur.fetchone()[0]
            slots = max(0, max_videos - current_count)
            new_ids = new_ids[:slots]

            if not new_ids:
                _mark_checked(conn, channel_id)
                processed_channels += 1
                continue

            # Fetch full metadata for new videos, skip Shorts
            items = [i for i in _fetch_video_metadata(youtube, new_ids) if not _is_short(i)]
            if not items:
                _mark_checked(conn, channel_id)
                processed_channels += 1
                continue

            with conn.cursor() as cur:
                for item in items:
                    snippet = item.get("snippet", {})
                    stats = item.get("statistics", {})
                    cur.execute(
                        """
                        INSERT INTO videos
                            (id, channel_id, title, description,
                             view_count, like_count, published_at, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, 'discovered')
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (
                            item["id"],
                            channel_id,
                            snippet.get("title", "")[:500],
                            snippet.get("description", ""),
                            int(stats.get("viewCount", 0) or 0),
                            int(stats.get("likeCount",  0) or 0),
                            snippet.get("publishedAt"),
                        ),
                    )
            _mark_checked(conn, channel_id)
            conn.commit()

            # Dispatch SQS messages in batches of 10
            messages = [
                {
                    "Id": vid,
                    "MessageBody": json.dumps({"video_id": vid, "channel_id": channel_id}),
                }
                for vid in new_ids
            ]
            for i in range(0, len(messages), 10):
                sqs.send_message_batch(QueueUrl=queue_url, Entries=messages[i : i + 10])

            new_videos_queued += len(new_ids)
            processed_channels += 1
            print(f"[fetch] channel={channel['name']}  new={len(new_ids)}")

        except Exception as exc:
            err = str(exc)
            print(f"[fetch] ERROR channel={channel.get('name', '?')}: {err}", file=sys.stderr)
            if "404" in err:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE channels SET is_active = FALSE WHERE id = %s", (channel_id,)
                    )
                conn.commit()
            elif "403" in err:
                # Quota exhausted — stop processing remaining channels this run
                break

    conn.close()
    return {"processed_channels": processed_channels, "new_videos_queued": new_videos_queued}


def _mark_checked(conn, channel_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE channels SET last_checked_at = NOW() WHERE id = %s", (channel_id,)
        )
    conn.commit()
