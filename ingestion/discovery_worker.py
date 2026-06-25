"""
ingestion/discovery_worker.py
Drains the pending_guest_discovery queue using the YouTube Data API.

Uses YOUTUBE_DISCOVERY_API_KEY if set, falling back to YOUTUBE_API_KEY.
This lets discovery run from a separate GCP project with its own quota pool.

Lambda schedule: EventBridge cron, e.g. every 6 hours.
Local:           uv run python run_local.py --discover-pending
"""
import os
import sys

from core.channel_discovery import process_discovery_queue
from core.db import get_connection


def run_discovery(db_conn, max_guests: int = 20) -> int:
    api_key = os.environ.get("YOUTUBE_DISCOVERY_API_KEY") or os.environ.get("YOUTUBE_API_KEY", "")
    if not api_key:
        print("[discovery] no API key set (YOUTUBE_DISCOVERY_API_KEY or YOUTUBE_API_KEY) — skipping", file=sys.stderr)
        return 0
    n = process_discovery_queue(db_conn, api_key, max_guests=max_guests)
    print(f"[discovery] {n} channel(s) discovered and queued for approval")
    return n


def lambda_handler(event: dict, context) -> dict:
    db_conn = get_connection()
    try:
        n = run_discovery(db_conn)
        return {"discovered": n}
    finally:
        db_conn.close()
