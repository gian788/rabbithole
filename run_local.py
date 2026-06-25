"""
run_local.py — Local development runner.

Calls Lambda handlers directly without AWS infrastructure.
Requires a .env file with DATABASE_URL, OPENAI_API_KEY, ANTHROPIC_API_KEY,
and either PINECONE_* vars or VECTOR_STORE=chroma + CHROMA_PATH.

Usage
-----
# Process a specific video
uv run python run_local.py --video VIDEO_ID --channel CHANNEL_ID

# Run the fetch loop against all active channels in the DB
uv run python run_local.py --fetch

# Process every 'discovered' video currently in the DB
uv run python run_local.py --process-pending

# Fetch new articles from all active website RSS feeds
uv run python run_local.py --fetch-articles

# Process every 'discovered' article currently in the DB
uv run python run_local.py --process-pending-articles

# Process a single article by UUID
uv run python run_local.py --article ARTICLE_UUID
"""
import argparse
import os
import sys

# Load .env before importing project modules
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is a dev dependency; skip if somehow missing


def _ensure_video_in_db(db_conn, video_id: str, channel_id: str) -> None:
    """Insert video metadata from YouTube API if the video isn't already in the DB.

    When using --video directly (bypassing fetch_lambda), the video row may not
    exist yet, leaving title/description empty and breaking guest extraction.
    """
    with db_conn.cursor() as cur:
        cur.execute("SELECT id FROM videos WHERE id = %s", (video_id,))
        if cur.fetchone():
            return

    api_key = os.environ.get("YOUTUBE_API_KEY", "")
    if not api_key:
        print(f"[run_local] WARNING: {video_id} not in DB and no YOUTUBE_API_KEY — title/description will be empty")
        return

    import httpx
    resp = httpx.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params={"part": "snippet", "id": video_id, "key": api_key},
        timeout=10,
    )
    items = resp.json().get("items", [])
    if not items:
        print(f"[run_local] WARNING: YouTube API returned no metadata for {video_id}")
        return

    snippet = items[0]["snippet"]
    with db_conn.cursor() as cur:
        cur.execute(
            """INSERT INTO videos (id, channel_id, title, description, status)
               VALUES (%s, %s, %s, %s, 'discovered')
               ON CONFLICT (id) DO NOTHING""",
            (video_id, channel_id, snippet.get("title", ""), snippet.get("description", "")),
        )
    db_conn.commit()
    print(f"[run_local] fetched and inserted video: {snippet.get('title', '')[:80]}")


def run_fetch() -> None:
    from ingestion.fetch_lambda import lambda_handler
    result = lambda_handler({}, None)
    print(f"\nFetch complete: {result}")


def run_process(video_id: str, channel_id: str) -> None:
    import boto3

    from core.db import get_connection
    from core.gateway import ModelGateway
    from core.vector_store import get_vector_store
    from ingestion.worker_lambda import process_video

    db_conn = get_connection()
    _ensure_video_in_db(db_conn, video_id, channel_id)
    s3_client = boto3.client(
        "s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    store = get_vector_store()
    gateway = ModelGateway(db_conn=db_conn)

    process_video(
        video_id=video_id,
        channel_id=channel_id,
        db_conn=db_conn,
        s3_client=s3_client,
        store=store,
        gateway=gateway,
    )
    db_conn.close()


def run_process_pending() -> None:
    import boto3
    import psycopg2.extras

    from core.db import get_connection
    from core.gateway import ModelGateway
    from core.vector_store import get_vector_store
    from ingestion.worker_lambda import process_video

    db_conn = get_connection()
    s3_client = boto3.client(
        "s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    store = get_vector_store()
    gateway = ModelGateway(db_conn=db_conn)

    with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, channel_id FROM videos WHERE status = 'discovered' ORDER BY created_at"
        )
        pending = cur.fetchall()

    print(f"Found {len(pending)} discovered video(s) to process.")
    for row in pending:
        print(f"  → processing {row['id']} …")
        try:
            process_video(
                video_id=row["id"],
                channel_id=row["channel_id"],
                db_conn=db_conn,
                s3_client=s3_client,
                store=store,
                gateway=gateway,
            )
        except Exception as exc:
            print(f"    FAILED: {exc}", file=sys.stderr)

    db_conn.close()


def run_discover_pending() -> None:
    from core.db import get_connection
    from ingestion.discovery_worker import run_discovery

    db_conn = get_connection()
    run_discovery(db_conn)
    db_conn.close()


def run_fetch_articles() -> None:
    from ingestion.article_fetch_lambda import lambda_handler
    result = lambda_handler({}, None)
    print(f"\nArticle fetch complete: {result}")


def run_process_pending_articles() -> None:
    import boto3
    import psycopg2.extras

    from core.db import get_connection
    from core.gateway import ModelGateway
    from core.vector_store import get_vector_store
    from ingestion.article_worker import process_article

    db_conn = get_connection()
    s3_client = boto3.client(
        "s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    store = get_vector_store()
    gateway = ModelGateway(db_conn=db_conn)

    with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, url, website_id FROM articles WHERE status = 'discovered' ORDER BY created_at"
        )
        pending = cur.fetchall()

    print(f"Found {len(pending)} discovered article(s) to process.")
    for row in pending:
        print(f"  → processing {row['id']} ({row['url']}) …")
        try:
            process_article(
                article_id=str(row["id"]),
                url=row["url"],
                website_id=row["website_id"] or "manual",
                db_conn=db_conn,
                store=store,
                gateway=gateway,
                s3_client=s3_client,
            )
        except Exception as exc:
            print(f"    FAILED: {exc}", file=sys.stderr)

    db_conn.close()


def run_process_article(article_id: str) -> None:
    import boto3
    import psycopg2.extras

    from core.db import get_connection
    from core.gateway import ModelGateway
    from core.vector_store import get_vector_store
    from ingestion.article_worker import process_article

    db_conn = get_connection()
    s3_client = boto3.client(
        "s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    store = get_vector_store()
    gateway = ModelGateway(db_conn=db_conn)

    with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id, url, website_id FROM articles WHERE id = %s", (article_id,))
        row = cur.fetchone()

    if not row:
        print(f"Article {article_id} not found in DB", file=sys.stderr)
        db_conn.close()
        sys.exit(1)

    process_article(
        article_id=str(row["id"]),
        url=row["url"],
        website_id=row["website_id"] or "manual",
        db_conn=db_conn,
        store=store,
        gateway=gateway,
        s3_client=s3_client,
    )
    db_conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local dev runner for youtube-topic-rag")
    parser.add_argument("--video",                    help="YouTube video ID to process")
    parser.add_argument("--channel",                  help="YouTube channel ID (required with --video)")
    parser.add_argument("--fetch",                    action="store_true", help="Run fetch_lambda (YouTube)")
    parser.add_argument("--process-pending",          action="store_true", help="Process all discovered videos")
    parser.add_argument("--discover-pending",         action="store_true", help="Drain the guest discovery queue")
    parser.add_argument("--fetch-articles",           action="store_true", help="Run article_fetch_lambda (RSS)")
    parser.add_argument("--process-pending-articles", action="store_true", help="Process all discovered articles")
    parser.add_argument("--article",                  help="Article UUID to process")
    args = parser.parse_args()

    if args.fetch:
        run_fetch()
    elif args.discover_pending:
        run_discover_pending()
    elif args.process_pending:
        run_process_pending()
    elif args.fetch_articles:
        run_fetch_articles()
    elif args.process_pending_articles:
        run_process_pending_articles()
    elif args.article:
        run_process_article(args.article)
    elif args.video and args.channel:
        run_process(args.video, args.channel)
    else:
        parser.print_help()
        sys.exit(1)
