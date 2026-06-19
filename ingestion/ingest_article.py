"""
ingestion/ingest_article.py — CLI entry point for one-off article ingestion.

Equivalent to: run_local.py --video <id> --channel <id>

Usage
-----
uv run python -m ingestion.ingest_article https://example.com/some-post
uv run python -m ingestion.ingest_article https://example.com/post --website-id hubermanlab.com
uv run python -m ingestion.ingest_article https://example.com/post --topic consciousness
"""
import argparse
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a single article URL")
    parser.add_argument("url", help="Article URL to ingest")
    parser.add_argument(
        "--website-id",
        default="manual",
        help="Website ID to associate the article with (default: 'manual')",
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="Topic hint for classification (e.g. consciousness)",
    )
    args = parser.parse_args()

    import boto3

    from core.db import get_connection, get_topic_names
    from core.gateway import ModelGateway
    from core.vector_store import get_vector_store
    from ingestion.article_worker import process_article

    db_conn = get_connection()
    store = get_vector_store()
    gateway = ModelGateway(db_conn=db_conn)
    s3_client = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

    # Validate topic hint
    if args.topic:
        available = get_topic_names(db_conn)
        if args.topic not in available:
            print(f"Unknown topic '{args.topic}'. Available: {available}", file=sys.stderr)
            sys.exit(1)

    # Ensure the website row exists (upsert a minimal row for manual ingestion)
    _ensure_website(db_conn, args.website_id, args.url)

    # Insert the article row (or skip if URL already exists)
    article_id = _ensure_article(db_conn, args.url, args.website_id, args.topic)
    if article_id is None:
        print(f"URL already ingested or failed to insert: {args.url}")
        db_conn.close()
        return

    print(f"Processing article {article_id} …")
    try:
        process_article(
            article_id=article_id,
            url=args.url,
            website_id=args.website_id,
            db_conn=db_conn,
            store=store,
            gateway=gateway,
            s3_client=s3_client,
        )
        print(f"Done: {article_id}")
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        db_conn.close()


def _ensure_website(conn, website_id: str, article_url: str) -> None:
    from urllib.parse import urlparse
    hostname = urlparse(article_url).hostname or website_id
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO websites (id, name, base_url)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (website_id, website_id, f"https://{hostname}"),
        )
    conn.commit()


def _ensure_article(conn, url: str, website_id: str, topic_hint: str | None) -> str | None:
    """Insert article row; return UUID string, or None if URL already exists."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO articles (website_id, url, title, primary_topic, status)
            VALUES (%s, %s, %s, %s, 'discovered')
            ON CONFLICT (url) DO NOTHING
            RETURNING id
            """,
            (website_id, url, url[:500], topic_hint),
        )
        row = cur.fetchone()
    conn.commit()
    return str(row[0]) if row else None


if __name__ == "__main__":
    main()
