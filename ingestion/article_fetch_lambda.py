"""
ingestion/article_fetch_lambda.py
Triggered by EventBridge cron (same or separate schedule to fetch_lambda).

Polls the RSS feed of each active website, detects new articles,
inserts them into the DB with status='discovered', and dispatches
SQS messages for the article_worker to process.

Parallel to fetch_lambda.py (YouTube channel polling).
"""
import json
import os
import sys

import boto3
import feedparser
import psycopg2.extras

from core.db import get_connection


def _fetch_rss_urls(rss_url: str, max_results: int) -> list[dict]:
    """Parse an RSS/Atom feed and return up to max_results entries as {url, title}."""
    feed = feedparser.parse(rss_url)
    results = []
    for entry in feed.entries[:max_results]:
        url = entry.get("link", "")
        title = entry.get("title", "")
        if url:
            results.append({"url": url, "title": title})
    return results


def lambda_handler(event: dict, context) -> dict:
    conn = get_connection()
    sqs = boto3.client("sqs", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    queue_url = os.environ["ARTICLE_SQS_QUEUE_URL"]

    processed_websites = 0
    new_articles_queued = 0

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, name, rss_url, articles_to_fetch, max_articles
            FROM websites
            WHERE is_active = TRUE AND rss_url IS NOT NULL
            ORDER BY last_checked_at ASC NULLS FIRST
            """
        )
        websites = cur.fetchall()

    for website in websites:
        website_id = website["id"]
        try:
            entries = _fetch_rss_urls(
                website["rss_url"],
                website["articles_to_fetch"] or 10,
            )
            if not entries:
                _mark_checked(conn, website_id)
                processed_websites += 1
                continue

            candidate_urls = [e["url"] for e in entries]

            # Filter out already-known URLs
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT url FROM articles WHERE url = ANY(%s)", (candidate_urls,)
                )
                known_urls = {row[0] for row in cur.fetchall()}

            new_entries = [e for e in entries if e["url"] not in known_urls]

            # Respect max_articles cap
            max_articles = website["max_articles"] or 100
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM articles WHERE website_id = %s", (website_id,)
                )
                current_count = cur.fetchone()[0]
            slots = max(0, max_articles - current_count)
            new_entries = new_entries[:slots]

            if not new_entries:
                _mark_checked(conn, website_id)
                processed_websites += 1
                continue

            # Insert new article rows and collect their UUIDs for SQS dispatch
            inserted = []
            with conn.cursor() as cur:
                for entry in new_entries:
                    cur.execute(
                        """
                        INSERT INTO articles (website_id, url, title, status)
                        VALUES (%s, %s, %s, 'discovered')
                        ON CONFLICT (url) DO NOTHING
                        RETURNING id
                        """,
                        (website_id, entry["url"], entry["title"][:500]),
                    )
                    row = cur.fetchone()
                    if row:
                        inserted.append({"id": str(row[0]), "url": entry["url"]})
            _mark_checked(conn, website_id)
            conn.commit()

            if not inserted:
                processed_websites += 1
                continue

            # Dispatch SQS messages in batches of 10
            messages = [
                {
                    "Id": item["id"].replace("-", "")[:80],  # SQS Id constraints
                    "MessageBody": json.dumps({
                        "article_id": item["id"],
                        "url":        item["url"],
                        "website_id": website_id,
                    }),
                }
                for item in inserted
            ]
            for i in range(0, len(messages), 10):
                sqs.send_message_batch(QueueUrl=queue_url, Entries=messages[i : i + 10])

            new_articles_queued += len(inserted)
            processed_websites += 1
            print(f"[article-fetch] website={website['name']}  new={len(inserted)}")

        except Exception as exc:
            err = str(exc)
            print(f"[article-fetch] ERROR website={website.get('name', '?')}: {err}", file=sys.stderr)

    conn.close()
    return {"processed_websites": processed_websites, "new_articles_queued": new_articles_queued}


def _mark_checked(conn, website_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE websites SET last_checked_at = NOW() WHERE id = %s", (website_id,)
        )
    conn.commit()
