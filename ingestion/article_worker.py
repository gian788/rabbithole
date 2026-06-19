"""
ingestion/article_worker.py
Triggered by SQS (ARTICLE_SQS_QUEUE_URL). For each message:
  1. Fetches the article HTML
  2. Extracts section-based chunks
  3. Classifies the article into 1-N topics (Claude Haiku)
  4. Embeds each chunk (OpenAI text-embedding-3-small)
  5. Upserts vectors to the configured VectorStore (Pinecone or Chroma)
  6. Saves structured JSON to S3 (or local disk in dev mode)
  7. Updates article status in PostgreSQL

Parallel to worker_lambda.py (YouTube video processing).
"""
import json
import os
import pathlib
import sys

import boto3

from core.article_fetcher import extract_sections, fetch_article
from core.db import get_channel_default_topic, get_connection, get_topic_names
from core.gateway import ModelGateway
from core.topics import classify_topics
from core.vector_store import VectorStore, get_vector_store


# ---------------------------------------------------------------------------
# Storage helper (mirrors worker_lambda._save_payload)
# ---------------------------------------------------------------------------

def _save_payload(s3_client, bucket: str, key: str, payload: dict) -> str:
    local_root = os.environ.get("S3_LOCAL_PATH")
    if local_root:
        dest = pathlib.Path(local_root) / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        return str(dest)
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, ensure_ascii=False),
        ContentType="application/json",
    )
    return f"s3://{bucket}/{key}"


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_article(
    article_id: str,
    url: str,
    website_id: str,
    db_conn,
    store: VectorStore,
    gateway: ModelGateway,
    s3_client=None,
) -> None:
    # Idempotency: skip already-completed articles
    with db_conn.cursor() as cur:
        cur.execute("SELECT status FROM articles WHERE id = %s", (article_id,))
        row = cur.fetchone()
    if row and row[0] == "completed":
        print(f"[article-worker] {article_id} already completed — skipping")
        return

    with db_conn.cursor() as cur:
        cur.execute("UPDATE articles SET status = 'processing' WHERE id = %s", (article_id,))
    db_conn.commit()

    try:
        # -- Fetch + parse HTML -----------------------------------------------
        article = fetch_article(url)
        chunks = extract_sections(article["html_body"], url)

        if not chunks:
            _fail(db_conn, article_id, "No content could be extracted from the article")
            return

        # -- Update title/author/published_at from parsed HTML ----------------
        with db_conn.cursor() as cur:
            cur.execute(
                """
                UPDATE articles
                SET title        = %s,
                    author       = %s,
                    published_at = %s
                WHERE id = %s
                """,
                (
                    (article["title"] or "")[:500],
                    (article["author"] or "")[:255],
                    article["published_at"],
                    article_id,
                ),
            )
        db_conn.commit()

        # -- Topic classification ---------------------------------------------
        available_topics = get_topic_names(db_conn)
        default_hint = _website_default_topic(db_conn, website_id) or available_topics[0]
        text_excerpt = " ".join(c["text_content"] for c in chunks[:3])
        article_topics = classify_topics(
            title=article["title"] or "",
            text_excerpt=text_excerpt,
            available_topics=available_topics,
            default_hint=default_hint,
            gateway=gateway,
        )
        primary_topic = article_topics[0] if article_topics else available_topics[0]

        with db_conn.cursor() as cur:
            cur.execute(
                "UPDATE articles SET topics = %s, primary_topic = %s WHERE id = %s",
                (article_topics, primary_topic, article_id),
            )
        db_conn.commit()

        # -- Embed + upsert to VectorStore ------------------------------------
        total_tokens = 0
        total_cost = 0.0
        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict] = []
        texts: list[str] = []

        for chunk in chunks:
            embed = gateway.get_embedding(chunk["text_content"], associated_id=article_id)
            total_tokens += embed.input_tokens
            total_cost += embed.cost

            ids.append(f"{article_id}_{chunk['chunk_id']}")
            embeddings.append(embed.embedding_vector)
            metadatas.append({
                "source_type":   "article",
                "article_id":    article_id,
                "website_id":    website_id,
                "topics":        article_topics,        # list — Pinecone $in filter
                "primary_topic": primary_topic,         # scalar — Chroma $eq filter
                "chapter":       chunk["associated_chapter"],
                "section_slug":  chunk.get("section_slug", ""),
                "deep_link":     chunk["deep_link"],
            })
            texts.append(chunk["text_content"])

        store.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, texts=texts)

        # -- S3 payload -------------------------------------------------------
        publisher = article.get("publisher") or website_id
        s3_key = f"articles/{primary_topic}/{website_id}/{article_id}_structured.json"
        payload = {
            "article_id":    article_id,
            "article_title": article["title"],
            "website_id":    website_id,
            "url":           url,
            "author":        article.get("author"),
            "publisher":     publisher,
            "topics":        article_topics,
            "total_sections": len(chunks),
            "sections": [
                {
                    "chunk_id":           c["chunk_id"],
                    "associated_chapter": c["associated_chapter"],
                    "section_slug":       c.get("section_slug", ""),
                    "deep_link":          c["deep_link"],
                    "text_content":       c["text_content"],
                }
                for c in chunks
            ],
        }
        saved_path = _save_payload(
            s3_client, os.environ.get("S3_BUCKET_NAME", ""), s3_key, payload
        )

        # -- Mark completed ---------------------------------------------------
        with db_conn.cursor() as cur:
            cur.execute(
                """
                UPDATE articles
                SET status           = 'completed',
                    s3_path          = %s,
                    ingestion_tokens = %s,
                    ingestion_cost   = %s,
                    processed_at     = NOW()
                WHERE id = %s
                """,
                (saved_path, total_tokens, total_cost, article_id),
            )
        db_conn.commit()
        print(
            f"[article-worker] {article_id} done — "
            f"{len(chunks)} sections, topics={article_topics}, ${total_cost:.6f}"
        )

    except Exception as exc:
        _fail(db_conn, article_id, str(exc)[:500])
        raise  # let SQS retry (up to maxReceiveCount)


def _website_default_topic(db_conn, website_id: str) -> str | None:
    """Return the default topic name for a website, or None."""
    try:
        with db_conn.cursor() as cur:
            cur.execute(
                """
                SELECT t.name FROM websites w
                JOIN topics t ON t.id = w.default_topic_id
                WHERE w.id = %s
                """,
                (website_id,),
            )
            row = cur.fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _fail(conn, article_id: str, message: str) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE articles SET status = 'failed', error_message = %s WHERE id = %s",
                (message, article_id),
            )
        conn.commit()
    except Exception as exc:
        print(f"[article-worker] could not write failure for {article_id}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Lambda entry point
# ---------------------------------------------------------------------------

def lambda_handler(event: dict, context) -> dict:
    db_conn = get_connection()
    s3_client = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    store = get_vector_store()
    gateway = ModelGateway(db_conn=db_conn)

    failed: list[dict] = []
    for record in event.get("Records", []):
        msg_id = record.get("messageId", "?")
        try:
            body = json.loads(record["body"])
            process_article(
                article_id=body["article_id"],
                url=body["url"],
                website_id=body["website_id"],
                db_conn=db_conn,
                store=store,
                gateway=gateway,
                s3_client=s3_client,
            )
        except Exception as exc:
            print(f"[article-worker] FAILED msg={msg_id}: {exc}", file=sys.stderr)
            failed.append({"itemIdentifier": msg_id})

    db_conn.close()
    return {"batchItemFailures": failed}
