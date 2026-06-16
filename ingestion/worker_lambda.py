"""
worker_lambda.py
Triggered by SQS. For each message:
  1. Downloads the YouTube transcript
  2. Detects or generates chapters
  3. Segments into paragraph chunks
  4. Classifies the video into 1-N topics (Claude Haiku)
  5. Embeds each chunk (OpenAI text-embedding-3-small)
  6. Upserts dense + sparse vectors to Pinecone
  7. Saves structured JSON to S3 (or local disk in dev mode)
  8. Updates video status in Neon PostgreSQL
"""
import json
import os
import pathlib
import re
import sys

import boto3
import psycopg2
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from core.chunker import (
    extract_chapters_from_description,
    fetch_sponsor_segments,
    filter_sponsored_srt,
    fixed_word_chunking,
    generate_chapters_with_llm,
    segment_into_paragraphs,
)
from core.db import get_channel_default_topic, get_connection, get_topic_names
from core.gateway import ModelGateway

_bm25 = BM25Encoder.default()
_PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "")


# ---------------------------------------------------------------------------
# Topic classification
# ---------------------------------------------------------------------------

def _classify_topics(
    title: str,
    description: str,
    default_hint: str,
    available_topics: list[str],
    gateway: ModelGateway,
) -> list[str]:
    system_prompt = (
        f"This channel primarily covers {default_hint}. "
        f"Classify this video into ALL relevant topics from this list: {available_topics}. "
        "Return ONLY a valid JSON array of strings. "
        'Example: ["consciousness", "spirituality"]. Output nothing else.'
    )
    try:
        resp = gateway.get_completion(
            prompt=f"Title: {title}\n\nDescription (first 500 chars): {description[:500]}",
            system_prompt=system_prompt,
            model="claude-haiku-4-5-20251001",
            provider="anthropic",
            associated_id="topic_classify",
        )
        raw = resp.text_content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[^\n]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())
        topics = json.loads(raw)
        validated = [t for t in topics if t in available_topics]
        return validated if validated else [default_hint]
    except Exception:
        return [default_hint] if default_hint else available_topics[:1]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _save_payload(s3_client, bucket: str, key: str, payload: dict) -> str:
    """Write JSON to local disk (dev) or S3 (prod)."""
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

def process_video(
    video_id: str,
    channel_id: str,
    db_conn,
    s3_client,
    pinecone_index,
    gateway: ModelGateway,
) -> None:
    # Idempotency: skip already-completed videos
    with db_conn.cursor() as cur:
        cur.execute("SELECT status FROM videos WHERE id = %s", (video_id,))
        row = cur.fetchone()
    if row and row[0] == "completed":
        print(f"[worker] {video_id} already completed — skipping")
        return

    with db_conn.cursor() as cur:
        cur.execute("UPDATE videos SET status = 'processing' WHERE id = %s", (video_id,))
    db_conn.commit()

    try:
        # -- Transcript -------------------------------------------------------
        try:
            fetched = YouTubeTranscriptApi().fetch(video_id)
            srt = [
                {"text": s.text, "start": s.start, "duration": s.duration}
                for s in fetched.snippets
            ]
        except (TranscriptsDisabled, NoTranscriptFound) as exc:
            _fail(db_conn, video_id, str(exc))
            return  # no retry benefit

        # -- Filter sponsored segments ----------------------------------------
        sponsor_segments = fetch_sponsor_segments(video_id)
        srt = filter_sponsored_srt(srt, sponsor_segments)

        # -- Video metadata ---------------------------------------------------
        with db_conn.cursor() as cur:
            cur.execute("SELECT title, description FROM videos WHERE id = %s", (video_id,))
            meta = cur.fetchone()
        title       = meta[0] if meta else ""
        description = meta[1] if meta else ""

        # -- Chapter strategy -------------------------------------------------
        chapters = extract_chapters_from_description(description)
        if len(chapters) < 3:
            full_text = " ".join(seg["text"] for seg in srt)
            chapters = generate_chapters_with_llm(full_text, gateway)

        if len(chapters) >= 3:
            chunks = segment_into_paragraphs(srt, chapters, video_id)
        else:
            full_text = " ".join(seg["text"] for seg in srt)
            chunks = fixed_word_chunking(full_text, video_id)

        # -- Topic classification ---------------------------------------------
        available_topics = get_topic_names(db_conn)
        default_hint     = get_channel_default_topic(db_conn, channel_id) or available_topics[0]
        # Use transcript excerpt as fallback when description is missing (e.g. manual inserts)
        classify_text = description.strip() if description and description.strip() else (
            " ".join(seg["text"] for seg in srt[:120])  # first ~2 min of transcript
        )
        video_topics = _classify_topics(title, classify_text, default_hint, available_topics, gateway)

        with db_conn.cursor() as cur:
            cur.execute("UPDATE videos SET topics = %s WHERE id = %s", (video_topics, video_id))
        db_conn.commit()

        # -- Embed + build Pinecone vectors -----------------------------------
        total_tokens = 0
        total_cost   = 0.0
        vectors      = []

        for chunk in chunks:
            embed = gateway.get_embedding(chunk["text_content"], associated_id=video_id)
            total_tokens += embed.input_tokens
            total_cost   += embed.cost

            sparse = _bm25.encode_documents([chunk["text_content"]])[0]
            if not sparse.get("indices"):
                continue  # too short / all stopwords — not useful for retrieval
            vectors.append({
                "id":            f"{video_id}_{chunk['chunk_id']}",
                "values":        embed.embedding_vector,
                "sparse_values": sparse,
                "metadata": {
                    "video_id":      video_id,
                    "channel_id":    channel_id,
                    "topics":        video_topics,
                    "chapter":       chunk["associated_chapter"],
                    "start_seconds": chunk["start_seconds"],
                    "deep_link":     chunk["deep_link"],
                    "text_content":  chunk["text_content"][:1000],
                },
            })

        # Upsert in batches of 100
        for i in range(0, len(vectors), 100):
            pinecone_index.upsert(vectors=vectors[i : i + 100], namespace=_PINECONE_NAMESPACE)

        # -- S3 payload -------------------------------------------------------
        primary_topic = video_topics[0] if video_topics else "general"
        s3_key  = f"transcripts/{primary_topic}/{channel_id}/{video_id}_structured.json"
        payload = {
            "video_id":        video_id,
            "video_title":     title,
            "channel_id":      channel_id,
            "video_base_url":  "https://youtu.be",
            "topics":          video_topics,
            "total_paragraphs": len(chunks),
            "paragraphs": [
                {
                    "chunk_id":          c["chunk_id"],
                    "associated_chapter": c["associated_chapter"],
                    "start_seconds":     c["start_seconds"],
                    "deep_link":         c["deep_link"],
                    "text_content":      c["text_content"],
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
                UPDATE videos
                SET status           = 'completed',
                    s3_path          = %s,
                    ingestion_tokens = %s,
                    ingestion_cost   = %s,
                    processed_at     = NOW()
                WHERE id = %s
                """,
                (saved_path, total_tokens, total_cost, video_id),
            )
        db_conn.commit()
        print(
            f"[worker] {video_id} done — "
            f"{len(chunks)} chunks, topics={video_topics}, ${total_cost:.6f}"
        )

    except Exception as exc:
        _fail(db_conn, video_id, str(exc)[:500])
        raise  # let SQS retry (up to maxReceiveCount)


def _fail(conn, video_id: str, message: str) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE videos SET status = 'failed', error_message = %s WHERE id = %s",
                (message, video_id),
            )
        conn.commit()
    except Exception as exc:
        print(f"[worker] could not write failure for {video_id}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Lambda entry point
# ---------------------------------------------------------------------------

def lambda_handler(event: dict, context) -> dict:
    db_conn       = get_connection()
    s3_client     = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    pinecone_index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(
        os.environ["PINECONE_INDEX_NAME"]
    )
    gateway = ModelGateway(db_conn=db_conn)

    failed: list[dict] = []
    for record in event.get("Records", []):
        msg_id = record.get("messageId", "?")
        try:
            body = json.loads(record["body"])
            process_video(
                video_id=body["video_id"],
                channel_id=body["channel_id"],
                db_conn=db_conn,
                s3_client=s3_client,
                pinecone_index=pinecone_index,
                gateway=gateway,
            )
        except Exception as exc:
            print(f"[worker] FAILED msg={msg_id}: {exc}", file=sys.stderr)
            failed.append({"itemIdentifier": msg_id})

    db_conn.close()
    return {"batchItemFailures": failed}
