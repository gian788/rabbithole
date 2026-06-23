"""
worker_lambda.py
Triggered by SQS. For each message:
  1. Downloads the YouTube transcript
  2. Detects or generates chapters
  3. Segments into paragraph chunks
  4. Classifies the video into 1-N topics (Claude Haiku)
  5. Embeds each chunk (OpenAI text-embedding-3-small)
  6. Upserts vectors to the configured VectorStore (Pinecone or Chroma)
  7. Saves structured JSON to S3 (or local disk in dev mode)
  8. Updates video status in Neon PostgreSQL
"""
import json
import os
import pathlib
import sys

import boto3
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
from core.channel_discovery import discover_guest_channels
from core.entities import extract_chunk_entities
from core.topics import classify_video_meta
from core.vector_store import VectorStore, get_vector_store


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
    store: VectorStore,
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

        with db_conn.cursor() as cur:
            cur.execute("SELECT name FROM channels WHERE id = %s", (channel_id,))
            ch_row = cur.fetchone()
        channel_name = ch_row[0] if ch_row else ""

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

        # -- Topic classification + people extraction -------------------------
        available_topics = get_topic_names(db_conn)
        default_hint     = get_channel_default_topic(db_conn, channel_id) or available_topics[0]
        classify_text = description.strip() if description and description.strip() else (
            " ".join(seg["text"] for seg in srt[:120])
        )
        video_meta = classify_video_meta(
            title=title,
            channel_name=channel_name,
            text_excerpt=classify_text,
            available_topics=available_topics,
            default_hint=default_hint,
            gateway=gateway,
        )
        video_topics  = video_meta.topics
        primary_topic = video_topics[0] if video_topics else available_topics[0]

        # -- Guest channel discovery ------------------------------------------
        api_key = os.environ.get("YOUTUBE_API_KEY", "")
        if api_key and video_meta.guests:
            discover_guest_channels(
                guest_names=video_meta.guests,
                source_video_id=video_id,
                db_conn=db_conn,
                youtube_api_key=api_key,
            )

        with db_conn.cursor() as cur:
            cur.execute("UPDATE videos SET topics = %s WHERE id = %s", (video_topics, video_id))
        db_conn.commit()

        # -- Embed + upsert to VectorStore ------------------------------------
        total_tokens = 0
        total_cost   = 0.0
        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict] = []
        texts: list[str] = []

        for chunk in chunks:
            embed = gateway.get_embedding(chunk["text_content"], associated_id=video_id)
            total_tokens += embed.input_tokens
            total_cost   += embed.cost

            entities = extract_chunk_entities(chunk["text_content"], gateway)
            ids.append(f"{video_id}_{chunk['chunk_id']}")
            embeddings.append(embed.embedding_vector)
            metadatas.append({
                "source_type":   "youtube_video",
                "video_id":      video_id,
                "channel_id":    channel_id,
                "topics":        video_topics,       # list — Pinecone $in filter
                "primary_topic": primary_topic,      # scalar — Chroma $eq filter
                "chapter":       chunk["associated_chapter"],
                "start_seconds": chunk["start_seconds"],
                "deep_link":     chunk["deep_link"],
                "entities":      entities,
                "host":          video_meta.host,
                "guests":        video_meta.guests,
            })
            texts.append(chunk["text_content"])

        store.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, texts=texts)

        # -- S3 payload -------------------------------------------------------
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
                    "chunk_id":           c["chunk_id"],
                    "associated_chapter": c["associated_chapter"],
                    "start_seconds":      c["start_seconds"],
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
    db_conn   = get_connection()
    s3_client = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    store     = get_vector_store()
    gateway   = ModelGateway(db_conn=db_conn)

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
                store=store,
                gateway=gateway,
            )
        except Exception as exc:
            print(f"[worker] FAILED msg={msg_id}: {exc}", file=sys.stderr)
            failed.append({"itemIdentifier": msg_id})

    db_conn.close()
    return {"batchItemFailures": failed}
