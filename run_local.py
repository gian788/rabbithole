"""
run_local.py — Local development runner.

Calls Lambda handlers directly without AWS infrastructure.
Requires a .env file with DATABASE_URL, OPENAI_API_KEY, ANTHROPIC_API_KEY,
PINECONE_API_KEY, PINECONE_INDEX_NAME, and S3_LOCAL_PATH.

Usage
-----
# Process a specific video
uv run python run_local.py --video VIDEO_ID --channel CHANNEL_ID

# Run the fetch loop against all active channels in the DB
uv run python run_local.py --fetch

# Process every 'discovered' video currently in the DB
uv run python run_local.py --process-pending
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


def run_fetch() -> None:
    from ingestion.fetch_lambda import lambda_handler
    result = lambda_handler({}, None)
    print(f"\nFetch complete: {result}")


def run_process(video_id: str, channel_id: str) -> None:
    import boto3
    from pinecone import Pinecone

    from core.db import get_connection
    from core.gateway import ModelGateway
    from ingestion.worker_lambda import process_video

    db_conn = get_connection()
    s3_client = boto3.client(
        "s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(
        os.environ["PINECONE_INDEX_NAME"]
    )
    gateway = ModelGateway(db_conn=db_conn)

    process_video(
        video_id=video_id,
        channel_id=channel_id,
        db_conn=db_conn,
        s3_client=s3_client,
        pinecone_index=index,
        gateway=gateway,
    )
    db_conn.close()


def run_process_pending() -> None:
    import boto3
    from pinecone import Pinecone

    from core.db import get_connection
    from core.gateway import ModelGateway
    from ingestion.worker_lambda import process_video

    db_conn = get_connection()
    s3_client = boto3.client(
        "s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(
        os.environ["PINECONE_INDEX_NAME"]
    )
    gateway = ModelGateway(db_conn=db_conn)

    import psycopg2.extras
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
                pinecone_index=index,
                gateway=gateway,
            )
        except Exception as exc:
            print(f"    FAILED: {exc}", file=sys.stderr)

    db_conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local dev runner for youtube-topic-rag")
    parser.add_argument("--video",            help="YouTube video ID to process")
    parser.add_argument("--channel",          help="YouTube channel ID (required with --video)")
    parser.add_argument("--fetch",            action="store_true", help="Run fetch_lambda")
    parser.add_argument("--process-pending",  action="store_true", help="Process all discovered videos")
    args = parser.parse_args()

    if args.fetch:
        run_fetch()
    elif args.process_pending:
        run_process_pending()
    elif args.video and args.channel:
        run_process(args.video, args.channel)
    else:
        parser.print_help()
        sys.exit(1)
