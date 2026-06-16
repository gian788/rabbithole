"""
dashboard/app.py
Streamlit developer ops portal.

Run:  streamlit run dashboard/app.py
"""
import os

from dotenv import load_dotenv
load_dotenv()

import boto3
import pandas as pd
import psycopg2
import psycopg2.extras
import streamlit as st

st.set_page_config(page_title="YouTube RAG Ops", layout="wide")


# ---------------------------------------------------------------------------
# Cached connections
# ---------------------------------------------------------------------------

def get_db():
    if "db_conn" not in st.session_state or st.session_state.db_conn.closed:
        st.session_state.db_conn = psycopg2.connect(os.environ["DATABASE_URL"])
    return st.session_state.db_conn


@st.cache_resource
def get_sqs():
    return boto3.client(
        "sqs", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )


def queue_depth(sqs, url: str) -> str:
    try:
        resp = sqs.get_queue_attributes(
            QueueUrl=url, AttributeNames=["ApproximateNumberOfMessages"]
        )
        return resp["Attributes"]["ApproximateNumberOfMessages"]
    except Exception:
        return "N/A"


# ---------------------------------------------------------------------------
# Header + live metrics
# ---------------------------------------------------------------------------

st.title("YouTube Topic RAG — Ops Dashboard")

conn = get_db()
sqs  = get_sqs()

with conn.cursor() as cur:
    cur.execute("SELECT COUNT(*) FROM videos WHERE status = 'completed'")
    indexed = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM videos WHERE status = 'processing'")
    processing = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM videos WHERE status = 'failed'")
    failed = cur.fetchone()[0]

main_q = queue_depth(sqs, os.environ.get("SQS_QUEUE_URL", ""))
dlq    = queue_depth(sqs, os.environ.get("SQS_DLQ_URL", ""))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Indexed",    indexed)
c2.metric("Processing", processing)
c3.metric("Failed",     failed)
c4.metric("SQS Queue",  main_q)
c5.metric("DLQ",        dlq, delta_color="inverse")

st.divider()

# ---------------------------------------------------------------------------
# Sidebar: register a channel
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Register Channel")
    with st.form("add_channel"):
        channel_id   = st.text_input("Channel ID (UC...)")
        channel_name = st.text_input("Channel Name")

        with conn.cursor() as cur:
            cur.execute("SELECT name FROM topics ORDER BY name")
            topic_names = [r[0] for r in cur.fetchall()]

        default_topic    = st.selectbox("Default Topic (hint)", topic_names)
        videos_to_fetch = st.number_input(
            "Videos to Fetch per Run", min_value=5, max_value=50, value=10,
            help="How many latest videos to check on each 6-hour cron run."
        )
        max_videos = st.number_input(
            "Max Videos (total cap)", min_value=10, max_value=10000, value=100,
            help="Hard ceiling on how many videos are ever indexed for this channel."
        )
        submitted = st.form_submit_button("Register")

    if submitted:
        if not channel_id.startswith("UC"):
            st.error("Channel ID must start with 'UC'")
        elif not channel_name.strip():
            st.error("Channel name is required")
        else:
            uploads_playlist_id = "UU" + channel_id[2:]
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO channels
                            (id, name, uploads_playlist_id, default_topic_id,
                             videos_to_fetch, max_videos)
                        SELECT %s, %s, %s, t.id, %s, %s
                        FROM topics t WHERE t.name = %s
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (
                            channel_id,
                            channel_name.strip(),
                            uploads_playlist_id,
                            int(videos_to_fetch),
                            int(max_videos),
                            default_topic,
                        ),
                    )
                conn.commit()
                st.success(f"Channel '{channel_name}' registered.")
            except Exception as exc:
                conn.rollback()
                st.error(f"DB error: {exc}")

# ---------------------------------------------------------------------------
# Value Attribution Table
# ---------------------------------------------------------------------------

st.subheader("Channel Value Attribution")

with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
    cur.execute(
        """
        SELECT
            c.id,
            c.name,
            t.name                                                              AS default_topic,
            c.is_active,
            COUNT(DISTINCT v.id) FILTER (WHERE v.status = 'completed')         AS indexed_videos,
            COALESCE(SUM(v.ingestion_cost), 0)                                  AS total_cost,
            COUNT(DISTINCT q.id)                                                AS search_count,
            CASE WHEN COUNT(DISTINCT q.id) = 0 THEN NULL
                 ELSE COALESCE(SUM(v.ingestion_cost), 0) / COUNT(DISTINCT q.id)
            END                                                                 AS cost_per_search
        FROM channels c
        LEFT JOIN topics t      ON t.id = c.default_topic_id
        LEFT JOIN videos v      ON v.channel_id = c.id
        LEFT JOIN rag_queries q ON c.id = ANY(q.video_ids)
        GROUP BY c.id, c.name, t.name, c.is_active
        ORDER BY total_cost DESC
        """
    )
    rows = cur.fetchall()

if not rows:
    st.info("No channels registered yet. Add one using the sidebar.")
else:
    header = st.columns([3, 2, 1, 2, 2, 2, 1, 1])
    for col, label in zip(
        header, ["Channel", "Topic", "Indexed", "Ingestion $", "Searches", "$/Search", "Active", ""]
    ):
        col.markdown(f"**{label}**")
    st.divider()

    for row in rows:
        searches   = row["search_count"]
        total_cost = float(row["total_cost"])
        cps        = float(row["cost_per_search"]) if row["cost_per_search"] else None

        # Colour coding
        if searches > 10 and cps is not None and cps < 0.01:
            badge = "🟢"
        elif searches < 2 and total_cost > 1.0:
            badge = "🔴"
        else:
            badge = "⚪"

        channel_url = f"https://www.youtube.com/channel/{row['id']}"

        cols = st.columns([3, 2, 1, 2, 2, 2, 1, 1])
        cols[0].markdown(f"{badge} {row['name']} &nbsp;[↗]({channel_url})")
        cols[1].write(row["default_topic"] or "—")
        cols[2].write(row["indexed_videos"])
        cols[3].write(f"${total_cost:.4f}")
        cols[4].write(searches)
        cols[5].write(f"${cps:.4f}" if cps is not None else "—")

        new_state = cols[6].toggle("", value=row["is_active"], key=f"toggle_{row['id']}")
        if new_state != row["is_active"]:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE channels SET is_active = %s WHERE id = %s",
                    (new_state, row["id"]),
                )
            conn.commit()
            st.rerun()

        if cols[7].button("▶", key=f"drill_{row['id']}", help="View videos"):
            st.session_state["drill_channel"] = (
                None if st.session_state.get("drill_channel") == row["id"] else row["id"]
            )
            st.session_state["drill_channel_name"] = row["name"]

# ---------------------------------------------------------------------------
# Channel drill-down: videos
# ---------------------------------------------------------------------------

drill_id = st.session_state.get("drill_channel")
if drill_id:
    st.subheader(f"Videos — {st.session_state.get('drill_channel_name', drill_id)}")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                v.id,
                v.title,
                v.status,
                v.topics,
                v.ingestion_cost,
                v.error_message,
                v.created_at
            FROM videos v
            WHERE v.channel_id = %s
            ORDER BY v.created_at DESC
            """,
            (drill_id,),
        )
        videos = cur.fetchall()

    if not videos:
        st.info("No videos found for this channel.")
    else:
        df_videos = pd.DataFrame(videos)
        df_videos["ingestion_cost"] = df_videos["ingestion_cost"].apply(
            lambda x: f"${float(x):.6f}" if x is not None else "—"
        )
        df_videos["created_at"] = pd.to_datetime(df_videos["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(df_videos, use_container_width=True, hide_index=True)
    st.divider()


# ---------------------------------------------------------------------------
# Model Spend Summary
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Model Spend Summary")

with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
    cur.execute(
        """
        SELECT
            provider,
            model,
            transaction_type,
            COUNT(*)               AS calls,
            SUM(input_tokens)      AS total_input_tokens,
            SUM(output_tokens)     AS total_output_tokens,
            ROUND(AVG(latency_ms)) AS avg_latency_ms,
            SUM(cost)              AS total_cost
        FROM model_telemetry
        GROUP BY provider, model, transaction_type
        ORDER BY total_cost DESC
        """
    )
    telemetry = cur.fetchall()

if telemetry:
    df = pd.DataFrame(telemetry)
    df["total_cost"] = df["total_cost"].apply(lambda x: f"${float(x):.6f}")
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No model telemetry recorded yet.")

# ---------------------------------------------------------------------------
# Failed Videos
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Failed Videos")

with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
    cur.execute(
        """
        SELECT v.id, v.title, c.name AS channel, v.error_message, v.created_at
        FROM videos v
        JOIN channels c ON c.id = v.channel_id
        WHERE v.status = 'failed'
        ORDER BY v.created_at DESC
        LIMIT 50
        """
    )
    failures = cur.fetchall()

if failures:
    df_fail = pd.DataFrame(failures)
    st.dataframe(df_fail, use_container_width=True, hide_index=True)
else:
    st.success("No failed videos.")
