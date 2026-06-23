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

st.set_page_config(page_title="Topic RAG Ops", layout="wide")


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

st.title("Topic RAG — Ops Dashboard")

conn = get_db()
sqs  = get_sqs()

with conn.cursor() as cur:
    cur.execute("SELECT COUNT(*) FROM videos WHERE status = 'completed'")
    indexed_videos = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM videos WHERE status = 'failed'")
    failed_videos = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM articles WHERE status = 'completed'")
    indexed_articles = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM articles WHERE status = 'failed'")
    failed_articles = cur.fetchone()[0]

main_q = queue_depth(sqs, os.environ.get("SQS_QUEUE_URL", ""))
dlq    = queue_depth(sqs, os.environ.get("SQS_DLQ_URL", ""))

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Videos Indexed",   indexed_videos)
c2.metric("Articles Indexed", indexed_articles)
c3.metric("Videos Failed",    failed_videos,    delta_color="inverse")
c4.metric("Articles Failed",  failed_articles,  delta_color="inverse")
c5.metric("SQS Queue",        main_q)
c6.metric("DLQ",              dlq,              delta_color="inverse")

st.divider()

# ---------------------------------------------------------------------------
# Sidebar: register a channel + register a website
# ---------------------------------------------------------------------------

with conn.cursor() as cur:
    cur.execute("SELECT name FROM topics ORDER BY name")
    topic_names = [r[0] for r in cur.fetchall()]

with st.sidebar:
    st.header("Register Channel")
    with st.form("add_channel"):
        channel_id   = st.text_input("Channel ID (UC...)")
        channel_name = st.text_input("Channel Name")
        default_topic    = st.selectbox("Default Topic (hint)", topic_names, key="ch_topic")
        videos_to_fetch = st.number_input(
            "Videos to Fetch per Run", min_value=5, max_value=50, value=10,
        )
        max_videos = st.number_input(
            "Max Videos (total cap)", min_value=10, max_value=10000, value=100,
        )
        submitted_ch = st.form_submit_button("Register")

    if submitted_ch:
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
                             videos_to_fetch, max_videos, is_approved, source)
                        SELECT %s, %s, %s, t.id, %s, %s, TRUE, 'manual'
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

    st.divider()

    st.header("Register Website")
    with st.form("add_website"):
        website_id   = st.text_input("Website ID (e.g. hubermanlab.com)")
        website_name = st.text_input("Website Name")
        base_url     = st.text_input("Base URL (https://...)")
        rss_url      = st.text_input("RSS Feed URL (optional)")
        default_topic_w = st.selectbox("Default Topic (hint)", topic_names, key="ws_topic")
        articles_to_fetch = st.number_input(
            "Articles to Fetch per Run", min_value=1, max_value=50, value=10,
        )
        max_articles = st.number_input(
            "Max Articles (total cap)", min_value=10, max_value=10000, value=100,
        )
        submitted_ws = st.form_submit_button("Register")

    if submitted_ws:
        if not website_id.strip():
            st.error("Website ID is required")
        elif not website_name.strip():
            st.error("Website name is required")
        elif not base_url.strip():
            st.error("Base URL is required")
        else:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO websites
                            (id, name, base_url, rss_url, default_topic_id,
                             articles_to_fetch, max_articles)
                        SELECT %s, %s, %s, %s, t.id, %s, %s
                        FROM topics t WHERE t.name = %s
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (
                            website_id.strip(),
                            website_name.strip(),
                            base_url.strip(),
                            rss_url.strip() or None,
                            int(articles_to_fetch),
                            int(max_articles),
                            default_topic_w,
                        ),
                    )
                conn.commit()
                st.success(f"Website '{website_name}' registered.")
            except Exception as exc:
                conn.rollback()
                st.error(f"DB error: {exc}")

# ---------------------------------------------------------------------------
# Tabs: Channels | Websites | Spend | Failures
# ---------------------------------------------------------------------------

tab_channels, tab_websites, tab_spend, tab_failures = st.tabs(
    ["Channels", "Websites", "Model Spend", "Failures"]
)

# ── Channels ──────────────────────────────────────────────────────────────
with tab_channels:
    # ── Pending Channel Approvals ──────────────────────────────────────────
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT c.id, c.name, c.discovered_guest_name,
                   c.discovered_from_video_id, c.subscriber_count,
                   v.title AS source_video_title
            FROM channels c
            LEFT JOIN videos v ON v.id = c.discovered_from_video_id
            WHERE c.is_approved = FALSE AND c.is_rejected = FALSE
            ORDER BY c.created_at DESC
            """
        )
        pending = cur.fetchall()

    if pending:
        st.subheader(f"Pending Channel Approvals ({len(pending)})")
        hdr = st.columns([3, 2, 3, 2, 1, 1])
        for col, label in zip(hdr, ["Channel", "Guest Name", "Source Video", "Subscribers", "", ""]):
            col.markdown(f"**{label}**")
        st.divider()

        for row in pending:
            channel_url = f"https://www.youtube.com/channel/{row['id']}"
            cols = st.columns([3, 2, 3, 2, 1, 1])
            cols[0].markdown(f"[{row['name']}]({channel_url})")
            cols[1].write(row["discovered_guest_name"] or "—")
            if row["discovered_from_video_id"]:
                video_url = f"https://youtu.be/{row['discovered_from_video_id']}"
                label = row["source_video_title"] or row["discovered_from_video_id"]
                cols[2].markdown(f"[{label}]({video_url})")
            else:
                cols[2].write("—")
            subs = row["subscriber_count"]
            cols[3].write(f"{subs:,}" if subs else "—")

            if cols[4].button("Approve", key=f"approve_{row['id']}"):
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE channels SET is_approved = TRUE, is_active = TRUE WHERE id = %s",
                        (row["id"],),
                    )
                conn.commit()
                st.rerun()

            if cols[5].button("Reject", key=f"reject_{row['id']}"):
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE channels SET is_rejected = TRUE WHERE id = %s",
                        (row["id"],),
                    )
                conn.commit()
                st.rerun()

        st.divider()

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
            WHERE c.is_approved = TRUE
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

            new_state = cols[6].toggle("", value=row["is_active"], key=f"toggle_ch_{row['id']}")
            if new_state != row["is_active"]:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE channels SET is_active = %s WHERE id = %s",
                        (new_state, row["id"]),
                    )
                conn.commit()
                st.rerun()

            if cols[7].button("▶", key=f"drill_ch_{row['id']}", help="View videos"):
                st.session_state["drill_channel"] = (
                    None if st.session_state.get("drill_channel") == row["id"] else row["id"]
                )
                st.session_state["drill_channel_name"] = row["name"]

    drill_ch_id = st.session_state.get("drill_channel")
    if drill_ch_id:
        st.subheader(f"Videos — {st.session_state.get('drill_channel_name', drill_ch_id)}")
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT v.id, v.title, v.status, v.topics, v.ingestion_cost,
                       v.error_message, v.created_at
                FROM videos v WHERE v.channel_id = %s ORDER BY v.created_at DESC
                """,
                (drill_ch_id,),
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

# ── Websites ───────────────────────────────────────────────────────────────
with tab_websites:
    st.subheader("Website Value Attribution")

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                w.id,
                w.name,
                t.name                                                               AS default_topic,
                w.is_active,
                w.last_checked_at,
                COUNT(DISTINCT a.id) FILTER (WHERE a.status = 'completed')          AS indexed_articles,
                COALESCE(SUM(a.ingestion_cost), 0)                                   AS total_cost,
                COUNT(DISTINCT q.id)                                                 AS search_count,
                CASE WHEN COUNT(DISTINCT q.id) = 0 THEN NULL
                     ELSE COALESCE(SUM(a.ingestion_cost), 0) / COUNT(DISTINCT q.id)
                END                                                                  AS cost_per_search
            FROM websites w
            LEFT JOIN topics t      ON t.id = w.default_topic_id
            LEFT JOIN articles a    ON a.website_id = w.id
            LEFT JOIN rag_queries q ON w.id = ANY(q.article_ids)
            GROUP BY w.id, w.name, t.name, w.is_active, w.last_checked_at
            ORDER BY total_cost DESC
            """
        )
        ws_rows = cur.fetchall()

    if not ws_rows:
        st.info("No websites registered yet. Add one using the sidebar.")
    else:
        header = st.columns([3, 2, 1, 2, 2, 2, 1, 1])
        for col, label in zip(
            header, ["Website", "Topic", "Indexed", "Ingestion $", "Searches", "$/Search", "Active", ""]
        ):
            col.markdown(f"**{label}**")
        st.divider()

        for row in ws_rows:
            searches   = row["search_count"]
            total_cost = float(row["total_cost"])
            cps        = float(row["cost_per_search"]) if row["cost_per_search"] else None

            if searches > 10 and cps is not None and cps < 0.01:
                badge = "🟢"
            elif searches < 2 and total_cost > 1.0:
                badge = "🔴"
            else:
                badge = "⚪"

            cols = st.columns([3, 2, 1, 2, 2, 2, 1, 1])
            cols[0].write(f"{badge} {row['name']}")
            cols[1].write(row["default_topic"] or "—")
            cols[2].write(row["indexed_articles"])
            cols[3].write(f"${total_cost:.4f}")
            cols[4].write(searches)
            cols[5].write(f"${cps:.4f}" if cps is not None else "—")

            new_state = cols[6].toggle("", value=row["is_active"], key=f"toggle_ws_{row['id']}")
            if new_state != row["is_active"]:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE websites SET is_active = %s WHERE id = %s",
                        (new_state, row["id"]),
                    )
                conn.commit()
                st.rerun()

            if cols[7].button("▶", key=f"drill_ws_{row['id']}", help="View articles"):
                st.session_state["drill_website"] = (
                    None if st.session_state.get("drill_website") == row["id"] else row["id"]
                )
                st.session_state["drill_website_name"] = row["name"]

    drill_ws_id = st.session_state.get("drill_website")
    if drill_ws_id:
        st.subheader(f"Articles — {st.session_state.get('drill_website_name', drill_ws_id)}")
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT a.id, a.title, a.url, a.status, a.topics, a.ingestion_cost,
                       a.error_message, a.created_at
                FROM articles a WHERE a.website_id = %s ORDER BY a.created_at DESC
                """,
                (drill_ws_id,),
            )
            articles = cur.fetchall()

        if not articles:
            st.info("No articles found for this website.")
        else:
            df_articles = pd.DataFrame(articles)
            df_articles["ingestion_cost"] = df_articles["ingestion_cost"].apply(
                lambda x: f"${float(x):.6f}" if x is not None else "—"
            )
            df_articles["created_at"] = pd.to_datetime(df_articles["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(df_articles, use_container_width=True, hide_index=True)

# ── Model Spend ────────────────────────────────────────────────────────────
with tab_spend:
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

# ── Failures ──────────────────────────────────────────────────────────────
with tab_failures:
    col_vf, col_af = st.columns(2)

    with col_vf:
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
            failures_v = cur.fetchall()

        if failures_v:
            st.dataframe(pd.DataFrame(failures_v), use_container_width=True, hide_index=True)
        else:
            st.success("No failed videos.")

    with col_af:
        st.subheader("Failed Articles")
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT a.id, a.title, a.url, w.name AS website, a.error_message, a.created_at
                FROM articles a
                LEFT JOIN websites w ON w.id = a.website_id
                WHERE a.status = 'failed'
                ORDER BY a.created_at DESC
                LIMIT 50
                """
            )
            failures_a = cur.fetchall()

        if failures_a:
            st.dataframe(pd.DataFrame(failures_a), use_container_width=True, hide_index=True)
        else:
            st.success("No failed articles.")
