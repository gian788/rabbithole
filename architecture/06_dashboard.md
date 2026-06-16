# Stage 6: Dashboard

## Responsibility

`dashboard/app.py` — a Streamlit developer portal for monitoring ingestion health, managing channels, and analysing the cost vs. value of indexed content.

---

## Purpose

The dashboard answers three operational questions:

1. **Health:** is the pipeline running? how many videos are indexed vs. failing?
2. **Management:** add new channels, configure their topic and fetch depth
3. **Economics:** which channels generate real user queries? which are costing money without being used?

---

## Live Metrics (top row)

Three metric tiles updated on every page load:

| Metric | Source | Query / API call |
|---|---|---|
| Videos indexed | Neon PostgreSQL | `SELECT COUNT(*) FROM videos WHERE status = 'completed'` |
| In SQS queue | AWS SQS | `get_queue_attributes(AttributeNames=["ApproximateNumberOfMessages"])` |
| DLQ failures | AWS SQS | Same call on the DLQ URL |

SQS calls are wrapped in `try/except` — if IAM permissions are not configured locally, the tile shows `"N/A"` instead of crashing.

---

## Sidebar: Add Channel

Fields:
- **YouTube Channel ID** (`UC...`) — must start with `UC`
- **Channel Name** — display name
- **Default Topic** — hint for LLM classification (dropdown from `topics` table)
- **Videos to Fetch** — how many latest videos to poll per run (5–50, default 10)

On submit:
```python
uploads_playlist_id = "UU" + channel_id[2:]   # UC → UU (1 quota unit per 50 videos)

INSERT INTO channels (id, name, uploads_playlist_id, default_topic_id, videos_to_fetch)
SELECT %(id)s, %(name)s, %(playlist)s, t.id, %(n)s
FROM topics t WHERE t.name = %(topic)s
ON CONFLICT (id) DO NOTHING
```

The `ON CONFLICT DO NOTHING` means re-submitting the same channel ID is a no-op — safe to click multiple times.

---

## Value Attribution Table

The core analytics view. Shows every channel with aggregated cost and usage data:

```sql
SELECT
    c.id,
    c.name,
    t.name                                AS default_topic,
    c.is_active,
    COALESCE(SUM(v.ingestion_cost), 0)    AS total_cost,
    COUNT(DISTINCT q.id)                  AS search_count,
    CASE
        WHEN COUNT(DISTINCT q.id) = 0 THEN NULL
        ELSE COALESCE(SUM(v.ingestion_cost), 0) / COUNT(DISTINCT q.id)
    END                                   AS cost_per_search
FROM channels c
LEFT JOIN topics t ON t.id = c.default_topic_id
LEFT JOIN videos v ON v.channel_id = c.id AND v.status = 'completed'
LEFT JOIN rag_queries q ON c.id = ANY(q.video_ids)
GROUP BY c.id, c.name, t.name, c.is_active
ORDER BY cost_per_search DESC NULLS LAST
```

### Row Colour Coding

| Condition | Colour | Meaning |
|---|---|---|
| `search_count > 10 AND cost_per_search < 0.01` | Green | High-value channel — keep and grow |
| `search_count < 2 AND total_cost > 1.0` | Red | Low-value, high-cost — candidate for deactivation |
| All others | Default | Monitor |

### Active Toggle

Each row has a toggle button. Flipping it writes immediately to the DB:

```python
is_active = st.toggle("Active", value=row.is_active, key=row.channel_id)
if is_active != row.is_active:
    cur.execute(
        "UPDATE channels SET is_active = %s WHERE id = %s",
        (is_active, row.channel_id)
    )
    conn.commit()
    st.rerun()
```

Setting `is_active = FALSE` stops the channel from being polled in future `fetch_lambda` runs. Its already-indexed videos remain in Pinecone and S3.

---

## Connection Management

```python
@st.cache_resource
def get_db():
    return psycopg2.connect(os.environ["DATABASE_URL"])

@st.cache_resource
def get_sqs():
    return boto3.client("sqs", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
```

`@st.cache_resource` ensures connections are created once per Streamlit session and reused across reruns, rather than reconnecting on every interaction.

---

## Running Locally

```bash
export $(cat .env | grep -v '^#' | xargs)
streamlit run dashboard/app.py
```

SQS metrics will show `"N/A"` if AWS credentials are not configured — everything else works against Neon PostgreSQL directly.

---

## Future Additions (Phase 2)

- **Model spend breakdown** — pie chart of cost by provider/model from `model_telemetry`
- **Ingestion timeline** — line chart of videos processed per day
- **Failed video requeue** — button to move DLQ messages back to main queue
- **Topic distribution** — how many videos per topic, stacked bar per channel
