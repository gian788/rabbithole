# Stage 1: Fetching

## Responsibility

`ingestion/fetch_lambda.py` — polls YouTube for new videos from known channels and queues them for processing.

---

## YouTube Quota Management

The YouTube Data API v3 gives 10,000 quota units per day. Naive usage burns quota fast:

| Endpoint | Quota cost | Notes |
|---|---|---|
| `videos.list` | 1 unit per call | Returns up to 50 videos per call |
| `playlistItems.list` | 1 unit per call | Returns up to 50 items per call |
| `search.list` | 100 units per call | **Avoided** — 100× more expensive |

**The `UC → UU` playlist trick:**
Every YouTube channel has a hidden uploads playlist. Its ID is derived by replacing the `UC` prefix in the channel ID with `UU`:

```
Channel ID:          UCxxxxxxxxxxxxxxxxxxxxxx
Uploads playlist:    UUxxxxxxxxxxxxxxxxxxxxxx
```

Calling `playlistItems.list(playlistId="UU...")` returns the channel's latest uploads using just **1 quota unit per 50 videos** — the same cost as a single `videos.list` call. This is the primary fetch mechanism.

---

## Flow

```
1. Query DB for active channels
   SELECT id, name, uploads_playlist_id, default_topic_id, videos_to_fetch
   FROM channels WHERE is_active = TRUE

2. For each channel → call YouTube API
   playlistItems.list(playlistId=uploads_playlist_id, maxResults=videos_to_fetch)
   → returns list of video IDs

3. Filter out already-known videos
   SELECT id FROM videos WHERE id = ANY(%(ids)s)
   new_ids = api_ids - db_ids

4. Fetch full metadata for new videos (1 API call per batch of 50)
   videos.list(part="snippet,statistics", id=",".join(new_ids))
   → title, description, publishedAt, viewCount, likeCount

5. Insert new video rows into DB (status = 'discovered')
   INSERT INTO videos (id, channel_id, title, description,
                       view_count, like_count, published_at, status)
   ON CONFLICT (id) DO NOTHING

6. Dispatch to SQS in batches of 10
   Message body: {"video_id": "...", "channel_id": "..."}

7. Update channel last_checked_at
   UPDATE channels SET last_checked_at = NOW() WHERE id = %(id)s
```

---

## SQS Message Structure

```json
{
  "video_id": "dQw4w9WgXcQ",
  "channel_id": "UCxxxxxxxxxxxxxxxxxxxxxx"
}
```

Topic classification is intentionally deferred to the worker Lambda, which has the full transcript context available.

---

## Queue Configuration

| Setting | Value | Reason |
|---|---|---|
| Long-polling wait time | 20 seconds | Reduces empty-receive API calls → cost saving |
| Message retention | 1 day | Videos are re-queued on next cron run if lost |
| DLQ redrive policy | maxReceiveCount = 3 | After 3 failures, message moves to DLQ |
| DLQ retention | 7 days | Time to investigate + manually reprocess |

---

## EventBridge Schedule

```
cron(0 */6 * * ? *)   →   runs every 6 hours
```

Each run fetches `videos_to_fetch` (default: 10) latest videos per active channel. For a channel uploading daily, this catches new content within 6 hours.

---

## Video State Machine

```
             fetch_lambda
                  │
                  ▼
            [ discovered ]
                  │
         SQS → worker_lambda
                  │
            [ processing ]
           /              \
    (success)            (error)
          │                  │
    [ completed ]        [ failed ]
                        error_message stored
```

Failed videos are retried up to 3× by SQS before landing in the DLQ. Failures due to missing transcripts (`TranscriptsDisabled`, `NoTranscriptFound`) skip SQS retry since retrying is pointless.

---

## Error Handling

| Error | Behaviour |
|---|---|
| YouTube 404 on playlist | Auto-deactivates channel: `UPDATE channels SET is_active=FALSE` |
| YouTube 403 (quota exceeded) | Logs warning, stops processing remaining channels for this run |
| DB connection error | Fatal — re-raises, Lambda execution fails, EventBridge records it in CloudWatch |
| New videos = 0 | Silent success — `last_checked_at` still updated |
