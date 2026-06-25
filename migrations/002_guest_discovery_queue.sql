-- migrations/002_guest_discovery_queue.sql
-- Decouples guest name extraction from YouTube API calls.
-- Video processing enqueues guests here; a separate discovery job drains the queue.
CREATE TABLE IF NOT EXISTS pending_guest_discovery (
    id                BIGSERIAL PRIMARY KEY,
    guest_name        VARCHAR(255) NOT NULL,
    source_video_id   VARCHAR(50)  NOT NULL,
    source_channel_id VARCHAR(50),
    status            VARCHAR(20)  DEFAULT 'pending',  -- pending|discovered|not_found|skipped
    attempts          INT          DEFAULT 0,
    last_attempted_at TIMESTAMP WITH TIME ZONE,
    linked_channel_id VARCHAR(50),  -- set on auto or manual link
    created_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (guest_name, source_video_id)
);
