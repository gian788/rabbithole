-- ============================================================
-- YouTube Topic RAG -- PostgreSQL Schema
-- Target: Neon Serverless PostgreSQL
-- ============================================================

CREATE TABLE topics (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL UNIQUE,  -- 'consciousness', 'biohacking', etc.
    description TEXT
);

CREATE TABLE channels (
    id                  VARCHAR(50) PRIMARY KEY,  -- YouTube Channel ID (UC...)
    name                VARCHAR(255) NOT NULL,
    handle              VARCHAR(100),
    uploads_playlist_id VARCHAR(50)  NOT NULL,    -- UU... (derived from UC...; 1 quota unit = 50 videos)
    default_topic_id    INT REFERENCES topics(id) ON DELETE SET NULL,  -- classification hint only
    videos_to_fetch     INT DEFAULT 10,
    max_videos          INT DEFAULT 100,          -- hard cap on total indexed videos per channel
    is_active           BOOLEAN DEFAULT TRUE,
    last_checked_at     TIMESTAMP WITH TIME ZONE,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE videos (
    id               VARCHAR(50) PRIMARY KEY,       -- YouTube Video ID
    channel_id       VARCHAR(50) REFERENCES channels(id) ON DELETE CASCADE,
    title            VARCHAR(500) NOT NULL,
    description      TEXT,
    view_count       BIGINT  DEFAULT 0,
    like_count       BIGINT  DEFAULT 0,
    published_at     TIMESTAMP WITH TIME ZONE,
    topics           TEXT[]  DEFAULT '{}',          -- multi-topic: ['consciousness', 'biohacking']
    status           VARCHAR(20) DEFAULT 'discovered',  -- discovered|processing|completed|failed
    error_message    TEXT,
    s3_path          VARCHAR(512),
    ingestion_tokens INT     DEFAULT 0,
    ingestion_cost   NUMERIC(10, 6) DEFAULT 0,
    processed_at     TIMESTAMP WITH TIME ZONE,
    created_at       TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Logs every user query with the topic it was classified into and which videos were cited.
-- Enables cost attribution (which channels drive real user value).
CREATE TABLE rag_queries (
    id             BIGSERIAL PRIMARY KEY,
    user_query     TEXT    NOT NULL,
    queried_topic  VARCHAR(100),
    video_ids      TEXT[]  DEFAULT '{}',  -- video IDs returned as citations
    retrieval_cost NUMERIC(10, 6) DEFAULT 0,
    queried_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Tracks every LLM/embedding API call: latency, tokens, cost, provider.
-- Used by the dashboard for cost attribution and budget monitoring.
CREATE TABLE model_telemetry (
    id               BIGSERIAL PRIMARY KEY,
    transaction_type VARCHAR(20)  NOT NULL,  -- 'embedding' | 'completion'
    provider         VARCHAR(50)  NOT NULL,  -- 'openai' | 'anthropic'
    model            VARCHAR(100) NOT NULL,
    input_tokens     INT     DEFAULT 0,
    output_tokens    INT     DEFAULT 0,
    latency_ms       INT     NOT NULL,
    cost             NUMERIC(10, 7) DEFAULT 0,
    associated_id    VARCHAR(100),           -- video_id or 'intent_router', 'chapter_gen', etc.
    created_at       TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Stores multi-turn chat sessions. session_id is a client-generated UUID (no accounts).
CREATE TABLE conversations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  TEXT NOT NULL,
    title       TEXT,                    -- set from the first user message
    topic       TEXT,                    -- last predicted topic
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    citations       JSONB,               -- non-null for assistant messages
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_conversations_session   ON conversations(session_id);
CREATE INDEX idx_messages_conversation   ON messages(conversation_id, created_at);
CREATE INDEX idx_channels_topic  ON channels(default_topic_id);
CREATE INDEX idx_videos_channel  ON videos(channel_id);
CREATE INDEX idx_videos_status   ON videos(status);
CREATE INDEX idx_videos_topics   ON videos USING GIN(topics);  -- fast array containment queries
CREATE INDEX idx_telemetry_model ON model_telemetry(model);

-- Seed topics
INSERT INTO topics (name, description) VALUES
    ('consciousness',       'Human consciousness, mind, awareness, and perception'),
    ('alternative_history', 'Alternative history, ancient Egypt, Atlantis, lost civilizations'),
    ('biohacking',          'Biohacking, longevity, nootropics, and performance optimization'),
    ('spirituality',        'Spirituality, metaphysics, meditation, and esoteric knowledge');
