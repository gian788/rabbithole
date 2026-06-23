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
    id                       VARCHAR(50) PRIMARY KEY,
    name                     VARCHAR(255) NOT NULL,
    handle                   VARCHAR(100),
    uploads_playlist_id      VARCHAR(50)  NOT NULL,
    default_topic_id         INT REFERENCES topics(id) ON DELETE SET NULL,
    videos_to_fetch          INT DEFAULT 10,
    max_videos               INT DEFAULT 100,
    is_active                BOOLEAN DEFAULT TRUE,
    is_approved              BOOLEAN DEFAULT FALSE,
    is_rejected              BOOLEAN DEFAULT FALSE,
    source                   VARCHAR(20) DEFAULT 'manual',
    discovered_from_video_id VARCHAR(50),
    discovered_guest_name    VARCHAR(255),
    subscriber_count         BIGINT,
    last_checked_at          TIMESTAMP WITH TIME ZONE,
    created_at               TIMESTAMP WITH TIME ZONE DEFAULT NOW()
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

-- Stores multi-turn chat sessions. user_id is an external identifier from the host app.
CREATE TABLE conversations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      TEXT NOT NULL,
    user_id         TEXT,                    -- external user ID from host app JWT (nullable for anonymous)
    title           TEXT,                    -- set from the first user message
    topic           TEXT,                    -- last predicted topic
    last_message_at TIMESTAMPTZ,             -- updated on every save_message call for efficient sorting
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
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
CREATE INDEX idx_conversations_user_id   ON conversations(user_id);
CREATE INDEX idx_conversations_user_time ON conversations(user_id, last_message_at DESC);
CREATE INDEX idx_messages_conversation   ON messages(conversation_id, created_at);
CREATE INDEX idx_channels_topic  ON channels(default_topic_id);
CREATE INDEX idx_videos_channel  ON videos(channel_id);
CREATE INDEX idx_videos_status   ON videos(status);
CREATE INDEX idx_videos_topics   ON videos USING GIN(topics);  -- fast array containment queries
CREATE INDEX idx_telemetry_model ON model_telemetry(model);

-- ---------------------------------------------------------------------------
-- Article ingestion (symmetric with channels/videos)
-- ---------------------------------------------------------------------------

CREATE TABLE websites (
    id                VARCHAR(100) PRIMARY KEY,
    name              VARCHAR(255) NOT NULL,
    base_url          TEXT NOT NULL,
    rss_url           TEXT,
    default_topic_id  INT REFERENCES topics(id) ON DELETE SET NULL,
    articles_to_fetch INT DEFAULT 10,
    max_articles      INT DEFAULT 100,
    is_active         BOOLEAN DEFAULT TRUE,
    is_approved       BOOLEAN DEFAULT FALSE,
    source            VARCHAR(20) DEFAULT 'manual',
    last_checked_at   TIMESTAMP WITH TIME ZONE,
    created_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE articles (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    website_id       VARCHAR(100) REFERENCES websites(id) ON DELETE CASCADE,
    url              TEXT NOT NULL UNIQUE,       -- deduplication key
    title            VARCHAR(500) NOT NULL,
    author           VARCHAR(255),
    published_at     TIMESTAMP WITH TIME ZONE,
    topics           TEXT[]  DEFAULT '{}',
    primary_topic    VARCHAR(100),
    status           VARCHAR(20) DEFAULT 'discovered',  -- discovered|processing|completed|failed
    error_message    TEXT,
    s3_path          VARCHAR(512),
    ingestion_tokens INT     DEFAULT 0,
    ingestion_cost   NUMERIC(10, 6) DEFAULT 0,
    processed_at     TIMESTAMP WITH TIME ZONE,
    created_at       TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_articles_status  ON articles(status);
CREATE INDEX idx_articles_website ON articles(website_id);
CREATE INDEX idx_articles_topics  ON articles USING GIN(topics);

ALTER TABLE rag_queries ADD COLUMN article_ids TEXT[] DEFAULT '{}';

-- Seed topics
INSERT INTO topics (name, description) VALUES
    ('consciousness',       'Human consciousness, mind, awareness, and perception'),
    ('alternative_history', 'Alternative history, ancient Egypt, Atlantis, lost civilizations'),
    ('biohacking',          'Biohacking, longevity, nootropics, and performance optimization'),
    ('spirituality',        'Spirituality, metaphysics, meditation, and esoteric knowledge');
