-- Migration 001: Add channel discovery support
-- Safe to re-run: all statements use IF NOT EXISTS / WHERE guards.

ALTER TABLE channels
  ADD COLUMN IF NOT EXISTS is_approved              BOOLEAN     DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS is_rejected              BOOLEAN     DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS source                   VARCHAR(20) DEFAULT 'manual',
  ADD COLUMN IF NOT EXISTS discovered_from_video_id VARCHAR(50),
  ADD COLUMN IF NOT EXISTS discovered_guest_name    VARCHAR(255),
  ADD COLUMN IF NOT EXISTS subscriber_count         BIGINT;

-- All rows that existed before this migration were manually registered.
UPDATE channels SET is_approved = TRUE WHERE is_approved = FALSE AND source = 'manual';

ALTER TABLE websites
  ADD COLUMN IF NOT EXISTS is_approved BOOLEAN     DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS source      VARCHAR(20) DEFAULT 'manual';

UPDATE websites SET is_approved = TRUE WHERE is_approved = FALSE AND source = 'manual';
