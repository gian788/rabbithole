import json
import os
from typing import Optional

import psycopg2
import psycopg2.extras


def get_connection() -> psycopg2.extensions.connection:
    return psycopg2.connect(os.environ["DATABASE_URL"])


def get_topic_names(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT name FROM topics ORDER BY name")
        return [row[0] for row in cur.fetchall()]


def get_channel_default_topic(conn, channel_id: str) -> Optional[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT t.name
            FROM channels c
            JOIN topics t ON t.id = c.default_topic_id
            WHERE c.id = %s
            """,
            (channel_id,),
        )
        row = cur.fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------

def create_conversation(
    conn, session_id: str, user_id: Optional[str] = None, title: Optional[str] = None
) -> str:
    """Insert a new conversation row and return its UUID."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO conversations (session_id, user_id, title)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (session_id, user_id, title),
        )
        conversation_id = str(cur.fetchone()[0])
    conn.commit()
    return conversation_id


def get_conversation_messages(conn, conversation_id: str, limit: int = 6) -> list[dict]:
    """Return the last `limit` messages for a conversation, oldest first."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT role, content, citations, created_at
            FROM messages
            WHERE conversation_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (conversation_id, limit),
        )
        rows = cur.fetchall()
    # Reverse so oldest is first (DESC fetch → reverse for chronological order)
    return [dict(r) for r in reversed(rows)]


def save_message(
    conn,
    conversation_id: str,
    role: str,
    content: str,
    citations: Optional[list] = None,
) -> None:
    citations_json = json.dumps(citations) if citations is not None else None
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO messages (conversation_id, role, content, citations)
            VALUES (%s, %s, %s, %s)
            """,
            (conversation_id, role, content, citations_json),
        )
        cur.execute(
            "UPDATE conversations SET last_message_at = NOW() WHERE id = %s",
            (conversation_id,),
        )
    conn.commit()


def update_conversation(conn, conversation_id: str, topic: Optional[str] = None, title: Optional[str] = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE conversations
            SET updated_at = NOW(),
                topic      = COALESCE(%s, topic),
                title      = COALESCE(%s, title)
            WHERE id = %s
            """,
            (topic, title, conversation_id),
        )
    conn.commit()


def get_conversation_owner(conn, conversation_id: str) -> Optional[str]:
    """Return the user_id that owns this conversation, or None if anonymous/not found."""
    with conn.cursor() as cur:
        cur.execute("SELECT user_id FROM conversations WHERE id = %s", (conversation_id,))
        row = cur.fetchone()
    return row[0] if row else None


def get_user_conversations(conn, user_id: str, limit: int = 30) -> list[dict]:
    """Return conversation summaries for a user, most recently active first."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                c.id,
                c.title,
                c.topic,
                c.last_message_at,
                (
                    SELECT content
                    FROM messages m
                    WHERE m.conversation_id = c.id AND m.role = 'assistant'
                    ORDER BY m.created_at DESC
                    LIMIT 1
                ) AS last_assistant_message
            FROM conversations c
            WHERE c.user_id = %s AND c.last_message_at IS NOT NULL
            ORDER BY c.last_message_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        )
        rows = cur.fetchall()
    return [
        {
            "id": str(r["id"]),
            "title": r["title"] or "Untitled",
            "topic": r["topic"],
            "last_message_at": r["last_message_at"].isoformat() if r["last_message_at"] else None,
            "preview": (r["last_assistant_message"] or "")[:120],
        }
        for r in rows
    ]
