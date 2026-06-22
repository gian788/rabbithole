"""
core/topics.py
Shared topic classification used by both video and article ingestion workers.
"""
import json
import re
from dataclasses import dataclass, field

from core.gateway import ModelGateway


def classify_topics(
    title: str,
    text_excerpt: str,
    available_topics: list[str],
    default_hint: str | None,
    gateway: ModelGateway,
) -> list[str]:
    """
    Use Claude Haiku to classify content into one or more available topics.

    Returns a validated list of topic strings from available_topics.
    Falls back to [default_hint] on any error.
    """
    hint = default_hint or (available_topics[0] if available_topics else "")
    system_prompt = (
        f"This content primarily covers {hint}. "
        f"Classify it into ALL relevant topics from this list: {available_topics}. "
        "Return ONLY a valid JSON array of strings. "
        'Example: ["consciousness", "spirituality"]. Output nothing else.'
    )
    try:
        resp = gateway.get_completion(
            prompt=f"Title: {title}\n\nExcerpt (first 500 chars): {text_excerpt[:500]}",
            system_prompt=system_prompt,
            model="claude-haiku-4-5-20251001",
            provider="anthropic",
            associated_id="topic_classify",
        )
        raw = resp.text_content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[^\n]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())
        topics = json.loads(raw)
        validated = [t for t in topics if t in available_topics]
        return validated if validated else ([hint] if hint else available_topics[:1])
    except Exception:
        return [hint] if hint else available_topics[:1]


@dataclass
class VideoMeta:
    topics: list[str]
    host: str | None
    guests: list[str] = field(default_factory=list)


def classify_video_meta(
    title: str,
    channel_name: str,
    text_excerpt: str,
    available_topics: list[str],
    default_hint: str | None,
    gateway: ModelGateway,
) -> VideoMeta:
    """Classify a YouTube video into topics and extract host + guests via a single Haiku call.

    Returns a VideoMeta with fallback topics on any failure — never raises.
    """
    hint = default_hint or (available_topics[0] if available_topics else "")
    fallback = VideoMeta(
        topics=[hint] if hint else available_topics[:1],
        host=None,
        guests=[],
    )
    system_prompt = (
        "Classify this YouTube video and extract people. "
        "Return ONLY a valid JSON object with exactly three keys: "
        f'"topics" (array of strings from this list only: {available_topics}; '
        f"include ALL relevant ones; hint: {hint}), "
        '"host" (string or null — the channel\'s regular host inferred from the channel name), '
        '"guests" (array of strings — guest names from the title; empty array if none). '
        'Example: {"topics": ["consciousness"], "host": "Joe Rogan", "guests": ["Graham Hancock"]}. '
        "Output nothing else."
    )
    try:
        resp = gateway.get_completion(
            prompt=(
                f"Title: {title}\n"
                f"Channel: {channel_name}\n"
                f"Excerpt (first 500 chars): {text_excerpt[:500]}"
            ),
            system_prompt=system_prompt,
            model="claude-haiku-4-5-20251001",
            provider="anthropic",
            associated_id="video_meta_classify",
        )
        raw = resp.text_content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[^\n]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())
        data = json.loads(raw)
        validated = [t for t in (data.get("topics") or []) if t in available_topics]
        topics = validated if validated else ([hint] if hint else available_topics[:1])
        host = data.get("host") or None
        guests = [
            g for g in (data.get("guests") or [])
            if isinstance(g, str) and g.strip()
        ]
        return VideoMeta(topics=topics, host=host, guests=guests)
    except Exception:
        return fallback
