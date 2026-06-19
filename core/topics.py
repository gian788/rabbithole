"""
core/topics.py
Shared topic classification used by both video and article ingestion workers.
"""
import json
import re

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
