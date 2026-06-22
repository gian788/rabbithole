"""core/entities.py — Per-chunk entity extraction via Claude Haiku."""
import json
import re

from core.gateway import ModelGateway

_SYSTEM_PROMPT = (
    "Extract the 5 to 8 most important concepts, frameworks, and people explicitly "
    "mentioned in this text. Return ONLY a valid JSON array of lowercase strings. "
    'Example: ["non-duality", "consciousness", "rupert spira", "advaita vedanta"]. '
    "Output nothing else."
)


def extract_chunk_entities(text: str, gateway: ModelGateway) -> list[str]:
    """Return a flat list of lowercase entity strings for a chunk of text.

    Returns [] on any failure — never raises.
    """
    try:
        resp = gateway.get_completion(
            prompt=text[:1000],
            system_prompt=_SYSTEM_PROMPT,
            model="claude-haiku-4-5-20251001",
            provider="anthropic",
            associated_id="entity_extract",
        )
        raw = resp.text_content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[^\n]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())
        result = json.loads(raw)
        return [e.lower() for e in result if isinstance(e, str) and e.strip()]
    except Exception:
        return []
