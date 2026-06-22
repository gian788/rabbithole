"""Unit tests for core/entities.py — per-chunk entity extraction."""
from unittest.mock import MagicMock

import pytest

from core.entities import extract_chunk_entities


def _make_gateway(response_text: str):
    gw = MagicMock()
    gw.get_completion.return_value = MagicMock(text_content=response_text, cost=0.0001)
    return gw


def test_happy_path_returns_lowercase_list():
    gw = _make_gateway('["Non-Duality", "Consciousness", "Rupert Spira"]')
    result = extract_chunk_entities("some text about consciousness", gw)
    assert result == ["non-duality", "consciousness", "rupert spira"]


def test_bad_json_returns_empty_list():
    gw = _make_gateway("not valid json at all")
    result = extract_chunk_entities("text", gw)
    assert result == []


def test_api_error_returns_empty_list():
    gw = MagicMock()
    gw.get_completion.side_effect = RuntimeError("API down")
    result = extract_chunk_entities("text", gw)
    assert result == []


def test_filters_non_string_items():
    gw = _make_gateway('["concept", 42, null, "person"]')
    result = extract_chunk_entities("text", gw)
    assert result == ["concept", "person"]


def test_strips_code_fence():
    gw = _make_gateway('```json\n["concept"]\n```')
    result = extract_chunk_entities("text", gw)
    assert result == ["concept"]


def test_empty_array_returns_empty_list():
    gw = _make_gateway("[]")
    result = extract_chunk_entities("text", gw)
    assert result == []


def test_text_truncated_to_1000_chars():
    gw = _make_gateway('["concept"]')
    long_text = "a" * 2000
    extract_chunk_entities(long_text, gw)
    call_kwargs = gw.get_completion.call_args.kwargs
    assert len(call_kwargs["prompt"]) <= 1000


def test_whitespace_only_strings_filtered():
    gw = _make_gateway('["concept", "   ", "person"]')
    result = extract_chunk_entities("text", gw)
    assert result == ["concept", "person"]
