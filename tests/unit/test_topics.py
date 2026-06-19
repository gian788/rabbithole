"""Unit tests for core/topics.py — topic classification via LLM."""
import json
from unittest.mock import MagicMock

from core.topics import classify_topics

AVAILABLE = ["consciousness", "biohacking", "spirituality", "alternative_history"]


def _make_gateway(response_text):
    gw = MagicMock()
    gw.get_completion.return_value = MagicMock(text_content=response_text, cost=0.001)
    return gw


def test_happy_path_returns_valid_topics():
    gw = _make_gateway('["consciousness", "spirituality"]')
    result = classify_topics("Test", "some text", AVAILABLE, "consciousness", gw)
    assert result == ["consciousness", "spirituality"]


def test_filters_out_invalid_topics():
    gw = _make_gateway('["consciousness", "quantum_physics"]')
    result = classify_topics("Test", "some text", AVAILABLE, "consciousness", gw)
    assert result == ["consciousness"]
    assert "quantum_physics" not in result


def test_falls_back_to_default_hint_on_bad_json():
    gw = _make_gateway("Not valid JSON at all")
    result = classify_topics("Test", "some text", AVAILABLE, "biohacking", gw)
    assert result == ["biohacking"]


def test_falls_back_to_first_available_when_no_hint():
    gw = _make_gateway("bad json")
    result = classify_topics("Test", "some text", AVAILABLE, None, gw)
    assert result == [AVAILABLE[0]]


def test_strips_markdown_code_block():
    response = '```json\n["consciousness"]\n```'
    gw = _make_gateway(response)
    result = classify_topics("Test", "some text", AVAILABLE, "consciousness", gw)
    assert result == ["consciousness"]


def test_strips_plain_code_fence():
    response = '```\n["biohacking"]\n```'
    gw = _make_gateway(response)
    result = classify_topics("Test", "some text", AVAILABLE, "biohacking", gw)
    assert result == ["biohacking"]


def test_all_invalid_falls_back_to_hint():
    gw = _make_gateway('["bogus", "also_bogus"]')
    result = classify_topics("Test", "some text", AVAILABLE, "spirituality", gw)
    assert result == ["spirituality"]


def test_gateway_exception_falls_back():
    gw = MagicMock()
    gw.get_completion.side_effect = RuntimeError("API down")
    result = classify_topics("Test", "some text", AVAILABLE, "consciousness", gw)
    assert result == ["consciousness"]


def test_empty_validated_list_falls_back_to_hint():
    gw = _make_gateway("[]")
    result = classify_topics("Test", "some text", AVAILABLE, "biohacking", gw)
    assert result == ["biohacking"]


def test_returns_multiple_valid_topics():
    gw = _make_gateway('["consciousness", "biohacking", "spirituality"]')
    result = classify_topics("Test", "some text", AVAILABLE, "consciousness", gw)
    assert len(result) == 3
    assert set(result) == {"consciousness", "biohacking", "spirituality"}


def test_empty_available_topics():
    gw = _make_gateway("[]")
    result = classify_topics("Test", "some text", [], None, gw)
    assert result == []
