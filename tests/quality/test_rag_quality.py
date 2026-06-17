"""
Quality tests: run against real services (Pinecone, OpenAI, PostgreSQL).

Required setup:
  1. Copy tests/quality/fixtures/seed_chunks.example.json → seed_chunks.json
     and fill with real transcript excerpts from indexed videos.
  2. Set PINECONE_NAMESPACE=test in your environment (conftest does this automatically).
  3. Run: uv run pytest -m quality tests/quality/ -v

These tests are excluded from the default run (addopts = "-m 'not quality'").
"""
import re

import pytest

pytestmark = pytest.mark.quality

KNOWN_TOPICS = {"consciousness", "biohacking", "spirituality", "alternative_history"}

PROHIBITED_PHRASES = [
    "the text", "the source", "as stated in", "the passage",
    "according to the text", "as mentioned in the text",
]

GOLDEN_QUERIES = [
    "What is the demiurge according to Gnostic tradition?",
    "How does meditation affect consciousness and the mind?",
]

CLIP_URL_RE = re.compile(r"https://youtu\.be/\w+\?t=\d+")


# ---------------------------------------------------------------------------
# Structural assertions helper
# ---------------------------------------------------------------------------

def _assert_response_structure(body: dict, expect_sources: bool = True) -> None:
    assert "answer" in body
    assert "topic" in body
    assert "sources" in body
    assert "conversation_id" in body

    if expect_sources:
        assert len(body["sources"]) > 0, "Expected non-empty sources for a known query"
        assert body["topic"] in KNOWN_TOPICS, f"Unknown topic: {body['topic']}"

        for src in body["sources"]:
            assert src.get("video_id"), "source missing video_id"
            assert src.get("title"),    "source missing title"
            assert src.get("channel"),  "source missing channel"
            assert "speaker" in src
            clips = src.get("clips", [])
            assert len(clips) > 0, "source has no clips"
            assert len(clips) <= 2, f"more than 2 clips for source {src['video_id']}"
            for clip in clips:
                assert CLIP_URL_RE.match(clip.get("url", "")), f"bad clip URL: {clip.get('url')}"
                assert isinstance(clip["start_seconds"], int)
                assert clip["start_seconds"] >= 0


def _assert_no_generic_phrases(answer: str) -> None:
    lower = answer.lower()
    for phrase in PROHIBITED_PHRASES:
        assert phrase not in lower, f"Answer contains generic phrase: {phrase!r}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_sources_structure_and_attribution(real_client):
    resp = real_client.post("/v1/chat", json={"query": "What is the demiurge?"})
    assert resp.status_code == 200
    body = resp.json()
    _assert_response_structure(body)
    _assert_no_generic_phrases(body["answer"])


def test_answer_mentions_speaker(real_client):
    resp = real_client.post("/v1/chat", json={"query": "What is the demiurge?"})
    body = resp.json()
    if body["sources"]:
        speakers = [s["speaker"] for s in body["sources"] if s.get("speaker")]
        if speakers:
            answer_lower = body["answer"].lower()
            speaker_words = {w.lower() for s in speakers for w in s.split()}
            # At least one speaker word should appear in the answer
            matched = any(w in answer_lower for w in speaker_words if len(w) > 3)
            assert matched, f"Answer does not mention any speaker. Speakers: {speakers}"


def test_topic_routing_consciousness(real_client):
    resp = real_client.post("/v1/chat", json={"query": "What is non-dual awareness in meditation?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["topic"] in {"consciousness", "spirituality"}


def test_topic_routing_biohacking(real_client):
    resp = real_client.post("/v1/chat", json={"query": "How can I optimise sleep and testosterone levels?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["topic"] == "biohacking"


def test_max_two_clips_per_source(real_client):
    resp = real_client.post("/v1/chat", json={"query": "What is consciousness?"})
    body = resp.json()
    for src in body.get("sources", []):
        assert len(src["clips"]) <= 2


def test_graceful_no_match(real_client):
    resp = real_client.post("/v1/chat", json={"query": "zxqkjf basketball 2024 playoffs xyz"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sources"] == []
    assert isinstance(body["answer"], str)


def test_streaming_completeness(real_client):
    from tests.conftest import parse_sse_events

    resp = real_client.post("/v1/chat", json={"query": "What is the demiurge?", "stream": True})
    assert resp.status_code == 200
    events = parse_sse_events(resp.text)
    assert len(events) > 0

    token_events = [e for e in events if e["type"] == "token"]
    done_event   = next((e for e in events if e["type"] == "done"), None)

    assert done_event is not None, "No 'done' event in stream"
    assert "answer" in done_event, "done event missing 'answer' field"
    assert "sources" in done_event

    # Concatenated tokens must equal the answer in the done event
    tokens_text = "".join(e["content"] for e in token_events)
    assert tokens_text == done_event["answer"]

    if done_event["sources"]:
        _assert_response_structure(done_event, expect_sources=True)


def test_conversation_continuity(real_client):
    resp1 = real_client.post("/v1/chat", json={"query": "What is the demiurge?"})
    assert resp1.status_code == 200
    conv_id = resp1.json()["conversation_id"]

    resp2 = real_client.post("/v1/chat", json={
        "query": "Who are the main thinkers associated with this concept?",
        "conversation_id": conv_id,
    })
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert len(body2["answer"]) > 50
    assert body2["conversation_id"] == conv_id


# ---------------------------------------------------------------------------
# LLM-as-judge tests (golden queries only)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", GOLDEN_QUERIES)
def test_golden_query_llm_judge(real_client, query):
    from tests.quality.conftest import _llm_judge

    resp = real_client.post("/v1/chat", json={"query": query})
    assert resp.status_code == 200
    body = resp.json()

    if not body["sources"]:
        pytest.skip(f"No sources returned for golden query: {query!r} — add relevant seed chunks")

    verdict = _llm_judge(query, body["answer"], body["sources"])
    assert verdict["relevance"]   >= 7, f"Low relevance ({verdict['relevance']}): {verdict['reason']}"
    assert verdict["grounding"]   >= 7, f"Low grounding ({verdict['grounding']}): {verdict['reason']}"
    assert verdict["attribution"] >= 7, f"Poor attribution ({verdict['attribution']}): {verdict['reason']}"
