import importlib
import json

import pytest

# ---------------------------------------------------------------------------
# Helpers: load module-level private functions without triggering heavy init
# ---------------------------------------------------------------------------

def _get_helpers():
    """Import helpers without triggering CrossEncoder/get_vector_store init."""
    from unittest.mock import MagicMock, patch
    with patch("sentence_transformers.CrossEncoder", return_value=MagicMock()), \
         patch("retrieval.main.get_connection", return_value=MagicMock()), \
         patch("retrieval.main.get_vector_store", return_value=MagicMock()), \
         patch("retrieval.main.ModelGateway", return_value=MagicMock()):
        import retrieval.main as m
        importlib.reload(m)
        return m


@pytest.fixture(scope="module")
def main_mod():
    return _get_helpers()


# ---------------------------------------------------------------------------
# _nearest_topic
# ---------------------------------------------------------------------------

def test_nearest_topic_single(main_mod, monkeypatch):
    monkeypatch.setattr(main_mod, "_topic_vectors", {
        "consciousness": [1.0] + [0.0] * 1535,
    })
    assert main_mod._nearest_topic([1.0] + [0.0] * 1535) == "consciousness"


def test_nearest_topic_picks_best(main_mod, monkeypatch):
    monkeypatch.setattr(main_mod, "_topic_vectors", {
        "consciousness": [1.0, 0.0],
        "biohacking":    [0.0, 1.0],
    })
    assert main_mod._nearest_topic([1.0, 0.0]) == "consciousness"
    assert main_mod._nearest_topic([0.0, 1.0]) == "biohacking"


def test_nearest_topic_empty(main_mod, monkeypatch):
    monkeypatch.setattr(main_mod, "_topic_vectors", {})
    assert main_mod._nearest_topic([1.0, 0.0]) == ""


# ---------------------------------------------------------------------------
# _extract_speaker
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title,expected", [
    ("Mind over Matter | Joe Dispenza", "Joe Dispenza"),
    ("Episode 42 | Dr. Andrew Huberman", "Dr. Andrew Huberman"),
    ("No Pipe Here", ""),
    ("| Leading pipe", "Leading pipe"),
    ("Multiple | pipes | last", "pipes | last"),  # first pipe match — multi-pipe is not a real case
    ("Trailing Whitespace |  Speaker  ", "Speaker"),
])
def test_extract_speaker(main_mod, title, expected):
    assert main_mod._extract_speaker(title) == expected


# ---------------------------------------------------------------------------
# _merge_adjacent_chunks
# ---------------------------------------------------------------------------

def _chunk(video_id: str, start: int, text: str = None) -> dict:
    return {"metadata": {
        "source_type": "youtube_video",
        "video_id": video_id,
        "start_seconds": start,
        "text_content": text or f"text@{start}",
        "chapter": "Intro",
        "deep_link": f"https://youtu.be/{video_id}?t={start}",
    }}


def test_merge_empty(main_mod):
    assert main_mod._merge_adjacent_chunks([]) == []


def test_merge_single(main_mod):
    result = main_mod._merge_adjacent_chunks([_chunk("vid1", 0)])
    assert len(result) == 1
    assert result[0]["metadata"]["video_id"] == "vid1"


def test_merge_same_video_close(main_mod):
    chunks = [_chunk("vid1", 0, "hello"), _chunk("vid1", 20, "world")]
    result = main_mod._merge_adjacent_chunks(chunks)
    assert len(result) == 1
    assert "hello" in result[0]["metadata"]["text_content"]
    assert "world" in result[0]["metadata"]["text_content"]


def test_merge_same_video_far_apart(main_mod):
    chunks = [_chunk("vid1", 0), _chunk("vid1", 31)]
    result = main_mod._merge_adjacent_chunks(chunks)
    assert len(result) == 2


def test_merge_different_video(main_mod):
    chunks = [_chunk("vid1", 0), _chunk("vid2", 0)]
    result = main_mod._merge_adjacent_chunks(chunks)
    assert len(result) == 2


def test_merge_three_same_video(main_mod):
    chunks = [_chunk("vid1", 0), _chunk("vid1", 20), _chunk("vid1", 40)]
    result = main_mod._merge_adjacent_chunks(chunks)
    assert len(result) == 1


def test_merge_preserves_first_chapter(main_mod):
    c1 = {"metadata": {"source_type": "youtube_video", "video_id": "v", "start_seconds": 0,
                        "text_content": "a", "chapter": "Intro",
                        "deep_link": "https://youtu.be/v?t=0"}}
    c2 = {"metadata": {"source_type": "youtube_video", "video_id": "v", "start_seconds": 10,
                        "text_content": "b", "chapter": "Main",
                        "deep_link": "https://youtu.be/v?t=10"}}
    result = main_mod._merge_adjacent_chunks([c1, c2])
    assert result[0]["metadata"]["chapter"] == "Intro"
    assert result[0]["metadata"]["deep_link"] == "https://youtu.be/v?t=0"


def test_merge_mixed(main_mod):
    chunks = [
        _chunk("vid1", 0),
        _chunk("vid1", 20),   # merges with vid1@0
        _chunk("vid2", 0),    # separate video
    ]
    result = main_mod._merge_adjacent_chunks(chunks)
    assert len(result) == 2


def test_merge_article_chunks_not_merged(main_mod):
    """Article chunks (no video_id) from the same source should not merge."""
    art1 = {"metadata": {
        "source_type": "article", "article_id": "abc", "start_seconds": None,
        "text_content": "intro text", "chapter": "Intro",
        "deep_link": "https://example.com/post#intro",
    }}
    art2 = {"metadata": {
        "source_type": "article", "article_id": "abc", "start_seconds": None,
        "text_content": "body text", "chapter": "Body",
        "deep_link": "https://example.com/post#body",
    }}
    result = main_mod._merge_adjacent_chunks([art1, art2])
    # Article chunks should NOT merge (no video_id)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# _sse
# ---------------------------------------------------------------------------

def test_sse_format(main_mod):
    out = main_mod._sse("token", {"content": "hello"})
    assert out.startswith("data: ")
    assert out.endswith("\n\n")


def test_sse_type_field(main_mod):
    out = main_mod._sse("done", {"sources": []})
    parsed = json.loads(out[6:])
    assert parsed["type"] == "done"
    assert parsed["sources"] == []


def test_sse_extra_keys_present(main_mod):
    out = main_mod._sse("token", {"content": "hi", "extra": 42})
    parsed = json.loads(out[6:])
    assert parsed["content"] == "hi"
    assert parsed["extra"] == 42


import pytest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# _parse_list_meta
# ---------------------------------------------------------------------------

def test_parse_list_meta_native_list(main_mod):
    assert main_mod._parse_list_meta(["a", "b"]) == ["a", "b"]


def test_parse_list_meta_json_string(main_mod):
    assert main_mod._parse_list_meta('["a", "b"]') == ["a", "b"]


def test_parse_list_meta_none(main_mod):
    assert main_mod._parse_list_meta(None) == []


def test_parse_list_meta_bad_json(main_mod):
    assert main_mod._parse_list_meta("not json") == []


# ---------------------------------------------------------------------------
# _entity_overlap_score
# ---------------------------------------------------------------------------

def test_entity_overlap_partial_match(main_mod):
    meta = {"entities": ["consciousness", "non-duality"]}
    score = main_mod._entity_overlap_score("tell me about consciousness", meta)
    assert score == pytest.approx(0.5)


def test_entity_overlap_all_match(main_mod):
    meta = {"entities": ["concept"]}
    assert main_mod._entity_overlap_score("concept here", meta) == pytest.approx(1.0)


def test_entity_overlap_no_entities(main_mod):
    assert main_mod._entity_overlap_score("query", {"entities": []}) == 0.0


def test_entity_overlap_missing_field(main_mod):
    assert main_mod._entity_overlap_score("query", {}) == 0.0


def test_entity_overlap_chroma_json_string(main_mod):
    meta = {"entities": '["consciousness", "non-duality"]'}
    score = main_mod._entity_overlap_score("consciousness", meta)
    assert score == pytest.approx(0.5)


def test_entity_overlap_no_match(main_mod):
    meta = {"entities": ["zen", "taoism"]}
    assert main_mod._entity_overlap_score("biohacking protocols", meta) == 0.0


# ---------------------------------------------------------------------------
# _people_bonus
# ---------------------------------------------------------------------------

def test_people_bonus_host_match(main_mod):
    meta = {"host": "Joe Rogan", "guests": [], "author": None}
    assert main_mod._people_bonus("what did joe rogan say", meta) == pytest.approx(0.15)


def test_people_bonus_guest_match(main_mod):
    meta = {"host": "Joe Rogan", "guests": ["Graham Hancock"], "author": None}
    assert main_mod._people_bonus("graham hancock on consciousness", meta) == pytest.approx(0.15)


def test_people_bonus_multiple_guests_capped(main_mod):
    meta = {"host": "Host", "guests": ["Guest A", "Guest B"], "author": None}
    assert main_mod._people_bonus("guest a and guest b discussed", meta) == pytest.approx(0.15)


def test_people_bonus_no_match(main_mod):
    meta = {"host": "Joe Rogan", "guests": ["Guest"], "author": None}
    assert main_mod._people_bonus("something completely unrelated", meta) == 0.0


def test_people_bonus_author_match(main_mod):
    meta = {"host": None, "guests": [], "author": "Mark Manson"}
    assert main_mod._people_bonus("mark manson on meaning", meta) == pytest.approx(0.15)


def test_people_bonus_chroma_json_guests(main_mod):
    meta = {"host": None, "guests": '["Graham Hancock"]', "author": None}
    assert main_mod._people_bonus("graham hancock", meta) == pytest.approx(0.15)


def test_people_bonus_empty_meta(main_mod):
    assert main_mod._people_bonus("query", {}) == 0.0


# ---------------------------------------------------------------------------
# _rerank — entity boost changes ordering
# ---------------------------------------------------------------------------

def _make_yt_chunk(video_id: str, start: int, entities: list, text: str = "text") -> dict:
    return {"metadata": {
        "source_type":   "youtube_video",
        "video_id":      video_id,
        "start_seconds": start,
        "text_content":  text,
        "chapter":       "Intro",
        "deep_link":     f"https://youtu.be/{video_id}?t={start}",
        "entities":      entities,
        "host":          None,
        "guests":        [],
        "author":        None,
    }}


def test_rerank_entity_boost_changes_order(main_mod, monkeypatch):
    # chunk_a: CE=0.4, no entity match → final=0.4
    # chunk_b: CE=0.2, both entities match → final=0.2 + 0.3*(2/2)=0.5
    chunk_a = _make_yt_chunk("v1", 0, entities=[])
    chunk_b = _make_yt_chunk("v2", 0, entities=["consciousness", "awareness"])

    monkeypatch.setattr(
        main_mod, "_reranker",
        MagicMock(predict=lambda pairs: [0.4, 0.2])
    )

    result = main_mod._rerank("consciousness and awareness", [chunk_a, chunk_b], top_n=2)
    assert result[0]["metadata"]["video_id"] == "v2"


def test_rerank_people_bonus_changes_order(main_mod, monkeypatch):
    # chunk_a: CE=0.4, no people match → final=0.4
    # chunk_b: CE=0.2, host match → final=0.2 + 0.15=0.35  → chunk_a still wins
    # Adjust: chunk_b CE=0.3, host match → final=0.3 + 0.15=0.45 → chunk_b wins
    chunk_a = _make_yt_chunk("v1", 0, entities=[])
    chunk_b = _make_yt_chunk("v2", 0, entities=[])
    chunk_b["metadata"]["host"] = "Graham Hancock"

    monkeypatch.setattr(
        main_mod, "_reranker",
        MagicMock(predict=lambda pairs: [0.4, 0.3])
    )

    result = main_mod._rerank("what did graham hancock say", [chunk_a, chunk_b], top_n=2)
    assert result[0]["metadata"]["video_id"] == "v2"
