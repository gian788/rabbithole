import json
import importlib
import pytest


# ---------------------------------------------------------------------------
# Helpers: load module-level private functions without triggering heavy init
# ---------------------------------------------------------------------------

def _get_helpers():
    """Import helpers without triggering BM25/CrossEncoder/Pinecone init."""
    # Patch the expensive module-level calls before import
    from unittest.mock import patch, MagicMock
    with patch("pinecone_text.sparse.BM25Encoder.default", return_value=MagicMock()), \
         patch("sentence_transformers.CrossEncoder", return_value=MagicMock()), \
         patch("retrieval.main.get_connection", return_value=MagicMock()), \
         patch("retrieval.main.Pinecone", return_value=MagicMock()), \
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
    c1 = {"metadata": {"video_id": "v", "start_seconds": 0, "text_content": "a",
                        "chapter": "Intro", "deep_link": "https://youtu.be/v?t=0"}}
    c2 = {"metadata": {"video_id": "v", "start_seconds": 10, "text_content": "b",
                        "chapter": "Main", "deep_link": "https://youtu.be/v?t=10"}}
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
