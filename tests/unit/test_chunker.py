import pytest

from core.chunker import (
    _assign_chapter,
    _parse_timestamp,
    extract_chapters_from_description,
    filter_sponsored_srt,
    fixed_word_chunking,
    segment_into_paragraphs,
)

# ---------------------------------------------------------------------------
# _parse_timestamp
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ts,expected", [
    ("1:30",   90),
    ("0:05",   5),
    ("59:59",  3599),
    ("1:00:00", 3600),
    ("1:23:45", 5025),
    ("0:00",   0),
])
def test_parse_timestamp(ts, expected):
    assert _parse_timestamp(ts) == expected


# ---------------------------------------------------------------------------
# extract_chapters_from_description
# ---------------------------------------------------------------------------

def test_extract_chapters_empty():
    assert extract_chapters_from_description("") == []


def test_extract_chapters_no_timestamps():
    assert extract_chapters_from_description("Check out my links below!") == []


def test_extract_chapters_basic():
    desc = "0:00 Intro\n3:45 Deep Dive\n12:30 Outro"
    chapters = extract_chapters_from_description(desc)
    assert len(chapters) == 3
    assert chapters[0] == {"start_seconds": 0,   "title": "Intro"}
    assert chapters[1] == {"start_seconds": 225, "title": "Deep Dive"}
    assert chapters[2] == {"start_seconds": 750, "title": "Outro"}


def test_extract_chapters_sorted():
    desc = "12:30 Outro\n0:00 Intro\n3:45 Main"
    chapters = extract_chapters_from_description(desc)
    starts = [c["start_seconds"] for c in chapters]
    assert starts == sorted(starts)


def test_extract_chapters_bracketed():
    desc = "[1:23] Big Topic\n[5:00] End"
    chapters = extract_chapters_from_description(desc)
    assert len(chapters) == 2
    assert chapters[0]["start_seconds"] == 83


def test_extract_chapters_hhmmss():
    desc = "1:23:45 Deep Section"
    chapters = extract_chapters_from_description(desc)
    assert chapters[0]["start_seconds"] == 5025


# ---------------------------------------------------------------------------
# _assign_chapter
# ---------------------------------------------------------------------------

def test_assign_chapter_empty():
    assert _assign_chapter(0, []) == "General"


def test_assign_chapter_before_any():
    chapters = [{"start_seconds": 10, "title": "A"}]
    assert _assign_chapter(0, chapters) == "A"


def test_assign_chapter_exact_boundary(chapters_3):
    assert _assign_chapter(0,   chapters_3) == "Intro"
    assert _assign_chapter(120, chapters_3) == "Main"
    assert _assign_chapter(600, chapters_3) == "Outro"


def test_assign_chapter_just_before_boundary(chapters_3):
    assert _assign_chapter(119, chapters_3) == "Intro"
    assert _assign_chapter(599, chapters_3) == "Main"


def test_assign_chapter_far_future(chapters_3):
    assert _assign_chapter(9999, chapters_3) == "Outro"


# ---------------------------------------------------------------------------
# filter_sponsored_srt
# ---------------------------------------------------------------------------

def test_filter_sponsored_no_segments(make_srt):
    srt = make_srt([("hello", 0.0, 1.0), ("world", 1.5, 1.0)])
    result = filter_sponsored_srt(srt, [])
    assert result is srt


def test_filter_sponsored_removes_within_window(make_srt):
    srt = make_srt([("ad", 30.0, 1.0), ("content", 40.0, 1.0)])
    result = filter_sponsored_srt(srt, [(25.0, 35.0)])
    assert len(result) == 1
    assert result[0]["text"] == "content"


def test_filter_sponsored_boundary_kept(make_srt):
    srt = make_srt([("keep", 35.0, 1.0)])
    result = filter_sponsored_srt(srt, [(25.0, 35.0)])
    assert len(result) == 1


def test_filter_sponsored_boundary_removed(make_srt):
    srt = make_srt([("remove", 25.0, 1.0)])
    result = filter_sponsored_srt(srt, [(25.0, 35.0)])
    assert len(result) == 0


def test_filter_sponsored_multiple_windows(make_srt):
    srt = make_srt([
        ("ad1", 10.0, 1.0),
        ("ok",  20.0, 1.0),
        ("ad2", 50.0, 1.0),
    ])
    result = filter_sponsored_srt(srt, [(5.0, 15.0), (45.0, 55.0)])
    assert len(result) == 1
    assert result[0]["text"] == "ok"


def test_filter_sponsored_all_removed(make_srt):
    srt = make_srt([("ad", 5.0, 1.0)])
    result = filter_sponsored_srt(srt, [(0.0, 60.0)])
    assert result == []


# ---------------------------------------------------------------------------
# segment_into_paragraphs
# ---------------------------------------------------------------------------

def _words(n: int) -> str:
    return " ".join([f"word{i}" for i in range(n)])


def test_segment_empty_srt(chapters_3):
    result = segment_into_paragraphs([], chapters_3, "vid1")
    assert result == []


def test_segment_chunk_ids_sequential(make_srt, chapters_3):
    # 3 groups of 130 words each, each ending in "."
    entries = []
    for group in range(3):
        base = group * 200.0
        for i in range(13):  # 13 segments × 10 words each = 130 words
            text = _words(10) + ("." if i == 12 else "")
            entries.append((text, base + i * 1.0, 0.9))
    srt = make_srt(entries)
    chunks = segment_into_paragraphs(srt, chapters_3, "vid1", min_words=120, max_words=200, overlap_words=10)
    ids = [c["chunk_id"] for c in chunks]
    assert ids[0] == "p_001"
    assert ids[1] == "p_002"


def test_segment_deep_link_contains_video_id_and_timestamp(make_srt, chapters_3):
    entries = [((_words(10) + ("." if i == 12 else "")), i * 1.0, 0.9) for i in range(13)]
    srt = make_srt(entries)
    chunks = segment_into_paragraphs(srt, chapters_3, "myvid", min_words=120, max_words=200)
    assert all("myvid" in c["deep_link"] for c in chunks)
    assert all("?t=" in c["deep_link"] for c in chunks)


def test_segment_flush_on_max_words(make_srt, chapters_3):
    # 210 words in one long run (no sentence endings, no gaps)
    entries = [(_words(10), i * 0.5, 0.4) for i in range(21)]
    srt = make_srt(entries)
    chunks = segment_into_paragraphs(srt, chapters_3, "vid", min_words=120, max_words=200, overlap_words=0)
    assert len(chunks) >= 1
    for c in chunks[:-1]:  # all but possibly the last
        assert len(c["text_content"].split()) <= 210  # max_words + tolerance for overlap


def test_segment_short_chunks_not_emitted(make_srt, chapters_3):
    # Only 5 words total — too short for any chunk
    srt = make_srt([("hello world foo bar baz", 0.0, 1.0)])
    chunks = segment_into_paragraphs(srt, chapters_3, "vid")
    # 5 words is below the 20-word minimum flush threshold
    assert chunks == []


def test_segment_assigns_correct_chapter(make_srt, chapters_3):
    # Build 130 words starting at t=150 (should be in "Main" chapter, which starts at 120)
    entries = [(_words(10) + ("." if i == 12 else ""), 150.0 + i, 0.9) for i in range(13)]
    srt = make_srt(entries)
    chunks = segment_into_paragraphs(srt, chapters_3, "vid", min_words=120, max_words=200)
    assert chunks[0]["associated_chapter"] == "Main"


def test_segment_silence_gap_flush(make_srt, chapters_3):
    # 70 words then a 6-second gap — should flush at the gap even though < min_words
    entries = [(_words(10), i * 0.5, 0.4) for i in range(7)]   # 70 words
    entries.append((_words(10), entries[-1][1] + 6.5, 0.9))      # gap > 5s
    srt = make_srt(entries)
    chunks = segment_into_paragraphs(srt, chapters_3, "vid", min_words=120, max_words=200, overlap_words=0)
    assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# fixed_word_chunking
# ---------------------------------------------------------------------------

def test_fixed_single_chunk_short_text():
    chunks = fixed_word_chunking(_words(50), "vid1")
    assert len(chunks) == 1
    assert chunks[0]["start_seconds"] == 0
    assert chunks[0]["associated_chapter"] == "General"
    assert "?t=" not in chunks[0]["deep_link"]
    assert "vid1" in chunks[0]["deep_link"]


def test_fixed_chunk_ids():
    chunks = fixed_word_chunking(_words(600), "vid1")
    assert chunks[0]["chunk_id"] == "w_001"
    assert chunks[1]["chunk_id"] == "w_002"


def test_fixed_two_chunks_with_overlap():
    # 600 words, chunk_size=300, overlap=50 → stride=250 → 2 full chunks
    text = _words(600)
    chunks = fixed_word_chunking(text, "vid1", chunk_size=300, overlap=50)
    assert len(chunks) >= 2
    # Second chunk should overlap: last words of chunk 1 should appear at start of chunk 2
    c1_words = set(chunks[0]["text_content"].split()[-50:])
    c2_words = set(chunks[1]["text_content"].split()[:50])
    assert c1_words & c2_words  # non-empty intersection
