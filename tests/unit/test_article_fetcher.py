"""Unit tests for core/article_fetcher.py — section extraction and word chunking."""
import pytest

from core.article_fetcher import _slugify, _word_chunks, extract_sections


# ---------------------------------------------------------------------------
# _slugify
# ---------------------------------------------------------------------------

def test_slugify_basic():
    assert _slugify("Hello World") == "hello-world"

def test_slugify_special_chars():
    assert _slugify("What is DMT?") == "what-is-dmt"

def test_slugify_accents():
    assert _slugify("Café Résumé") == "cafe-resume"

def test_slugify_numbers():
    assert _slugify("Chapter 3: Overview") == "chapter-3-overview"

def test_slugify_empty():
    assert _slugify("") == ""

def test_slugify_result_is_lowercase():
    assert _slugify("UPPERCASE") == "uppercase"

def test_slugify_no_spaces():
    result = _slugify("Multiple Words Here")
    assert " " not in result


# ---------------------------------------------------------------------------
# extract_sections — with H2/H3 headings
# ---------------------------------------------------------------------------

_LONG_PARA = (
    "This section has many words. Padding words to exceed fifty-word minimum. "
    "Adding more filler content here to ensure section is not merged. "
    "The test needs this paragraph to be long enough to pass reliably always. "
)

ARTICLE_WITH_HEADINGS = f"""
<html><body>
<h2>Introduction</h2>
<p>{_LONG_PARA}</p>
<p>Additional paragraph for the introduction section with more text.</p>
<h2>Main Topic</h2>
<p>{_LONG_PARA}</p>
<h3>Subsection</h3>
<p>{_LONG_PARA}</p>
</body></html>
"""


def test_extract_sections_with_headings_returns_chunks():
    chunks = extract_sections(ARTICLE_WITH_HEADINGS, "https://example.com/post")
    assert len(chunks) >= 1


def test_extract_sections_chunk_required_fields():
    chunks = extract_sections(ARTICLE_WITH_HEADINGS, "https://example.com/post")
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "associated_chapter" in chunk
        assert "section_slug" in chunk
        assert "deep_link" in chunk
        assert "text_content" in chunk


def test_extract_sections_chunk_id_starts_with_s():
    chunks = extract_sections(ARTICLE_WITH_HEADINGS, "https://example.com/post")
    assert all(c["chunk_id"].startswith("s_") for c in chunks)


def test_extract_sections_deep_link_has_anchor():
    chunks = extract_sections(ARTICLE_WITH_HEADINGS, "https://example.com/post")
    anchored = [c for c in chunks if "#" in c["deep_link"]]
    assert len(anchored) >= 1


def test_extract_sections_slug_is_lowercase_no_spaces():
    chunks = extract_sections(ARTICLE_WITH_HEADINGS, "https://example.com/post")
    for chunk in chunks:
        slug = chunk["section_slug"]
        if slug:
            assert slug == slug.lower()
            assert " " not in slug


def test_extract_sections_text_content_not_empty():
    chunks = extract_sections(ARTICLE_WITH_HEADINGS, "https://example.com/post")
    assert all(c["text_content"].strip() for c in chunks)


# ---------------------------------------------------------------------------
# extract_sections — fallback to word chunks (no headings)
# ---------------------------------------------------------------------------

NO_HEADING_HTML = """
<html><body>
<p>First paragraph. This is a test article without any headings whatsoever.</p>
<p>Second paragraph. We deliberately omit h2 and h3 tags so the fallback
word chunking path is exercised by this test case.</p>
</body></html>
"""


def test_extract_sections_no_headings_returns_chunks():
    chunks = extract_sections(NO_HEADING_HTML, "https://example.com/noh")
    assert len(chunks) >= 1


def test_extract_sections_no_headings_chunk_id_starts_with_w():
    chunks = extract_sections(NO_HEADING_HTML, "https://example.com/noh")
    assert all(c["chunk_id"].startswith("w_") for c in chunks)


def test_extract_sections_no_headings_deep_link_has_no_anchor():
    chunks = extract_sections(NO_HEADING_HTML, "https://example.com/noh")
    assert all(c["deep_link"] == "https://example.com/noh" for c in chunks)


def test_extract_sections_no_headings_chapter_is_general():
    chunks = extract_sections(NO_HEADING_HTML, "https://example.com/noh")
    assert all(c["associated_chapter"] == "General" for c in chunks)


# ---------------------------------------------------------------------------
# Short section merging — tail section shorter than 50 words merges into preceding
# ---------------------------------------------------------------------------

_SHORT = "Too short."
_LONG = _LONG_PARA * 2  # definitely > 50 words

LONG_THEN_SHORT = f"""
<html><body>
<h2>Long Section</h2>
<p>{_LONG}</p>
<h2>Tiny</h2>
<p>{_SHORT}</p>
</body></html>
"""


def test_short_section_merged_into_preceding():
    chunks = extract_sections(LONG_THEN_SHORT, "https://example.com/merge")
    # "Tiny" has < 50 words, merges into "Long Section" → single chunk
    assert len(chunks) == 1
    assert "Too short" in chunks[0]["text_content"]


def test_merged_chunk_keeps_first_heading():
    chunks = extract_sections(LONG_THEN_SHORT, "https://example.com/merge")
    assert chunks[0]["associated_chapter"] == "Long Section"


# ---------------------------------------------------------------------------
# _word_chunks — sliding window
# ---------------------------------------------------------------------------

def test_word_chunks_small_text():
    chunks = _word_chunks("hello world", "https://example.com")
    assert len(chunks) == 1
    assert chunks[0]["text_content"] == "hello world"


def test_word_chunks_overlap():
    words = " ".join([f"w{i}" for i in range(400)])
    chunks = _word_chunks(words, "https://example.com", chunk_size=100, overlap=20)
    assert len(chunks) >= 2
    end_of_first = set(chunks[0]["text_content"].split()[-20:])
    start_of_second = set(chunks[1]["text_content"].split()[:20])
    assert end_of_first & start_of_second


def test_word_chunks_chapter_and_deep_link():
    chunks = _word_chunks("word " * 50, "https://example.com/a")
    for chunk in chunks:
        assert chunk["associated_chapter"] == "General"
        assert chunk["deep_link"] == "https://example.com/a"
        assert chunk["section_slug"] == ""
