"""
core/article_fetcher.py
Fetches and parses a blog article URL into section-based chunks.
"""
import hashlib
import re
import unicodedata
from datetime import datetime

import httpx
from bs4 import BeautifulSoup


_USER_AGENT = (
    "Mozilla/5.0 (compatible; youtube-topic-rag/1.0; "
    "https://github.com/user/youtube-topic-rag)"
)


def fetch_article(url: str) -> dict:
    """
    HTTP GET the URL and extract article metadata + HTML body.

    Returns:
        url, title, author, publisher, published_at (datetime|None), html_body (str)
    """
    resp = httpx.get(url, headers={"User-Agent": _USER_AGENT}, follow_redirects=True, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Title: prefer Open Graph, fall back to <title>
    og_title = soup.find("meta", property="og:title")
    title = (
        (og_title.get("content") if og_title else None)
        or (soup.title.string.strip() if soup.title else "")
    )

    # Author
    author_meta = soup.find("meta", attrs={"name": "author"})
    og_author = soup.find("meta", property="article:author")
    author = (
        (author_meta.get("content") if author_meta else None)
        or (og_author.get("content") if og_author else None)
        or ""
    )

    # Publisher / site name
    og_site = soup.find("meta", property="og:site_name")
    publisher = og_site.get("content") if og_site else _hostname(url)

    # Published date
    published_at: datetime | None = None
    time_tag = soup.find("time")
    if time_tag and time_tag.get("datetime"):
        try:
            published_at = datetime.fromisoformat(
                str(time_tag["datetime"]).replace("Z", "+00:00")
            )
        except ValueError:
            pass

    # Article body: prefer <article>, then <main>, then <body>
    body_tag = soup.find("article") or soup.find("main") or soup.body

    return {
        "url":          url,
        "title":        title,
        "author":       author,
        "publisher":    publisher,
        "published_at": published_at,
        "html_body":    str(body_tag) if body_tag else resp.text,
    }


def extract_sections(html_body: str, article_url: str) -> list[dict]:
    """
    Parse HTML into section chunks keyed by H2/H3 headings.

    Falls back to fixed-size word chunking when no headings are found.
    Sections shorter than 50 words are merged into the preceding section.

    Returns list of:
        chunk_id, associated_chapter, section_slug, deep_link, text_content
    """
    soup = BeautifulSoup(html_body, "html.parser")

    # Collect block-level elements in document order
    blocks = soup.find_all(["h2", "h3", "p"])
    has_headings = any(b.name in ("h2", "h3") for b in blocks)

    if not has_headings:
        return _word_chunks(soup.get_text(separator=" ", strip=True), article_url)

    sections: list[dict] = []
    current_heading = "Introduction"
    current_parts: list[str] = []

    for block in blocks:
        if block.name in ("h2", "h3"):
            text = " ".join(current_parts).strip()
            if text:
                sections.append({"heading": current_heading, "text": text})
            current_heading = block.get_text(strip=True) or "Section"
            current_parts = []
        elif block.name == "p":
            t = block.get_text(strip=True)
            if t:
                current_parts.append(t)

    # Flush final section
    text = " ".join(current_parts).strip()
    if text:
        sections.append({"heading": current_heading, "text": text})

    if not sections:
        return _word_chunks(soup.get_text(separator=" ", strip=True), article_url)

    # Merge sections shorter than 50 words into the preceding section
    merged: list[dict] = []
    for section in sections:
        if merged and len(section["text"].split()) < 50:
            merged[-1]["text"] += " " + section["text"]
        else:
            merged.append(section)

    chunks = []
    for i, section in enumerate(merged):
        slug = _slugify(section["heading"])
        chunk_id = f"s_{i + 1:03d}"
        deep_link = f"{article_url}#{slug}" if slug else article_url
        chunks.append({
            "chunk_id":           chunk_id,
            "associated_chapter": section["heading"],
            "section_slug":       slug,
            "deep_link":          deep_link,
            "text_content":       section["text"],
        })

    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word_chunks(
    text: str,
    article_url: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[dict]:
    """Fixed sliding-window word chunks for articles without headings."""
    words = text.split()
    stride = chunk_size - overlap
    chunks = []
    for i, idx in enumerate(range(0, max(len(words), 1), stride), start=1):
        window = words[idx : idx + chunk_size]
        if not window:
            break
        chunks.append({
            "chunk_id":           f"w_{i:03d}",
            "associated_chapter": "General",
            "section_slug":       "",
            "deep_link":          article_url,
            "text_content":       " ".join(window),
        })
    return chunks


def _hostname(url: str) -> str:
    from urllib.parse import urlparse
    return urlparse(url).hostname or url


def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_-]+", "-", text)
