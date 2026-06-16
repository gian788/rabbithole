import bisect
import json
import re
import sys
import urllib.error
import urllib.request

CHAPTER_REGEX = re.compile(r"(?:\[)?(\d{1,2}:\d{2}(?::\d{2})?)(?:\])?\s+(.*)")

_SPONSOR_CATEGORIES = ["sponsor", "selfpromo", "interaction", "intro", "outro"]
_SPONSORBLOCK_URL = "https://sponsor.ajay.app/api/skipSegments"


def fetch_sponsor_segments(video_id: str) -> list[tuple[float, float]]:
    """
    Fetch sponsor/selfpromo/intro/outro segments from SponsorBlock for a video.
    Returns a list of (start_seconds, end_seconds) float tuples.
    Returns [] if the video has no submissions or the API is unreachable.
    """
    import json as _json

    categories_param = urllib.request.quote(json.dumps(_SPONSOR_CATEGORIES))
    url = f"{_SPONSORBLOCK_URL}?videoID={video_id}&categories={categories_param}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "youtube-topic-rag/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = _json.loads(resp.read())
        return [(seg["segment"][0], seg["segment"][1]) for seg in data if "segment" in seg]
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []  # no submissions for this video
        print(f"[sponsorblock] HTTP {exc.code} for {video_id}", file=sys.stderr)
        return []
    except Exception as exc:
        print(f"[sponsorblock] error for {video_id}: {exc}", file=sys.stderr)
        return []


def filter_sponsored_srt(
    srt: list[dict], sponsor_segments: list[tuple[float, float]]
) -> list[dict]:
    """
    Remove SRT entries whose start time falls within any sponsor segment window.
    Returns the filtered SRT list; returns original list unchanged if no segments.
    """
    if not sponsor_segments:
        return srt
    filtered = []
    for entry in srt:
        t = entry["start"]
        if not any(start <= t < end for start, end in sponsor_segments):
            filtered.append(entry)
    removed = len(srt) - len(filtered)
    if removed:
        print(f"[sponsorblock] filtered {removed} SRT segments across {len(sponsor_segments)} windows")
    return filtered


def _parse_timestamp(ts: str) -> int:
    parts = ts.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0


def extract_chapters_from_description(description: str) -> list[dict]:
    """Parse YouTube chapter timestamps from a video description."""
    if not description:
        return []
    try:
        matches = CHAPTER_REGEX.findall(description)
        chapters = [
            {"start_seconds": _parse_timestamp(ts), "title": title.strip()}
            for ts, title in matches
            if title.strip()
        ]
        return sorted(chapters, key=lambda c: c["start_seconds"])
    except Exception:
        return []


def generate_chapters_with_llm(transcript_text: str, gateway) -> list[dict]:
    """Ask Claude Haiku to identify logical chapter boundaries in a transcript."""
    system_prompt = (
        "You are a transcript analyst. Given a YouTube video transcript, identify 5-12 logical "
        "section boundaries where the speaker transitions to a new topic or subtopic. "
        'Return ONLY valid JSON — an array of objects with "title" (string) and '
        '"start_seconds" (integer). '
        'Example: [{"title": "Introduction", "start_seconds": 0}, ...]. '
        "Output nothing else."
    )
    try:
        response = gateway.get_completion(
            prompt=transcript_text[:12000],
            system_prompt=system_prompt,
            model="claude-haiku-4-5-20251001",
            provider="anthropic",
            associated_id="chapter_gen",
        )
        raw = response.text_content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```[^\n]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())
        chapters = json.loads(raw)
        valid = [
            c for c in chapters
            if isinstance(c.get("start_seconds"), int) and isinstance(c.get("title"), str)
        ]
        return sorted(valid, key=lambda c: c["start_seconds"])
    except Exception:
        return []


def _assign_chapter(start_seconds: int, chapters: list[dict]) -> str:
    """Return the chapter title that contains the given timestamp."""
    if not chapters:
        return "General"
    starts = [c["start_seconds"] for c in chapters]
    idx = bisect.bisect_right(starts, start_seconds) - 1
    return chapters[max(idx, 0)]["title"]


def segment_into_paragraphs(
    srt: list[dict],
    chapters: list[dict],
    video_id: str,
    min_words: int = 120,
    max_words: int = 200,
    overlap_words: int = 40,
) -> list[dict]:
    """
    Chunk an SRT transcript into semantically rich paragraphs.

    Flush conditions (evaluated in order):
      1. Hard ceiling: accumulated word count >= max_words
      2. Soft flush: word count >= min_words AND segment ends a sentence
      3. Topic break: silence gap > 5 s AND word count >= 60

    After each flush, the last `overlap_words` words are carried forward into
    the next chunk so concepts spanning a boundary aren't silently dropped.
    """
    chunks: list[dict] = []
    # Each entry: (srt_index, text)
    acc_segments: list[tuple[int, str]] = []
    chunk_index = 1

    def _flush(acc: list[tuple[int, str]]) -> list[tuple[int, str]]:
        nonlocal chunk_index
        if not acc:
            return acc
        start_seg = srt[acc[0][0]]
        paragraph_start = int(start_seg["start"])
        text = " ".join(t for _, t in acc)
        if len(text.split()) >= 20:
            chunks.append({
                "chunk_id":           f"p_{chunk_index:03d}",
                "associated_chapter": _assign_chapter(paragraph_start, chapters),
                "start_seconds":      paragraph_start,
                "deep_link":          f"https://youtu.be/{video_id}?t={paragraph_start}",
                "text_content":       text,
            })
            chunk_index += 1
        # Carry the last overlap_words worth of segments into next chunk
        carry: list[tuple[int, str]] = []
        carry_words = 0
        for item in reversed(acc):
            w = len(item[1].split())
            if carry_words + w > overlap_words:
                break
            carry.insert(0, item)
            carry_words += w
        return carry

    for i, segment in enumerate(srt):
        acc_segments.append((i, segment["text"]))
        current_words = sum(len(t.split()) for _, t in acc_segments)
        ends_sentence = segment["text"].strip().endswith((".", "!", "?"))
        gap = 0.0
        if i + 1 < len(srt):
            gap = srt[i + 1]["start"] - (segment["start"] + segment["duration"])

        if current_words >= max_words:
            acc_segments = _flush(acc_segments)
        elif current_words >= min_words and ends_sentence:
            acc_segments = _flush(acc_segments)
        elif gap > 5.0 and current_words >= 60:
            acc_segments = _flush(acc_segments)

    # Final flush
    _flush(acc_segments)
    return chunks


def fixed_word_chunking(
    text: str,
    video_id: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[dict]:
    """Fallback: fixed sliding-window word chunks when SRT is unavailable."""
    words = text.split()
    stride = chunk_size - overlap
    chunks: list[dict] = []
    chunk_index = 1

    for i in range(0, max(len(words), 1), stride):
        window = words[i : i + chunk_size]
        if not window:
            break
        chunks.append({
            "chunk_id":          f"w_{chunk_index:03d}",
            "associated_chapter": "General",
            "start_seconds":      0,
            "deep_link":          f"https://youtu.be/{video_id}",
            "text_content":       " ".join(window),
        })
        chunk_index += 1

    return chunks
