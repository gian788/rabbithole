import json

import pytest


@pytest.fixture
def make_srt():
    """Factory: make_srt([(text, start, duration), ...]) → list[dict]"""
    def _make(entries):
        return [{"text": t, "start": s, "duration": d} for t, s, d in entries]
    return _make


@pytest.fixture
def chapters_3():
    return [
        {"start_seconds": 0,   "title": "Intro"},
        {"start_seconds": 120, "title": "Main"},
        {"start_seconds": 600, "title": "Outro"},
    ]


def parse_sse_events(raw: str) -> list[dict]:
    return [
        json.loads(line[6:])
        for line in raw.strip().splitlines()
        if line.startswith("data: ")
    ]
