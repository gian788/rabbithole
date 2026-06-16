import pytest
from ingestion.fetch_lambda import _is_short


def _item(duration: str = "", title: str = "", description: str = "") -> dict:
    return {
        "snippet": {"title": title, "description": description},
        "contentDetails": {"duration": duration},
    }


@pytest.mark.parametrize("duration,expected", [
    ("PT30S",  True),
    ("PT60S",  True),
    ("PT61S",  False),
    ("PT1M0S", True),
    ("PT1M1S", False),
    ("PT1H",   False),
    ("PT1H0M0S", False),
    ("PT2M",   False),
    ("",       False),
])
def test_is_short_duration(duration, expected):
    assert _is_short(_item(duration=duration)) == expected


@pytest.mark.parametrize("title", ["My #shorts video", "Check #Shorts out", "COOL #SHORTS"])
def test_is_short_title_hashtag(title):
    assert _is_short(_item(duration="PT5M", title=title)) is True


@pytest.mark.parametrize("description", ["Watch on #shorts", "posted as #Shorts daily"])
def test_is_short_description_hashtag(description):
    assert _is_short(_item(duration="PT5M", description=description)) is True


def test_is_short_missing_content_details():
    item = {"snippet": {"title": "Normal video", "description": ""}}
    assert _is_short(item) is False


def test_is_short_normal_video():
    assert _is_short(_item(duration="PT10M", title="Normal Podcast Episode")) is False
