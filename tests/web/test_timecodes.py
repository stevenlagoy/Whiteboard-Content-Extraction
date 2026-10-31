import pytest

from whiteboard_extraction.web.domain.errors import ValidationError
from whiteboard_extraction.web.timecodes import format_timecode, parse_timecode


@pytest.mark.parametrize(
    "seconds, expected",
    [(0, "0:00"), (9, "0:09"), (61, "1:01"), (3599, "59:59"), (3600, "1:00:00"), (5716, "1:35:16"), (-4, "0:00"), (61.9, "1:01")],
)
def test_format_timecode(seconds, expected):
    assert format_timecode(seconds) == expected


@pytest.mark.parametrize(
    "text, expected",
    [("90", 90), ("1:30", 90), ("1:02:03", 3723), ("1:30.5", 90.5), (" 2:00 ", 120), ("0:00", 0), ("05", 5)],
)
def test_parse_timecode(text, expected):
    assert parse_timecode(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "1:75", "1:2:3:4", "-5", "1:", ":30", "1.5:30", "1:30:75"])
def test_parse_timecode_rejects_bad_input(text):
    with pytest.raises(ValidationError):
        parse_timecode(text)


def test_error_message_names_the_field():
    with pytest.raises(ValidationError, match="Trim start"):
        parse_timecode("x", "Trim start")


def test_round_trip():
    for seconds in (0, 59, 60, 3661, 7325):
        assert parse_timecode(format_timecode(seconds)) == seconds
