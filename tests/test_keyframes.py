"""Tests for whiteboard_extraction.video.keyframes."""

import numpy as np

from whiteboard_extraction.video.keyframes import extract_keyframes, frames_differ


def blank() -> np.ndarray:
    return np.zeros((100, 100), dtype=np.uint8)


def with_rows(n_rows: int) -> np.ndarray:
    """A blank frame with the top n_rows filled white."""
    frame = blank()
    frame[:n_rows, :] = 255
    return frame


def test_identical_frames_do_not_differ():
    assert not frames_differ(blank(), blank())


def test_change_above_threshold_differs():
    assert frames_differ(blank(), with_rows(5))


def test_change_below_threshold_does_not_differ():
    assert not frames_differ(blank(), with_rows(1))


def test_small_intensity_change_is_ignored():
    dim = blank() + 20
    assert not frames_differ(blank(), dim)


def custom_threshold():
    assert frames_differ(blank(), with_rows(1), threshold=0.005)


def test_uint8_subtraction_does_not_wrap():
    """Guards int16 cast: 0 - 255 with uint8 wraps to 1."""
    assert frames_differ(with_rows(10), blank())
    assert frames_differ(blank(), with_rows(10))


def test_color_frames_supported():
    a = np.zeros((100, 100, 3), dtype=np.uint8)
    b = a.copy()
    b[:10, :, :] = 255
    assert frames_differ(a, b)


def test_empty_input_returns_empty_list():
    assert extract_keyframes([]) == []


def test_first_frame_is_always_kept():
    result = extract_keyframes([(3.0, blank())])
    assert [t for t, _ in result] == [3.0]


def test_identical_frames_collapse_to_first():
    frames = [(float(t), blank()) for t in range(5)]
    assert [t for t, _ in extract_keyframes(frames)] == [0.0]


def test_compares_against_last_kept_keyframe_not_previous_frame():
    step1 = blank()
    step1[0, :] = 255
    step1[1, :50] = 255
    step2 = blank()
    step2[:3, :] = 255
    frames = [(0.0, blank()), (2.0, step1), (4.0, step2)]
    assert [t for t, _ in extract_keyframes(frames)] == [0.0, 4.0]


def test_returns_original_frame_objects():
    first = blank()
    result = extract_keyframes([(0.0, first)])
    assert result[0][1] is first