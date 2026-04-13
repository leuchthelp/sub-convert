from contextlib import nullcontext as does_not_raise
import pytest

from pathlib import Path

from main import (
    check_if_adjacent_exists,
    check_aged,
    get_candidates,
    get_classes,
    import_class,
)


def test_check_aged_pos_offset():
    path = Path("tests/files/for-main/test-adjacent-exists.srt")
    assert check_aged(path, offset="s+1") is True
    assert check_aged(path, offset="+1") is True
    assert check_aged(path, offset="1") is True


def test_check_aged_neg_offset():
    path = Path("tests/files/for-main/test-adjacent-exists.srt")
    assert check_aged(path, offset="s-1") is False
    assert check_aged(path, offset="-1") is False


@pytest.mark.parametrize(
    "input, expectation",
    [
        ("1", does_not_raise()),
        ("+1", does_not_raise()),
        ("p+1", does_not_raise()),
        ("ü+1", does_not_raise()),
        ("h", pytest.raises(ValueError)),
        ("+h", pytest.raises(ValueError)),
        ("h+", pytest.raises(ValueError)),
        ("h1", pytest.raises(ValueError)),
        ("1h", pytest.raises(ValueError)),
        ("h-", pytest.raises(ValueError)),
        ("h1-", pytest.raises(ValueError)),
    ],
)
def test_check_aged_inputs(input, expectation):
    path = Path("tests/files/for-main/test-adjacent-exists.srt")
    with expectation:
        check_aged(path, offset=input)


def test_check_if_adjacent_exists():
    path1 = Path("tests/files/for-main/test-adjacent-exists.mkv")
    path2 = Path("tests/files/for-main/test-no-adjacent.mkv")
    assert check_if_adjacent_exists(path1) is True
    assert check_if_adjacent_exists(path2) is False


@pytest.mark.parametrize(
    "input, expectation",
    [
        ({"skip_if_existing": False, "convert_aged": ""}, does_not_raise()),
        ({"skip_if_existing": False, "convert_aged": "s+1"}, does_not_raise()),
        ({"skip_if_existing": True, "convert_aged": ""}, does_not_raise()),
        ({"skip_if_existing": True, "convert_aged": "s+1"}, does_not_raise()),
        ({"skip_if_existing": True}, pytest.raises(KeyError)),
        ({}, pytest.raises(KeyError)),
    ],
)
def test_get_candidates_inputs(input, expectation):
    with expectation:
        list(get_candidates(Path("tests/files/for-main"), options=input))


@pytest.mark.parametrize(
    "input, needed, expectation",
    [
        (
            {"skip_if_existing": False, "convert_aged": ""},
            [
                "tests/files/for-main/test-adjacent-exists.mkv",
                "tests/files/for-main/test-no-adjacent.mkv",
            ],
            True,
        ),
        (
            {"skip_if_existing": False, "convert_aged": "s+1"},
            [
                "tests/files/for-main/test-adjacent-exists.mkv",
                "tests/files/for-main/test-no-adjacent.mkv",
            ],
            True,
        ),
        (
            {"skip_if_existing": True, "convert_aged": ""},
            ["tests/files/for-main/test-no-adjacent.mkv"],
            True,
        ),
        (
            {"skip_if_existing": True, "convert_aged": ""},
            ["tests/files/for-main/test-adjacent-exists.mkv"],
            False,
        ),
        (
            {"skip_if_existing": True, "convert_aged": "s+1"},
            ["tests/files/for-main/test-no-adjacent.mkv"],
            True,
        ),
    ],
)
def test_get_candidates_results(input, needed, expectation):
    tmp = list(get_candidates(Path("tests/files/for-main"), options=input))
    
    if not tmp:
        assert len(tmp) == len(needed)

    def is_in_substring(x: list):
        for entry in x:
            if res.name in entry:
                return True
        return False

    for res in tmp:
        assert is_in_substring(needed) is expectation
