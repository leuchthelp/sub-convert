from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pytest

from sub_convert.model import ocr_model_core, language_model_core
from sub_convert.cli import (
    check_if_adjacent_exists,
    check_aged,
    get_candidates,
    get_classes,
    import_class,
)


def test_check_aged_pos_offset():
    path = Path("tests/files/for-main/test-adjacent-exists.srt")
    assert check_aged(path, offset="s+1") is True
    # assert check_aged(path, offset="+1") is True
    # assert check_aged(path, offset="1") is True


def test_check_aged_neg_offset():
    path = Path("tests/files/for-main/test-adjacent-exists.srt")
    assert check_aged(path, offset="s-1") is False
    # assert check_aged(path, offset="-1") is False


@pytest.mark.parametrize(
    "input1, expectation",
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
def test_check_aged_inputs(input1, expectation):
    path = Path("tests/files/for-main/test-adjacent-exists.srt")
    with expectation:
        check_aged(path, offset=input1)


def test_check_if_adjacent_exists():
    path1 = Path("tests/files/for-main/test-adjacent-exists.mkv")
    path2 = Path("tests/files/for-main/test-no-adjacent.mkv")
    assert check_if_adjacent_exists(path1) is True
    assert check_if_adjacent_exists(path2) is False


@pytest.mark.parametrize(
    "input1, expectation",
    [
        ({"skip_if_existing": False, "convert_aged": ""}, does_not_raise()),
        ({"skip_if_existing": False, "convert_aged": "s+1"}, does_not_raise()),
        ({"skip_if_existing": True, "convert_aged": ""}, does_not_raise()),
        ({"skip_if_existing": True, "convert_aged": "s+1"}, does_not_raise()),
        ({"skip_if_existing": True}, pytest.raises(KeyError)),
        ({}, pytest.raises(KeyError)),
    ],
)
def test_get_candidates_inputs(input1, expectation):
    with expectation:
        list(get_candidates(Path("tests/files/for-main"), options=input1))


@pytest.mark.parametrize(
    "input1, needed, expectation",
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
def test_get_candidates_results(input1, needed, expectation):
    tmp = list(get_candidates(Path("tests/files/for-main"), options=input1))

    if not tmp:
        assert len(tmp) == len(needed)

    def is_in_substring(x: list):
        for entry in x:
            if res.name in entry:
                return True
        return False

    for res in tmp:
        assert is_in_substring(needed) is expectation


@pytest.mark.parametrize(
    "input1, needed, expectation",
    [
        (ocr_model_core, ["OCRModelCore", "PaddleModelCore"], True),
        (language_model_core, ["LanguageModelCore", "LangDetectModelCore"], True),
        (ocr_model_core, ["LanguageModelCore", "LangDetectModelCore"], False),
        (language_model_core, ["OCRModelCore", "PaddleModelCore"], False),
    ],
)
def test_get_classes_inputs(input1, needed, expectation):
    tmp = get_classes(input1)

    assert bool(set(needed).intersection(tmp)) is expectation


def test_import_class_inputs():
    assert import_class("OCRModelCore", ocr_model_core.__name__)(options={})
    assert import_class("LanguageModelCore", language_model_core.__name__)(options={})


def test_import_class_invalid_inputs():
    with pytest.raises(AttributeError):
        assert import_class("OCRModelSchmore", ocr_model_core.__name__)(options={})
        assert import_class("LanguageModelSchmore", language_model_core.__name__)(
            options={}
        )
