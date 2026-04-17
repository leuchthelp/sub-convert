import pytest

from src.subtitle.subtitle_group import Pgs

def test_pgs():
    pgs = Pgs(tmp_location="tests/files/for-pgs/test.sup")
    assert bool(pgs.items) is True