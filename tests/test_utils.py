import pytest

from src.utils import utils
from src.utils import torch_utils


@pytest.mark.parametrize(
    "input1, expectation",
    [
        (b"\x00\x01", 1),
        (b"\x01\x02", 258),
    ],
)
def test_from_hex_inputs(input1, expectation):
    assert utils.from_hex(input1) == expectation


@pytest.mark.parametrize(
    "input1, input2, expectation",
    [
        (b"\x00\x01", 0, 0),
        (b"\x00\x01", 1, 1),
        (b"\x02", 0, 2),
        (b"\x02", 6, 0),
    ],
)
def test_safe_get_inputs(input1, input2, expectation):
    assert utils.safe_get(input1, input2) == expectation


def test_check_torch_cuda_inputs():
    assert bool(torch_utils.check_torch_cuda({})) is True
