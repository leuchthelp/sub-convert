from pysrt import SubRipTime
import typing


def from_hex(b: bytes):
    return int(b.hex(), base=16)


def safe_get(b: bytes, i: int, default_value=0):
    try:
        return b[i]
    except IndexError:
        return default_value


def to_time(value: int):
    return SubRipTime.from_ordinal(value)


T = typing.TypeVar('T')
