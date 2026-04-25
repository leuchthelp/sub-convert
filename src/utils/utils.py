from pysrt import SubRipTime


def from_hex(b: bytes):
    try:
        return int(b.hex(), base=16)
    except ValueError:
        return -1


def safe_get(b: bytes, i: int, default_value=0):
    try:
        return b[i]
    except IndexError:
        return default_value


def to_time(value: int):
    return SubRipTime.from_ordinal(value)
