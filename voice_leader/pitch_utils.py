import math
from functools import reduce
import typing as t


def put_in_range(p, low=None, high=None, tet=12, fail_silently: bool = False):
    """Used by voice_lead_pitches().

    >>> put_in_range(0, low=58, high=74)
    60

    Moves pitch as little as possible while keeping it within the specified
    range:

    >>> put_in_range(72, low=32, high=62)
    60
    >>> put_in_range(48, low=58, high=100)
    60

    Raises an exception if the pitch-class isn't found between the bounds,
    unless fail_silently is True, in which case it returns a pitch below the
    lower bound:

    >>> put_in_range(60, low=49, high=59)
    Traceback (most recent call last):
        raise ValueError(
    ValueError: pitch-class 0 does not occur between low=49 and high=59

    >>> put_in_range(60, low=49, high=59, fail_silently=True)
    48
    """
    if low is not None:
        below = low - p
        if below > 0:
            octaves_below = math.ceil((below) / tet)
            p += octaves_below * tet
    if high is not None:
        above = p - high
        if above > 0:
            octaves_above = math.ceil(above / tet)
            p -= octaves_above * tet
    if not fail_silently and low is not None:
        if p < low:
            raise ValueError(
                f"pitch-class {p % 12} does not occur between "
                f"low={low} and high={high}"
            )
    return p


def get_all_in_range(
    p: t.Union[t.Sequence[int], int],
    low: int,
    high: int,
    tet: int = 12,
    sorted: bool = False,
) -> t.List[int]:
    """Bounds are inclusive.

    >>> get_all_in_range(60, low=58, high=72)
    [60, 72]
    >>> get_all_in_range(60, low=58, high=59)
    []
    >>> get_all_in_range(58, low=58, high=85)
    [58, 70, 82]
    >>> get_all_in_range([58, 60], low=58, high=83, sorted=True)
    [58, 60, 70, 72, 82]
    """
    if not isinstance(p, int):
        if not p:
            return []
        out = reduce(
            lambda x, y: x + y,
            [get_all_in_range(pp, low, high, tet) for pp in p],
        )
        if sorted:
            out.sort()
        return out
    pc = p % tet
    low_octave, low_pc = divmod(low, tet)
    low_octave += pc < low_pc
    high_octave, high_pc = divmod(high, tet)
    high_octave -= pc > high_pc
    return [pc + octave * tet for octave in range(low_octave, high_octave + 1)]


def next_pc_up_from_pitch(pitch: int, pc: int, tet: int = 12) -> int:
    """
    >>> next_pc_up_from_pitch(62, 0)
    72
    >>> next_pc_up_from_pitch(62, 11)
    71
    """
    src_octave, src_pc = divmod(pitch, tet)
    if pc > src_pc:
        return src_octave * tet + pc
    return (src_octave + 1) * tet + pc


def next_pc_down_from_pitch(pitch: int, pc: int, tet: int = 12) -> int:
    """
    >>> next_pc_down_from_pitch(62, 0)
    60
    >>> next_pc_down_from_pitch(62, 11)
    59
    """
    src_octave, src_pc = divmod(pitch, tet)
    if pc < src_pc:
        return src_octave * tet + pc
    return (src_octave - 1) * tet + pc
