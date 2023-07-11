from itertools import count, cycle
import math
from functools import reduce
import random
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


def next_pc_up_from_pitch(
    pitch: int, pc: int, tet: int = 12, allow_unison: bool = True
) -> int:
    """
    >>> next_pc_up_from_pitch(62, 0)
    72
    >>> next_pc_up_from_pitch(62, 11)
    71
    >>> next_pc_up_from_pitch(62, 2)
    62
    >>> next_pc_up_from_pitch(62, 2, allow_unison=False)
    74
    """
    src_octave, src_pc = divmod(pitch, tet)
    if (pc >= src_pc) if allow_unison else (pc > src_pc):
        return src_octave * tet + pc
    return (src_octave + 1) * tet + pc


def next_pc_down_from_pitch(
    pitch: int, pc: int, tet: int = 12, allow_unison: bool = True
) -> int:
    """
    >>> next_pc_down_from_pitch(62, 0)
    60
    >>> next_pc_down_from_pitch(62, 11)
    59
    >>> next_pc_down_from_pitch(62, 2)
    62
    >>> next_pc_down_from_pitch(62, 2, allow_unison=False)
    50
    """
    src_octave, src_pc = divmod(pitch, tet)
    if (pc <= src_pc) if allow_unison else (pc < src_pc):
        return src_octave * tet + pc
    return (src_octave - 1) * tet + pc


def closest_pc_iter(
    pitch: int,
    pc: int,
    tet: int = 12,
    prefer_down: bool | None = True,
    max_results: int | None = None,
) -> t.Iterable[int]:
    """Yields the closest realization of `pc` to `pitch`, then the 2nd closest, etc.

    >>> [i for i in closest_pc_iter(pitch=60, pc=2, max_results=5)]
    [62, 50, 74, 38, 86]

    >>> [i for i in closest_pc_iter(pitch=60, pc=11, max_results=5)]
    [59, 71, 47, 83, 35]

    ---------
    Tritones:
    ---------

    >>> [i for i in closest_pc_iter(pitch=60, pc=6, max_results=5)]
    [54, 66, 42, 78, 30]

    >>> [i for i in closest_pc_iter(pitch=60, pc=6, max_results=5, prefer_down=False)]
    [66, 54, 78, 42, 90]

    If `prefer_down=None` then the direction is chosen randomly:
    >>> [next(closest_pc_iter(pitch=60, pc=6, prefer_down=None)) for _ in range(10)]
    ... # doctest: +SKIP
    [54, 66, 66, 66, 66, 54, 54, 54, 54, 66]

    --------
    Unisons:
    --------

    >>> [i for i in closest_pc_iter(pitch=60, pc=0, max_results=5)]
    [60, 48, 72, 36, 84]

    >>> [i for i in closest_pc_iter(pitch=60, pc=0, max_results=5, prefer_down=False)]
    [60, 72, 48, 84, 36]

    If `prefer_down=None` then the direction is chosen randomly:
    >>> [tuple(closest_pc_iter(pitch=60, pc=0, prefer_down=None, max_results=2))[1]
    ...  for _ in range(10)]  # doctest: +SKIP
    [48, 48, 48, 72, 48, 72, 48, 48, 48, 48]
    """
    pc_int = pc - (pitch % tet)

    halftet = tet // 2

    if pc_int > halftet:
        pc_int -= tet
    elif pc_int < -halftet:
        pc_int += tet
    elif abs(pc_int) == halftet:
        if prefer_down is None:
            pc_int *= random.choice([1, -1])
        elif prefer_down:
            pc_int = -halftet
        else:
            pc_int = halftet

    if max_results is not None:
        octaves = range(0, max_results * 12, 12)
    else:
        octaves = count(start=0, step=12)

    if pc_int == 0:
        if prefer_down is None:
            octave_signs = random.choice([[1, -1], [-1, 1]])
        elif prefer_down:
            octave_signs = [1, -1]
        else:
            octave_signs = [-1, 1]
    else:
        octave_signs = [1, -1] if pc_int > 0 else [-1, 1]

    octave_cycle = cycle(octave_signs)

    out = pitch + pc_int
    for octave, octave_sign in zip(octaves, octave_cycle):
        out += octave * octave_sign
        yield out
