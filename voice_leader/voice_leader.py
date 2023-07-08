import math
import random
import typing as t
from dataclasses import dataclass
from itertools import combinations

import numpy as np

# In a VoiceLeadingMotion,
# - an int indicates that a voice proceeds by that interval
# - a tuple of int indicates that a voice splits into two or more voices
# - None indicates a voice that vanishes
VoiceLeadingMotion = t.Tuple[t.Union[int, t.Tuple[int, ...], None], ...]


class NoMoreVoiceLeadingsError(Exception):
    pass


class CardinalityDiffersTooMuch(Exception):
    pass


def apply_vl(vl: VoiceLeadingMotion, chord: t.Sequence[int]) -> t.List[int]:
    """
    >>> apply_vl(vl=(0, 1, 2), chord=(60, 64, 67))
    [60, 65, 69]

    >>> apply_vl(vl=(-3, 1, None), chord=(60, 64, 67))
    [57, 65]

    >>> apply_vl(vl=((-3, 0), (-1, 1), 2), chord=(60, 64, 67))
    [57, 60, 63, 65, 69]

    # >>> apply_vl(vl=(-3, 0), chord=(60, 64, 67))
    # Traceback (innermost last):
    # AssertionError

    # Raises:
    #     AssertionError if len(vl) != len(chord)
    """
    assert len(vl) == len(chord)
    out = []
    for i, motion in enumerate(vl):
        if isinstance(motion, tuple):
            for m in motion:
                out.append((chord[i] + m))
        elif motion is None:
            continue
        else:
            out.append((chord[i] + motion))
    return out


def indices_to_vl(
    indices: t.Iterable[int], chord1: t.Sequence[int], chord2: t.Sequence[int], tet: int
) -> t.Tuple[int, ...]:
    """
    >>> indices_to_vl(indices=(0, 1, 2), chord1=(60, 64, 67), chord2=(60, 65, 69), tet=12)
    (0, 1, 2)
    >>> indices_to_vl(indices=(1, 0, 2), chord1=(60, 64, 67), chord2=(60, 65, 69), tet=12)
    (5, -4, 2)
    >>> indices_to_vl(indices=(0, 0, 2), chord1=(60, 64, 67), chord2=(60, 65, 69), tet=12)
    (0, -4, 2)

    Args:
        indices: the index of the pitch in `chord2` to which the corresponding pitch in
            `chord1` should proceed. For example, (1, 0, 2) means
                - chord1[0] -> chord2[1]
                - chord1[1] -> chord2[0]
                - chord1[2] -> chord2[2]

    """
    voice_leading = []
    for i, j in enumerate(indices):
        interval = chord2[j] - chord1[i]
        if abs(interval) <= tet // 2:
            voice_leading.append(interval)
        elif interval > 0:
            voice_leading.append(interval - tet)
        else:
            voice_leading.append(interval + tet)
    return tuple(voice_leading)


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


def voice_lead_pitches_iter(
    chord1_pitches: t.Sequence[int],
    chord2_pcs: t.Sequence[int],
    preserve_root: bool = False,
    avoid_bass_crossing: bool = True,
    tet: int = 12,
    allow_different_cards: bool = True,
    min_pitch: t.Optional[int] = None,
    max_pitch: t.Optional[int] = None,
    min_bass_pitch: t.Optional[int] = None,
    max_bass_pitch: t.Optional[int] = None,
) -> t.Iterator[t.List[int]]:
    """ """

    def _get_max_motions_up(pitches, start_i):
        if max_pitch is None:
            return None
        return {i: max_pitch - pitch for i, pitch in enumerate(pitches)}

    def _get_max_motions_down(pitches, start_i):
        if min_pitch is None:
            return None
        return {i: min_pitch - pitch for i, pitch in enumerate(pitches)}

    i = 0  # stop pylance complaining that motions is possibly unbound below
    motions = []  # stop pylance complaining that motions is possibly unbound below

    if preserve_root and len(chord1_pitches) < len(chord2_pcs):
        min_displacement = float("inf")
        # If chord1 has fewer pitches than chord2 has pcs, then at least
        #   one pitch from chord1 will be "split" into multiple pitches in
        #   chord2. Ordinarily when preserve_root is True, we exclude the
        #   root from the voice-leading calculation, but since we might
        #   want to split the root, we need to try including it here.
        for j in (0, 1):
            # really we should do a cartesian product of indices to alternately
            # exclude the root of both chords too but that seems like overkill
            try:
                temp_motions, temp_displacement = efficient_voice_leading(
                    [p % tet for p in chord1_pitches[j:]],
                    chord2_pcs[j:],
                    tet,
                    allow_different_cards=allow_different_cards,
                    max_motions_down=_get_max_motions_down(chord1_pitches[j:], j),
                    max_motions_up=_get_max_motions_up(chord1_pitches[j:], j),
                )
            except NotImplementedError:
                # TODO what is this exception?
                continue
            if temp_displacement < min_displacement:
                motions = temp_motions
                min_displacement = temp_displacement
                i = j
        if min_displacement == float("inf"):
            raise NotImplementedError(
                f"Too many excess pitches between {chord1_pitches} and {chord2_pcs}"
            )
    else:
        i = 1 if preserve_root else 0
        motions, _ = efficient_voice_leading(
            [p % tet for p in chord1_pitches[i:]],
            chord2_pcs[i:],
            tet,
            allow_different_cards=allow_different_cards,
            max_motions_down=_get_max_motions_down(chord1_pitches[i:], i),
            max_motions_up=_get_max_motions_up(chord1_pitches[i:], i),
        )

    for motion in motions:
        if i == 0:
            # either we are not preserving the root, or the voice-leading proceeds
            # from the root as well
            out = apply_vl(motion, chord1_pitches)
            if preserve_root:
                # find the index of the root in the next chord
                while out[i] % tet != chord2_pcs[0]:
                    i += 1
                if i != 0:
                    out[i] -= tet
                    out.sort()
                if avoid_bass_crossing:
                    max_bass_pitch = (
                        out[1]
                        if max_bass_pitch is None
                        else min(out[1], max_bass_pitch)
                    )
                out[0] = put_in_range(out[0], low=min_bass_pitch, high=max_bass_pitch)
            yield out
            return
        else:
            root_int = (chord2_pcs[0] - chord1_pitches[0]) % tet
            if root_int > tet // 2:
                root_int -= tet
            new_root = chord1_pitches[0] + root_int
            upper_parts = apply_vl(motion, chord1_pitches[1:])
            if avoid_bass_crossing:
                if upper_parts:
                    max_bass_pitch = (
                        upper_parts[0]
                        if max_bass_pitch is None
                        else min(upper_parts[0], max_bass_pitch)
                    )
            new_root = put_in_range(new_root, low=min_bass_pitch, high=max_bass_pitch)
            # while new_root > upper_parts[0] and avoid_bass_crossing:
            #     new_root -= tet
            # if max_pitch is not None:
            #     assert all(p <= max_pitch for p in upper_parts)
            yield [new_root] + upper_parts


def voice_lead_pitches(
    chord1_pitches: t.Sequence[int],
    chord2_pcs: t.Sequence[int],
    preserve_root: bool = False,
    avoid_bass_crossing: bool = True,
    tet: int = 12,
    allow_different_cards: bool = True,
    min_pitch: t.Optional[int] = None,
    max_pitch: t.Optional[int] = None,
    min_bass_pitch: t.Optional[int] = None,
    max_bass_pitch: t.Optional[int] = None,
    return_first: bool = True,
) -> t.List[int]:
    """
    >>> voice_lead_pitches([60, 64, 67], [5, 8, 0])
    [60, 65, 68]
    >>> voice_lead_pitches([60, 64, 67], [5, 8, 0], preserve_root=True)
    [53, 60, 68]

    If preserve_root is True, the bass voice can exceed 'min_pitch'

    >>> voice_lead_pitches([60, 64, 67], [7, 11, 2], min_pitch=60)
    [62, 67, 71]
    >>> voice_lead_pitches([60, 64, 67], [7, 11, 2], preserve_root=True,
    ...     min_pitch=60)
    [55, 62, 71]

    >>> voice_lead_pitches([60, 64, 67], [7, 11, 2, 5], preserve_root=True,
    ...     min_pitch=60)
    [55, 62, 65, 71]
    """
    iterator = voice_lead_pitches_iter(
        chord1_pitches,
        chord2_pcs,
        preserve_root,
        avoid_bass_crossing,
        tet,
        allow_different_cards,
        min_pitch,
        max_pitch,
        min_bass_pitch,
        max_bass_pitch,
    )
    if return_first:
        return next(iterator)

    return random.choice(list(iterator))


def _remap_from_doubled_indices(
    mapping: t.Dict[int, t.Any], doubled_indices: t.Container[int]
):
    """
    Used by growing_cardinality_handler()

    >>> _remap_from_doubled_indices({0:"a", 1:"b", 2:"c"}, [1,])
    {0: 'a', 1: 'b', 2: 'b', 3: 'c'}

    >>> _remap_from_doubled_indices({0:"a", 2:"c"}, [1,])
    {0: 'a', 3: 'c'}

    >>> _remap_from_doubled_indices({0:"a", 1:"b", 2:"c"}, [3,])
    {0: 'a', 1: 'b', 2: 'c'}

    >>> _remap_from_doubled_indices({}, [1, 2])
    {}

    """
    # This function feels inelegant...
    if not mapping:
        return {}
    out = {}
    j = 0
    for i in range(max(mapping) + 1):
        if i in mapping:
            out[j] = mapping[i]
        if i in doubled_indices:
            j += 1
            if i in mapping:
                out[j] = mapping[i]
        j += 1
    return out


def growing_cardinality_handler(
    chord1: t.Sequence[int],
    chord2: t.Sequence[int],
    *args,
    sort: bool = True,
    exclude_motions: t.Optional[t.Dict[int, t.List[int]]] = None,
    max_motions_up: t.Optional[t.Dict[int, int]] = None,
    max_motions_down: t.Optional[t.Dict[int, int]] = None,
    **kwargs,
) -> t.Tuple[t.List[VoiceLeadingMotion], int]:
    # This algorithm has horrible complexity but in practice I think that
    # chords are small enough (we will almost always be going from 3 to 4
    # notes) that it doesn't really matter. But nevertheless, if you are
    # voice-leading between chords of like ~20 elements, don't use this!

    card1, card2 = len(chord1), len(chord2)
    excess = card2 - card1
    if excess > card1:
        raise CardinalityDiffersTooMuch(
            f"chord1 has {card1} two items; chord2 has {card2} items; "
            "when increasing cardinality, the number of items can differ by "
            "at most the number of items in chord1"
        )
    least_displacement = 2**31
    best_vls, best_indices = [], []

    if exclude_motions is None:
        exclude_motions = {}
    if max_motions_up is None:
        max_motions_up = {}
    if max_motions_down is None:
        max_motions_down = {}

    previously_doubled_ps: t.Set[t.Tuple[int, ...]] = set()  # Only used if sort==True

    for doubled_indices in combinations(range(card1), excess):
        if sort:
            doubled_ps = tuple(chord1[i] for i in doubled_indices)
            if doubled_ps in previously_doubled_ps:
                continue
            previously_doubled_ps.add(doubled_ps)
        temp_chord = []
        for i, p in enumerate(chord1):
            temp_chord.extend([p, p] if i in doubled_indices else [p])
        vls, total_displacement = efficient_voice_leading(
            temp_chord,
            chord2,
            *args,
            exclude_motions=_remap_from_doubled_indices(
                exclude_motions, doubled_indices
            ),
            max_motions_up=_remap_from_doubled_indices(max_motions_up, doubled_indices),
            max_motions_down=_remap_from_doubled_indices(
                max_motions_down, doubled_indices
            ),
            **kwargs,
        )
        if total_displacement < least_displacement:
            best_indices = [doubled_indices]
            best_vls = [vls]
            least_displacement = total_displacement
        elif total_displacement == least_displacement:
            best_indices.append(doubled_indices)
            best_vls.append(vls)

    out = []
    for indices, vls in zip(best_indices, best_vls):
        for vl in vls:
            vl_out = []
            vl_i = 0
            for i in range(card1):
                if i in indices:
                    vl_out.append((vl[vl_i], vl[vl_i + 1]))
                    vl_i += 2
                else:
                    vl_out.append(vl[vl_i])
                    vl_i += 1
            out.append(tuple(vl_out))
    return out, least_displacement


def _remap_from_omitted_indices(
    mapping: t.Dict[int, t.Any], omitted_indices: t.Sequence[int]
):
    """Used by shrinking_cardinality_handler()

    >>> _remap_from_omitted_indices({0:"a", 1:"b", 2:"c"}, [1,])
    {0: 'a', 1: 'c'}

    >>> _remap_from_omitted_indices({0:"a", 2:"c"}, [1,])
    {0: 'a', 1: 'c'}

    >>> _remap_from_omitted_indices({0:"a", 1:"b", 2:"c"}, [3,])
    {0: 'a', 1: 'b', 2: 'c'}

    >>> _remap_from_omitted_indices({}, [1, 2])
    {}
    """
    # This function feels inelegant...
    if not mapping:
        return {}
    out = {}
    j = 0
    for i in range(max(mapping) + 1):
        if i in mapping:
            out[j] = mapping[i]
        if i in omitted_indices:
            j -= 1
        j += 1
    return out


def shrinking_cardinality_handler(
    chord1: t.Sequence[int],
    chord2: t.Sequence[int],
    *args,
    sort: bool = True,
    exclude_motions: t.Optional[t.Dict[int, t.List[int]]] = None,
    max_motions_up: t.Optional[t.Dict[int, int]] = None,
    max_motions_down: t.Optional[t.Dict[int, int]] = None,
    **kwargs,
) -> t.Tuple[t.List[VoiceLeadingMotion], int]:
    card1, card2 = len(chord1), len(chord2)
    excess = card1 - card2
    if excess > card2:
        raise CardinalityDiffersTooMuch(
            f"chord1 has {card1} two items; chord2 has {card2} items; "
            "when decreasing cardinality, the number of items can differ by "
            "at most the number of items in chord2"
        )
    least_displacement = 2**31
    best_vls, best_indices = [], []

    if exclude_motions is None:
        exclude_motions = {}
    if max_motions_up is None:
        max_motions_up = {}
    if max_motions_down is None:
        max_motions_down = {}

    previously_omitted_ps: t.Set[t.Tuple[int, ...]] = set()  # Only used if sort==True

    for omitted_indices in combinations(range(card1), excess):
        if sort:
            omitted_ps = tuple(chord1[i] for i in omitted_indices)
            if omitted_ps in previously_omitted_ps:
                continue
            previously_omitted_ps.add(omitted_ps)
        temp_chord = [p for (i, p) in enumerate(chord1) if i not in omitted_indices]
        vls, total_displacement = efficient_voice_leading(
            temp_chord,
            chord2,
            *args,
            sort=sort,
            exclude_motions=_remap_from_omitted_indices(
                exclude_motions, omitted_indices
            ),
            max_motions_up=_remap_from_omitted_indices(max_motions_up, omitted_indices),
            max_motions_down=_remap_from_omitted_indices(
                max_motions_down, omitted_indices
            ),
            **kwargs,
        )
        if total_displacement < least_displacement:
            best_indices = [omitted_indices]
            best_vls = [vls]
            least_displacement = total_displacement
        elif total_displacement == least_displacement:
            best_indices.append(omitted_indices)
            best_vls.append(vls)
    out = []
    for indices, vls in zip(best_indices, best_vls):
        for vl in vls:
            vl_out = []
            vl_i = 0
            for i in range(card1):
                if i in indices:
                    vl_out.append(None)
                else:
                    vl_out.append(vl[vl_i])
                    vl_i += 1
            out.append(tuple(vl_out))
    return out, least_displacement


def different_cardinality_handler(
    chord1: t.Sequence[int], chord2: t.Sequence[int], *args, **kwargs
) -> t.Tuple[t.List[VoiceLeadingMotion], int]:
    if len(chord1) > len(chord2):
        return shrinking_cardinality_handler(chord1, chord2, *args, **kwargs)
    return growing_cardinality_handler(chord1, chord2, *args, **kwargs)


def efficient_voice_leading(
    chord1: t.Sequence[int],
    chord2: t.Sequence[int],
    tet: int = 12,
    displacement_more_than: t.Optional[int] = None,
    exclude_motions: t.Optional[t.Dict[int, t.List[int]]] = None,
    max_motions_up: t.Optional[t.Dict[int, int]] = None,
    max_motions_down: t.Optional[t.Dict[int, int]] = None,
    allow_different_cards: bool = False,
    sort: bool = True,
) -> t.Tuple[t.List[VoiceLeadingMotion], int]:
    """Returns efficient voice-leading(s) between two chords.

    Keyword args:
        displacement_more_than: optional int; voice-leading will be enforced to
            have a greater total displacement than this interval.
        exclude_motions: optional dict of form int: list[int]. The note in
            chord1 whose index matches the key will be prevented from proceeding
            by any interval in the list.
        max_motions_up: optional dict. The note in chord1 whose index matches key
            can ascend by at most the indicated interval in the list.
        max_motions_down: optional dict. The note in chord1 whose index matches key
            can descend by at most the indicated interval in the list. (These
            intervals should be negative.)
        allow_different_cards: if True, then will provide voice-leadings for
            chords of different cardinalities. (Otherwise, raises a ValueError.)
            If this occurs, the returned
        sort: if True, then if chord1 contains any unisons, the voice-
            leadings applied to those unisons will be sorted in ascending
            order. Note that this means that `exclude_motions` and similar may
            fail (at least until the implementation gets more robust) if one
            of the pitches of the unison has an excluded motion but the others
            do not.

    Returns:
        Two-tuple consisting of
            - a list of voice-leadings. (A list because there may be more than
                one equally-efficient voice-leading.)

                Each voice leading is a tuple. The elements of the tuple can
                be ints, tuples of ints (in the case of a voice-leading to a
                chord of larger cardinality) or None (in the case of a voice-
                leading to a chord of smaller cardinality).
            - interval indicating the total displacement.

    Raises:
        NoMoreVoiceLeadingsError if there is no voice leading that satisfies
            displacement_more_than and exclude_motions
    """

    if displacement_more_than is None:
        displacement_more_than = -1
    if exclude_motions is None:
        exclude_motions = {}
    if max_motions_down is None:
        max_motions_down = {}
    if max_motions_up is None:
        max_motions_up = {}

    def _voice_leading_sub(in_indices, out_indices, current_sum):
        """This is a recursive function for calculating most efficient
        voice leadings.

        The idea is that a bijective voice-leading between two
        ordered chords can be represented by a single set of indexes,
        where the first index *x* maps the first note of the first chord
        on to the *x*th note of the second chord, and so on.

        Should be initially called as follows:
            _voice_leading_sub(list(range(cardinality of chords)),
                               [],
                               0)

        The following variables are found in the enclosing scope:
            chord1
            chord2
            best_sum
            best_vl_indices
            halftet
            tet
            displacement_more_than

        """
        # LONGTERM try multisets somehow?
        nonlocal best_sum, best_vl_indices
        if not in_indices:
            if current_sum > displacement_more_than:
                if best_sum > current_sum:
                    best_sum = current_sum
                    best_vl_indices = [out_indices]
                elif best_sum == current_sum:
                    best_vl_indices.append(out_indices)
        else:
            chord1_i = len(out_indices)
            this_p = chord1[chord1_i]
            unique = not any((this_p == p) for p in chord1[chord1_i + 1 :])
            for i, chord2_i in enumerate(in_indices):
                motion = chord2[chord2_i] - chord1[chord1_i]
                if motion > halftet:
                    motion -= tet
                elif motion < -halftet:
                    motion += tet
                if chord1_i in max_motions_up:
                    if motion > max_motions_up[chord1_i]:
                        continue
                if chord1_i in max_motions_down:
                    if motion < max_motions_down[chord1_i]:
                        continue
                if chord1_i in exclude_motions:
                    if motion in exclude_motions[chord1_i]:
                        #      MAYBE expand to include combinations of
                        #       multiple voice leading motions
                        continue
                displacement = abs(motion)
                present_sum = current_sum + displacement
                if present_sum > best_sum:
                    continue
                _voice_leading_sub(
                    in_indices[:i] + in_indices[i + 1 :],
                    out_indices + [chord2_i],
                    present_sum,
                )
                if not unique:
                    # if the pitch-class is not unique, all other mappings of
                    # it will already have been checked in the recursive calls,
                    # and so we can break here
                    break

    card = len(chord1)
    if card != len(chord2):
        if allow_different_cards:
            return different_cardinality_handler(
                chord1,
                chord2,
                tet=tet,
                displacement_more_than=displacement_more_than,
                exclude_motions=exclude_motions,
                max_motions_up=max_motions_up,
                max_motions_down=max_motions_down,
                sort=sort,
            )
        raise ValueError(f"{chord1} and {chord2} have different lengths.")

    best_sum = starting_sum = tet**8
    best_vl_indices = []
    halftet = tet // 2

    _voice_leading_sub(list(range(card)), [], 0)

    # If best_sum hasn't changed, then we haven't found any
    # voice-leadings.
    if best_sum == starting_sum:
        raise NoMoreVoiceLeadingsError

    # When there are unisons in chord2, there can be duplicate voice-leadings.
    # There is probably a more efficient way of avoiding generating these in
    # the first place, but for now, we get rid of them by casting to a set.
    voice_leading_intervals = list(
        {indices_to_vl(indices, chord1, chord2, tet) for indices in best_vl_indices}
    )

    if len(voice_leading_intervals) > 1:
        voice_leading_intervals.sort(key=np.var)

    if sort:
        unisons = tuple(p for p in set(chord1) if chord1.count(p) > 1)
        unison_indices = tuple(
            tuple(i for i, p1 in enumerate(chord1) if p1 == p2) for p2 in unisons
        )
        for j, vl in enumerate(voice_leading_intervals):
            vl_copy = list(vl)
            for indices in unison_indices:
                motions = [vl[i] for i in indices]
                motions.sort()
                for i, m in zip(indices, motions):
                    vl_copy[i] = m
            voice_leading_intervals[j] = tuple(vl_copy)

    return voice_leading_intervals, best_sum
