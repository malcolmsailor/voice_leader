from itertools import combinations
import numpy as np
import random

import typing as t


class NoMoreVoiceLeadingsError(Exception):
    pass


def apply_vl(vl, chord):
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


def indices_to_vl(indices, chord1, chord2, tet):
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


def voice_lead_pitches(
    chord1_pitches: t.Sequence[int],
    chord2_pcs: t.Sequence[int],
    preserve_root: bool = False,
    return_first: bool = True,
    avoid_bass_crossing: bool = True,
    tet: int = 12,
    allow_different_cards: bool = True,
    min_pitch: t.Optional[int] = None,
    max_pitch: t.Optional[int] = None,
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

    # TODO fix this example
    # >>> voice_lead_pitches([60, 64, 67], [7, 11, 2, 5], preserve_root=True,
    # ...     min_pitch=60)
    # [55, 62, 65, 71]
    """

    # TODO implement min_pitch and max_pitch for bass
    def _get_max_motions_up(pitches, start_i):
        if max_pitch is None:
            return None
        return {i: max_pitch - pitch for i, pitch in enumerate(pitches)}

    def _get_max_motions_down(pitches, start_i):
        if min_pitch is None:
            return None
        return {i: min_pitch - pitch for i, pitch in enumerate(pitches)}

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
                    max_motions_down=_get_max_motions_down(
                        chord1_pitches[j:], j
                    ),
                    max_motions_up=_get_max_motions_up(chord1_pitches[j:], j),
                )
            except NotImplementedError:
                # TODO document this exception?
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
    if return_first or len(motions) == 1:
        motion = motions[0]
    else:
        motion = random.choice(motions)
    if i == 0:
        # either we are not preserving the root, or the voice-leading proceeds
        # from the root as well
        out = apply_vl(motion, chord1_pitches)
        if preserve_root:
            while out[i] % tet != chord2_pcs[0]:
                i += 1
            if i != 0:
                out[i] -= tet
                out.sort()
        return out
    root_int = (chord2_pcs[0] - chord1_pitches[0]) % tet
    if root_int > tet // 2:
        root_int -= tet
    new_root = chord1_pitches[0] + root_int
    upper_parts = apply_vl(motion, chord1_pitches[1:])
    while new_root > upper_parts[0] and avoid_bass_crossing:
        new_root -= tet
    if max_pitch is not None:
        assert all(p <= max_pitch for p in upper_parts)
    return [new_root] + upper_parts


def _remap_from_doubled_indices(
    mapping: t.Dict[int, t.Any], doubled_indices: t.Sequence[int]
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
) -> t.Tuple[t.List[t.Tuple[t.Union[int, t.Tuple[int]], None]], int]:
    # This algorithm has horrible complexity but in practice I think that
    # chords are small enough (we will almost always be going from 3 to 4
    # notes) that it doesn't really matter. But nevertheless, if you are
    # voice-leading between chords of like ~20 elements, don't use this!

    card1, card2 = len(chord1), len(chord2)
    excess = card2 - card1
    if excess > card1:
        raise NotImplementedError
    least_displacement = 2 ** 31
    best_vls, best_indices = None, None
    if sort:
        previously_doubled_ps = set()
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
            max_motions_up=_remap_from_doubled_indices(
                max_motions_up, doubled_indices
            ),
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
):

    card1, card2 = len(chord1), len(chord2)
    excess = card1 - card2
    if excess > card2:
        raise NotImplementedError
    least_displacement = 2 ** 31
    best_vls, best_indices = None, None
    if sort:
        previously_omitted_ps = set()
    for omitted_indices in combinations(range(card1), excess):
        if sort:
            omitted_ps = tuple(chord1[i] for i in omitted_indices)
            if omitted_ps in previously_omitted_ps:
                continue
            previously_omitted_ps.add(omitted_ps)
        temp_chord = [
            p for (i, p) in enumerate(chord1) if i not in omitted_indices
        ]
        vls, total_displacement = efficient_voice_leading(
            temp_chord,
            chord2,
            *args,
            sort=sort,
            exclude_motions=_remap_from_omitted_indices(
                exclude_motions, omitted_indices
            ),
            max_motions_up=_remap_from_omitted_indices(
                max_motions_up, omitted_indices
            ),
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
) -> t.Tuple[t.List[t.Tuple[t.Union[int, t.Tuple[int]], None]], int]:
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
) -> t.Tuple[t.List[t.Tuple[t.Union[int, t.Tuple[int]], None]], int]:
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

    if displacement_more_than is None:
        displacement_more_than = -1
    if exclude_motions is None:
        exclude_motions = {}
    if max_motions_down is None:
        max_motions_down = {}
    if max_motions_up is None:
        max_motions_up = {}

    best_sum = starting_sum = tet ** 8
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
        {
            indices_to_vl(indices, chord1, chord2, tet)
            for indices in best_vl_indices
        }
    )

    if len(voice_leading_intervals) > 1:
        voice_leading_intervals.sort(key=np.var)

    if sort:
        unisons = tuple(p for p in set(chord1) if chord1.count(p) > 1)
        unison_indices = tuple(
            tuple(i for i, p1 in enumerate(chord1) if p1 == p2)
            for p2 in unisons
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
