from collections import defaultdict
from copy import deepcopy
from functools import reduce
import math
import random
from types import MappingProxyType
import typing as t
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import logging

from voice_leader.pitch_utils import (
    next_pc_down_from_pitch,
    next_pc_up_from_pitch,
    get_all_in_range,
)

LOGGER = logging.getLogger(__name__)


# In a VoiceLeadingAtom,
# - an int indicates that a voice proceeds by that interval
# - a tuple of int indicates that a voice splits into two or more voices
# - None indicates a voice that vanishes
VoiceLeadingAtom = t.Union[int, t.Tuple[int, ...], None]
VoiceLeadingMotion = t.Tuple[VoiceLeadingAtom, ...]

BijectiveVoiceLeadingAtom = int
BijectiveVoiceLeadingMotion = t.Tuple[BijectiveVoiceLeadingAtom, ...]

# In EquivalentVoiceLeadingMotions
# - the first item is a list of VoiceLeadingMotion
# - the second item is the total displacement for any of the VoiceLeadingMotion
EquivalentVoiceLeadingMotions = t.Tuple[t.List[VoiceLeadingMotion], int]

Index = int

PitchClass = int
Pitch = int
VoiceAssignments = t.Tuple[int, ...]


class NoMoreVoiceLeadingsError(Exception):
    pass


class CardinalityDiffersTooMuch(Exception):
    pass


def get_vl_atom_displacement(vl_atom: VoiceLeadingAtom) -> int:
    """
    >>> get_vl_atom_displacement(None)
    0
    >>> get_vl_atom_displacement(-2)
    2
    >>> get_vl_atom_displacement((3, -1))
    4
    """
    if vl_atom is None:
        return 0
    elif isinstance(vl_atom, int):
        return abs(vl_atom)
    return sum(abs(x) for x in vl_atom)


def apply_vl(
    vl: VoiceLeadingMotion, chord: t.Sequence[int]
) -> t.Tuple[t.Tuple[int], VoiceAssignments]:
    """
    `VoiceAssignments` are relative to the notes in `chord`. Not sure if this is exactly
    what we want.

    >>> apply_vl(vl=(0, 1, 2), chord=(60, 64, 67))
    ((60, 65, 69), (0, 1, 2))

    >>> apply_vl(vl=(-3, 1, None), chord=(60, 64, 67))
    ((57, 65), (0, 1))

    >>> apply_vl(vl=((-3, 0), (-1, 1), 2), chord=(60, 64, 67))
    ((57, 60, 63, 65, 69), (0, 0, 1, 1, 2))

    The returned pitches are sorted in ascending order:
    >>> apply_vl(vl=(5, -4, 2), chord=(60, 64, 67))
    ((60, 65, 69), (1, 0, 2))

    # >>> apply_vl(vl=(-3, 0), chord=(60, 64, 67))  # +doctest: IGNORE_EXCEPTION_DETAIL
    # Traceback (most recent call last):
    # AssertionError:

    # Raises:
    #     AssertionError if len(vl) != len(chord)
    """
    assert len(vl) == len(chord)

    out = []
    for i, motion in enumerate(vl):
        if isinstance(motion, tuple):
            for m in motion:
                pitch = chord[i] + m
                voice_assignment = i
                out.append((pitch, voice_assignment))
        elif motion is None:
            continue
        else:
            pitch = chord[i] + motion
            voice_assignment = i
            out.append((pitch, voice_assignment))

    # Sort by pitch
    out.sort(key=lambda x: x[0])

    return tuple(zip(*out))


def indices_to_vl(
    indices: t.Iterable[int], chord1: t.Sequence[int], chord2: t.Sequence[int], tet: int
) -> BijectiveVoiceLeadingMotion:
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


def get_preserve_bass_vl_atom(
    src_pitches: t.Sequence[Pitch],
    dst_pcs: t.Sequence[PitchClass],
    min_bass_pitch: t.Optional[int] = None,
    max_bass_pitch: t.Optional[int] = None,
) -> t.Iterator[BijectiveVoiceLeadingAtom]:
    """
    The return values are sorted by displacement.

    >>> src_pitches = [55, 62, 71]
    >>> dst_pcs = [0, 4, 7]

    If `min_bass_pitch` and `max_bass_pitch` arguments are not provided, we return the
    next lower/higher pitches respectively.
    >>> list(get_preserve_bass_vl_atom(src_pitches, dst_pcs))
    [5, -7]
    >>> list(get_preserve_bass_vl_atom(src_pitches, dst_pcs, max_bass_pitch=59))
    [-7]
    >>> list(get_preserve_bass_vl_atom(src_pitches, dst_pcs, min_bass_pitch=49))
    [5]
    >>> list(get_preserve_bass_vl_atom(src_pitches, dst_pcs, min_bass_pitch=49, max_bass_pitch=59))
    []
    >>> list(get_preserve_bass_vl_atom(src_pitches, dst_pcs, min_bass_pitch=32, max_bass_pitch=84))
    [5, -7, 17, -19, 29]
    """
    src_root_pitch = src_pitches[0]
    dst_root_pc = dst_pcs[0]
    if min_bass_pitch is None and max_bass_pitch is None:
        dst_pitch_options = [
            next_pc_up_from_pitch(src_root_pitch, dst_root_pc),
            next_pc_down_from_pitch(src_root_pitch, dst_root_pc),
        ]
    elif min_bass_pitch is None:
        dst_pitch_options = get_all_in_range(
            dst_root_pc, low=src_root_pitch, high=max_bass_pitch  # type:ignore
        ) + [next_pc_down_from_pitch(src_root_pitch, dst_root_pc)]
    elif max_bass_pitch is None:
        dst_pitch_options = get_all_in_range(
            dst_root_pc, low=min_bass_pitch, high=src_root_pitch
        ) + [next_pc_up_from_pitch(src_root_pitch, dst_root_pc)]
    else:
        dst_pitch_options = get_all_in_range(
            dst_root_pc, low=min_bass_pitch, high=max_bass_pitch
        )
    atoms = [dst_pitch - src_root_pitch for dst_pitch in dst_pitch_options]
    atoms.sort(key=abs)

    yield from atoms


def pop_voice_leading_from_options(
    options: t.List[EquivalentVoiceLeadingMotions],
    indices: t.Tuple[int, int],
    vl_iters: t.Optional[t.List[t.Iterator[EquivalentVoiceLeadingMotions]]] = None,
) -> VoiceLeadingMotion:
    """
    >>> options = [
    ...     ([(1, -1, 0), (-1, 1, 0)], 2),
    ...     ([(0, 0, 1)], 1),
    ...     ([(0, 0, 5)], 5),
    ... ]
    >>> pop_voice_leading_from_options(options, (0, 1))
    (-1, 1, 0)
    >>> options  # doctest: +NORMALIZE_WHITESPACE
    [([(1, -1, 0)], 2),
    ([(0, 0, 1)], 1),
    ([(0, 0, 5)], 5)]
    >>> pop_voice_leading_from_options(options, (1, 0))
    (0, 0, 1)
    >>> options  # doctest: +NORMALIZE_WHITESPACE
    [([(1, -1, 0)], 2),
    ([(0, 0, 5)], 5)]
    >>> pop_voice_leading_from_options(options, (1, 1))
    Traceback (most recent call last):
    IndexError: pop index out of range

    If we provide the `vl_iters` argument, when the EquivalentVoiceLeadingMotions at
    a given index are exhausted, they will be refreshed with the next item from the
    corresponding voice-leading iterator.

    Note: the `vl_iters` below are musically meaningless and just to exhibit the
    function logic.

    >>> vl_iters = [efficient_pc_voice_leading_iter((0, 4, 7), (0, 3, 8))] * 2
    >>> pop_voice_leading_from_options(options, (1, 0), vl_iters=vl_iters)
    (0, 0, 5)
    >>> options  # doctest: +NORMALIZE_WHITESPACE
    [([(1, -1, 0)], 2),
    ([(0, -1, 1)], 2)]
    """
    option_i, vl_motion_i = indices
    out = options[option_i][0].pop(vl_motion_i)
    if not options[option_i][0]:
        if vl_iters is not None:
            try:
                options[option_i] = next(vl_iters[option_i])
            except StopIteration:
                options.pop(option_i)
        else:
            options.pop(option_i)
    return out


def choose_voice_leading_from_options(
    options: t.List[EquivalentVoiceLeadingMotions],
) -> t.Tuple[int, int]:
    """
    >>> options = [
    ...     ([(1, -1, 0), (-1, 1, 0)], 2),
    ...     ([(0, 0, 1)], 1),
    ...     ([(0, 0, 5)], 5),
    ... ]
    >>> choose_voice_leading_from_options(options)
    (1, 0)
    >>> pop_voice_leading_from_options(options, (1, 0))
    (0, 0, 1)
    >>> choose_voice_leading_from_options(options)
    (0, 0)

    Longterm we might wish to sample from options inversely proportional to their
    displacement rather than just take argmin.
    """
    if not options:
        raise ValueError("There must be at least one option")

    # argmin
    min_i, min_displacement = 0, options[0][1]
    for i, option in enumerate(options[1:], start=1):
        if option[1] < min_displacement:
            min_i = i
            min_displacement = option[1]

    # for now we just take the first option
    return min_i, 0


def apply_next_vl_from_vl_iters(
    vl_iters: t.List[t.Iterator[EquivalentVoiceLeadingMotions]],
    src_pitches: t.Sequence[Pitch],
) -> t.Iterator[t.Tuple[t.Sequence[Pitch], VoiceAssignments]]:
    # Initialize options
    options = []
    for vl_iter in vl_iters:
        try:
            options.append(next(vl_iter))
        except StopIteration:
            pass

    # Yield from options
    while options:
        choice = choose_voice_leading_from_options(options)
        vl_motion = pop_voice_leading_from_options(options, choice, vl_iters)
        yield apply_vl(vl_motion, src_pitches)


def insert_in_vl_motion(
    vl_motion: VoiceLeadingMotion, insertions: t.Mapping[Index, VoiceLeadingAtom]
) -> VoiceLeadingMotion:
    """
    >>> insert_in_vl_motion((-1, 2, 0), {0: 3, 4: 0})
    (3, -1, 2, 0, 0)
    """
    vl_motion_list = list(vl_motion)
    for insertion_i in sorted(insertions.keys()):
        vl_motion_list.insert(insertion_i, insertions[insertion_i])
    return tuple(vl_motion_list)


def prepend_to_vl_motion(
    vl_motion: VoiceLeadingMotion,
    prependings: t.Mapping[Index, BijectiveVoiceLeadingAtom],
) -> VoiceLeadingMotion:
    """
    >>> prepend_to_vl_motion((-1, None, (1, 2)), {0: 0, 1: 2, 2: 0})
    ((0, -1), 2, (0, 1, 2))
    """
    vl_motion_list = list(vl_motion)
    for insertion_i, motion in prependings.items():
        if vl_motion_list[insertion_i] is None:
            vl_motion_list[insertion_i] = motion
        elif isinstance(vl_motion_list[insertion_i], int):
            vl_motion_list[insertion_i] = (  # type:ignore
                motion,
                vl_motion_list[insertion_i],
            )
        elif isinstance(vl_motion_list[insertion_i], tuple):
            vl_motion_list[insertion_i] = (motion,) + vl_motion_list[
                insertion_i
            ]  # type:ignore
        else:
            raise ValueError(
                "vl atoms must be None, int, or tuple of ints; "
                f"got {vl_motion_list[insertion_i]}"
            )
    return tuple(vl_motion_list)


def postpend_to_vl_motion(
    vl_motion: VoiceLeadingMotion,
    prependings: t.Mapping[Index, BijectiveVoiceLeadingAtom],
) -> VoiceLeadingMotion:
    """
    >>> postpend_to_vl_motion((-1, None, (1, 2)), {0: 0, 1: 2, 2: 0})
    ((-1, 0), 2, (1, 2, 0))
    """
    vl_motion_list = list(vl_motion)
    for insertion_i, motion in prependings.items():
        if vl_motion_list[insertion_i] is None:
            vl_motion_list[insertion_i] = motion
        elif isinstance(vl_motion_list[insertion_i], int):
            vl_motion_list[insertion_i] = (  # type:ignore
                vl_motion_list[insertion_i],
                motion,
            )
        elif isinstance(vl_motion_list[insertion_i], tuple):
            vl_motion_list[insertion_i] = vl_motion_list[insertion_i] + (
                motion,
            )  # type:ignore
        else:
            raise ValueError(
                "vl atoms must be None, int, or tuple of ints; "
                f"got {vl_motion_list[insertion_i]}"
            )
    return tuple(vl_motion_list)


def vl_iter_wrapper(
    vl_iter: t.Iterator[EquivalentVoiceLeadingMotions],
    insertions: t.Mapping[Index, VoiceLeadingAtom] = MappingProxyType({}),
    prependings: t.Mapping[Index, BijectiveVoiceLeadingAtom] = MappingProxyType({}),
    postpendings: t.Mapping[Index, BijectiveVoiceLeadingAtom] = MappingProxyType({}),
) -> t.Iterator[EquivalentVoiceLeadingMotions]:
    """
    Insertions are applied *before* prependings/postpendings.
    >>> vl_iter = (([(0, None, (1, 2))], 3) for _ in range(4))
    >>> next(vl_iter_wrapper(vl_iter, insertions={0: -1}, prependings={2: 4}))
    ([(-1, 0, 4, (1, 2))], 8)

    """
    for vl_motions, displacement in vl_iter:
        if insertions:
            vl_motions = [
                insert_in_vl_motion(vl_motion, insertions) for vl_motion in vl_motions
            ]
            displacement += sum(
                get_vl_atom_displacement(x) for x in insertions.values()
            )
        if prependings:
            vl_motions = [
                prepend_to_vl_motion(vl_motion, prependings) for vl_motion in vl_motions
            ]
            displacement += sum(abs(x) for x in prependings.values())
        if postpendings:
            vl_motions = [
                postpend_to_vl_motion(vl_motion, postpendings)
                for vl_motion in vl_motions
            ]
            displacement += sum(abs(x) for x in postpendings.values())
        yield vl_motions, displacement


def ignore_voice_assignments_wrapper(
    apply_iter: t.Iterator[t.Tuple[t.Sequence[Pitch], VoiceAssignments]]
) -> t.Iterator[t.Tuple[t.Sequence[Pitch], VoiceAssignments]]:
    memory = set()
    for out, voice_assignments in apply_iter:
        if tuple(out) in memory:
            continue
        memory.add(tuple(out))
        yield out, voice_assignments


def efficient_pitch_voice_leading_iter(
    src_pitches: t.Sequence[Pitch],
    dst_pcs: t.Sequence[PitchClass],
    tet: int = 12,
    allow_different_cards: bool = True,
    min_pitch: t.Optional[int] = None,
    max_pitch: t.Optional[int] = None,
) -> t.Iterator[EquivalentVoiceLeadingMotions]:
    """
    >>> src_pitches = (55, 62, 67)
    >>> dst_pcs = (0, 4, 7)
    >>> vl_iter = efficient_pitch_voice_leading_iter(src_pitches, dst_pcs)
    >>> next(vl_iter)[0], next(vl_iter)[0]
    ([(-3, -2, 0), (0, -2, -3)], [(5, 2, 0), (0, 2, 5)])
    """

    def _get_max_motions_up(pitches):
        if max_pitch is None:
            return None
        return {i: max_pitch - pitch for i, pitch in enumerate(pitches)}

    def _get_max_motions_down(pitches):
        if min_pitch is None:
            return None
        return {i: min_pitch - pitch for i, pitch in enumerate(pitches)}

    return efficient_pc_voice_leading_iter(
        [p % tet for p in src_pitches],
        dst_pcs,
        tet=tet,
        allow_different_cards=allow_different_cards,
        max_motions_down=_get_max_motions_down(src_pitches),
        max_motions_up=_get_max_motions_up(src_pitches),
        sort=False,
    )


def preserve_and_split_root_vl_iters(
    src_pitches: t.Sequence[Pitch],
    dst_pcs: t.Sequence[PitchClass],
    avoid_bass_crossing: bool = True,
    min_pitch: t.Optional[int] = None,
    max_pitch: t.Optional[int] = None,
    min_bass_pitch: t.Optional[int] = None,
    max_bass_pitch: t.Optional[int] = None,
    **vl_kwargs,
) -> t.List[t.Iterator[EquivalentVoiceLeadingMotions]]:
    """
    >>> src_pitches = [60, 64, 67]
    >>> dst_pcs = [2, 5, 7, 11]
    >>> vl_iters = preserve_and_split_root_vl_iters(src_pitches, dst_pcs)
    >>> [next(vl_iter) for vl_iter in vl_iters]
    [([((2, 5), 3, 4)], 14), ([((-10, -1), 1, 0)], 12)]
    """
    # Since `preserve_bass` is True, if we are splitting the bass, one of
    # the bass voice-leading motions must be the two roots, and the root of
    # the second chord must be lower than the other parts.
    if max_bass_pitch is None:
        max_bass_pitch = max_pitch

    root_motions = get_preserve_bass_vl_atom(
        src_pitches,
        dst_pcs,
        min_bass_pitch=min_bass_pitch,
        max_bass_pitch=max_bass_pitch,
    )
    vl_iters: t.List[t.Iterator[EquivalentVoiceLeadingMotions]] = []
    this_min_pitch = min_pitch

    for root_motion in root_motions:
        if avoid_bass_crossing:
            dst_bass = src_pitches[0] + root_motion
            this_min_pitch = dst_bass if min_pitch is None else max(min_pitch, dst_bass)
        sub_iter = efficient_pitch_voice_leading_iter(
            src_pitches,
            dst_pcs[1:],
            min_pitch=this_min_pitch,
            max_pitch=max_pitch,
            **vl_kwargs,
        )
        vl_iter = vl_iter_wrapper(sub_iter, prependings={0: root_motion})
        vl_iters.append(vl_iter)
    return vl_iters


def preserve_bass_vl_iters(
    src_pitches: t.Sequence[Pitch],
    dst_pcs: t.Sequence[PitchClass],
    avoid_bass_crossing: bool = True,
    min_pitch: t.Optional[int] = None,
    max_pitch: t.Optional[int] = None,
    min_bass_pitch: t.Optional[int] = None,
    max_bass_pitch: t.Optional[int] = None,
    **vl_kwargs,
) -> t.List[t.Iterator[EquivalentVoiceLeadingMotions]]:
    """Note: this will not produce voice-leadings where the bass-note of the second
    chord is doubled.

    >>> src_pitches = [60, 64, 67]
    >>> dst_pcs = [5, 8, 0]
    >>> vl_iters = preserve_bass_vl_iters(src_pitches, dst_pcs)
    >>> [next(vl_iter) for vl_iter in vl_iters]
    [([(5, 4, 5)], 14), ([(-7, -4, 1)], 12)]

    >>> src_pitches = [60, 64, 67]
    >>> dst_pcs = [2, 5, 7, 11]
    >>> vl_iters = preserve_bass_vl_iters(src_pitches, dst_pcs)
    >>> [next(vl_iter) for vl_iter in vl_iters]
    [([(2, 1, (0, 4))], 7), ([(-10, 1, (0, 4))], 15)]

    """
    if max_bass_pitch is None:
        max_bass_pitch = max_pitch

    root_motions = get_preserve_bass_vl_atom(
        src_pitches,
        dst_pcs,
        min_bass_pitch=min_bass_pitch,
        max_bass_pitch=max_bass_pitch,
    )
    vl_iters: t.List[t.Iterator[EquivalentVoiceLeadingMotions]] = []
    this_min_pitch = min_pitch

    for root_motion in root_motions:
        if avoid_bass_crossing:
            dst_bass = src_pitches[0] + root_motion
            this_min_pitch = dst_bass if min_pitch is None else max(min_pitch, dst_bass)
        sub_iter = efficient_pitch_voice_leading_iter(
            src_pitches[1:],
            dst_pcs[1:],
            min_pitch=this_min_pitch,
            max_pitch=max_pitch,
            **vl_kwargs,
        )
        vl_iter = vl_iter_wrapper(sub_iter, insertions={0: root_motion})
        vl_iters.append(vl_iter)
    return vl_iters


def dont_preserve_bass_vl_iters(
    src_pitches: t.Sequence[Pitch],
    dst_pcs: t.Sequence[PitchClass],
    avoid_bass_crossing: bool = True,
    **vl_kwargs,
) -> t.List[t.Iterator[EquivalentVoiceLeadingMotions]]:
    """Note: this will not produce voice-leadings where the bass-note of the second
    chord is doubled.

    >>> src_pitches = [60, 64, 67]
    >>> dst_pcs = [2, 5, 7, 11]
    >>> vl_iters = dont_preserve_bass_vl_iters(src_pitches, dst_pcs)
    >>> [next(vl_iter) for vl_iter in vl_iters]
    [([((-1, 2), 1, 0), (-1, (-2, 1), 0)], 4)]
    """

    # min_bass_pitch and max_bass_pitch are unused
    vl_kwargs = vl_kwargs.copy()
    vl_kwargs.pop("min_bass_pitch", None)
    vl_kwargs.pop("max_bass_pitch", None)

    if avoid_bass_crossing:
        LOGGER.warning(
            "`dont_preserve_bass_vl_iters()` called with "
            "`avoid_bass_crossing=True`. This parameter is not implemented yet and "
            "will have no effect."
        )
    return [efficient_pitch_voice_leading_iter(src_pitches, dst_pcs, **vl_kwargs)]


def get_voice_lead_pitches_iters(
    chord1_pitches: t.Sequence[Pitch],
    chord2_pcs: t.Sequence[PitchClass],
    preserve_bass: bool = False,
    avoid_bass_crossing: bool = True,
    tet: int = 12,
    allow_different_cards: bool = True,
    min_pitch: t.Optional[int] = None,
    max_pitch: t.Optional[int] = None,
    min_bass_pitch: t.Optional[int] = None,
    max_bass_pitch: t.Optional[int] = None,
) -> t.List[t.Iterator[EquivalentVoiceLeadingMotions]]:
    """
    >>> chord1_pitches = (47, 55, 62, 67)
    >>> chord2_pcs = (0, 0, 4, 7)
    >>> vl_iter = get_voice_lead_pitches_iters(chord1_pitches, chord2_pcs, preserve_bass=True)
    >>> [next(v) for v in vl_iter]
    [([(1, -3, -2, 0), (1, 0, -2, -3)], 6), ([(-11, -3, -2, 0), (-11, 0, -2, -3)], 16)]
    """
    subroutine_kwargs = {
        "avoid_bass_crossing": avoid_bass_crossing,
        "tet": tet,
        "allow_different_cards": allow_different_cards,
        "min_pitch": min_pitch,
        "max_pitch": max_pitch,
        "min_bass_pitch": min_bass_pitch,
        "max_bass_pitch": max_bass_pitch,
    }

    if preserve_bass and not avoid_bass_crossing:
        LOGGER.warning(
            "`preserve_bass=True` and `avoid_bass_crossing=False`: bass is not "
            "guaranteed to be preserved"
        )

    if preserve_bass and len(chord1_pitches) < len(chord2_pcs):
        # If chord1 has fewer pitches than chord2 has pcs, then at least
        # one pitch from chord1 will be "split" into multiple pitches in
        # chord2. Ordinarily when preserve_bass is True, we exclude the
        # root from the voice-leading calculation, but since we might
        # want to split the root, we need to try including it here.

        # -----------------------------------------------------------------------
        # Case 1: split root
        # -----------------------------------------------------------------------

        split_root_iters = preserve_and_split_root_vl_iters(
            chord1_pitches, chord2_pcs, **subroutine_kwargs
        )

        # -----------------------------------------------------------------------
        # Case 2: don't split root
        # -----------------------------------------------------------------------

        dont_split_root_iters = preserve_bass_vl_iters(
            chord1_pitches, chord2_pcs, **subroutine_kwargs
        )
        vl_iters = split_root_iters + dont_split_root_iters

    elif preserve_bass:
        vl_iters = preserve_bass_vl_iters(
            chord1_pitches, chord2_pcs, **subroutine_kwargs
        )
    else:
        vl_iters = dont_preserve_bass_vl_iters(
            chord1_pitches, chord2_pcs, **subroutine_kwargs
        )
    return vl_iters


def voice_lead_pitches_multiple_options_iter(
    chord1_pitches: t.Sequence[Pitch],
    chord2_pcs_options: t.Iterator[t.Sequence[PitchClass]],
    ignore_voice_assignments: bool = True,
    **get_voice_lead_pitches_iters_kwargs,
) -> t.Iterator[t.Tuple[t.Sequence[Pitch], VoiceAssignments]]:
    """
    >>> chord1_pitches = [60, 64, 67]
    >>> chord2_pcs_options = [[2, 5, 7, 11], [5, 7, 11], [2, 5, 7]]
    >>> vl_iter = voice_lead_pitches_multiple_options_iter(chord1_pitches, chord2_pcs_options)
    >>> next(vl_iter)[0], next(vl_iter)[0], next(vl_iter)[0]
    ((59, 65, 67), (62, 65, 67), (59, 62, 65, 67))
    """
    vl_iters = []
    for chord2_pcs in chord2_pcs_options:
        vl_iters += get_voice_lead_pitches_iters(
            chord1_pitches, chord2_pcs, **get_voice_lead_pitches_iters_kwargs
        )
    apply_iter = apply_next_vl_from_vl_iters(vl_iters, chord1_pitches)

    if ignore_voice_assignments:
        apply_iter = ignore_voice_assignments_wrapper(apply_iter)

    yield from apply_iter


# TODO: (Malcolm) rename preserve_bass to the more accurate preserve_bass
def voice_lead_pitches_iter(
    chord1_pitches: t.Sequence[Pitch],
    chord2_pcs: t.Sequence[PitchClass],
    ignore_voice_assignments: bool = True,
    **get_voice_lead_pitches_iters_kwargs,
) -> t.Iterator[t.Tuple[t.Sequence[Pitch], VoiceAssignments]]:
    """
    >>> chord1_pitches = [60, 64, 67]
    >>> chord2_pcs = [2, 5, 7, 11]
    >>> vl_iter = voice_lead_pitches_iter(chord1_pitches, chord2_pcs)
    >>> next(vl_iter)[0], next(vl_iter)[0], next(vl_iter)[0]
    ((59, 62, 65, 67), (62, 65, 67, 71), (55, 62, 65, 71))

    If `ignore_voice_assignments` is True, then voice-leadings that result in exactly
    the same pitches (differing only in which voice is led where) are considered
    equivalent and not returned separately.

    >>> vl_iter = voice_lead_pitches_iter(chord1_pitches, chord2_pcs,
    ...                                   ignore_voice_assignments=False)
    >>> next(vl_iter), next(vl_iter)
    (((59, 62, 65, 67), (0, 0, 1, 2)), ((59, 62, 65, 67), (0, 1, 1, 2)))

    >>> chord1_pitches = (47, 55, 62, 67)
    >>> chord2_pcs = (0, 0, 4, 7)
    >>> vl_iter = voice_lead_pitches_iter(chord1_pitches, chord2_pcs, preserve_bass=True)
    >>> next(vl_iter)[0], next(vl_iter)[0], next(vl_iter)[0]
    ((48, 52, 60, 67), (48, 55, 60, 64), (48, 60, 64, 67))
    """
    vl_iters = get_voice_lead_pitches_iters(
        chord1_pitches, chord2_pcs, **get_voice_lead_pitches_iters_kwargs
    )

    apply_iter = apply_next_vl_from_vl_iters(vl_iters, chord1_pitches)

    if ignore_voice_assignments:
        apply_iter = ignore_voice_assignments_wrapper(apply_iter)

    yield from apply_iter


def voice_lead_pitches(
    chord1_pitches: t.Sequence[Pitch],
    chord2_pcs: t.Sequence[PitchClass],
    preserve_bass: bool = False,
    avoid_bass_crossing: bool = True,
    tet: int = 12,
    allow_different_cards: bool = True,
    min_pitch: t.Optional[int] = None,
    max_pitch: t.Optional[int] = None,
    min_bass_pitch: t.Optional[int] = None,
    max_bass_pitch: t.Optional[int] = None,
    return_first: bool = True,
) -> t.Sequence[Pitch]:
    """
    >>> voice_lead_pitches([60, 64, 67], [5, 8, 0])
    (60, 65, 68)
    >>> voice_lead_pitches([60, 64, 67], [5, 8, 0], preserve_bass=True)
    (53, 60, 68)

    >>> voice_lead_pitches([60, 64, 67], [0, 4, 9], preserve_bass=True)
    (60, 64, 69)

    If preserve_bass is True, the bass voice can exceed 'min_pitch'

    >>> voice_lead_pitches([60, 64, 67], [7, 11, 2], min_pitch=60)
    (62, 67, 71)
    >>> voice_lead_pitches([60, 64, 67], [7, 11, 2], preserve_bass=True,
    ...     min_pitch=60)
    (55, 62, 71)

    TODO what happens with min_bass_pitch?
    # >>> voice_lead_pitches([60, 64, 67], [7, 11, 2], preserve_bass=True,
    # ...     min_bass_pitch=60)
    # (55, 62, 71)

    >>> voice_lead_pitches([60, 64, 67], [7, 11, 2, 5], preserve_bass=True,
    ...     min_pitch=60)
    (55, 62, 65, 71)

    Note that when shrinking cardinality, pitches will not be doubled.
    >>> voice_lead_pitches([64, 65, 67, 71], [0, 4, 7], preserve_bass=True)
    (60, 64, 67)

    However, it is possible to double pitch-classes in `chord2_pcs`.
    >>> voice_lead_pitches([64, 65, 67, 71], [0, 0, 4, 7], preserve_bass=True)
    (60, 64, 67, 72)
    """
    iterator = voice_lead_pitches_iter(
        chord1_pitches,
        chord2_pcs,
        preserve_bass=preserve_bass,
        avoid_bass_crossing=avoid_bass_crossing,
        tet=tet,
        allow_different_cards=allow_different_cards,
        min_pitch=min_pitch,
        max_pitch=max_pitch,
        min_bass_pitch=min_bass_pitch,
        max_bass_pitch=max_bass_pitch,
    )
    if return_first:
        return next(iterator)[0]

    # TODO: (Malcolm) I think we need to constrain the maximum number of options here
    return random.choice(list(iterator))[0]


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
    src_pcs: t.Sequence[PitchClass],
    dst_pcs: t.Sequence[PitchClass],
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

    card1, card2 = len(src_pcs), len(dst_pcs)
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
            doubled_ps = tuple(src_pcs[i] for i in doubled_indices)
            if doubled_ps in previously_doubled_ps:
                continue
            previously_doubled_ps.add(doubled_ps)
        temp_chord = []
        for i, p in enumerate(src_pcs):
            temp_chord.extend([p, p] if i in doubled_indices else [p])
        vls, total_displacement = efficient_pc_voice_leading(
            temp_chord,
            dst_pcs,
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
    src_pcs: t.Sequence[PitchClass],
    dst_pcs: t.Sequence[PitchClass],
    *args,
    sort: bool = True,
    exclude_motions: t.Optional[t.Dict[int, t.List[int]]] = None,
    max_motions_up: t.Optional[t.Dict[int, int]] = None,
    max_motions_down: t.Optional[t.Dict[int, int]] = None,
    **kwargs,
) -> t.Tuple[t.List[VoiceLeadingMotion], int]:
    card1, card2 = len(src_pcs), len(dst_pcs)
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
            omitted_ps = tuple(src_pcs[i] for i in omitted_indices)
            if omitted_ps in previously_omitted_ps:
                continue
            previously_omitted_ps.add(omitted_ps)
        temp_chord = [p for (i, p) in enumerate(src_pcs) if i not in omitted_indices]
        vls, total_displacement = efficient_pc_voice_leading(
            temp_chord,
            dst_pcs,
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
    src_pcs: t.Sequence[PitchClass], dst_pcs: t.Sequence[PitchClass], *args, **kwargs
) -> t.Tuple[t.List[VoiceLeadingMotion], int]:
    if len(src_pcs) > len(dst_pcs):
        return shrinking_cardinality_handler(src_pcs, dst_pcs, *args, **kwargs)
    return growing_cardinality_handler(src_pcs, dst_pcs, *args, **kwargs)


# TODO: (Malcolm) rather than using this function for pitch voice-leading, I may want
# to use a new function (in particular this function will never produce intervals > 6)
def efficient_pc_voice_leading(
    src_pcs: t.Sequence[PitchClass],
    dst_pcs: t.Sequence[PitchClass],
    *,
    tet: int = 12,
    displacement_more_than: t.Optional[int] = None,
    exclude_motions: t.Optional[t.Dict[int, t.List[int]]] = None,
    max_motions_up: t.Optional[t.Dict[int, int]] = None,
    max_motions_down: t.Optional[t.Dict[int, int]] = None,
    allow_different_cards: bool = False,
    sort: bool = True,
) -> EquivalentVoiceLeadingMotions:
    """Returns efficient voice-leading(s) between two chords.

    >>> src_pcs = [7, 0, 2]
    >>> dst_pcs = [4, 8, 11]
    >>> vl_motions, total_displacement = efficient_pc_voice_leading(src_pcs, dst_pcs)
    >>> vl_motions
    [(1, -1, 2)]
    >>> total_displacement
    4

    >>> src_pcs = (7, 2, 7)
    >>> dst_pcs = (0, 4, 7)
    >>> efficient_pc_voice_leading(src_pcs, dst_pcs)
    ([(-3, -2, 0)], 5)
    >>> efficient_pc_voice_leading(src_pcs, dst_pcs, sort=False)
    ([(-3, -2, 0), (0, -2, -3)], 5)

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
            - integer indicating the total displacement.

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
            this_p = src_pcs[chord1_i]
            unique = not any((this_p == p) for p in src_pcs[:chord1_i])
            for i, chord2_i in enumerate(in_indices):
                motion = dst_pcs[chord2_i] - src_pcs[chord1_i]
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
                    # TODO: (Malcolm) I'm not sure that this is correct; what about
                    # cases of tripled tones etc.?
                    break

    card = len(src_pcs)
    if card != len(dst_pcs):
        if allow_different_cards:
            return different_cardinality_handler(
                src_pcs,
                dst_pcs,
                tet=tet,
                displacement_more_than=displacement_more_than,
                exclude_motions=exclude_motions,
                max_motions_up=max_motions_up,
                max_motions_down=max_motions_down,
                sort=sort,
            )
        raise ValueError(f"{src_pcs} and {dst_pcs} have different lengths.")

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
        {indices_to_vl(indices, src_pcs, dst_pcs, tet) for indices in best_vl_indices}
    )

    if len(voice_leading_intervals) > 1:
        voice_leading_intervals.sort(key=np.var)

    if sort:
        unisons = tuple(p for p in set(src_pcs) if src_pcs.count(p) > 1)
        unison_indices = tuple(
            tuple(i for i, p1 in enumerate(src_pcs) if p1 == p2) for p2 in unisons
        )
        for j, vl in enumerate(voice_leading_intervals):
            vl_copy = list(vl)
            for indices in unison_indices:
                motions = [vl[i] for i in indices]
                motions.sort()
                for i, m in zip(indices, motions):
                    vl_copy[i] = m
            voice_leading_intervals[j] = tuple(vl_copy)

        voice_leading_intervals = list(set(voice_leading_intervals))

    return voice_leading_intervals, best_sum  # type:ignore


def efficient_pc_voice_leading_iter(
    *args, **kwargs
) -> t.Iterator[EquivalentVoiceLeadingMotions]:
    """
    >>> src_pcs = (7, 2, 7)
    >>> dst_pcs = (0, 4, 7)
    >>> vl_iter = efficient_pc_voice_leading_iter(src_pcs, dst_pcs)
    >>> next(vl_iter)[0], next(vl_iter)[0]
    ([(-3, -2, 0)], [(0, 2, 5)])
    """
    displacement_more_than = kwargs.pop("displacement_more_than", -1)
    while True:
        try:
            vl_motions, total_displacement = efficient_pc_voice_leading(
                *args, displacement_more_than=displacement_more_than, **kwargs
            )
            yield vl_motions, total_displacement
            displacement_more_than = total_displacement
        except NoMoreVoiceLeadingsError:
            return
