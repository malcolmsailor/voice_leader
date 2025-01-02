import logging
import math
import random
import typing as t
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
from itertools import combinations
from types import MappingProxyType

import numpy as np

from voice_leader.pitch_utils import (
    closest_pc_iter,
    get_all_in_range,
    next_pc_down_from_pitch,
    next_pc_up_from_pitch,
)

LOGGER = logging.getLogger(__name__)

ChromaticInterval = int
Index = int

PitchClass = int
Pitch = int
PitchOrPitchClass = int

# In a VoiceLeadingAtom,
# - an int indicates that a voice proceeds by that interval
# - a tuple of int indicates that a voice splits into two or more voices
# - None indicates a voice that vanishes
VoiceLeadingAtom = t.Union[ChromaticInterval, t.Tuple[ChromaticInterval, ...], None]
VoiceLeadingMotion = t.Tuple[VoiceLeadingAtom, ...]

BijectiveVoiceLeadingAtom = ChromaticInterval
BijectiveVoiceLeadingMotion = t.Tuple[BijectiveVoiceLeadingAtom, ...]

# In EquivalentVoiceLeadingMotions
# - the first item is a list of VoiceLeadingMotion
# - the second item is the total displacement for any of the VoiceLeadingMotion
# Semantically, EquivalentVoiceLeadingMotions are meant to contain voice-leadings that have the same
#   number of "destinations" (see `count_vl_destinations` below)
EquivalentVoiceLeadingMotions = t.Tuple[t.List[VoiceLeadingMotion], int]


VoiceAssignments = t.Tuple[int, ...]


def softmax(x, temperature=1.0):
    """
    >>> weights = [1 / 2, 2 / 3]
    >>> softmax(weights)
    array([0.45842952, 0.54157048])
    >>> softmax(weights, temperature=5.0)
    array([0.49166744, 0.50833256])
    >>> softmax(weights, temperature=0.2)
    array([0.30294072, 0.69705928])
    """
    exp = np.exp(np.array(x) / temperature)
    return exp / exp.sum()


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


def count_vl_destinations(vl: VoiceLeadingMotion) -> int:
    """
    - a bijective atom like `(1,)` has 1 destination
    - an atom of None has 0 destinations
    - an atom like (2, 3) has 2 destinations
    The output is the sum of all atoms of the voice-leading.

    >>> count_vl_destinations((11, 15, 7))
    3
    >>> count_vl_destinations((-1, None, 0))
    2
    >>> count_vl_destinations((-1, (-2, -2), 0))
    4
    >>> count_vl_destinations((None, (1, 2, 3), (4, 5)))
    5
    >>> count_vl_destinations(())
    0
    >>> count_vl_destinations((None, None, None))
    0
    """
    out = 0
    for vl_atom in vl:
        if vl_atom is None:
            continue
        elif isinstance(vl_atom, tuple):
            out += len(vl_atom)
        else:
            out += 1
    return out


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
    >>> indices_to_vl(
    ...     indices=(0, 1, 2), chord1=(60, 64, 67), chord2=(60, 65, 69), tet=12
    ... )
    (0, 1, 2)
    >>> indices_to_vl(
    ...     indices=(1, 0, 2), chord1=(60, 64, 67), chord2=(60, 65, 69), tet=12
    ... )
    (5, -4, 2)
    >>> indices_to_vl(
    ...     indices=(0, 0, 2), chord1=(60, 64, 67), chord2=(60, 65, 69), tet=12
    ... )
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
    >>> list(
    ...     get_preserve_bass_vl_atom(
    ...         src_pitches, dst_pcs, min_bass_pitch=49, max_bass_pitch=59
    ...     )
    ... )
    []
    >>> list(
    ...     get_preserve_bass_vl_atom(
    ...         src_pitches, dst_pcs, min_bass_pitch=32, max_bass_pitch=84
    ...     )
    ... )
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


def get_root_only_vl_motion(
    bass_vl_atom: BijectiveVoiceLeadingAtom, n_atoms: int
) -> VoiceLeadingMotion:
    assert n_atoms > 0
    motion = (bass_vl_atom,) + (None,) * (n_atoms - 1)
    return motion


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
    normalize_displacement_by_voice_count: bool = True,
    weights: t.Sequence[float | None] | None = None,
    choose_max: bool = True,
    temperature: float = 1.0,
) -> t.Tuple[int, int]:
    """

    ------------------------------------------------------------------------------------
    Unweighted
    ------------------------------------------------------------------------------------

    >>> options = [
    ...     ([(1, -1, 0), (-1, 1, 0)], 2),  # 2nd least displacement
    ...     ([(0, 0, 1)], 1),  # least displacement
    ...     ([(0, 0, 5)], 5),  # most displacement
    ... ]

    We first retrieve the option with the least displacement:
    >>> choose_voice_leading_from_options(options)
    (1, 0)

    We can retrieve the chosen voice-leading by providing the return value as argument to
    `pop_voice_leading_from_options`:
    >>> pop_voice_leading_from_options(options, (1, 0))
    (0, 0, 1)

    After popping that option, we next retrieve the option with the 2nd-least displacement:
    >>> choose_voice_leading_from_options(options)
    (0, 0)

    In the case of ties, we take the first item of the tie:
    >>> options = [
    ...     ([(1, -1, 0), (-1, 1, 0)], 2),
    ...     ([(0, 0, 2)], 2),
    ...     ([(2, 0, 0)], 2),
    ... ]
    >>> choose_voice_leading_from_options(options)
    (0, 0)

    ------------------------------------------------------------------------------------
    Weighted
    ------------------------------------------------------------------------------------

    >>> options = [([(2, 2, 2, 2)], 8), ([(2, 2, 0, 2)], 6)]
    >>> choose_voice_leading_from_options(options)  # Unweighted
    (1, 0)
    >>> choose_voice_leading_from_options(options, weights=(2, 0))  # Weighted
    (0, 0)

    ------------------------------------------------------------------------------------
    Choose maximum vs probabilistically
    ------------------------------------------------------------------------------------

    By default we return the option with the maximum score. We can instead
    sample from softmax(scores) by using `choose_max=False`.
    >>> options = [([(2, 2, 2, 2)], 8), ([(2, 2, 0, 2)], 6)]
    >>> [choose_voice_leading_from_options(options) for _ in range(10)]
    [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]
    >>> [
    ...     choose_voice_leading_from_options(options, choose_max=False)
    ...     for _ in range(10)
    ... ]  # doctest: +SKIP
    [(1, 0), (1, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 0)]

    If `choose_max=False` then we suply the `temperature` argument to the softmax:
    >>> [
    ...     choose_voice_leading_from_options(
    ...         options, choose_max=False, temperature=0.1
    ...     )
    ...     for _ in range(10)
    ... ]  # doctest: +SKIP
    [(0, 0), (1, 0), (1, 0), (1, 0), (1, 0), (0, 0), (1, 0), (0, 0), (1, 0), (0, 0)]

    """
    if not options:
        raise ValueError("There must be at least one option")

    scores = []
    for i, option in enumerate(options):
        option_displacement = option[1]
        if normalize_displacement_by_voice_count:
            # TODO: (Malcolm 2023-07-19) here we are assuming that all motions within
            #   the equivalent voice-leading motions have the same number of output
            #   voices. This should be enforced somehow.
            voice_count = count_vl_destinations(option[0][0])
            # TODO: (Malcolm 2023-07-19) here we are assuming that voice_count != 0.
            #   What should we do if it does?
            option_displacement /= voice_count

        # TODO: (Malcolm 2023-07-19) revise this expression?
        # We add 1 to avoid dividing by 0.
        score = 1 / (option_displacement + 1)
        if weights is None or weights[i] is None:
            scores.append(score)
        else:
            scores.append(score + weights[i])  # type:ignore

    if choose_max:
        choice_i = np.argmax(scores)
    else:
        print(scores, softmax(scores, temperature))
        choice_i = random.choices(
            range(len(scores)), weights=softmax(scores, temperature), k=1
        )[0]

    # for now we just take the first option from the EquivalentVoiceLeadingMotions
    equivalent_voice_leading_choice = 0
    return choice_i, equivalent_voice_leading_choice  # type:ignore


def apply_next_vl_from_vl_iters(
    vl_iters: t.List[t.Iterator[EquivalentVoiceLeadingMotions]],
    src_pitches: t.Sequence[Pitch],
    vl_iter_weights: t.Sequence[float] | None = None,
) -> t.Iterator[t.Tuple[t.Tuple[Pitch, ...], VoiceAssignments]]:
    # Initialize options
    options = []
    for vl_iter in vl_iters:
        try:
            options.append(next(vl_iter))
        except StopIteration:
            pass

    # Yield from options
    while options:
        choice = choose_voice_leading_from_options(options, weights=vl_iter_weights)
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
    apply_iter: t.Iterator[t.Tuple[t.Tuple[Pitch, ...], VoiceAssignments]]
) -> t.Iterator[t.Tuple[t.Tuple[Pitch, ...], VoiceAssignments]]:
    memory = set()
    for out, voice_assignments in apply_iter:
        if tuple(out) in memory:
            continue
        memory.add(tuple(out))
        yield out, voice_assignments


def _remap_from_doubled_indices(
    mapping: t.Mapping[int, t.Any], doubled_indices: t.Container[int]
):
    """
    Used by growing_cardinality_handler()

    >>> _remap_from_doubled_indices({0: "a", 1: "b", 2: "c"}, [1])
    {0: 'a', 1: 'b', 2: 'b', 3: 'c'}

    >>> _remap_from_doubled_indices({0: "a", 2: "c"}, [1])
    {0: 'a', 3: 'c'}

    >>> _remap_from_doubled_indices({0: "a", 1: "b", 2: "c"}, [3])
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


def _growing_cardinality_handler(
    vl_func: t.Callable[..., EquivalentVoiceLeadingMotions],
    src_pcs: t.Sequence[PitchClass],
    dst_pcs: t.Sequence[PitchClass],
    *args,
    normalize: bool = True,
    exclude_motions: t.Mapping[int, t.Sequence[int]] = MappingProxyType({}),
    max_motions_up: t.Mapping[int, int] = MappingProxyType({}),
    max_motions_down: t.Mapping[int, int] = MappingProxyType({}),
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

    previously_doubled_ps: t.Set[
        t.Tuple[int, ...]
    ] = set()  # Only used if normalize==True

    for doubled_indices in combinations(range(card1), excess):
        if normalize:
            doubled_ps = tuple(src_pcs[i] for i in doubled_indices)
            if doubled_ps in previously_doubled_ps:
                continue
            previously_doubled_ps.add(doubled_ps)
        temp_chord = []
        for i, p in enumerate(src_pcs):
            temp_chord.extend([p, p] if i in doubled_indices else [p])
        vls, total_displacement = vl_func(
            temp_chord,
            dst_pcs,
            *args,
            normalize=normalize,
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
    mapping: t.Mapping[int, t.Any], omitted_indices: t.Sequence[int]
):
    """Used by shrinking_cardinality_handler()

    >>> _remap_from_omitted_indices({0: "a", 1: "b", 2: "c"}, [1])
    {0: 'a', 1: 'c'}

    >>> _remap_from_omitted_indices({0: "a", 2: "c"}, [1])
    {0: 'a', 1: 'c'}

    >>> _remap_from_omitted_indices({0: "a", 1: "b", 2: "c"}, [3])
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


def _shrinking_cardinality_handler(
    vl_func: t.Callable[..., EquivalentVoiceLeadingMotions],
    src_pcs: t.Sequence[PitchClass],
    dst_pcs: t.Sequence[PitchClass],
    *args,
    normalize: bool = True,
    exclude_motions: t.Mapping[int, t.Sequence[int]] = MappingProxyType({}),
    max_motions_up: t.Mapping[int, int] = MappingProxyType({}),
    max_motions_down: t.Mapping[int, int] = MappingProxyType({}),
    **kwargs,
) -> t.Tuple[t.List[VoiceLeadingMotion], int]:
    card1, card2 = len(src_pcs), len(dst_pcs)
    excess = card1 - card2
    if excess > card2:
        raise CardinalityDiffersTooMuch(
            f"chord1 has {card1} items; chord2 has {card2} items; "
            "when decreasing cardinality, the number of items can differ by "
            "at most the number of items in chord2"
        )
    least_displacement = 2**31
    best_vls, best_indices = [], []

    previously_omitted_ps: t.Set[
        t.Tuple[int, ...]
    ] = set()  # Only used if normalize==True

    for omitted_indices in combinations(range(card1), excess):
        if normalize:
            omitted_ps = tuple(src_pcs[i] for i in omitted_indices)
            if omitted_ps in previously_omitted_ps:
                continue
            previously_omitted_ps.add(omitted_ps)
        temp_chord = [p for (i, p) in enumerate(src_pcs) if i not in omitted_indices]
        vls, total_displacement = vl_func(
            temp_chord,
            dst_pcs,
            *args,
            normalize=normalize,
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
    vl_func: t.Callable[..., EquivalentVoiceLeadingMotions],
    src_pcs: t.Sequence[PitchClass],
    dst_pcs: t.Sequence[PitchClass],
    *args,
    **kwargs,
) -> t.Tuple[t.List[VoiceLeadingMotion], int]:
    if len(src_pcs) > len(dst_pcs):
        return _shrinking_cardinality_handler(
            vl_func, src_pcs, dst_pcs, *args, **kwargs
        )
    return _growing_cardinality_handler(vl_func, src_pcs, dst_pcs, *args, **kwargs)


def efficient_pc_voice_leading(
    src_pcs: t.Sequence[PitchClass],
    dst_pcs: t.Sequence[PitchClass],
    *,
    tet: int = 12,
    displacement_more_than: t.Optional[int] = None,
    exclude_motions: t.Mapping[int, t.Sequence[int]] = MappingProxyType({}),
    max_motions_up: t.Mapping[int, int] = MappingProxyType({}),
    max_motions_down: t.Mapping[int, int] = MappingProxyType({}),
    allow_different_cards: bool = False,
    normalize: bool = True,
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
    >>> efficient_pc_voice_leading(src_pcs, dst_pcs, normalize=False)
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
        normalize: if True, then if chord1 contains any unisons, the voice-
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

    def _voice_leading_sub(in_indices, out_indices, current_displacement):
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
        nonlocal best_displacement, best_vl_indices
        if not in_indices:
            if current_displacement > displacement_more_than:
                if best_displacement > current_displacement:
                    best_displacement = current_displacement
                    best_vl_indices = [out_indices]
                elif best_displacement == current_displacement:
                    best_vl_indices.append(out_indices)
        else:
            src_pcs_i = len(out_indices)
            src_pc = src_pcs[src_pcs_i]
            unique = not any((src_pc == pc) for pc in src_pcs[:src_pcs_i])
            for i, dst_pcs_i in enumerate(in_indices):
                motion = dst_pcs[dst_pcs_i] - src_pcs[src_pcs_i]
                if motion > halftet:
                    motion -= tet
                elif motion < -halftet:
                    motion += tet
                if src_pcs_i in max_motions_up:
                    if motion > max_motions_up[src_pcs_i]:
                        continue
                if src_pcs_i in max_motions_down:
                    if motion < max_motions_down[src_pcs_i]:
                        continue
                if src_pcs_i in exclude_motions:
                    if motion in exclude_motions[src_pcs_i]:
                        #      MAYBE expand to include combinations of
                        #       multiple voice leading motions
                        continue
                this_displacement = current_displacement + abs(motion)
                if this_displacement > best_displacement:
                    continue
                _voice_leading_sub(
                    in_indices[:i] + in_indices[i + 1 :],
                    out_indices + [dst_pcs_i],
                    this_displacement,
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
                efficient_pc_voice_leading,
                src_pcs,
                dst_pcs,
                tet=tet,
                displacement_more_than=displacement_more_than,
                exclude_motions=exclude_motions,
                max_motions_up=max_motions_up,
                max_motions_down=max_motions_down,
                normalize=normalize,
            )
        raise ValueError(f"{src_pcs} and {dst_pcs} have different lengths.")

    best_displacement = starting_displacement = 2**31
    best_vl_indices = []
    halftet = tet // 2

    _voice_leading_sub(list(range(card)), [], 0)

    # If best_sum hasn't changed, then we haven't found any
    # voice-leadings.
    if best_displacement == starting_displacement:
        raise NoMoreVoiceLeadingsError

    # When there are unisons in chord2, there can be duplicate voice-leadings.
    # There is probably a more efficient way of avoiding generating these in
    # the first place, but for now, we get rid of them by casting to a set.
    voice_leading_intervals = list(
        {indices_to_vl(indices, src_pcs, dst_pcs, tet) for indices in best_vl_indices}
    )

    if len(voice_leading_intervals) > 1:
        voice_leading_intervals.sort(key=np.var)

    if normalize:
        voice_leading_intervals = normalize_voice_leading_motions_and_remove_duplicates(
            src_pcs, voice_leading_intervals
        )

    assert len(set(count_vl_destinations(motion) for motion in voice_leading_intervals))
    return voice_leading_intervals, best_displacement  # type:ignore


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


def normalize_voice_leading_motions_and_remove_duplicates(
    src_pcs_or_pitches: t.Sequence[PitchOrPitchClass],
    voice_leading_motions: t.Sequence[BijectiveVoiceLeadingMotion],
) -> t.List[VoiceLeadingMotion]:
    """
    By "normalizing" a voice-leading motion, we mean the following: if two motions
    proceed from the same pitch or pitch-class, they should be in nondecreasing order.
    >>> normalize_voice_leading_motions_and_remove_duplicates([0, 0], [(2, -2)])
    [(-2, 2)]
    >>> normalize_voice_leading_motions_and_remove_duplicates(
    ...     [0, 4, 7, 0], [(2, -2, 0, -1)]
    ... )
    [(-1, -2, 0, 2)]

    Normalizing in this way also allows us to remove duplicate voice-leadings that
    differ only in the order the motions are specified.
    >>> normalize_voice_leading_motions_and_remove_duplicates(
    ...     [0, 0], [(2, -2), (2, -2), (-2, 2)]
    ... )
    [(-2, 2)]

    If there are no duplicate pitches/pitch-classes in `src_pcs_or_pitches`, this
    function has no effect.
    >>> normalize_voice_leading_motions_and_remove_duplicates(
    ...     [0, 4, 7], [(2, 3, 1), (1, 3, 2)]
    ... )
    [(2, 3, 1), (1, 3, 2)]

    Note: this function only works with *bijective* voice-leadings.
    >>> normalize_voice_leading_motions_and_remove_duplicates(
    ...     [0, 0, 0], [((-2, 2), 5, None)]
    ... )
    Traceback (most recent call last):
    ValueError: Voice-leading must be bijective
    """
    p_indices: t.DefaultDict[PitchOrPitchClass, t.List[int]] = defaultdict(list)

    if not all(
        isinstance(atom, int) for motion in voice_leading_motions for atom in motion
    ):
        raise ValueError("Voice-leading must be bijective")

    for i, p in enumerate(src_pcs_or_pitches):
        p_indices[p].append(i)

    unison_indices = [indices for indices in p_indices.values() if len(indices) > 1]

    if not unison_indices:
        return list(voice_leading_motions)

    unique_accumulator = set()
    for motion in voice_leading_motions:
        motion_copy = list(motion)
        for indices in unison_indices:
            unison_atoms = sorted(motion[i] for i in indices)
            for idx, unison_atom in zip(indices, unison_atoms):
                motion_copy[idx] = unison_atom
        unique_accumulator.add(tuple(motion_copy))

    return list(unique_accumulator)


def efficient_pitch_voice_leading(
    src_pitches: t.Sequence[Pitch],
    dst_pcs: t.Sequence[PitchClass],
    *,
    tet: int = 12,
    displacement_more_than: int | None = None,
    exclude_motions: t.Mapping[int, t.Sequence[int]] = MappingProxyType({}),
    max_motions_up: t.Mapping[int, int] = MappingProxyType({}),
    max_motions_down: t.Mapping[int, int] = MappingProxyType({}),
    allow_different_cards: bool = False,
    normalize: bool = True,
) -> EquivalentVoiceLeadingMotions:
    """
    >>> src_pcs = [55, 62, 72]
    >>> dst_pcs = [4, 8, 11]
    >>> vl_motions, total_displacement = efficient_pitch_voice_leading(src_pcs, dst_pcs)
    >>> vl_motions
    [(1, 2, -1)]
    >>> total_displacement
    4
    """

    if displacement_more_than is None:
        displacement_more_than = -1

    def _voice_leading_sub(in_indices, out_motions, current_displacement):
        nonlocal best_displacement, best_motions

        if not in_indices:
            # base condition
            if current_displacement > displacement_more_than:
                if best_displacement > current_displacement:
                    best_displacement = current_displacement
                    best_motions = [tuple(out_motions)]
                elif best_displacement == current_displacement:
                    best_motions.append(tuple(out_motions))
            return

        src_pitch_i = len(out_motions)
        src_pitch = src_pitches[src_pitch_i]

        for i, dst_pcs_i in enumerate(in_indices):
            dst_pc = dst_pcs[dst_pcs_i]
            for dst_pitch in closest_pc_iter(
                pitch=src_pitch, pc=dst_pc, prefer_down=None, max_results=3, tet=tet
            ):
                motion = dst_pitch - src_pitch
                if (
                    (
                        src_pitch_i in max_motions_up
                        and motion > max_motions_up[src_pitch_i]
                    )
                    or (
                        src_pitch_i in max_motions_down
                        and motion < max_motions_down[src_pitch_i]
                    )
                    or (
                        src_pitch_i in exclude_motions
                        and motion in exclude_motions[src_pitch_i]
                    )
                ):
                    continue
                this_displacement = current_displacement + abs(motion)
                if this_displacement > best_displacement:
                    continue
                _voice_leading_sub(
                    in_indices[:i] + in_indices[i + 1 :],
                    out_motions + [motion],
                    this_displacement,
                )

    best_displacement = starting_displacement = 2**31
    best_motions = []

    src_card = len(src_pitches)
    if src_card != len(dst_pcs):
        if allow_different_cards:
            return different_cardinality_handler(
                efficient_pitch_voice_leading,
                src_pitches,
                dst_pcs,
                tet=tet,
                displacement_more_than=displacement_more_than,
                exclude_motions=exclude_motions,
                max_motions_up=max_motions_up,
                max_motions_down=max_motions_down,
                normalize=normalize,
            )
        raise ValueError(f"{src_pitches} and {dst_pcs} have different lengths.")

    _voice_leading_sub(list(range(src_card)), [], 0)

    # If best_displacement hasn't changed, then we haven't found any
    # voice-leadings.
    if best_displacement == starting_displacement:
        raise NoMoreVoiceLeadingsError

    # Get rid of duplicates
    best_motions = list(set(best_motions))

    # TODO: (Malcolm) Am I sure I want to sort like this?
    if len(best_motions) > 1:
        best_motions.sort(key=np.var)

    if normalize:
        best_motions = normalize_voice_leading_motions_and_remove_duplicates(
            src_pitches, best_motions
        )
    assert len(set(count_vl_destinations(motion) for motion in best_motions))
    return best_motions, best_displacement


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
) -> t.Tuple[Pitch, ...]:
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
    >>> voice_lead_pitches([60, 64, 67], [7, 11, 2], preserve_bass=True, min_pitch=60)
    (55, 62, 71)

    However, it will not exceed `min_bass_pitch`.
    >>> voice_lead_pitches(
    ...     [60, 64, 67], [7, 11, 2], preserve_bass=True, min_bass_pitch=60
    ... )
    (67, 71, 74)

    >>> voice_lead_pitches(
    ...     [60, 64, 67], [7, 11, 2, 5], preserve_bass=True, min_pitch=60
    ... )
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
        try:
            return next(iterator)[0]
        except StopIteration:
            raise NoMoreVoiceLeadingsError()

    # We set a maximum number of times to run the iterator when making a random choice.
    # This speeds up calling this function with `return_first=False` quite a bit.
    # `max_choies` could be made a function parameter.
    max_choices = 10

    return random.choice([v for i, v in zip(range(max_choices), iterator)])[0]


def efficient_pitch_voice_leading_iter(
    src_pitches: t.Sequence[Pitch],
    *args,
    min_pitch: t.Optional[int] = None,
    max_pitch: t.Optional[int] = None,
    **kwargs,
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
            return {}
        return {i: max_pitch - pitch for i, pitch in enumerate(pitches)}

    def _get_max_motions_down(pitches):
        if min_pitch is None:
            return {}
        return {i: min_pitch - pitch for i, pitch in enumerate(pitches)}

    displacement_more_than = kwargs.pop("displacement_more_than", -1)

    while True:
        try:
            vl_motions, total_displacement = efficient_pitch_voice_leading(
                src_pitches,
                *args,
                displacement_more_than=displacement_more_than,
                max_motions_up=_get_max_motions_up(src_pitches),
                max_motions_down=_get_max_motions_down(src_pitches),
                **kwargs,
            )
            yield vl_motions, total_displacement
            displacement_more_than = total_displacement
        except NoMoreVoiceLeadingsError:
            return


def preserve_and_split_root_vl_iters(
    src_pitches: t.Sequence[Pitch],
    dst_pcs: t.Sequence[PitchClass],
    avoid_bass_crossing: bool = True,
    min_pitch: t.Optional[int] = None,
    max_pitch: t.Optional[int] = None,
    min_bass_pitch: t.Optional[int] = None,
    max_bass_pitch: t.Optional[int] = None,
    allow_different_cards: bool = True,
    exclude_motions: t.Mapping[int, t.Sequence[int]] = MappingProxyType({}),
    **vl_kwargs,
) -> t.List[t.Iterator[EquivalentVoiceLeadingMotions]]:
    """
    >>> src_pitches = [60, 64, 67]
    >>> dst_pcs = [2, 5, 7, 11]
    >>> vl_iters = preserve_and_split_root_vl_iters(src_pitches, dst_pcs)
    >>> [next(vl_iter) for vl_iter in vl_iters]  # doctest: +NORMALIZE_WHITESPACE
    [([((2, 5), 3, 4), ((2, 7), 1, 4), ((2, 5), 7, 0), ((2, 11), 1, 0)], 14),
     ([((-10, -1), 1, 0)], 12)]
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
            allow_different_cards=allow_different_cards,
            exclude_motions=exclude_motions,
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
    allow_different_cards: bool = True,
    exclude_motions: t.Mapping[int, t.Sequence[int]] = MappingProxyType({}),
    **vl_kwargs,
) -> t.List[t.Iterator[EquivalentVoiceLeadingMotions]]:
    """Note: this will not produce voice-leadings where the bass-note of the second
    chord is doubled.

    >>> src_pitches = [60, 64, 67]
    >>> dst_pcs = [5, 8, 0]
    >>> vl_iters = preserve_bass_vl_iters(src_pitches, dst_pcs)
    >>> [next(vl_iter) for vl_iter in vl_iters]
    [([(5, 4, 5), (5, 8, 1)], 14), ([(-7, -4, 1)], 12)]

    >>> src_pitches = [60, 64, 67]
    >>> dst_pcs = [2, 5, 7, 11]
    >>> vl_iters = preserve_bass_vl_iters(src_pitches, dst_pcs)
    >>> [next(vl_iter) for vl_iter in vl_iters]
    [([(2, 1, (0, 4))], 7), ([(-10, 1, (0, 4))], 15)]

    If dst_pcs contains only 1 item, we assume it is the bass:
    >>> src_pitches = [60, 64, 67]
    >>> dst_pcs = [2]
    >>> vl_iters = preserve_bass_vl_iters(src_pitches, dst_pcs)
    >>> [next(vl_iter) for vl_iter in vl_iters]
    [([(2, None, None)], 2), ([(-10, None, None)], 10)]
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

    exclude_motions = _remap_from_omitted_indices(exclude_motions, omitted_indices=[0])

    for root_motion in root_motions:
        if len(dst_pcs) == 1:
            vl_motion = get_root_only_vl_motion(root_motion, len(src_pitches))
            vl_iters.append(
                iter([([vl_motion], get_vl_atom_displacement(root_motion))])
            )
            continue
        if avoid_bass_crossing:
            dst_bass = src_pitches[0] + root_motion
            this_min_pitch = dst_bass if min_pitch is None else max(min_pitch, dst_bass)
        sub_iter = efficient_pitch_voice_leading_iter(
            src_pitches[1:],
            dst_pcs[1:],
            min_pitch=this_min_pitch,
            max_pitch=max_pitch,
            allow_different_cards=allow_different_cards,
            exclude_motions=exclude_motions,
            **vl_kwargs,
        )
        vl_iter = vl_iter_wrapper(sub_iter, insertions={0: root_motion})
        vl_iters.append(vl_iter)
    return vl_iters


def dont_preserve_bass_vl_iters(
    src_pitches: t.Sequence[Pitch],
    dst_pcs: t.Sequence[PitchClass],
    avoid_bass_crossing: bool = True,
    allow_different_cards: bool = True,
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
    return [
        efficient_pitch_voice_leading_iter(
            src_pitches,
            dst_pcs,
            allow_different_cards=allow_different_cards,
            **vl_kwargs,
        )
    ]


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
    exclude_motions: t.Mapping[int, t.Sequence[int]] = MappingProxyType({}),
) -> t.List[t.Iterator[EquivalentVoiceLeadingMotions]]:
    """
    >>> chord1_pitches = (47, 55, 62, 67)
    >>> chord2_pcs = (0, 0, 4, 7)
    >>> vl_iter = get_voice_lead_pitches_iters(
    ...     chord1_pitches, chord2_pcs, preserve_bass=True
    ... )
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
        "exclude_motions": exclude_motions,
    }

    if preserve_bass and not avoid_bass_crossing:
        LOGGER.warning(
            "`preserve_bass=True` and `avoid_bass_crossing=False`: bass is not "
            "guaranteed to be preserved"
        )
    if not chord2_pcs and not preserve_bass:
        return []
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
    chord2_pcs_options: t.Iterable[t.Sequence[PitchClass]],
    chord2_option_weights: t.Sequence[float] | None = None,
    ignore_voice_assignments: bool = True,
    **get_voice_lead_pitches_iters_kwargs,
) -> t.Iterator[t.Tuple[t.Tuple[Pitch, ...], VoiceAssignments]]:
    """
    >>> chord1_pitches = [60, 64, 67]
    >>> chord2_pcs_options = [[2, 5, 7, 11], [5, 7, 11], [2, 5, 7]]
    >>> vl_iter = voice_lead_pitches_multiple_options_iter(
    ...     chord1_pitches, chord2_pcs_options
    ... )
    >>> next(vl_iter)[0], next(vl_iter)[0], next(vl_iter)[0]
    ((59, 65, 67), (59, 62, 65, 67), (62, 65, 67))
    """

    # if not any(chord2_pcs_options):
    #     yield (), ()
    #     return

    vl_iters = []
    vl_iters_option_weights = []

    if chord2_option_weights is None:
        chord2_option_weights = [None] * len(chord2_pcs_options)  # type:ignore

    for chord2_pcs, option_weights in zip(
        chord2_pcs_options, chord2_option_weights  # type:ignore
    ):
        these_vl_iters = get_voice_lead_pitches_iters(
            chord1_pitches,
            chord2_pcs,
            **get_voice_lead_pitches_iters_kwargs,
        )
        vl_iters += these_vl_iters
        vl_iters_option_weights += [option_weights] * len(these_vl_iters)

    apply_iter = apply_next_vl_from_vl_iters(
        vl_iters, chord1_pitches, vl_iters_option_weights
    )

    if ignore_voice_assignments:
        apply_iter = ignore_voice_assignments_wrapper(apply_iter)

    yield from apply_iter


def voice_lead_pitches_iter(
    chord1_pitches: t.Sequence[Pitch],
    chord2_pcs: t.Sequence[PitchClass],
    ignore_voice_assignments: bool = True,
    **get_voice_lead_pitches_iters_kwargs,
) -> t.Iterator[t.Tuple[t.Tuple[Pitch, ...], VoiceAssignments]]:
    """
    >>> chord1_pitches = [60, 64, 67]
    >>> chord2_pcs = [2, 5, 7, 11]
    >>> vl_iter = voice_lead_pitches_iter(chord1_pitches, chord2_pcs)
    >>> next(vl_iter)[0], next(vl_iter)[0], next(vl_iter)[0]
    ((59, 62, 65, 67), (62, 65, 67, 71), (59, 65, 67, 74))

    If `ignore_voice_assignments` is True, then voice-leadings that result in exactly
    the same pitches (differing only in which voice is led where) are considered
    equivalent and not returned separately.

    >>> vl_iter = voice_lead_pitches_iter(
    ...     chord1_pitches, chord2_pcs, ignore_voice_assignments=False
    ... )
    >>> next(vl_iter), next(vl_iter)
    (((59, 62, 65, 67), (0, 0, 1, 2)), ((59, 62, 65, 67), (0, 1, 1, 2)))

    >>> chord1_pitches = (47, 55, 62, 67)
    >>> chord2_pcs = (0, 0, 4, 7)
    >>> vl_iter = voice_lead_pitches_iter(
    ...     chord1_pitches, chord2_pcs, preserve_bass=True
    ... )
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
