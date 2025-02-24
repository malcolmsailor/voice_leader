import itertools
import warnings

import pytest

from voice_leader import (
    NoMoreVoiceLeadingsError,
    apply_vl,
    efficient_pc_voice_leading,
    voice_lead_pitches,
)


def test_remove_synonyms():
    tests = [
        ((2, 5, 7), (0, 0, 4)),
        ((0, 0, 4), (2, 5, 7)),
        ((0, 0, 0), (2, 5, 7)),
        ((0, 0, 0, 0), (2, 5)),
        ((0, 0, 4), (2, 5)),
        ((2, 5, 7), (0, 0, 0, 0)),
        ((2, 5), (0, 0, 0, 0)),
    ]
    for chord1, chord2 in tests:
        motions, _ = efficient_pc_voice_leading(
            chord1, chord2, allow_different_cards=True
        )
        if len(motions) == 1:
            continue
        # TODO I think there may be cases where there will be more than one
        #   possibility with syonyms, find and test those
        raise NotImplementedError
        # unisons = tuple(p for p in set(chord1) if chord1.count(p) > 1)
        # unison_indices = tuple(
        #     tuple(i for i, p1 in enumerate(chord1) if p1 == p2)
        #     for p2 in unisons
        # )
        # for indices in unison_indices:
        #     unison = []
        #     other = []
        #     for i in len


@pytest.mark.parametrize(
    "src_pitches, dst_pcs",
    (
        ((48, 64, 67, 70), (5, 9, 0)),
        ((48, 48, 52, 55), (2, 5, 7, 11)),
        ((48, 52, 55), (2, 5, 7, 11)),
        ((48, 52, 55, 58), (5, 7, 11)),
        # After making a fix, the following test fails when preserving root
        # because the first chord has 3 remaining pitches and the second one
        # only one pc, which leads to an "excess" error
        # ((48, 48, 52, 58), (7, 11)),
    ),
)
@pytest.mark.parametrize(
    "preserve_bass, avoid_bass_crossing", ((True, True), (False, True), (False, False))
)
@pytest.mark.parametrize(
    "min_pitch, max_pitch", ((False, False), (True, False), (False, True))
)
def test_voice_lead_pitches(
    src_pitches, dst_pcs, preserve_bass, avoid_bass_crossing, min_pitch, max_pitch
):
    min_pitch = min(src_pitches) if min_pitch else None
    max_pitch = max(src_pitches) + 2 if max_pitch else None
    min_bass_pitch = min_pitch - 12 if min_pitch is not None else None
    for start_i in range(len(dst_pcs)):
        dst_pcs_rotation = dst_pcs[start_i:] + dst_pcs[:start_i]
        out = voice_lead_pitches(
            src_pitches,
            dst_pcs_rotation,
            preserve_bass=preserve_bass,
            return_first=False,
            avoid_bass_crossing=avoid_bass_crossing,
            min_pitch=min_pitch,
            max_pitch=max_pitch,
            min_bass_pitch=min_bass_pitch,
        )
        if preserve_bass:
            assert out[0] % 12 == dst_pcs_rotation[0]
        if avoid_bass_crossing:
            assert out[0] <= out[1]
        assert len(out) == len(dst_pcs_rotation)
        assert {pc % 12 for pc in out} == set(dst_pcs_rotation)
        if min_pitch is not None and len(out) > 1:
            assert min(out[1:]) >= min_pitch
        if min_bass_pitch is not None:
            assert min(out) >= min_bass_pitch
        if max_pitch is not None:
            assert max(out) <= max_pitch


TRIADS = [(0, 3, 7), (0, 4, 7), (0, 1, 2), (0, 4, 8), (0, 2, 6)]
DOUBLED_TRIADS = [(0, 0, 3, 7), (0, 0, 4, 7)]
TETRADS = [(0, 2, 4, 6), (0, 1, 2, 3), (0, 3, 6, 10), (0, 4, 8, 11)]
SEXTADS = [(0, 2, 4, 6, 8, 10), (0, 3, 4, 7, 8, 11)]

FEW_CHORD_PAIRS = list(itertools.product(TRIADS[:2] + TETRADS[-1:], repeat=2))
MANY_CHORD_PAIRS = list(
    itertools.product(TRIADS + DOUBLED_TRIADS + TETRADS, repeat=2)
) + list(itertools.product(SEXTADS, repeat=2))


@pytest.mark.parametrize(
    "chord_pairs, exclude_motions, max_motions_up, max_motions_down",
    (
        (FEW_CHORD_PAIRS, {}, {}, {}),
        (MANY_CHORD_PAIRS, {1: [0]}, {2: 1}, {3: -1}),
    ),
)
def test_efficient_voice_leading(
    chord_pairs, exclude_motions, max_motions_up, max_motions_down
):
    for c1, c2 in chord_pairs:
        for i in range(12):
            c3 = [(p + i) % 12 for p in c2]
            displacement = -1
            while True:
                try:
                    out1, displacement1 = efficient_pc_voice_leading(
                        c1,
                        c3,
                        tet=12,
                        displacement_more_than=displacement,
                        exclude_motions=exclude_motions,
                        max_motions_up=max_motions_up,
                        max_motions_down=max_motions_down,
                        allow_different_cards=True,
                        normalize=False,
                    )
                except NoMoreVoiceLeadingsError:
                    break
                for vl in out1:
                    assert set([p % 12 for p in apply_vl(vl, c1)[0]]) == set(c3)
                    for note_i, intervals in exclude_motions.items():
                        motions = vl[note_i]
                        if motions is None:
                            continue
                        if isinstance(motions, int):
                            motions = [motions]
                        for motion in motions:
                            assert motion not in intervals
                    for note_i, interval in max_motions_up.items():
                        try:
                            motions = vl[note_i]
                        except IndexError:
                            continue
                        if motions is None:
                            continue
                        if isinstance(motions, int):
                            motions = [motions]
                        for motion in motions:
                            assert motion <= interval
                    for note_i, interval in max_motions_down.items():
                        try:
                            motions = vl[note_i]
                        except IndexError:
                            continue
                        if motions is None:
                            continue
                        if isinstance(motions, int):
                            motions = [motions]
                        for motion in motions:
                            assert motion >= interval
                assert displacement1 > displacement
                displacement = displacement1


@pytest.mark.parametrize(
    "chord1, chord2",
    [
        ((0, 3, 7), (0, 4, 7, 11)),
        ((0, 3, 7), (0, 2, 4, 7, 9)),
    ],
)
@pytest.mark.parametrize("forward", (True, False))
def test_different_cardinality_handler(chord1, chord2, forward):
    if forward:
        c1, c2 = chord1, chord2
    else:
        c1, c2 = chord2, chord1
    for i in range(12):
        c3 = [(p + i) % 12 for p in c2]
        vls, _ = efficient_pc_voice_leading(c1, c3, allow_different_cards=True)
        for vl in vls:
            c4 = [p % 12 for p in apply_vl(vl, c1)[0]]
            assert set(p % 12 for p in c4) == set(c3)
