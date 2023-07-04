import itertools
import warnings

from voice_leader import (
    NoMoreVoiceLeadingsError,
    apply_vl,
    efficient_voice_leading,
    voice_lead_pitches,
)

# def test_apply_vl():
#     apply_vl()


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
        motions, _ = efficient_voice_leading(chord1, chord2, allow_different_cards=True)
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


def test_voice_lead_pitches():
    tests = [
        ((48, 64, 67, 70), (5, 9, 0)),
        ((48, 48, 52, 55), (2, 5, 7, 11)),
        ((48, 52, 55), (2, 5, 7, 11)),
        ((48, 52, 55, 58), (5, 7, 11)),
        # After making a fix, the following test fails when preserving root
        # because the first chord has 3 remaining pitches and the second one
        # only one pc, which leads to an "excess" error
        # ((48, 48, 52, 58), (7, 11)),
    ]
    total_tests = 0
    total_failures = 0
    for ps, ref_pcs in tests:
        for preserve_root, avoid_bass_crossing in itertools.product(
            [True, False], repeat=2
        ):
            for min_pitch, max_pitch in (
                (False, False),
                (True, False),
                (False, True),
            ):
                min_pitch = min(ps) if min_pitch else None
                max_pitch = max(ps) if max_pitch else None
            for start_i in range(len(ref_pcs)):
                total_tests += 1
                pcs = ref_pcs[start_i:] + ref_pcs[:start_i]
                try:
                    out = voice_lead_pitches(
                        ps,
                        pcs,
                        preserve_root=preserve_root,
                        return_first=False,
                        avoid_bass_crossing=avoid_bass_crossing,
                        min_pitch=min_pitch,
                        max_pitch=max_pitch,
                    )
                except NoMoreVoiceLeadingsError:
                    total_failures += 1
                    warnings.warn(
                        f"couldn't voice-lead {ps} to pcs {pcs} with "
                        f"preserve_root={preserve_root}, avoid_bass_crossing="
                        f"{avoid_bass_crossing}, min_pitch={min_pitch}, and "
                        f"max_pitch={max_pitch}"
                    )
                    continue
                if preserve_root:
                    assert out[0] % 12 == pcs[0]
                if avoid_bass_crossing:
                    assert out[0] <= out[1]
                assert len(out) == len(pcs)
                assert {pc % 12 for pc in out} == set(pcs)
                if min_pitch is not None:
                    assert min(out) >= min_pitch
                if max_pitch is not None:
                    assert max(out) <= max_pitch

    if total_failures:
        warnings.warn(f"{total_failures}/{total_tests} failed")


def test_efficient_voice_leading():
    triads = [(0, 3, 7), (0, 4, 7), (0, 1, 2), (0, 4, 8), (0, 2, 6)]
    doubled_triads = [(0, 0, 3, 7), (0, 0, 4, 7)]
    tetrads = [(0, 2, 4, 6), (0, 1, 2, 3), (0, 3, 6, 10), (0, 4, 8, 11)]
    sextads = [(0, 2, 4, 6, 8, 10), (0, 3, 4, 7, 8, 11)]
    # The following slow the test *way* down
    # heptads = [(0, 2, 4, 5, 7, 9, 11), (0, 2, 3, 5, 7, 9, 11)]

    chord_pairs = (
        list(itertools.product(triads + doubled_triads + tetrads, repeat=2))
        # + list(itertools.product(heptads, repeat=2))
        + list(itertools.product(sextads, repeat=2))
    )

    exclude_motions = {1: [0]}
    max_motions_up = {2: 1}
    max_motions_down = {3: -1}
    for c1, c2 in chord_pairs:
        for i in range(12):
            c3 = [(p + i) % 12 for p in c2]
            displacement = -1
            while True:
                try:
                    out1, displacement1 = efficient_voice_leading(
                        c1,
                        c3,
                        tet=12,
                        displacement_more_than=displacement,
                        exclude_motions=exclude_motions,
                        max_motions_up=max_motions_up,
                        max_motions_down=max_motions_down,
                        allow_different_cards=True,
                        sort=False,
                    )
                except NoMoreVoiceLeadingsError:
                    break
                for vl in out1:
                    # c4 = [p % 12 for p in apply_vl(vl, c1)]
                    assert set([p % 12 for p in apply_vl(vl, c1)]) == set(c3)
                    # assert set((p + i) % 12 for (p, i) in zip(c1, vl)) == set(
                    #     c3
                    # )
                    # for note_i, intervals in exclude_motions.items():
                    #     assert vl[note_i] not in intervals
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
                        # try:
                        #     assert vl[note_i] <= interval
                        # except IndexError:
                        #     pass
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
                    # for note_i, interval in max_motions_down.items():
                    #     try:
                    #         assert vl[note_i] >= interval
                    #     except IndexError:
                    #         pass
                assert displacement1 > displacement
                displacement = displacement1


def test_different_cardinality_handler():
    chord_pairs = [
        ((0, 3, 7), (0, 4, 7, 11)),
        ((0, 3, 7), (0, 2, 4, 7, 9)),
    ]
    for chord1, chord2 in chord_pairs:
        for c1, c2 in ((chord1, chord2), (chord2, chord1)):
            for i in range(12):
                c3 = [(p + i) % 12 for p in c2]
                vls, _ = efficient_voice_leading(c1, c3, allow_different_cards=True)
                for vl in vls:
                    c4 = [p % 12 for p in apply_vl(vl, c1)]
                    assert set(p % 12 for p in c4) == set(c3)


if __name__ == "__main__":
    test_efficient_voice_leading()
