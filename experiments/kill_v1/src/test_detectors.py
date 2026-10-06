"""Tests for detectors.py. No GPU, no network, no dataset download.

Verdict parsing and option shuffling are both named in PREREG.md section 12 as
things that invalidate the run if they go wrong, so they are tested here rather
than discovered on Colab.

    python test_detectors.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))
import detectors as D  # noqa: E402


def _fake_items(n=40):
    cats = ["STEM", "humanities", "social sciences", "other"]
    return [
        {
            "subject": f"subj_{i % 7}",
            "supercategory": cats[i % 4],
            "question": f"Question number {i}?",
            "choices": [f"opt{i}a", f"opt{i}b", f"opt{i}c", f"opt{i}d"],
            "answer": i % 4,
            "is_ok": (i % 5) != 0,
        }
        for i in range(n)
    ]


def test_permutation_is_fixed_and_a_real_permutation():
    for idx in (0, 1, 17, 5699):
        a, b = D.option_permutation(idx), D.option_permutation(idx)
        assert np.array_equal(a, b), "permutation must be reproducible"
        assert sorted(a.tolist()) == [0, 1, 2, 3], a
    many = [tuple(D.option_permutation(i)) for i in range(200)]
    assert len(set(many)) > 10, "permutations must actually vary across items"
    return f"per-item permutations are stable and varied ({len(set(many))} distinct over 200 items)"


def test_key_moves_with_its_text():
    """The whole point of shuffling: the claimed key follows its option text."""
    for idx in range(100):
        opts = ["alpha", "beta", "gamma", "delta"]
        key = idx % 4
        perm = D.option_permutation(idx)
        new_opts, new_key = D.apply_permutation(opts, key, perm)
        assert new_opts[new_key] == opts[key], (idx, opts[key], new_opts[new_key])
        assert sorted(new_opts) == sorted(opts)
    return "after shuffling, the key still names the same option text"


def test_scored_wrong_keys():
    for idx in range(200):
        for answer in range(4):
            picked = D.scored_wrong_keys(idx, answer)
            assert len(picked) == 2 and len(set(picked)) == 2, picked
            assert answer not in picked, (answer, picked)
            assert all(0 <= p <= 3 for p in picked)
            assert picked == D.scored_wrong_keys(idx, answer), "must be reproducible"
    # no option position should be systematically favoured
    counts = np.zeros(4)
    for idx in range(4000):
        for p in D.scored_wrong_keys(idx, idx % 4):
            counts[p] += 1
    spread = counts.max() / counts.min()
    assert spread < 1.15, counts
    return f"2-of-3 wrong keys are reproducible and position-balanced (spread {spread:.3f})"


def test_work_list_sizes():
    items = _fake_items(40)
    n_ok = sum(i["is_ok"] for i in items)
    sample = set(range(10))
    rows = D.build_work_list(items, sample)
    kinds = {k: sum(r.kind == k for r in rows) for k in (D.REDUX, D.INJECT, D.ORIGINAL)}
    assert kinds[D.REDUX] == len(items), kinds
    assert kinds[D.INJECT] == 2 * n_ok, kinds
    assert kinds[D.ORIGINAL] == len(sample), kinds
    assert len({r.row_id for r in rows}) == len(rows), "row ids must be unique"
    return f"work list sizes are right ({kinds[D.REDUX]} redux, {kinds[D.INJECT]} inject, {kinds[D.ORIGINAL]} original)"


def test_inject_rows_never_use_the_real_key():
    items = _fake_items(60)
    for row in D.build_inject_rows(items):
        assert row.claimed_key_original != items[row.item_index]["answer"], row
        assert items[row.item_index]["is_ok"], "only ok items can carry an injected error"
    return "injection rows always claim a wrong key, and only on ok items"


def test_original_rows_are_not_shuffled():
    items = _fake_items(20)
    rows = D.build_work_list(items, {3, 7})
    for row in rows:
        if row.kind == D.ORIGINAL:
            _, meta = D.render_row(row, items[row.item_index])
            assert meta["perm"] == [0, 1, 2, 3], meta
            assert meta["claimed_letter"] == D.LETTERS[items[row.item_index]["answer"]]
        else:
            assert row.shuffled
    return "the 600-item original-order rows keep the original option order"


def test_render_row_claims_the_right_text():
    items = _fake_items(30)
    rows = D.build_work_list(items, set())
    for row in rows[:60]:
        item = items[row.item_index]
        prompt, meta = D.render_row(row, item)
        claimed_text = item["choices"][row.claimed_key_original]
        letter = meta["claimed_letter"]
        line = [l for l in prompt.splitlines() if l.startswith(f"{letter}. ")][0]
        assert line == f"{letter}. {claimed_text}", (line, claimed_text)
        assert f"Claimed answer: {letter}" in prompt
    return "the prompt's claimed letter always points at the intended option text"


def test_parse_verdict():
    cases = [
        ("VERDICT: CORRECT\nREASON: it is right", D.CORRECT),
        ("VERDICT: INCORRECT\nREASON: key is wrong", D.INCORRECT),
        ("verdict: incorrect", D.INCORRECT),
        ("VERDICT:**INCORRECT**", D.INCORRECT),
        ("VERDICT - CORRECT", D.CORRECT),
        ("The claimed answer is INCORRECT.", D.INCORRECT),
        ("I think this is correct.", D.CORRECT),
        ("", D.UNPARSED),
        ("Hmm, let me think about the options.", D.UNPARSED),
        ("VERDICT: MAYBE", D.UNPARSED),
    ]
    for text, want in cases:
        got = D.parse_verdict(text)
        assert got == want, (text, got, want)
    # the substring trap: INCORRECT contains CORRECT
    assert D.parse_verdict("INCORRECT") == D.INCORRECT
    assert D.is_flag(D.INCORRECT) and not D.is_flag(D.CORRECT)
    assert not D.is_flag(D.UNPARSED), "unparsed must count as not flagged"
    return f"verdict parsing handles {len(cases)} formats including the INCORRECT/CORRECT substring trap"


def test_original_order_sample_contains_all_key_errors():
    items = _fake_items(1000)
    errors = list(range(0, 181))
    sample = D.original_order_sample(items, errors, n_total=600)
    assert set(errors) <= set(sample), "every key error must be in the 9a sample"
    assert len(sample) == 600, len(sample)
    assert sample == D.original_order_sample(items, errors, n_total=600), "must be reproducible"
    return f"the 9a sample holds all {len(errors)} key errors and totals {len(sample)}"


def test_smoke_check_passes_a_healthy_run():
    verdicts = [D.CORRECT] * 480 + [D.INCORRECT] * 30 + [D.UNPARSED] * 2
    r = D.smoke_check(verdicts, session_elapsed_s=300, generate_elapsed_s=60,
                      n_done=512, n_total=16360, budget_hours=1.0, label="llama3")
    assert r.ok, r.message
    assert abs(r.unparsed_rate - 2 / 512) < 1e-9
    assert abs(r.prompts_per_s - 512 / 60) < 1e-6
    assert "PASS" in r.message and "continuing automatically" in r.message
    return f"a healthy chunk passes (unparsed {r.unparsed_rate:.1%}, {r.projected_hours:.2f} h projected)"


def test_smoke_check_stops_on_unparsed():
    verdicts = [D.CORRECT] * 460 + [D.UNPARSED] * 52   # 10.2%
    r = D.smoke_check(verdicts, 300, 60, 512, 16360, budget_hours=5.0)
    assert not r.ok
    assert any("unparsed" in x for x in r.reasons), r.reasons
    assert "STOP" in r.message and "PREREG 12" in r.message
    return f"an unparsed rate of {r.unparsed_rate:.1%} stops the run"


def test_smoke_check_stops_on_budget():
    verdicts = [D.CORRECT] * 512
    # 512 rows took 600 s -> 0.85 prompts/s -> 16360 rows is well past 1 hour
    r = D.smoke_check(verdicts, 900, 600, 512, 16360, budget_hours=1.0)
    assert not r.ok
    assert any("projected" in x for x in r.reasons), r.reasons
    assert r.projected_hours > 1.0
    return f"a slow run stops ({r.prompts_per_s:.2f} prompts/s -> {r.projected_hours:.1f} h)"


def test_smoke_check_boundary_is_strictly_above():
    """Exactly 5% unparsed must pass; one more must not."""
    ok = D.smoke_check([D.UNPARSED] * 5 + [D.CORRECT] * 95, 10, 10, 100, 100, 10.0)
    bad = D.smoke_check([D.UNPARSED] * 6 + [D.CORRECT] * 94, 10, 10, 100, 100, 10.0)
    assert ok.ok, ok.message
    assert not bad.ok, bad.message
    return "the 5% unparsed limit is a strict >, so exactly 5% passes"


def test_smoke_check_handles_a_dead_run():
    r = D.smoke_check([D.UNPARSED] * 512, 60, 0.0, 512, 16360, budget_hours=1.0)
    assert not r.ok and r.projected_hours == float("inf")
    return "a zero-rate run projects to infinity and stops rather than dividing by zero"


TESTS = [
    test_permutation_is_fixed_and_a_real_permutation,
    test_key_moves_with_its_text,
    test_scored_wrong_keys,
    test_work_list_sizes,
    test_inject_rows_never_use_the_real_key,
    test_original_rows_are_not_shuffled,
    test_render_row_claims_the_right_text,
    test_parse_verdict,
    test_original_order_sample_contains_all_key_errors,
    test_smoke_check_passes_a_healthy_run,
    test_smoke_check_stops_on_unparsed,
    test_smoke_check_stops_on_budget,
    test_smoke_check_boundary_is_strictly_above,
    test_smoke_check_handles_a_dead_run,
]


def main():
    print("detector tests")
    ok = True
    for fn in TESTS:
        try:
            print(f"  PASS  {fn.__name__}: {fn()}")
        except AssertionError as exc:
            ok = False
            print(f"  FAIL  {fn.__name__}: {exc}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
