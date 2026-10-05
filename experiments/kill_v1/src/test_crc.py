"""Tests and the simulation coverage study for crc.py.

Two jobs:

1. Unit tests. Most importantly, that the jackknife weights derived in crc.py by
   Richardson extrapolation reproduce Burnham & Overton's published closed forms
   for orders 1, 2 and 3, so nothing depends on a coefficient typed in by hand.

2. The pre-registration's simulation requirement: measure the coverage of the
   95% interval under independent detectors, dependent detectors and
   heterogeneous items, BEFORE any GPU run. If coverage under realistic
   dependence is poor, the design is fixed now rather than after seeing results.

    python test_crc.py              # unit tests + full simulation study
    python test_crc.py --quick      # unit tests + a small simulation
    python test_crc.py --unit-only  # unit tests only
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import sys
from dataclasses import dataclass
from fractions import Fraction
from math import comb

import numpy as np

sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))
import crc  # noqa: E402


# ==========================================================================
# unit tests
# ==========================================================================


def _published_jackknife(k: int, f: list[int], order: int) -> Fraction:
    """Burnham & Overton (1978) closed forms for orders 1-3, in exact arithmetic."""
    d = Fraction(sum(f))
    f1, f2, f3 = (Fraction(f[i]) if i < len(f) else Fraction(0) for i in range(3))
    if order == 1:
        return d + f1 * Fraction(k - 1, k)
    if order == 2:
        return d + f1 * Fraction(2 * k - 3, k) - f2 * Fraction((k - 2) ** 2, k * (k - 1))
    if order == 3:
        return (
            d
            + f1 * Fraction(3 * k - 6, k)
            - f2 * Fraction(3 * k * k - 15 * k + 19, k * (k - 1))
            + f3 * Fraction((k - 3) ** 3, k * (k - 1) * (k - 2))
        )
    raise ValueError(order)


def test_jackknife_matches_burnham_overton():
    rng = np.random.default_rng(0)
    checked = 0
    for k in (4, 5, 6, 7):
        for _ in range(25):
            f = [int(x) for x in rng.integers(0, 60, size=k)]
            d = sum(f)
            for order in (1, 2, 3):
                if order > k - 1:
                    continue
                got = crc._jackknife_point(np.array(f), d, k, order)
                want = float(_published_jackknife(k, f, order))
                assert abs(got - want) < 1e-6 * max(1.0, abs(want)), (k, f, order, got, want)
                checked += 1
    return f"jackknife weights match Burnham & Overton orders 1-3 ({checked} cases)"


def test_capture_data_basics():
    x = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.int8)
    d = crc.CaptureData(x)
    assert d.n_found == 4
    assert list(d.frequencies) == [2, 1, 1]
    patterns, counts = d.cell_counts()
    assert len(patterns) == 7 and counts.sum() == 4
    try:
        crc.CaptureData(np.array([[0, 0, 0]], dtype=np.int8))
    except ValueError:
        pass
    else:
        raise AssertionError("all-zero history should be rejected")
    return "CaptureData counts, frequencies and cells behave"


def test_only_verified_flags_are_captures():
    """The lesson from the occupancy project: a flag on a non-error is not a capture."""
    flags = np.array([[1, 1], [1, 0], [0, 1], [1, 1]], dtype=np.int8)
    is_error = np.array([1, 0, 1, 0], dtype=bool)  # items 1 and 2 only
    d = crc.CaptureData.from_flags(flags, is_error)
    assert d.n_found == 2, d.n_found
    assert sorted(map(tuple, d.histories)) == [(0, 1), (1, 1)]
    return "from_flags counts verified flags only, false positives are not captures"


def test_chapman_known_value():
    # 60 captured by A, 40 by B, 20 by both -> Chapman = 61*41/21 - 1 = 118.09...
    rows = [[1, 1]] * 20 + [[1, 0]] * 40 + [[0, 1]] * 20
    d = crc.CaptureData(np.array(rows, dtype=np.int8))
    e = crc.chapman(d, 0, 1)
    want = 61 * 41 / 21 - 1
    assert abs(e.n_hat - want) < 1e-9, (e.n_hat, want)
    assert e.extra == {"n1": 60, "n2": 40, "m": 20, "se": e.extra["se"]}
    return f"Chapman reproduces its closed form ({want:.2f})"


def test_loglinear_recovers_noiseless_independence():
    """Exact expected cell counts under independence -> N-hat must be exact."""
    k, n = 4, 1000.0
    p = np.array([0.5, 0.4, 0.3, 0.2])
    patterns = np.array(
        [q for q in itertools.product((0, 1), repeat=k) if any(q)], dtype=np.int8
    )
    probs = np.array(
        [np.prod(np.where(q == 1, p, 1 - p)) for q in patterns]
    )
    counts = n * probs
    rows = []
    for q, c in zip(patterns, counts):
        rows.extend([q] * int(round(c * 100)))
    d = crc.CaptureData(np.array(rows, dtype=np.int8))
    est = crc.loglinear(d, n_boot=50, seed=0)
    ratio = est.n_hat / (n * 100)
    assert 0.99 < ratio < 1.01, (est.n_hat, n * 100)
    assert est.extra["gof_p"] > 0.99, est.extra["gof_p"]
    return f"log-linear recovers a noiseless independent population (ratio {ratio:.4f})"


def test_chao_behaves_like_a_bound():
    """Chao must never sit below what was found, and under the heterogeneity it
    is designed for it must sit at or below the truth most of the time.

    It is NOT a lower bound under near-homogeneous binomial capture, where it
    runs high; the simulation table records that, which is why it is reported
    as a floor only alongside the goodness-of-fit numbers.
    """
    rng = np.random.default_rng(3)
    never_below_found = True
    below_truth = 0
    n_true = 400
    for _ in range(40):
        base = np.log(np.array([0.40, 0.35, 0.30, 0.45]) / np.array([0.60, 0.65, 0.70, 0.55]))
        offset = rng.normal(0.0, 1.8, size=(n_true, 1))
        prob = 1 / (1 + np.exp(-(base[None, :] + offset)))
        x = (rng.random((n_true, 4)) < prob).astype(np.int8)
        x = x[x.sum(1) > 0]
        d = crc.CaptureData(x)
        est = crc.chao_lower(d)
        never_below_found &= est.n_hat >= d.n_found - 1e-9
        below_truth += est.n_hat <= n_true
    assert never_below_found, "chao_lower fell below the union count"
    assert below_truth >= 32, below_truth
    return f"Chao >= union always, and <= truth in {below_truth}/40 heterogeneous draws"


def test_unstable_fit_falls_back():
    """A degenerate, perfectly nested pattern must not produce a silent huge N."""
    rows = [[1, 1, 1, 1]] * 50 + [[1, 1, 1, 0]] * 2
    d = crc.CaptureData(np.array(rows, dtype=np.int8))
    est = crc.loglinear(d, n_boot=20, seed=0, max_ratio=10.0)
    assert est.n_hat <= 10 * d.n_found + 1e-6, est.n_hat
    return "the stability guard caps a degenerate fit instead of returning nonsense"


UNIT_TESTS = [
    test_jackknife_matches_burnham_overton,
    test_capture_data_basics,
    test_only_verified_flags_are_captures,
    test_chapman_known_value,
    test_loglinear_recovers_noiseless_independence,
    test_chao_behaves_like_a_bound,
    test_unstable_fit_falls_back,
]


def run_unit_tests() -> bool:
    print("unit tests")
    ok = True
    for fn in UNIT_TESTS:
        try:
            msg = fn()
            print(f"  PASS  {fn.__name__}: {msg}")
        except AssertionError as exc:
            ok = False
            print(f"  FAIL  {fn.__name__}: {exc}")
    print()
    return ok


# ==========================================================================
# simulation study
# ==========================================================================


@dataclass
class Scenario:
    key: str
    description: str
    n_true: int
    n_detectors: int

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


@dataclass
class Independent(Scenario):
    p: tuple[float, ...] = ()

    def draw(self, rng):
        p = np.array(self.p)
        return (rng.random((self.n_true, self.n_detectors)) < p).astype(np.int8)


@dataclass
class Heterogeneous(Scenario):
    """Mh: every item has its own detectability, shared by all detectors."""

    p: tuple[float, ...] = ()
    sd: float = 1.0

    def draw(self, rng):
        base = np.log(np.array(self.p) / (1 - np.array(self.p)))
        offset = rng.normal(0.0, self.sd, size=(self.n_true, 1))
        prob = 1 / (1 + np.exp(-(base[None, :] + offset)))
        return (rng.random((self.n_true, self.n_detectors)) < prob).astype(np.int8)


@dataclass
class DependentJudges(Scenario):
    """The project's own failure mode: the LLM judges all miss the same errors.

    The first `n_judges` detectors share a latent item factor with loading
    `rho`; the remaining detectors (confident learning) are conditionally
    independent of it, which is exactly why the design insists on a detector
    from a different family.
    """

    p: tuple[float, ...] = ()
    n_judges: int = 4
    rho: float = 1.5
    sd: float = 0.6

    def draw(self, rng):
        base = np.log(np.array(self.p) / (1 - np.array(self.p)))
        judge_factor = rng.normal(0.0, self.rho, size=(self.n_true, 1))
        item_noise = rng.normal(0.0, self.sd, size=(self.n_true, 1))
        loading = np.zeros(self.n_detectors)
        loading[: self.n_judges] = 1.0
        eta = base[None, :] + judge_factor * loading[None, :] + item_noise
        prob = 1 / (1 + np.exp(-eta))
        return (rng.random((self.n_true, self.n_detectors)) < prob).astype(np.int8)


def scenarios(kill_sizes=(181, 266)) -> list[Scenario]:
    five = (0.40, 0.35, 0.30, 0.25, 0.45)
    out = [
        Independent(
            "M0", "independent detectors, equal detectability", 181, 5, p=(0.30,) * 5
        ),
        Independent(
            "Mt", "independent detectors, detector-specific rates", 181, 5, p=five
        ),
        Heterogeneous(
            "Mh", "independent detectors, item detectability varies (sd 1.0)",
            181, 5, p=five, sd=1.0,
        ),
        Heterogeneous(
            "Mh-strong", "item detectability varies a lot (sd 1.8)", 181, 5, p=five, sd=1.8,
        ),
        DependentJudges(
            "dep-judges",
            "4 judges share a latent factor (rho 1.5), CL independent",
            181, 5, p=five, n_judges=4, rho=1.5,
        ),
        DependentJudges(
            "dep-strong",
            "judges strongly dependent (rho 2.5) -- the designed-to-fail case",
            181, 5, p=five, n_judges=4, rho=2.5,
        ),
        DependentJudges(
            "dep-4det",
            "only 4 detectors, 3 dependent judges -- Briand et al. 2000 warning",
            181, 4, p=(0.40, 0.35, 0.30, 0.45), n_judges=3, rho=1.5,
        ),
    ]
    for n in kill_sizes:
        if n != 181:
            out.append(
                DependentJudges(
                    f"dep-n{n}",
                    f"injection-sized population (N={n}), judges dependent",
                    n, 5, p=five, n_judges=4, rho=1.5,
                )
            )
    return out


# display name -> key produced by _one_replicate
ESTIMATORS = {
    "loglin-par": "loglinear",       # PRIMARY: parametric bootstrap interval
    "loglin-npar": "loglinear_np",   # same point estimate, resampling interval
    "jackknife": "jackknife_mh",
    "chao": "chao_lower",
    "union": "union",
}


def _one_replicate(args):
    scenario, seed, n_boot = args
    rng = np.random.default_rng(seed)
    x = scenario.draw(rng)
    x = x[x.sum(axis=1) > 0]
    if len(x) < 5:
        return None
    data = crc.CaptureData(x)
    out = {}
    for est in crc.summary(data, n_boot=n_boot, seed=seed, include_pairs=False):
        out[est.name] = (est.n_hat, est.lo, est.hi, est.ok)
    alt = crc.loglinear(data, n_boot=n_boot, seed=seed, ci_method="nonparametric")
    out["loglinear_np"] = (alt.n_hat, alt.lo, alt.hi, alt.ok)
    out["_found"] = data.n_found
    return out


def run_simulation(n_rep: int, n_boot: int, processes: int | None = None):
    print(f"simulation study: {n_rep} replicates per scenario, {n_boot} bootstrap draws")
    print(
        "coverage = share of replicates whose 95% interval contains the true count.\n"
        "ratio    = median estimate / truth. miss = share of true errors no detector found.\n"
    )
    header = f"{'scenario':12s} {'truth':>5s} {'miss':>5s} " + " ".join(
        f"{name:>21s}" for name in ESTIMATORS
    )
    print(header)
    print(f"{'':12s} {'':>5s} {'':>5s} " + " ".join(f"{'cover  ratio  unst':>21s}" for _ in ESTIMATORS))
    print("-" * len(header))

    rows = []
    pool = mp.Pool(processes) if (processes or 1) > 1 else None
    try:
        for sc in scenarios():
            jobs = [(sc, 10_000 + i, n_boot) for i in range(n_rep)]
            res = pool.map(_one_replicate, jobs) if pool else [_one_replicate(j) for j in jobs]
            res = [r for r in res if r]
            miss = 1 - np.mean([r["_found"] for r in res]) / sc.n_true
            line = f"{sc.key:12s} {sc.n_true:5d} {miss:5.0%} "
            row = {"scenario": sc.key, "description": sc.description,
                   "truth": sc.n_true, "miss": miss}
            for name, est in ESTIMATORS.items():
                vals = np.array([r[est] for r in res], dtype=float)
                with np.errstate(invalid="ignore"):
                    cover = np.nanmean(
                        ((vals[:, 1] <= sc.n_true) & (sc.n_true <= vals[:, 2])).astype(float)
                    )
                ratio = np.median(vals[:, 0]) / sc.n_true
                unstable = 1 - np.mean(vals[:, 3])
                line += f" {cover:6.0%} {ratio:6.2f} {unstable:5.0%}"
                row[name] = {"coverage": float(cover), "ratio": float(ratio),
                             "unstable": float(unstable)}
            print(line)
            rows.append(row)
    finally:
        if pool:
            pool.close()
            pool.join()
    print()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small, fast simulation")
    ap.add_argument("--unit-only", action="store_true")
    ap.add_argument("--n-rep", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--processes", type=int, default=mp.cpu_count())
    args = ap.parse_args()

    ok = run_unit_tests()
    if args.unit_only:
        return 0 if ok else 1

    n_rep = args.n_rep or (60 if args.quick else 400)
    n_boot = args.n_boot or (100 if args.quick else 300)
    run_simulation(n_rep, n_boot, args.processes)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
