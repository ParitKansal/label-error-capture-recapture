"""Capture-recapture estimators for counting label errors in a benchmark.

Each detector is one capture occasion. An item is "captured" by a detector only
if the detector flags it AND the flag is verified to be a real error, so the
population being estimated is the number of real label errors, including those
no detector flagged.

Estimators implemented
----------------------
union               naive baseline: the number of distinct verified errors found
chapman             bias-corrected Lincoln-Petersen, for one pair of detectors
loglinear           Fienberg (1972) multiple-systems model, main effects plus
                    two-way interactions chosen by forward AIC   [PRIMARY]
chao_lower          Chao (1984) lower bound under heterogeneous detectability
jackknife_mh        Burnham & Overton (1978, 1979) Mh jackknife

References are listed in literature/README.md. Every one has been opened and
verified. The jackknife coefficients are not transcribed from the paper: they
are re-derived here by Richardson extrapolation and checked against Burnham &
Overton's published closed forms for orders 1-3 in test_crc.py.

Only numpy and scipy are required.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from math import comb, log, sqrt

import numpy as np
from scipy import stats

__all__ = [
    "CaptureData",
    "Estimate",
    "union",
    "chapman",
    "loglinear",
    "chao_lower",
    "jackknife_mh",
    "summary",
]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


@dataclass
class CaptureData:
    """Capture histories of the verified errors that at least one detector found.

    `histories` is an (n_found, n_detectors) 0/1 array. Rows are verified errors,
    columns are detectors. An all-zero row is meaningless here (an error nobody
    found is by definition not in the data) and is rejected.
    """

    histories: np.ndarray
    detector_names: tuple[str, ...] = ()

    def __post_init__(self):
        self.histories = np.asarray(self.histories, dtype=np.int8)
        if self.histories.ndim != 2:
            raise ValueError("histories must be 2-D (items x detectors)")
        if self.histories.size and not np.isin(self.histories, (0, 1)).all():
            raise ValueError("histories must contain only 0 and 1")
        if self.histories.size and (self.histories.sum(axis=1) == 0).any():
            raise ValueError("an all-zero capture history cannot be observed")
        if not self.detector_names:
            self.detector_names = tuple(f"D{i + 1}" for i in range(self.n_detectors))
        if len(self.detector_names) != self.n_detectors:
            raise ValueError("detector_names length does not match histories width")

    @property
    def n_detectors(self) -> int:
        return self.histories.shape[1]

    @property
    def n_found(self) -> int:
        """D: distinct verified errors found by at least one detector."""
        return int(self.histories.shape[0])

    @property
    def frequencies(self) -> np.ndarray:
        """f[j-1] = number of errors captured by exactly j detectors, j = 1..K."""
        times = self.histories.sum(axis=1)
        return np.array(
            [int((times == j).sum()) for j in range(1, self.n_detectors + 1)]
        )

    def cell_counts(self) -> tuple[np.ndarray, np.ndarray]:
        """All 2^K - 1 observable patterns and their counts (zeros included)."""
        k = self.n_detectors
        patterns = np.array(
            [p for p in itertools.product((0, 1), repeat=k) if any(p)], dtype=np.int8
        )
        if self.n_found == 0:
            return patterns, np.zeros(len(patterns), dtype=float)
        index = {tuple(p): i for i, p in enumerate(map(tuple, patterns))}
        counts = np.zeros(len(patterns), dtype=float)
        for row in self.histories:
            counts[index[tuple(row)]] += 1
        return patterns, counts

    def bootstrap(self, rng: np.random.Generator) -> "CaptureData":
        """Nonparametric bootstrap: resample the found errors with replacement."""
        idx = rng.integers(0, self.n_found, size=self.n_found)
        return CaptureData(self.histories[idx], self.detector_names)

    @classmethod
    def from_flags(
        cls,
        flags: np.ndarray,
        is_error: np.ndarray,
        detector_names: tuple[str, ...] = (),
    ) -> "CaptureData":
        """Build capture histories from raw detector flags and verified truth.

        `flags` is (n_items, n_detectors) 0/1, `is_error` is (n_items,) 0/1.
        A capture is a flag on an item that is truly an error (a verified flag);
        flags on non-errors are false positives and are not captures.
        """
        flags = np.asarray(flags, dtype=np.int8)
        is_error = np.asarray(is_error, dtype=bool)
        captured = flags & is_error[:, None]
        rows = captured[captured.sum(axis=1) > 0]
        return cls(rows, detector_names)


@dataclass
class Estimate:
    """One estimator's answer. `n_hat` is the estimated total, found and unfound."""

    name: str
    n_hat: float
    lo: float = float("nan")
    hi: float = float("nan")
    ok: bool = True
    note: str = ""
    extra: dict = field(default_factory=dict)

    def covers(self, truth: float) -> bool:
        return bool(self.lo <= truth <= self.hi)


# --------------------------------------------------------------------------
# baseline and pairwise
# --------------------------------------------------------------------------


def union(data: CaptureData) -> Estimate:
    """The naive baseline: count what was actually found."""
    d = data.n_found
    return Estimate("union", float(d), float(d), float(d), note="baseline, not an estimator")


def chapman(data: CaptureData, i: int, j: int, conf: float = 0.95) -> Estimate:
    """Chapman's bias-corrected Lincoln-Petersen for detectors i and j.

    Cited through Rivest & Yauck (arXiv 2307.11705), which we have opened;
    Chapman (1951) itself could not be located (see literature/README.md).
    """
    a = data.histories[:, i].astype(bool)
    b = data.histories[:, j].astype(bool)
    n1, n2, m = int(a.sum()), int(b.sum()), int((a & b).sum())
    name = f"chapman[{data.detector_names[i]},{data.detector_names[j]}]"
    if n1 == 0 or n2 == 0:
        return Estimate(name, float("nan"), ok=False, note="a detector captured nothing")
    n_hat = (n1 + 1) * (n2 + 1) / (m + 1) - 1
    var = (
        (n1 + 1) * (n2 + 1) * (n1 - m) * (n2 - m) / ((m + 1) ** 2 * (m + 2))
    )
    se = sqrt(max(var, 0.0))
    z = stats.norm.ppf(0.5 + conf / 2)
    lo, hi = n_hat - z * se, n_hat + z * se
    note = "m = 0: unstable" if m == 0 else ""
    return Estimate(
        name,
        float(n_hat),
        float(max(lo, data.n_found)),
        float(hi),
        ok=(m > 0),
        note=note,
        extra={"n1": n1, "n2": n2, "m": m, "se": se},
    )


# --------------------------------------------------------------------------
# Chao lower bound
# --------------------------------------------------------------------------


def chao_lower(data: CaptureData, conf: float = 0.95) -> Estimate:
    """Chao (1984) lower bound, bias-corrected form, with a log-normal CI.

    This is a floor, not a point estimate: it is the smallest total consistent
    with the observed amount of single-capture mass. Reported as a sanity check
    on everything else.
    """
    d = data.n_found
    f = data.frequencies
    f1 = float(f[0]) if len(f) > 0 else 0.0
    f2 = float(f[1]) if len(f) > 1 else 0.0
    if f1 == 0:
        return Estimate("chao_lower", float(d), float(d), float(d), note="f1 = 0")
    # bias-corrected form, valid when f2 may be 0
    extra = f1 * (f1 - 1) / (2 * (f2 + 1))
    n_hat = d + extra
    # log-normal interval on the unseen part (Chao's standard construction)
    if extra <= 0:
        return Estimate("chao_lower", float(n_hat), float(d), float(n_hat), note="no unseen mass")
    var = (
        f1 * (f1 - 1) / (2 * (f2 + 1))
        + f1 * (2 * f1 - 1) ** 2 / (4 * (f2 + 1) ** 2)
        + f1**2 * f2 * (f1 - 1) ** 2 / (4 * (f2 + 1) ** 4)
    )
    se = sqrt(max(var, 1e-12))
    z = stats.norm.ppf(0.5 + conf / 2)
    c = np.exp(z * sqrt(log(1 + var / extra**2)))
    return Estimate(
        "chao_lower",
        float(n_hat),
        float(d + extra / c),
        float(d + extra * c),
        note="lower bound, not a point estimate",
        extra={"f1": f1, "f2": f2, "se": se},
    )


# --------------------------------------------------------------------------
# Burnham & Overton Mh jackknife
# --------------------------------------------------------------------------


def _jackknife_weights(k_occasions: int, order: int) -> np.ndarray:
    """Richardson-extrapolation weights that kill bias terms 1/K .. 1/K^order.

    Solves  sum_i w_i = 1  and  sum_i w_i (K-i)^-m = 0  for m = 1..order.
    test_crc.py checks that the resulting estimator equals Burnham & Overton's
    published closed forms for orders 1, 2 and 3.
    """
    n = order + 1
    a = np.empty((n, n), dtype=float)
    a[0, :] = 1.0
    for m in range(1, n):
        a[m, :] = [1.0 / (k_occasions - i) ** m for i in range(n)]
    b = np.zeros(n)
    b[0] = 1.0
    return np.linalg.solve(a, b)


def _jackknife_coefficients(k_occasions: int, order: int) -> np.ndarray:
    """a[j-1] in  N_hat_order = D + sum_j a[j-1] * f_j,  for j = 1..K.

    Derived from the delete-i jackknife, not transcribed: an item caught in
    exactly j occasions disappears when a random i-subset of occasions is
    deleted with probability C(K-j, i-j)/C(K, i), and the Richardson weights
    sum to 1, so the D terms cancel.
    """
    a = np.zeros(k_occasions)
    if order == 0:
        return a
    w = _jackknife_weights(k_occasions, order)
    for j in range(1, k_occasions + 1):
        a[j - 1] = -sum(
            w[i] * comb(k_occasions - j, i - j) / comb(k_occasions, i)
            for i in range(j, order + 1)
        )
    return a


def _jackknife_point(freqs: np.ndarray, d: int, k_occasions: int, order: int) -> float:
    """N-hat for the jackknife of the given order."""
    if order == 0:
        return float(d)
    return float(d + _jackknife_coefficients(k_occasions, order) @ np.asarray(freqs, float))


def _jackknife_select_order(freqs: np.ndarray, d: int, k: int, max_order: int, alpha: float) -> int:
    """Burnham & Overton's sequential test, with their variance of the difference.

    Start at order 0 (= take what was found) and step up while the next order
    differs significantly. Var(N_{o+1} - N_o) = sum_j (a_{o+1,j} - a_{o,j})^2 f_j
    under the usual multinomial approximation.
    """
    z_crit = stats.norm.ppf(1 - alpha / 2)
    f = np.asarray(freqs, dtype=float)
    for o in range(0, max_order):
        a_here = _jackknife_coefficients(k, o)
        a_next = _jackknife_coefficients(k, o + 1)
        diff = float((a_next - a_here) @ f)
        var = float(((a_next - a_here) ** 2) @ f)
        if var <= 0:
            return o
        if abs(diff) / sqrt(var) < z_crit:
            return o
    return max_order


def jackknife_mh(
    data: CaptureData,
    conf: float = 0.95,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
    order: int | None = None,
) -> Estimate:
    """Burnham & Overton (1978, 1979) Mh jackknife.

    SECONDARY estimator. The order is chosen by Burnham & Overton's sequential
    test, and the interval is a nonparametric bootstrap in which the order is
    re-selected inside every replicate, so the interval pays for order-selection
    uncertainty as well as sampling noise.

    The simulation study in test_crc.py shows this estimator runs high and its
    interval is too narrow when detectability is close to homogeneous. It is
    reported for comparison; it is not the estimator the GO rule depends on.
    """
    d = data.n_found
    k = data.n_detectors
    f = data.frequencies
    max_order = min(k - 1, 5)
    if d == 0 or max_order < 1:
        return Estimate("jackknife_mh", float(d), ok=False, note="not enough detectors")

    chosen = int(order) if order is not None else _jackknife_select_order(
        f, d, k, max_order, alpha
    )
    n_hat = _jackknife_point(f, d, k, chosen)

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        rs = data.bootstrap(rng)
        fb, db = rs.frequencies, rs.n_found
        ob = int(order) if order is not None else _jackknife_select_order(
            fb, db, k, max_order, alpha
        )
        boots[b] = _jackknife_point(fb, db, k, ob)
    lo, hi = np.percentile(boots, [100 * (1 - conf) / 2, 100 * (1 + conf) / 2])

    clipped = n_hat < d
    return Estimate(
        "jackknife_mh",
        float(max(n_hat, d)),
        float(max(lo, d)),
        float(max(hi, d)),
        ok=not clipped,
        note=f"order {chosen}" + (" (clipped at D)" if clipped else ""),
        extra={"order": chosen, "frequencies": f.tolist()},
    )


# --------------------------------------------------------------------------
# log-linear multiple-systems estimation  (primary)
# --------------------------------------------------------------------------


def _poisson_irls(x: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-10):
    """Poisson log-link GLM by IRLS. Returns (beta, converged)."""
    beta = np.zeros(x.shape[1])
    beta[0] = log(max(y.mean(), 1e-3))
    for _ in range(max_iter):
        eta = np.clip(x @ beta, -30, 30)
        mu = np.exp(eta)
        w = mu
        z = eta + (y - mu) / np.maximum(mu, 1e-10)
        xtw = x.T * w
        try:
            new = np.linalg.solve(xtw @ x + 1e-10 * np.eye(x.shape[1]), xtw @ z)
        except np.linalg.LinAlgError:
            return beta, False
        if not np.all(np.isfinite(new)):
            return beta, False
        if np.max(np.abs(new - beta)) < tol:
            return new, True
        beta = new
    return beta, False


def _design(patterns: np.ndarray, terms: list[tuple[int, ...]]) -> np.ndarray:
    cols = [np.ones(len(patterns))]
    for t in terms:
        col = np.ones(len(patterns))
        for idx in t:
            col = col * patterns[:, idx]
        cols.append(col.astype(float))
    return np.column_stack(cols)


def _fit(patterns, counts, terms):
    x = _design(patterns, terms)
    beta, ok = _poisson_irls(x, counts)
    mu = np.exp(np.clip(x @ beta, -30, 30))
    with np.errstate(divide="ignore", invalid="ignore"):
        dev_terms = np.where(counts > 0, counts * np.log(counts / mu), 0.0) - (counts - mu)
    deviance = float(2 * dev_terms.sum())
    aic = deviance + 2 * x.shape[1]
    return beta, ok, deviance, aic, x.shape[1]


def _forward_aic(patterns, counts, k, max_order=2):
    """Main effects, then greedily add the two-way term that lowers AIC most."""
    main = [(i,) for i in range(k)]
    candidates = [c for c in itertools.combinations(range(k), 2)] if max_order >= 2 else []
    terms = list(main)
    _, ok, _, best_aic, _ = _fit(patterns, counts, terms)
    improved = True
    while improved and candidates:
        improved = False
        best_term, best_new = None, best_aic
        for c in candidates:
            _, ok_c, _, aic_c, _ = _fit(patterns, counts, terms + [c])
            if ok_c and aic_c < best_new - 1e-9:
                best_term, best_new = c, aic_c
        if best_term is not None:
            terms.append(best_term)
            candidates.remove(best_term)
            best_aic = best_new
            improved = True
    return terms


def _all_patterns(k: int) -> np.ndarray:
    """All 2^k patterns, the all-zero (never-detected) cell first."""
    return np.array(list(itertools.product((0, 1), repeat=k)), dtype=np.int8)


def _parametric_draw(data: CaptureData, terms, rng: np.random.Generator) -> CaptureData | None:
    """Simulate a whole population from the fitted model, then keep what is seen.

    Unlike resampling the found errors, this regenerates the unobserved cell
    too, so the interval carries the uncertainty in the part of the population
    nobody detected. That is the part the estimate is actually about.
    """
    k = data.n_detectors
    patterns, counts = data.cell_counts()
    beta, ok, _, _, _ = _fit(patterns, counts, terms)
    if not ok:
        return None
    full = _all_patterns(k)
    mu = np.exp(np.clip(_design(full, terms) @ beta, -30, 30))
    total = mu.sum()
    if not np.isfinite(total) or total <= 0:
        return None
    n_star = rng.poisson(total)
    if n_star < 5:
        return None
    cell_counts = rng.multinomial(n_star, mu / total)
    rows = []
    for pattern, c in zip(full, cell_counts):
        if c and pattern.any():
            rows.append(np.repeat(pattern[None, :], c, axis=0))
    if not rows:
        return None
    return CaptureData(np.vstack(rows), data.detector_names)


def _loglinear_point(data: CaptureData, max_order: int, max_ratio: float):
    patterns, counts = data.cell_counts()
    k = data.n_detectors
    d = data.n_found
    terms = _forward_aic(patterns, counts, k, max_order)
    beta, ok, deviance, aic, n_par = _fit(patterns, counts, terms)
    unseen = float(np.exp(np.clip(beta[0], -30, 30)))
    n_hat = d + unseen
    stable = ok and np.isfinite(n_hat) and n_hat <= max_ratio * max(d, 1)
    if not stable:
        terms = [(i,) for i in range(k)]
        beta, ok, deviance, aic, n_par = _fit(patterns, counts, terms)
        unseen = float(np.exp(np.clip(beta[0], -30, 30)))
        n_hat = d + unseen
        n_hat = min(n_hat, max_ratio * max(d, 1))
    df = len(patterns) - n_par
    return n_hat, terms, deviance, df, stable


def loglinear(
    data: CaptureData,
    conf: float = 0.95,
    max_order: int = 2,
    n_boot: int = 2000,
    seed: int = 0,
    max_ratio: float = 10.0,
    ci_method: str = "parametric",
) -> Estimate:
    """PRIMARY estimator. Fienberg (1972) log-linear multiple-systems model.

    Main effects plus two-way detector interactions selected by forward AIC.
    Model selection is re-run inside every bootstrap replicate, so the interval
    pays for selection uncertainty as well as sampling noise.

    ci_method:
      "parametric"    simulate a whole population from the fitted model and keep
                      what is detected. Pre-registered default: the simulation
                      study in test_crc.py shows it reaches nominal coverage,
                      while conditioning on the found errors does not.
      "nonparametric" resample the found errors with replacement.

    `max_ratio` is the stability guard: if the selected model implies more than
    `max_ratio` x (found errors), the fit is treated as unstable and the
    independence model is used instead. Fixed here, in advance, from simulation
    only -- never retuned after seeing expert labels.
    """
    d = data.n_found
    if d == 0:
        return Estimate("loglinear", 0.0, 0.0, 0.0, ok=False, note="nothing found")

    n_hat, terms, deviance, df, stable = _loglinear_point(data, max_order, max_ratio)
    gof_p = float(stats.chi2.sf(deviance, df)) if df > 0 else float("nan")

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        if ci_method == "parametric":
            rs = _parametric_draw(data, terms, rng)
        elif ci_method == "nonparametric":
            rs = data.bootstrap(rng)
        else:
            raise ValueError(f"unknown ci_method {ci_method!r}")
        if rs is None:
            continue
        boots.append(_loglinear_point(rs, max_order, max_ratio)[0])
    boots = np.asarray(boots, dtype=float)
    if boots.size < 20:
        lo = hi = float("nan")
    else:
        lo, hi = np.percentile(boots, [100 * (1 - conf) / 2, 100 * (1 + conf) / 2])

    pretty = [":".join(data.detector_names[i] for i in t) for t in terms if len(t) > 1]
    return Estimate(
        "loglinear",
        float(n_hat),
        float(max(lo, d)) if np.isfinite(lo) else float("nan"),
        float(hi),
        ok=stable,
        note=("interactions: " + ", ".join(pretty) if pretty else "independence model")
        + ("" if stable else "; UNSTABLE, fell back to independence"),
        extra={
            "terms": terms,
            "deviance": deviance,
            "df": df,
            "gof_p": gof_p,
            "ci_method": ci_method,
            "boot_sd": float(boots.std(ddof=1)) if boots.size > 1 else float("nan"),
            "n_boot_used": int(boots.size),
        },
    )


# --------------------------------------------------------------------------


def summary(
    data: CaptureData,
    conf: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
    include_pairs: bool = True,
) -> list[Estimate]:
    """Run every estimator. The first entry is the pre-registered primary one."""
    out = [
        loglinear(data, conf=conf, n_boot=n_boot, seed=seed),
        jackknife_mh(data, conf=conf, n_boot=n_boot, seed=seed),
        chao_lower(data, conf=conf),
        union(data),
    ]
    if include_pairs:
        for i, j in itertools.combinations(range(data.n_detectors), 2):
            out.append(chapman(data, i, j, conf=conf))
    return out
