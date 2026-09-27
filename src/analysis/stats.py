"""
Statistics module for BASS benchmarking: turns raw per-trial measurements
into the numbers a manuscript actually reports.

This module consolidates and supersedes statistics code that used to be
duplicated (and drifting) across individual notebooks
(`Fidelity_Pr.ipynb`, `MPS_Comparison.ipynb`, `SchmidtWeightedTruncation.ipynb`)
and, before that, across ad hoc blocks inside `runner.py`. There is now
exactly one implementation of "how we report a paired comparison" in this
repository.

Design choices, and why (each one is a response to something checked
against either NumPy documentation or the statistical-methodology
literature, not picked arbitrarily):

- Median [IQR] rather than mean/std for absolute fidelities: fidelities in
this project span many orders of magnitude and are nowhere near Gaussian.

- Geometric-mean ratio with a percentile bootstrap CI for paired ratios:
ratios of positive, multiplicatively-distributed quantities are the
textbook case for geometric rather than arithmetic averaging.

- Wilson score interval (not the normal/Wald interval) for win rates: win
rates in this project are frequently at or near 0 or 1, exactly where
the Wald interval is known to fail.


"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ─── Legacy / general-purpose descriptive statistics ──────────────────────────


def compute_metrics(data_array, n_boot=4000, seed=42):
    """
    Computes rigorous, non-parametric statistics for heavy-tailed quantum data.
    - Uses Median/IQR for absolute probabilities (Fidelity)
    - Uses Geometric Mean / 95% Bootstrap CI for multiplicatively distributed data (PR, Runtime, Ratios)
    """
    data = np.asarray(data_array, dtype=float)
    data = data[~np.isnan(data)]

    if len(data) == 0:
        return {
            "median": np.nan,
            "iqr_25": np.nan,
            "iqr_75": np.nan,
            "p10": np.nan,
            "p90": np.nan,
            "arithmetic_mean": np.nan,
            "sem": np.nan,
            "geometric_mean": np.nan,
            "gm_ci95_lo": np.nan,
            "gm_ci95_hi": np.nan,
        }

    p10, p25, median, p75, p90 = np.percentile(data, [10, 25, 50, 75, 90])
    arithmetic_mean = np.mean(data)
    sem = stats.sem(data) if len(data) > 1 else 0.0

    positive_data = data[data > 0]
    if len(positive_data) > 0:
        log_data = np.log(positive_data)
        geometric_mean = np.exp(np.mean(log_data))
        if len(positive_data) > 1:
            rng = np.random.default_rng(seed=seed)
            boot_samples = rng.choice(
                log_data, size=(n_boot, len(log_data)), replace=True
            )
            boot_log_means = np.mean(boot_samples, axis=1)
            ci95_lo = np.exp(np.percentile(boot_log_means, 2.5))
            ci95_hi = np.exp(np.percentile(boot_log_means, 97.5))
        else:
            ci95_lo, ci95_hi = geometric_mean, geometric_mean
    else:
        geometric_mean, ci95_lo, ci95_hi = np.nan, np.nan, np.nan

    return {
        "median": median,
        "iqr_25": p25,
        "iqr_75": p75,
        "p10": p10,
        "p90": p90,
        "arithmetic_mean": arithmetic_mean,
        "sem": sem,
        "geometric_mean": geometric_mean,
        "gm_ci95_lo": ci95_lo,
        "gm_ci95_hi": ci95_hi,
    }


def bootstrap_ci(data, stat_fn=None, n_boot=4000, ci_level=95, seed=0):
    """Non-parametric bootstrap confidence interval for any statistic."""
    if stat_fn is None:
        stat_fn = lambda x: np.median(x, axis=1)

    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    if len(data) == 0:
        return np.nan, np.nan
    if len(data) == 1:
        v = float(stat_fn(data[np.newaxis, :]))
        return v, v

    rng_b = np.random.default_rng(seed)
    boots = rng_b.choice(data, size=(n_boot, len(data)), replace=True)
    boot_stats = stat_fn(boots)

    alpha = (100 - ci_level) / 2.0
    return (
        float(np.percentile(boot_stats, alpha)),
        float(np.percentile(boot_stats, 100 - alpha)),
    )


def trial_stats(data, n_boot=4000, ci_level=95, seed=0):
    """Full distribution summary for a 1-D array of trial measurements."""
    _nan = lambda keys: {k: np.nan for k in keys}
    _fields = [
        "n",
        "median",
        "q10",
        "q25",
        "q75",
        "q90",
        "mean",
        "std",
        "sem",
        "geomean",
        "gm_ci_lo",
        "gm_ci_hi",
        "med_ci_lo",
        "med_ci_hi",
    ]

    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    n = len(data)
    if n == 0:
        d = _nan(_fields)
        d["n"] = 0
        return d

    p10, p25, med, p75, p90 = np.percentile(data, [10, 25, 50, 75, 90])
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n)

    med_lo, med_hi = bootstrap_ci(
        data,
        stat_fn=lambda x: np.median(x, axis=1),
        n_boot=n_boot,
        ci_level=ci_level,
        seed=seed,
    )

    pos = data[data > 0]
    if len(pos) >= 2:
        geomean = float(np.exp(np.mean(np.log(pos))))
        gm_lo, gm_hi = bootstrap_ci(
            pos,
            stat_fn=lambda x: np.exp(np.mean(np.log(np.maximum(x, 1e-300)), axis=1)),
            n_boot=n_boot,
            ci_level=ci_level,
            seed=seed,
        )
    elif len(pos) == 1:
        geomean = gm_lo = gm_hi = float(pos[0])
    else:
        geomean = gm_lo = gm_hi = np.nan

    return dict(
        n=n,
        median=float(med),
        q10=float(p10),
        q25=float(p25),
        q75=float(p75),
        q90=float(p90),
        mean=mean,
        std=std,
        sem=sem,
        geomean=geomean,
        gm_ci_lo=float(gm_lo),
        gm_ci_hi=float(gm_hi),
        med_ci_lo=float(med_lo),
        med_ci_hi=float(med_hi),
    )


def ratio_stats(adaptive_arr, fixed_arr, n_boot=4000, ci_level=95, seed=0):
    """Paired ratio statistics: adaptive / fixed."""
    a = np.asarray(adaptive_arr, dtype=float)
    f = np.asarray(fixed_arr, dtype=float)
    tiny = np.finfo(float).tiny
    mask = np.isfinite(a) & np.isfinite(f) & (f > tiny)
    a, f = a[mask], f[mask]

    if len(a) == 0:
        return {"win_rate": np.nan, "n_trials": 0, "ratio_trials": np.array([])}

    ratio = a / f
    win = a > f
    rs = trial_stats(ratio, n_boot=n_boot, ci_level=ci_level, seed=seed)
    return dict(
        ratio_trials=ratio,
        delta_trials=a - f,
        win_rate=float(np.mean(win)),
        win_count=int(np.sum(win)),
        n_trials=len(ratio),
        **{f"ratio_{k}": v for k, v in rs.items()},
    )


# ─── Publication-grade paired significance testing ────────────────────────────


def _rank_biserial_effect_size(diffs) -> float:
    """
    Matched-pairs rank-biserial correlation, the standard effect-size
    companion to a Wilcoxon-signed-rank-style test (Kerby, 2014, "The
    simple difference formula"): r = (W+ - W-) / (W+ + W-), where W+/W-
    are the sums of the ranks of the positive/negative paired differences.

    r in [-1, 1]; r > 0 means `method` tends to exceed `baseline`.
    Returns 0.0 for the degenerate all-ties case.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diffs))
    w_pos = float(ranks[diffs > 0].sum())
    w_neg = float(ranks[diffs < 0].sum())
    total = w_pos + w_neg
    return (w_pos - w_neg) / total if total > 0 else 0.0


def _wilson_ci(k, n, ci_level=95) -> Tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion k/n.

    Preferred over the normal (Wald) interval used in earlier drafts
    because Wald intervals are unreliable exactly in the low-n /
    near-0-or-1 regime that win-rate statistics fall into here.
    """
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(0.5 + ci_level / 200.0)
    p = k / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


def holm_bonferroni(pvals, alpha=0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Holm-Bonferroni step-down family-wise error correction, implemented
    directly (no `statsmodels` dependency).

    Returns
    -------
    reject : bool ndarray, same order as `pvals`.
    p_adj  : adjusted p-values, same order as `pvals` (monotone
        non-decreasing along the sorted order, as required by the
        Holm procedure).
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    order = np.argsort(pvals)
    sorted_p = pvals[order]

    adj_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for i in range(m):
        candidate = (m - i) * sorted_p[i]
        running_max = max(running_max, candidate)
        adj_sorted[i] = min(running_max, 1.0)

    p_adj = np.empty(m, dtype=float)
    p_adj[order] = adj_sorted
    reject = p_adj < alpha
    return reject, p_adj


def _paired_permutation_test(
    log_diffs: np.ndarray,
    alternative: str = "greater",
    n_perm: int = 20000,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Paired sign-flip (Fisher/randomization) permutation test on
    `log_diffs = log(method) - log(baseline)`.

    Makes no symmetry assumption whatsoever (unlike Wilcoxon): under the
    null of "no systematic difference", each paired log-difference is
    equally likely to carry either sign, so randomly flipping signs and
    recomputing the sum gives an exact, assumption-free reference
    distribution for the observed sum of log-differences.

    Returns
    -------
    observed_stat : float, the observed sum of log-differences.
    p_value : float, one- or two-sided as requested.
    """
    d = np.asarray(log_diffs, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return float("nan"), float("nan")

    obs_stat = float(np.sum(d))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    perm_stats = (signs * d[np.newaxis, :]).sum(axis=1)

    if alternative == "greater":
        p = float((np.sum(perm_stats >= obs_stat) + 1) / (n_perm + 1))
    elif alternative == "less":
        p = float((np.sum(perm_stats <= obs_stat) + 1) / (n_perm + 1))
    else:
        p = float((np.sum(np.abs(perm_stats) >= abs(obs_stat)) + 1) / (n_perm + 1))
    return obs_stat, p


def paired_comparison_stats(
    method_vals,
    baseline_vals,
    label: Optional[str] = None,
    alternative: str = "greater",
    n_boot: int = 4000,
    n_perm: int = 20000,
    ci_level: int = 95,
    seed: int = 0,
    min_valid: int = 10,
) -> dict:
    """
    Rigorous, publication-grade paired-trial comparison of `method_vals`
    against `baseline_vals`, measured on the *same* circuit instances
    (paired design -- this is what makes every test below meaningful,
    rather than just comparing two independent samples).

    Returns a dict containing:
        - median [IQR] for both arms
        - geometric-mean ratio (method / baseline) with a bootstrap 95% CI
        - win rate with a Wilson score CI
        - **primary significance tests, both computed on
        log(method) - log(baseline)** (see module docstring for why):
            * `wilcoxon_log_p_raw`   -- Wilcoxon signed-rank on log-differences
            * `permutation_log_p_raw` -- sign-flip permutation test, same data
        - `wilcoxon_raw_p_raw` -- the original raw-scale Wilcoxon test, kept
        for continuity/comparison, NOT used to drive significance decisions
        - `effect_size_r` -- matched-pairs rank-biserial correlation, computed
        on the same log-differences as the primary tests

    Parameters
    ----------
    method_vals, baseline_vals : array-like, same shape
        Paired per-trial measurements (e.g. fidelity[trial] for BASS vs.
        fixed-basis on the *same* circuit instance and trial index).
    label : str, optional
        Free-text identifier carried through into `build_significance_table`.
    alternative : {"greater", "less", "two-sided"}
        Direction of the one-sided tests; default tests whether
        `method_vals` stochastically dominates `baseline_vals`.
    min_valid : int
        Minimum number of valid paired samples required to run any test.
        Below this, the record is flagged `insufficient_data=True`.

    Returns
    -------
    dict
    """
    m = np.asarray(method_vals, dtype=float)
    b = np.asarray(baseline_vals, dtype=float)
    if m.shape != b.shape:
        raise ValueError(
            "method_vals and baseline_vals must be paired (same shape); "
            f"got {m.shape} vs {b.shape}"
        )

    n_total = len(m)
    valid = np.isfinite(m) & np.isfinite(b) & (m >= 0) & (b >= 0)
    m, b = m[valid], b[valid]
    n_valid = len(m)

    out = {"label": label, "n_total": n_total, "n_valid": n_valid}

    if n_valid < min_valid:
        out["insufficient_data"] = True
        return out
    out["insufficient_data"] = False

    # Median [IQR]
    q_m = np.percentile(m, [50, 25, 75])
    q_b = np.percentile(b, [50, 25, 75])
    out["median_method"], out["q25_method"], out["q75_method"] = (float(x) for x in q_m)
    out["median_baseline"], out["q25_baseline"], out["q75_baseline"] = (
        float(x) for x in q_b
    )

    # Geometric-mean ratio + bootstrap CI. Ratio (and log) undefined at
    # baseline==0 or method==0; report exactly how many pairs were excluded.
    pos = (m > 0) & (b > 0)
    out["n_excluded_zero"] = int(n_valid - int(np.sum(pos)))
    if np.any(pos):
        rs = trial_stats(m[pos] / b[pos], n_boot=n_boot, ci_level=ci_level, seed=seed)
        out["ratio_gm"] = rs["geomean"]
        out["ratio_ci_lo"] = rs["gm_ci_lo"]
        out["ratio_ci_hi"] = rs["gm_ci_hi"]
    else:
        out["ratio_gm"] = out["ratio_ci_lo"] = out["ratio_ci_hi"] = float("nan")

    # Win rate + Wilson CI (uses all valid pairs, including exact zeros --
    # "0 vs positive" is still a perfectly well-defined win/loss).
    wins = int(np.sum(m > b))
    ties = int(np.sum(m == b))
    out["wins"], out["ties"] = wins, ties
    out["win_rate"] = wins / n_valid
    out["win_ci_lo"], out["win_ci_hi"] = _wilson_ci(wins, n_valid, ci_level)

    # --- Primary tests: Wilcoxon + permutation, both on log-differences ---
    if np.any(pos):
        log_diffs = np.log(m[pos]) - np.log(b[pos])
        try:
            w_stat, w_p = stats.wilcoxon(
                np.log(m[pos]),
                np.log(b[pos]),
                alternative=alternative,
                zero_method="wilcox",
            )
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        perm_stat, perm_p = _paired_permutation_test(
            log_diffs, alternative=alternative, n_perm=n_perm, seed=seed
        )
        out["wilcoxon_log_stat"] = (
            float(w_stat) if np.isfinite(w_stat) else float("nan")
        )
        out["wilcoxon_log_p_raw"] = float(w_p)
        out["permutation_log_stat"] = perm_stat
        out["permutation_log_p_raw"] = perm_p
        out["effect_size_r"] = _rank_biserial_effect_size(log_diffs)
    else:
        out["wilcoxon_log_stat"] = out["wilcoxon_log_p_raw"] = float("nan")
        out["permutation_log_stat"] = out["permutation_log_p_raw"] = float("nan")
        out["effect_size_r"] = 0.0

    # --- Secondary / legacy: raw-scale Wilcoxon, kept for continuity only ---
    diffs_raw = m - b
    if np.any(diffs_raw != 0):
        try:
            raw_stat, raw_p = stats.wilcoxon(
                m, b, alternative=alternative, zero_method="wilcox"
            )
        except ValueError:
            raw_stat, raw_p = float("nan"), float("nan")
    else:
        raw_stat, raw_p = float("nan"), 1.0
    out["wilcoxon_raw_stat"] = (
        float(raw_stat) if np.isfinite(raw_stat) else float("nan")
    )
    out["wilcoxon_raw_p_raw"] = float(raw_p)
    out["alternative"] = alternative

    return out


def build_significance_table(
    records: Sequence[dict],
    alpha: float = 0.05,
    correction_on: str = "permutation_log_p_raw",
) -> Tuple[pd.DataFrame, dict]:
    """
    Assemble a list of `paired_comparison_stats()` records into a single
    publication-ready table, with a *family-wise* Holm-Bonferroni
    correction applied across every test in `records` at once.

    Parameters
    ----------
    correction_on : {"permutation_log_p_raw", "wilcoxon_log_p_raw", "wilcoxon_raw_p_raw"}
        Which raw p-value column drives the Holm correction / reject-H0
        decision. Defaults to the assumption-free permutation test (see
        module docstring); the other two p-values are still reported in
        every row for transparency/comparison.

    Records with `insufficient_data=True` are excluded from the correction
    and reported by count (not silently dropped).

    Returns
    -------
    df : pandas.DataFrame
        One row per valid record. Human-readable "X [Y, Z]" summary
        columns for direct inclusion in the manuscript
        (`df.to_latex(...)`, `df.to_markdown(...)`), plus the underlying
        numeric columns (prefixed `_`) for downstream plotting/filtering.
    meta : dict
        {"n_tests", "n_skipped_insufficient_data", "correction_method", "correction_on", "alpha"}
    """
    valid = [r for r in records if not r.get("insufficient_data", False)]
    skipped = [r for r in records if r.get("insufficient_data", False)]

    pvals = [r[correction_on] for r in valid]
    if pvals:
        reject, p_adj = holm_bonferroni(pvals, alpha=alpha)
    else:
        reject, p_adj = np.array([], dtype=bool), np.array([], dtype=float)

    rows = []
    for r, p_c, rej in zip(valid, p_adj, reject):
        rows.append(
            {
                "label": r["label"],
                "n": r["n_valid"],
                "median_method [IQR]": (
                    f"{r['median_method']:.3e} [{r['q25_method']:.3e}, {r['q75_method']:.3e}]"
                ),
                "median_baseline [IQR]": (
                    f"{r['median_baseline']:.3e} [{r['q25_baseline']:.3e}, {r['q75_baseline']:.3e}]"
                ),
                "GM ratio [95% CI]": (
                    f"{r['ratio_gm']:.3g} [{r['ratio_ci_lo']:.3g}, {r['ratio_ci_hi']:.3g}]"
                ),
                "win rate [Wilson 95% CI]": (
                    f"{r['wins']}/{r['n_valid']} [{r['win_ci_lo']:.2f}, {r['win_ci_hi']:.2f}]"
                ),
                "Wilcoxon p (log, raw)": r["wilcoxon_log_p_raw"],
                "Permutation p (log, raw)": r["permutation_log_p_raw"],
                "Wilcoxon p (raw-scale, legacy)": r["wilcoxon_raw_p_raw"],
                f"p ({correction_on}, Holm)": p_c,
                "reject H0": bool(rej),
                "effect size r": round(r["effect_size_r"], 3),
                "n_excluded_zero": r["n_excluded_zero"],
                "_ratio_gm": r["ratio_gm"],
                "_ratio_ci_lo": r["ratio_ci_lo"],
                "_ratio_ci_hi": r["ratio_ci_hi"],
                "_win_rate": r["win_rate"],
            }
        )

    df = pd.DataFrame(rows)
    meta = {
        "n_tests": len(valid),
        "n_skipped_insufficient_data": len(skipped),
        "correction_method": "holm-bonferroni",
        "correction_on": correction_on,
        "alpha": alpha,
    }
    if skipped:
        print(
            f"[build_significance_table] WARNING: {len(skipped)} record(s) skipped "
            f"for insufficient data (< min_valid pairs): {[r['label'] for r in skipped]}"
        )
    return df, meta


def format_significance_table(df: pd.DataFrame, meta: dict) -> str:
    """Monospace console rendering of `build_significance_table` output
    for quick inspection inside a notebook. Use `df.to_latex()` /
    `df.to_markdown()` directly when exporting to the manuscript."""
    header = (
        f"Paired comparisons: {meta['n_tests']} tests "
        f"({meta['correction_method']} on {meta['correction_on']}, "
        f"family-wise alpha={meta['alpha']}); "
        f"{meta['n_skipped_insufficient_data']} skipped (insufficient data)"
    )
    display_cols = [c for c in df.columns if not c.startswith("_")]
    return header + "\n" + "-" * 100 + "\n" + df[display_cols].to_string(index=False)
