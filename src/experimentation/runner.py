r"""
QuantumBenchmarkRunner
======================
The central benchmarking and evaluation framework for comparative verification
of the Basis-Adaptive Sparse Simulator (BASS) against standard Fixed-Basis
and Exact statevector simulation baselines.

This module acts as a strict, unified experiment engine designed to guarantee
full code reproducibility and eliminate any structural, circuit, or statistical
variations between competing simulator paradigms. All automated multi-trial evaluation loops,
system-size scaling sweeps, and publication-ready tables are channeled through this backend.

Benchmark Circuit Topologies & Ansatzes
---------------------------------------
    1. 1D & 2D Brickwork Fabrics (`make_brickwork`, `make_brickwork_2d`)
        Generates hardware-efficient chaotic circuits consisting of alternating layers
        of Haar-random 2-qubit gates. The 2D grid utilizes row-major indexing with a period-4
        spatial cycle (horizontal even/odd, vertical even/odd column/row pairings) to ensure
        rapid, isotropic propagation of multi-qubit entanglement across the planar geometry.

    2. Deterministic Quantum Fourier Transform (`make_qft`)
        Constructs standard QFT cascades bounded to the nearest 4 qubits for execution speed.
        Serves as a deterministic structural baseline with long-range phase-rotation patterns.

    3. Correlated Condensed-Matter Ensembles (`make_tfim`, `make_rfim`)
        Generates disordered Transverse-Field and Longitudinal Random-Field Ising Model
        Trotter steps. The RFIM variant maps static, site-dependent longitudinal fields
        (\Delta_i \sim U[-W, W]) fixed across a single circuit realization to investigate
        the boundaries of Many-Body Localization (MBL) and Ergodic phase crossovers.

    4. Chemistry & Variational Ansatzes (`make_uccsd`, `make_qaoa`)
        - UCCSD: Prepares a single computational-basis Hartree-Fock reference followed by
            Jordan-Wigner encoded singles/doubles excitation cascades. Amplitudes stay bounded,
            verifying that BASS elegantly reverts to fixed-basis sparse simulation when states
            remain naturally Z-sparse.
        - QAOA MaxCut: Implements Farhi-ansatz levels on uniformly random 3-regular graphs
            generated via the Bollobás pairing model. Angles follow near-optimal Fourier
            schedules (quarter-period sinusoidal ramps) to closely approximate physical multi-variable
            optimization landscapes rather than unguided chaotic random-angle walks.

Key functions
-------------
    run_trial_fixed                -- one fixed basis trial against exact reference
    run_trial_bass                 -- one BASS trial against exact reference
    run_trial_timed                -- one self-consistent timed trial isolating simulation runtime
    run_sweep_k                    -- sweep k values for one or more simulators (averaged over trials)
    run_sweep_full                 -- comprehensive k-sweep collecting full per-trial statistical data
    run_n_scaling_full             -- system size (N) scaling benchmark with robust per-trial metrics
    run_comparison                 -- multi-trial comparison (fixed vs BASS) with aggregated statistics
    compute_metrics                -- robust, reviewer-proof statistics (Median/IQR, Geomean/Bootstrap CI)
    print_summary_table            -- publication-ready formatted console output for comparisons
    print_sweep_table              -- publication-ready table printing for full k-sweeps (Fidelity/PR)
    print_nscaling_table           -- publication-ready table printing for N-scaling benchmarks
"""

import time
import numpy as np
import pandas as pd
from scipy import stats

from src.core.seeding import ExperimentSeeder
from src.simulation.simulator import FixedBasisSimulator
from src.simulation.bass_simulator import BASS
from src.simulation.exact_simulator import ExactSimulator
from src.benchmarking.fidelity import compute_fidelity, compute_participation_ratio
from src.utils.random_circuits import (
    generate_random_circuit,
    generate_tfim_circuit,
    generate_rfim_circuit,
)
from src.core.gates import (
    TwoQubitGate,
    HGate,
    RXGate,
    RandomTwoQubitGate,
    RZGate,
    CNOTGate,
    XGate,
    RZZGate,
    SingleQubitGate,
)

# ── Exact reference ────────────────────────────────────────────────────────────


def exact_statevector(N, circuit):
    """Run exact simulation and return dense statevector. N must be ≤ 24."""
    return ExactSimulator(N, verbose=False).simulate(circuit)


def fidelity_bass(sim, state, exact_sv):
    """Extract full statevector from BASS to compute true fidelity overlap."""
    if sim.N > 24:
        return 0.0
    psi_approx = sim.to_statevector(state)
    overlap = np.vdot(exact_sv, psi_approx)
    return float(np.abs(overlap) ** 2)


# ── Statistics & Aggregation ───────────────────────────────────────────────────


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

    # 1. Percentiles (Robust to all skew/chaos)
    p10, p25, median, p75, p90 = np.percentile(data, [10, 25, 50, 75, 90])

    # 2. Arithmetic Statistics (For expectation values)
    arithmetic_mean = np.mean(data)
    sem = stats.sem(data) if len(data) > 1 else 0.0

    # 3. Geometric Statistics (For log-normal data: PR, Ratios)
    positive_data = data[data > 0]
    if len(positive_data) > 0:
        log_data = np.log(positive_data)
        geometric_mean = np.exp(np.mean(log_data))

        if len(positive_data) > 1:
            # Fast vectorized bootstrap for 95% CI
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


# ── Single-trial execution ─────────────────────────────────────────────────────
def run_trial_fixed(N, k, circuit, exact_sv=None):
    """Run one fixed-basis simulation."""
    sim = FixedBasisSimulator(N, k, verbose=False)
    t0 = time.perf_counter()
    state = sim.simulate(circuit)
    dt = time.perf_counter() - t0

    metrics = {"runtime_s": dt}
    if exact_sv is not None:
        metrics["fidelity"] = compute_fidelity(state, exact_sv)
    metrics["pr"] = compute_participation_ratio(state)
    return metrics


def run_trial(circuit_gen, N, k, seed=0, bass_kwargs=None, _exact_sv=None):
    """
    One independent trial.

    Parameters
    ----------
    circuit_gen : callable (N, rng) -> list[Gate]
    N           : system size
    k           : sparse budget
    seed        : RNG seed
    bass_kwargs : extra kwargs forwarded to BASS constructor
    _exact_sv   : pre-computed exact statevector (optional; avoids redundant
                exact simulation when the circuit is deterministic)

    Returns
    -------
    f_fixed, f_bass : float fidelities
    pr_fixed, pr_bass : float participation ratios (final sparse states)
    circuit : the circuit that was simulated (needed by caller to share exact_sv)
    """
    if bass_kwargs is None:
        bass_kwargs = {}

    rng = np.random.default_rng(seed)
    circuit = circuit_gen(N, rng)
    if _exact_sv is None:
        _exact_sv = exact_statevector(N, circuit)

    # fixedbasis simulator
    fixed_sim = FixedBasisSimulator(N, k, verbose=False)
    fixed_state = fixed_sim.simulate(circuit)
    f_fixed = compute_fidelity(fixed_state, _exact_sv)
    pr_fixed = compute_participation_ratio(fixed_state)

    # BASS
    BassClass = BASS
    bass_sim = BassClass(N, k, **bass_kwargs)
    b_state = bass_sim.simulate(circuit)
    f_bass = fidelity_bass(bass_sim, b_state, _exact_sv)
    pr_bass = compute_participation_ratio(b_state)

    return float(f_fixed), float(f_bass), float(pr_fixed), float(pr_bass)


# made an independent function for legacy tests files
def run_trial_timed(
    circuit_gen,
    N,
    k,
    seed=0,
    bass_kwargs=None,
    _exact_sv=None,
):
    """
    Run one fully self-consistent trial for FixedBasis and BASS.

    IMPORTANT
    ---------
    - The SAME circuit instance is used for:
        * exact simulation
        * FixedBasis simulation
        * BASS simulation
    - Timing and fidelity metrics are computed from the SAME simulator run.
    - No simulator is re-run for metrics after timing.
    - This avoids methodological inconsistencies in benchmarking.

    Parameters
    ----------
    circuit_gen : callable
        Signature: circuit_gen(N, rng) -> list[Gate]

    N : int
        System size.

    k : int
        Sparse budget.

    seed : int, default=0
        RNG seed for reproducibility.

    bass_kwargs : dict or None
        Extra kwargs forwarded to BASS constructor.

    _exact_sv : np.ndarray or None
        Optional precomputed exact statevector.
        Useful when the circuit is deterministic and reused across trials.

    Returns
    -------
    results : dict
        {
            "f_fixed"     : float,
            "f_bass"      : float,
            "pr_fixed"    : float,
            "pr_bass"     : float,
            "t_fixed"     : float,
            "t_bass"      : float,
            "circuit"     : object,
        }

    Notes
    -----
    Timing measurements:
    - Measured using time.perf_counter().
    - Timing includes ONLY the simulate(...) call.
    - Constructor overhead, fidelity evaluation, and PR computation
        are excluded to isolate simulator runtime.
    """

    if bass_kwargs is None:
        bass_kwargs = {}

    rng = np.random.default_rng(seed)

    # Generate ONE circuit for the entire trial
    circuit = circuit_gen(N, rng)

    # Exact reference state
    if _exact_sv is None:
        _exact_sv = exact_statevector(N, circuit)

    # FixedBasis simulator
    fixed_sim = FixedBasisSimulator(N, k, verbose=False)
    t0 = time.perf_counter()
    fixed_state = fixed_sim.simulate(circuit)
    t_fixed = time.perf_counter() - t0

    # Metrics computed from SAME run
    f_fixed = compute_fidelity(fixed_state, _exact_sv)
    pr_fixed = compute_participation_ratio(fixed_state)

    # BASS simulator
    bass_sim = BASS(N, k, **bass_kwargs)
    t0 = time.perf_counter()
    b_state = bass_sim.simulate(circuit)
    t_bass = time.perf_counter() - t0

    # Metrics computed from SAME run
    f_bass = fidelity_bass(bass_sim, b_state, _exact_sv)
    pr_bass = compute_participation_ratio(b_state)

    return {
        "f_fixed": float(f_fixed),
        "f_bass": float(f_bass),
        "pr_fixed": float(pr_fixed),
        "pr_bass": float(pr_bass),
        "t_fixed": float(t_fixed),
        "t_bass": float(t_bass),
        "circuit": circuit,
    }


def run_trial_bass(N, k, circuit, exact_sv=None, **bass_kwargs):
    """Run one BASS simulation."""
    sim = BASS(N, k, verbose=False, **bass_kwargs)
    t0 = time.perf_counter()
    state = sim.simulate(circuit)
    dt = time.perf_counter() - t0

    metrics = {"runtime_s": dt}
    if exact_sv is not None:
        metrics["fidelity"] = fidelity_bass(sim, state, exact_sv)
    metrics["pr"] = compute_participation_ratio(state)
    return metrics


# ── Multi-trial comparison ─────────────────────────────────────────────────────


def run_comparison(N, k, depth, n_trials, circuit_type="brickwork", bass_kwargs=None):
    """
    Run n_trials independent circuits, compare Fixed vs BASS,
    and aggregate results using robust non-parametric metrics.
    """
    if bass_kwargs is None:
        bass_kwargs = {}

    f_fixed_arr = np.zeros(n_trials)
    f_bass_arr = np.zeros(n_trials)
    pr_fixed_arr = np.zeros(n_trials)
    pr_bass_arr = np.zeros(n_trials)
    ratios_arr = np.zeros(n_trials)

    rng = np.random.default_rng(42)

    for i in range(n_trials):
        if circuit_type == "brickwork":
            circ = make_brickwork(N, depth, rng)
        else:
            circ = generate_random_circuit(N, depth, rng=rng)

        exact_sv = exact_statevector(N, circ)

        res_fix = run_trial_fixed(N, k, circ, exact_sv)
        res_bas = run_trial_bass(N, k, circ, exact_sv, **bass_kwargs)

        f_fixed_arr[i] = res_fix["fidelity"]
        f_bass_arr[i] = res_bas["fidelity"]
        pr_fixed_arr[i] = res_fix["pr"]
        pr_bass_arr[i] = res_bas["pr"]

        if res_fix["fidelity"] > 1e-12:
            ratios_arr[i] = res_bas["fidelity"] / res_fix["fidelity"]
        else:
            ratios_arr[i] = np.nan

    wins = np.sum(f_bass_arr > f_fixed_arr)
    win_rate = wins / n_trials

    # Flatten the robust metrics into the result dictionary
    result = {
        "N": N,
        "k": k,
        "depth": depth,
        "trials": n_trials,
        "win_rate": float(win_rate),
    }

    metric_groups = [
        ("f_fixed", f_fixed_arr),
        ("f_bass", f_bass_arr),
        ("pr_fixed", pr_fixed_arr),
        ("pr_bass", pr_bass_arr),
        ("ratio", ratios_arr),
    ]

    for prefix, arr in metric_groups:
        stats_dict = compute_metrics(arr)
        for key, val in stats_dict.items():
            result[f"{prefix}_{key}"] = val

    return result


# ── k-sweep ────────────────────────────────────────────────────────────────────


def run_sweep_k(
    circuit_gen,
    N,
    k_values,
    n_trials=1,
    seed_base=0,
    bass_kwargs=None,
    verbose=True,
    cache_exact=True,
    exp: ExperimentSeeder = None,
    family_label: str = None,
):
    """
    Sweep k values, averaging over n_trials independent circuits.

    cache_exact=True (default): compute the exact statevector once per trial
    (reusing across k values) by pre-generating the circuit at seed 0.
    Set False for stochastic circuits where exact state depends on k.

    Seeding
    -------
    If `exp` (an `ExperimentSeeder`) and `family_label` are given, each
    trial's circuit is seeded via `exp.seq(family_label, "sweepk", t)` --
    collision-free and order-independent, per `src.core.seeding`. If `exp`
    is omitted, falls back to the legacy `seed_base * 10**6 + t` arithmetic
    for backward compatibility with any pre-existing caller.

    Returns
    -------
    dict with keys k_values, f_fixed, f_bass, pr_fixed, pr_bass
    Each value is a list of length len(k_values).
    """
    if bass_kwargs is None:
        bass_kwargs = {}

    def _seed_for(t):
        if exp is not None:
            return exp.seq(family_label or "run_sweep_k", "sweepk", t)
        return seed_base * 10**6 + t

    results = {
        "k_values": k_values,
        "f_fixed": [],
        "f_bass": [],
        "pr_fixed": [],
        "pr_bass": [],
    }

    # Pre-compute exact state vectors per trial (shared across k-values)
    _exact_cache = {}
    if cache_exact:
        for t in range(n_trials):
            seed = _seed_for(t)
            rng = np.random.default_rng(seed)
            circ = circuit_gen(N, rng)
            if verbose:
                print(
                    f"  [exact N={N}] computing statevector for trial {t}...",
                    flush=True,
                )
            _exact_cache[t] = exact_statevector(N, circ)
            if verbose:
                print(f"  [exact] done", flush=True)

    for k in k_values:
        ft_list, fb_list, prt_list, prb_list = [], [], [], []
        for t in range(n_trials):
            seed = _seed_for(t)
            esv = _exact_cache.get(t)  # None if cache_exact=False
            f_fixed, f_bass, pr_fixed, pr_bass = run_trial(
                circuit_gen, N, k, seed=seed, bass_kwargs=bass_kwargs, _exact_sv=esv
            )
            ft_list.append(f_fixed)
            fb_list.append(f_bass)
            prt_list.append(pr_fixed)
            prb_list.append(pr_bass)
            if verbose:
                ratio = f_bass / f_fixed if f_fixed > 1e-30 else float("nan")
                print(
                    f"  k={k:>8,}  trial={t}  "
                    f"FixedBasis={f_fixed:.4f}  BASS={f_bass:.4f}  ratio={ratio:.2f}x  "
                    f"pr_fixed={pr_fixed:.0f}  pr_bass={pr_bass:.0f}",
                    flush=True,
                )

        # BUG FIX: paper uses geometric means throughout (fidelity is
        # log-normally distributed); arithmetic means here were inconsistent.
        def _geomean(arr):
            a = np.asarray(arr, dtype=float)
            a = a[a > 0]
            return float(np.exp(np.mean(np.log(a)))) if len(a) else 0.0

        results["f_fixed"].append(_geomean(ft_list))
        results["f_bass"].append(_geomean(fb_list))
        results["pr_fixed"].append(_geomean(prt_list))
        results["pr_bass"].append(_geomean(prb_list))

    return results


# ─── Full Sweep with Per-Trial Data ──────────────────────────────────────────


def run_n_scaling_full(
    circuit_gen,
    N_values,
    k_fn,
    n_trials,
    seed_base,
    bass_kwargs=None,
    verbose=True,
    n_boot=4000,
    exp: ExperimentSeeder = None,
    family_label: str = None,
):
    """
    N-scaling benchmark.

    Seeding: if `exp` (an `ExperimentSeeder`) and `family_label` are given,
    each (N, trial) circuit is seeded via
    `exp.seq(family_label, "nscaling", N, t)` -- collision-free and
    order-independent (see `src.core.seeding`). If `exp` is omitted, falls
    back to the legacy `seed_base * 10**6 + ni*1000 + t` arithmetic for
    backward compatibility.
    """

    if bass_kwargs is None:
        bass_kwargs = {}

    def _seed_for(ni, N, t):
        if exp is not None:
            return exp.seq(family_label or "run_n_scaling_full", "nscaling", N, t)
        return seed_base * 10**6 + ni * 1000 + t

    nN = len(N_values)

    k_values = [int(k_fn(N)) for N in N_values]

    f_fixed = np.full((n_trials, nN), np.nan)
    f_bass = np.full((n_trials, nN), np.nan)

    pr_fixed = np.full((n_trials, nN), np.nan)
    pr_bass = np.full((n_trials, nN), np.nan)

    for ni, N in enumerate(N_values):

        k = k_values[ni]

        exact_svs = []

        for t in range(n_trials):

            seed = _seed_for(ni, N, t)

            circ = circuit_gen(
                N,
                np.random.default_rng(seed),
            )

            exact_svs.append(exact_statevector(N, circ))

        for t in range(n_trials):

            seed = _seed_for(ni, N, t)

            ff, fb, prf, prb = run_trial(
                circuit_gen,
                N,
                k,
                seed=seed,
                bass_kwargs=bass_kwargs,
                _exact_sv=exact_svs[t],
            )

            f_fixed[t, ni] = ff
            f_bass[t, ni] = fb

            pr_fixed[t, ni] = prf
            pr_bass[t, ni] = prb

        if verbose:

            ratio_arr = f_bass[:, ni] / np.maximum(f_fixed[:, ni], 1e-300)

            pos = ratio_arr[ratio_arr > 0]

            gm = float(np.exp(np.mean(np.log(pos)))) if len(pos) else np.nan

            win = float(np.mean(f_bass[:, ni] > f_fixed[:, ni]))

            print(
                f"  N={N} k={k}: " f"ratio_gm={gm:.3f}x  " f"win={100*win:.0f}%",
                flush=True,
            )

    stats = {}

    for ni, N in enumerate(N_values):

        ratio = f_bass[:, ni] / np.maximum(f_fixed[:, ni], 1e-300)

        stats[N] = {
            "f_fixed": trial_stats(
                f_fixed[:, ni],
                n_boot=n_boot,
            ),
            "f_bass": trial_stats(
                f_bass[:, ni],
                n_boot=n_boot,
            ),
            "pr_fixed": trial_stats(
                pr_fixed[:, ni],
                n_boot=n_boot,
            ),
            "pr_bass": trial_stats(
                pr_bass[:, ni],
                n_boot=n_boot,
            ),
            "ratio": trial_stats(
                ratio,
                n_boot=n_boot,
            ),
        }

    return dict(
        N_values=N_values,
        k_values=k_values,
        f_fixed=f_fixed,
        f_bass=f_bass,
        pr_fixed=pr_fixed,
        pr_bass=pr_bass,
        stats=stats,
    )


def run_sweep_full(
    circuit_gen,
    N,
    k_values,
    n_trials,
    seed_base,
    bass_kwargs=None,
    verbose=True,
    n_boot=4000,
    exp: ExperimentSeeder = None,
    family_label: str = None,
):
    """
    Sweep sparse budget k over n_trials independent circuits.

    Seeding: if `exp` (an `ExperimentSeeder`) and `family_label` are given,
    each trial's circuit is seeded via `exp.seq(family_label, "sweepfull", t)`
    -- collision-free and order-independent (see `src.core.seeding`). If
    `exp` is omitted, falls back to the legacy `seed_base * 10**6 + t`
    arithmetic for backward compatibility.
    """

    if bass_kwargs is None:
        bass_kwargs = {}

    def _seed_for(t):
        if exp is not None:
            return exp.seq(family_label or "run_sweep_full", "sweepfull", t)
        return seed_base * 10**6 + t

    k_values = np.asarray(k_values, dtype=int)

    nk = len(k_values)

    f_fixed = np.full((n_trials, nk), np.nan)
    f_bass = np.full((n_trials, nk), np.nan)

    pr_fixed = np.full((n_trials, nk), np.nan)
    pr_bass = np.full((n_trials, nk), np.nan)

    exact_svs = []

    for t in range(n_trials):

        seed = _seed_for(t)

        circ = circuit_gen(
            N,
            np.random.default_rng(seed),
        )

        exact_svs.append(exact_statevector(N, circ))

        if verbose:
            print(
                f"  [exact] trial {t}/{n_trials}",
                flush=True,
            )

    for ki, k in enumerate(k_values):

        for t in range(n_trials):

            seed = _seed_for(t)

            ff, fb, prf, prb = run_trial(
                circuit_gen,
                N,
                int(k),
                seed=seed,
                bass_kwargs=bass_kwargs,
                _exact_sv=exact_svs[t],
            )

            f_fixed[t, ki] = ff
            f_bass[t, ki] = fb

            pr_fixed[t, ki] = prf
            pr_bass[t, ki] = prb

        if verbose:

            ratio_arr = f_bass[:, ki] / np.maximum(f_fixed[:, ki], 1e-300)

            pos = ratio_arr[ratio_arr > 0]

            gm = float(np.exp(np.mean(np.log(pos)))) if len(pos) else np.nan

            win = np.mean(f_bass[:, ki] > f_fixed[:, ki])

            print(
                f"  k={int(k):>8,}: " f"ratio_gm={gm:.3f}x  " f"win={100*win:.0f}%",
                flush=True,
            )

    stats = {}

    for ki, k in enumerate(k_values):

        ratio = f_bass[:, ki] / np.maximum(f_fixed[:, ki], 1e-300)

        stats[int(k)] = {
            "f_fixed": trial_stats(
                f_fixed[:, ki],
                n_boot=n_boot,
            ),
            "f_bass": trial_stats(
                f_bass[:, ki],
                n_boot=n_boot,
            ),
            "pr_fixed": trial_stats(
                pr_fixed[:, ki],
                n_boot=n_boot,
            ),
            "pr_bass": trial_stats(
                pr_bass[:, ki],
                n_boot=n_boot,
            ),
            "ratio": trial_stats(
                ratio,
                n_boot=n_boot,
            ),
        }

    return dict(
        k_values=k_values,
        f_fixed=f_fixed,
        f_bass=f_bass,
        pr_fixed=pr_fixed,
        pr_bass=pr_bass,
        stats=stats,
    )


# ─── Rigorous Statistics ─────────────────────────────────────


def bootstrap_ci(data, stat_fn=None, n_boot=4000, ci_level=95, seed=0):
    """
    Non-parametric bootstrap confidence interval for any statistic.
    """
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
    """
    Full distribution summary for a 1-D array of trial measurements.
    """
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
            stat_fn=lambda x: np.exp(
                np.mean(
                    np.log(np.maximum(x, 1e-300)),
                    axis=1,
                )
            ),
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


def ratio_stats(
    adaptive_arr,
    fixed_arr,
    n_boot=4000,
    ci_level=95,
    seed=0,
):
    """
    Paired ratio statistics: adaptive / fixed.
    """
    a = np.asarray(adaptive_arr, dtype=float)
    f = np.asarray(fixed_arr, dtype=float)

    tiny = np.finfo(float).tiny

    mask = np.isfinite(a) & np.isfinite(f) & (f > tiny)

    a, f = a[mask], f[mask]

    if len(a) == 0:
        return {
            "win_rate": np.nan,
            "n_trials": 0,
            "ratio_trials": np.array([]),
        }

    ratio = a / f
    win = a > f

    rs = trial_stats(
        ratio,
        n_boot=n_boot,
        ci_level=ci_level,
        seed=seed,
    )

    return dict(
        ratio_trials=ratio,
        delta_trials=a - f,
        win_rate=float(np.mean(win)),
        win_count=int(np.sum(win)),
        n_trials=len(ratio),
        **{f"ratio_{k}": v for k, v in rs.items()},
    )


def _rank_biserial_effect_size(diffs):
    """
    Matched-pairs rank-biserial correlation, the standard effect-size
    companion to a Wilcoxon signed-rank test (Kerby, 2014, "The simple
    difference formula"): r = (W+ - W-) / (W+ + W-), where W+/W- are the
    sums of the ranks of the positive/negative paired differences.

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


def _wilson_ci(k, n, ci_level=95):
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


def holm_bonferroni(pvals, alpha=0.05):
    """
    Holm-Bonferroni step-down family-wise error correction, implemented
    directly (no `statsmodels` dependency).

    Parameters
    ----------
    pvals : array-like of raw p-values, in whatever order you want the
        output aligned to.
    alpha : family-wise significance level.

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


def paired_comparison_stats(
    method_vals,
    baseline_vals,
    label=None,
    alternative="greater",
    n_boot=4000,
    ci_level=95,
    seed=0,
    min_valid=10,
):
    """
    Parameters
    ----------
    method_vals, baseline_vals : array-like, same shape
        Paired per-trial measurements (e.g. fidelity[trial] for BASS vs.
        fixed-basis on the *same* circuit instance and same trial index).
    label : str, optional
        Free-text identifier (e.g. "Brickwork, k=2000") carried through
        into `build_significance_table`.
    alternative : {"greater", "less", "two-sided"}
        Direction of the one-sided Wilcoxon test; default tests whether
        `method_vals` stochastically dominates `baseline_vals`.
    min_valid : int
        Minimum number of valid paired samples required to run the test.
        Below this, the record is flagged `insufficient_data=True` and
        no test is attempted (rather than reporting a spurious p-value
        from a handful of pairs).

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

    # Median [IQR] -- robust to the heavy tails seen throughout this project
    q_m = np.percentile(m, [50, 25, 75])
    q_b = np.percentile(b, [50, 25, 75])
    out["median_method"], out["q25_method"], out["q75_method"] = (float(x) for x in q_m)
    out["median_baseline"], out["q25_baseline"], out["q75_baseline"] = (
        float(x) for x in q_b
    )

    # Geometric-mean ratio + bootstrap CI. Ratio undefined at baseline==0;
    # we report exactly how many pairs were excluded rather than silently
    # dropping them from the estimate.
    pos = (m > 0) & (b > 0)
    out["n_excluded_zero"] = int(n_valid - int(np.sum(pos)))
    if np.any(pos):
        rs = trial_stats(m[pos] / b[pos], n_boot=n_boot, ci_level=ci_level, seed=seed)
        out["ratio_gm"] = rs["geomean"]
        out["ratio_ci_lo"] = rs["gm_ci_lo"]
        out["ratio_ci_hi"] = rs["gm_ci_hi"]
    else:
        out["ratio_gm"] = out["ratio_ci_lo"] = out["ratio_ci_hi"] = float("nan")

    # Win rate + Wilson CI
    wins = int(np.sum(m > b))
    ties = int(np.sum(m == b))
    out["wins"], out["ties"] = wins, ties
    out["win_rate"] = wins / n_valid
    out["win_ci_lo"], out["win_ci_hi"] = _wilson_ci(wins, n_valid, ci_level)

    # Paired Wilcoxon signed-rank test + effect size
    diffs = m - b
    if np.any(diffs != 0):
        try:
            stat, p = stats.wilcoxon(
                m, b, alternative=alternative, zero_method="wilcox"
            )
        except ValueError:
            stat, p = float("nan"), float("nan")
    else:
        stat, p = float("nan"), 1.0
    out["wilcoxon_stat"] = float(stat) if np.isfinite(stat) else float("nan")
    out["wilcoxon_p_raw"] = float(p)
    out["effect_size_r"] = _rank_biserial_effect_size(diffs)
    out["alternative"] = alternative

    return out


def build_significance_table(records, alpha=0.05):
    """
    Assemble a list of `paired_comparison_stats()` records into a single
    table, with a *family-wise* Holm-Bonferroni
    correction applied across every test in `records` at once (correcting
    each record independently would understate the true false-positive
    risk when many (family, k) configurations are tested together, which
    is exactly the situation in every table in this paper).

    Records with `insufficient_data=True` are excluded from the
    correction and reported by count (not silently dropped).

    Returns
    -------
    df : pandas.DataFrame
        One row per valid record. Human-readable "X [Y, Z]" summary
        columns for direct inclusion in the manuscript
        (`df.to_latex(...)`, `df.to_markdown(...)`), plus the underlying
        numeric columns for downstream plotting/filtering.
    meta : dict
        {"n_tests", "n_skipped_insufficient_data", "correction_method", "alpha"}
    """
    valid = [r for r in records if not r.get("insufficient_data", False)]
    skipped = [r for r in records if r.get("insufficient_data", False)]

    pvals = [r["wilcoxon_p_raw"] for r in valid]
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
                    f"{r['median_method']:.3e} "
                    f"[{r['q25_method']:.3e}, {r['q75_method']:.3e}]"
                ),
                "median_baseline [IQR]": (
                    f"{r['median_baseline']:.3e} "
                    f"[{r['q25_baseline']:.3e}, {r['q75_baseline']:.3e}]"
                ),
                "GM ratio [95% CI]": (
                    f"{r['ratio_gm']:.3g} [{r['ratio_ci_lo']:.3g}, {r['ratio_ci_hi']:.3g}]"
                ),
                "win rate [Wilson 95% CI]": (
                    f"{r['wins']}/{r['n_valid']} "
                    f"[{r['win_ci_lo']:.2f}, {r['win_ci_hi']:.2f}]"
                ),
                "Wilcoxon p (raw)": r["wilcoxon_p_raw"],
                "Wilcoxon p (Holm)": p_c,
                "reject H0": bool(rej),
                "effect size r": round(r["effect_size_r"], 3),
                "n_excluded_zero": r["n_excluded_zero"],
                # numeric duplicates for plotting without re-parsing strings
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
        "alpha": alpha,
    }
    if skipped:
        print(
            f"[build_significance_table] WARNING: {len(skipped)} record(s) skipped "
            f"for insufficient data (< min_valid pairs): "
            f"{[r['label'] for r in skipped]}"
        )
    return df, meta


def format_significance_table(df, meta):
    """Monospace console rendering of `build_significance_table` output
    for quick inspection inside a notebook."""
    header = (
        f"Paired comparisons: {meta['n_tests']} tests "
        f"({meta['correction_method']}, family-wise alpha={meta['alpha']}); "
        f"{meta['n_skipped_insufficient_data']} skipped (insufficient data)"
    )
    display_cols = [c for c in df.columns if not c.startswith("_")]
    return header + "\n" + "-" * 100 + "\n" + df[display_cols].to_string(index=False)


# ─── Plot Helpers ─────────────────────────────────────────────────────────────


def add_ci_bands(
    ax,
    x_vals,
    stats_by_x,
    color,
    alpha_outer=0.08,
    alpha_mid=0.17,
    alpha_inner=0.32,
    use_geomean=False,
    zorder=1,
    clip_lo=1e-100,
):
    """
    Add three nested shaded uncertainty bands to ax.
    """

    xs = []
    lo10 = []
    hi10 = []
    lo25 = []
    hi25 = []
    cilo = []
    cihi = []

    for x in x_vals:

        s = None

        for key in [
            x,
            int(x) if isinstance(x, float) else x,
            str(x),
        ]:
            if key in stats_by_x:
                s = stats_by_x[key]
                break

        if s is None or np.isnan(s.get("median", np.nan)):
            continue

        xs.append(x)

        lo10.append(max(s["q10"], clip_lo))
        hi10.append(max(s["q90"], clip_lo))

        lo25.append(max(s["q25"], clip_lo))
        hi25.append(max(s["q75"], clip_lo))

        if use_geomean:

            cilo.append(
                max(
                    s.get("gm_ci_lo", s["q25"]),
                    clip_lo,
                )
            )

            cihi.append(
                max(
                    s.get("gm_ci_hi", s["q75"]),
                    clip_lo,
                )
            )

        else:

            cilo.append(
                max(
                    s.get("med_ci_lo", s["q25"]),
                    clip_lo,
                )
            )

            cihi.append(
                max(
                    s.get("med_ci_hi", s["q75"]),
                    clip_lo,
                )
            )

    if not xs:
        return

    xs = np.asarray(xs)

    kw = dict(linewidth=0)

    ax.fill_between(
        xs,
        lo10,
        hi10,
        color=color,
        alpha=alpha_outer,
        zorder=zorder,
        **kw,
    )

    ax.fill_between(
        xs,
        lo25,
        hi25,
        color=color,
        alpha=alpha_mid,
        zorder=zorder + 1,
        **kw,
    )

    ax.fill_between(
        xs,
        cilo,
        cihi,
        color=color,
        alpha=alpha_inner,
        zorder=zorder + 2,
        **kw,
    )


def confidence_ellipse(
    x_data,
    y_data,
    ax,
    n_std=2.0,
    color="black",
    alpha=0.18,
    linewidth=1.0,
    linestyle="-",
):
    """
    Draw covariance ellipse for 2-D data.
    """

    from matplotlib.patches import Ellipse

    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    mask = np.isfinite(x_data) & np.isfinite(y_data)

    x_data = x_data[mask]
    y_data = y_data[mask]

    if len(x_data) < 3:
        return

    cov = np.cov(x_data, y_data)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = eigenvalues.argsort()[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    width, height = 2 * n_std * np.sqrt(np.abs(eigenvalues))

    ellipse = Ellipse(
        xy=(
            np.mean(x_data),
            np.mean(y_data),
        ),
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        alpha=alpha,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=2,
    )

    ax.add_patch(ellipse)


# ─── Publication Table Printing ──────────────────────────────────────────────


def print_sweep_table(
    k_values,
    family_results,
    title,
    metric="fidelity",
):
    """
    Print benchmark table with mathematically
    consistent ratios and win rates for both fidelity and PR metrics.

    metric:
        "fidelity" -> uses f_fixed / f_bass, ratio = BASS / Fixed (>1 is good)
        "pr"       -> uses pr_fixed / pr_bass, ratio = Fixed / BASS (>1 is good compression)
    """
    import numpy as np

    if metric == "fidelity":
        fixed_key = "f_fixed"
        bass_key = "f_bass"
        ratio_key = "ratio"  # Maps to f_bass / f_fixed
        ratio_label = "Ratio gm[CI95]"
        title = "Fidelity vs k"

    elif metric == "pr":
        fixed_key = "pr_fixed"
        bass_key = "pr_bass"
        ratio_key = (
            "pr_ratio"  # Adjust this to match your pipeline's key if saved separately,
        )
        # otherwise we calculate it below on-the-fly to be safe.
        ratio_label = "Comp. Factor gm[CI95]"
        title = "Participation Ratio vs k"

    else:
        raise ValueError(f"Unknown metric: {metric}")

    hdr = (
        f"{'Family':<18} {'k':>8}  "
        f"{'Fixed med[q25,q75]':<24}  "
        f"{'BASS med[q25,q75]':<24}  "
        f"{ratio_label:<22}  "
        f"{'Win%':>6}"
    )

    print(f"\n{'=' * len(hdr)}")
    print(title)
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    for label, res in family_results.items():
        for ki, k in enumerate(k_values):
            s = res["stats"].get(int(k), {})

            fixed_stats = s.get(fixed_key, {})
            bass_stats = s.get(bass_key, {})

            fixed_arr = res[fixed_key][:, ki]
            bass_arr = res[bass_key][:, ki]

            # BUG FIX: Win metrics are inverted for Participation Ratio
            if metric == "fidelity":
                win = float(np.mean(bass_arr > fixed_arr))
                rat = s.get(ratio_key, {})
            else:
                # BASS wins when it localized/compressed the state more than Fixed basis
                win = float(np.mean(bass_arr < fixed_arr))

                # BUG FIX: Recalculate geometric mean on-the-fly for PR to ensure
                # Ratio = Fixed / BASS (Compression factor). Prevents key-mismatch slop.
                with np.errstate(divide="ignore", invalid="ignore"):
                    pr_ratios = fixed_arr / bass_arr
                    # Exclude any NaNs or Infs from bad runs
                    valid_ratios = pr_ratios[np.isfinite(pr_ratios) & (pr_ratios > 0)]

                if len(valid_ratios) > 0:
                    log_ratios = np.log(valid_ratios)
                    geomean = np.exp(np.mean(log_ratios))
                    std_err = np.std(log_ratios, ddof=1) / np.sqrt(len(valid_ratios))
                    # 95% Confidence Interval for the log-normal distribution
                    gm_ci_lo = np.exp(np.mean(log_ratios) - 1.96 * std_err)
                    gm_ci_hi = np.exp(np.mean(log_ratios) + 1.96 * std_err)
                    rat = {
                        "geomean": geomean,
                        "gm_ci_lo": gm_ci_lo,
                        "gm_ci_hi": gm_ci_hi,
                    }
                else:
                    rat = {}

            def fmt_med_iqr(d, prec=".2e"):
                if not d or np.isnan(d.get("median", np.nan)):
                    return "N/A"
                return (
                    f"{d['median']:{prec}} "
                    f"[{d['q25']:{prec}},"
                    f"{d['q75']:{prec}}]"
                )

            def fmt_gm_ci(d):
                if not d or np.isnan(d.get("geomean", np.nan)):
                    return "N/A"
                return (
                    f"{d['geomean']:.3f}x "
                    f"[{d['gm_ci_lo']:.3f},"
                    f"{d['gm_ci_hi']:.3f}]"
                )

            kstr = f"{int(k):>8,}"

            print(
                f"{label:<18} {kstr}  "
                f"{fmt_med_iqr(fixed_stats):<24}  "
                f"{fmt_med_iqr(bass_stats):<24}  "
                f"{fmt_gm_ci(rat):<22}  "
                f"{100 * win:>5.1f}%"
            )

    print("=" * len(hdr))


def print_nscaling_table(
    family_results,
    title="N-scaling",
):
    """
    Print N-scaling table.
    """

    hdr = (
        f"{'Family':<20} {'N':>4} {'k':>7}  "
        f"{'F_fix  med[q25,q75]':<24}  "
        f"{'F_bass med[q25,q75]':<24}  "
        f"{'Ratio  gm[CI95]':<22}  "
        f"{'Win%':>5}"
    )

    print(f"\n{'='*len(hdr)}")
    print(title)
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    for label, res in family_results.items():

        for ni, N in enumerate(res["N_values"]):

            k = res["k_values"][ni]

            s = res["stats"].get(N, {})

            ff = s.get("f_fixed", {})
            fb = s.get("f_bass", {})
            rat = s.get("ratio", {})

            win = float(np.mean(res["f_bass"][:, ni] > res["f_fixed"][:, ni]))

            def fmt_med_iqr(d, prec=".2e"):

                if not d or np.isnan(d.get("median", np.nan)):
                    return "N/A"

                return (
                    f"{d['median']:{prec}} "
                    f"[{d['q25']:{prec}},"
                    f"{d['q75']:{prec}}]"
                )

            def fmt_gm_ci(d):

                if not d or np.isnan(d.get("geomean", np.nan)):
                    return "N/A"

                return (
                    f"{d['geomean']:.2f}x "
                    f"[{d['gm_ci_lo']:.2f},"
                    f"{d['gm_ci_hi']:.2f}]"
                )

            print(
                f"{label:<20} "
                f"{N:>4} "
                f"{k:>7,}  "
                f"{fmt_med_iqr(ff):<24}  "
                f"{fmt_med_iqr(fb):<24}  "
                f"{fmt_gm_ci(rat):<22}  "
                f"{100*win:>5.1f}%"
            )

    print("=" * len(hdr))


def print_summary_table(results):
    """
    Prints a summary table.
    Fidelity is reported as: Median [25th - 75th percentile]
    PR & Advantage reported as: Geometric Mean [95% CI]
    """
    print("=" * 140)
    print(
        f"{'N':<4} | {'k':<6} | {'Trials':<6} | "
        f"{'F_fixed (Med [IQR])':<27} | {'F_BASS (Med [IQR])':<27} | "
        f"{'PR_BASS (GM [95% CI])':<25} | {'Advantage (GM [95% CI])':<25}"
    )
    print("-" * 140)

    for res in results:
        # Format Fidelity (Median + IQR)
        f_fix = f"{res['f_fixed_median']:.2e} [{res['f_fixed_iqr_25']:.2e}-{res['f_fixed_iqr_75']:.2e}]"
        f_bas = f"{res['f_bass_median']:.2e} [{res['f_bass_iqr_25']:.2e}-{res['f_bass_iqr_75']:.2e}]"

        # Format Participation Ratio (Geometric Mean + 95% CI)
        pr_bas = f"{res['pr_bass_geometric_mean']:.1f} [{res['pr_bass_gm_ci95_lo']:.1f}-{res['pr_bass_gm_ci95_hi']:.1f}]"

        # Format Ratio Advantage (Geometric Mean + 95% CI)
        if not np.isnan(res["ratio_geometric_mean"]):
            adv = f"{res['ratio_geometric_mean']:.2f}x [{res['ratio_gm_ci95_lo']:.2f}-{res['ratio_gm_ci95_hi']:.2f}]"
        else:
            adv = "N/A"

        print(
            f"{res['N']:<4} | {res['k']:<6} | {res['trials']:<6} | "
            f"{f_fix:<27} | {f_bas:<27} | {pr_bas:<25} | {adv:<25}"
        )

    print("=" * 140)
