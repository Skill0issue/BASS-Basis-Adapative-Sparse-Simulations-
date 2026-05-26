"""
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

Rigorous Peer-Review Statistics Pipeline
----------------------------------------
    Quantum circuit approximations are highly susceptible to fat-tailed distributions,
    extreme data skew, and rare high-overlap outliers. Standard arithmetic means and
    standard deviations present misleading pictures of performance. To ensure
    reviewer-proof mathematical validity, the pipeline splits metrics into independent
    statistical treatments:

1. Non-Parametric Percentile Metrics (Fidelity)
    Absolute probabilities and state overlaps are characterized exclusively via Medians
    and Interquartile Ranges (IQR: 25th to 75th percentiles).

2. Multiplicatively Distributed Metrics (PR, Space-Ratios, Speedups)
    Data governed by multiplicative random variations (such as the State Participation
    Ratio or comparative speedup factors) are evaluated using Geometric Means.
    Confidence intervals (95% CI) are extracted via a non-parametric, vector-accelerated
    bootstrap sampling technique (typically 2,000 to 4,000 resamples).

Strict Timing & Validation Guarantees
-------------------------------------
- Hardware-Cache Equivalence: Single-trial workflows (`run_trial_timed`) enforce
that a single generated circuit instance is evaluated across the exact same reference
hardware state. Timing calculations (`time.perf_counter()`) isolate the internal
`.simulate()` call, explicitly excluding circuit assembly, tensor reshaping, or
subsequent metric aggregation.

- Numerical Stability: PR win-conditions and compression ratios are dynamically re-calculated
as Ratio = Fixed / BASS to safeguard against arithmetic division-by-zero slop or Inf loops
during severe state-budget truncations.

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
from scipy import stats

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

# ── Circuit factories ──────────────────────────────────────────────────────────


def make_brickwork(N, depth, rng):
    """Brickwork circuit: alternating even/odd layers of Haar-random 2-qubit gates."""
    gates = []
    for layer in range(depth):
        for i in range(layer % 2, N - 1, 2):
            gates.append(RandomTwoQubitGate(i, i + 1, seed=int(rng.integers(0, 2**31))))
    return gates


def make_brickwork_2d(rows, cols, depth, rng):
    """
    2D brickwork circuit on a rowsxcols grid (row-major qubit indexing).

    Alternates four sublayer types per period-4 cycle:
        0 — horizontal even columns  (pairs (r,c)-(r,c+1) for even c)
        1 — horizontal odd  columns
        2 — vertical   even rows     (pairs (r,c)-(r+1,c) for even r)
        3 — vertical   odd  rows

    This ensures every nearest-neighbour pair is covered within 4 layers,
    so entanglement can propagate in both spatial directions.
    Total qubits N = rows * cols.
    """
    gates = []
    for layer in range(depth):
        d = layer % 4
        pairs = []
        if d == 0:
            for r in range(rows):
                for c in range(0, cols - 1, 2):
                    pairs.append((r * cols + c, r * cols + c + 1))
        elif d == 1:
            for r in range(rows):
                for c in range(1, cols - 1, 2):
                    pairs.append((r * cols + c, r * cols + c + 1))
        elif d == 2:
            for r in range(0, rows - 1, 2):
                for c in range(cols):
                    pairs.append((r * cols + c, (r + 1) * cols + c))
        else:
            for r in range(1, rows - 1, 2):
                for c in range(cols):
                    pairs.append((r * cols + c, (r + 1) * cols + c))
        for q1, q2 in pairs:
            gates.append(RandomTwoQubitGate(q1, q2, seed=int(rng.integers(0, 2**31))))
    return gates


def make_haar(N, depth, rng):
    """Haar-random circuit (random 2-qubit gates on random pairs)."""
    np.random.seed(int(rng.integers(0, 2**31)))
    return generate_random_circuit(N, depth)


def make_qft(N, rng=None):
    """QFT circuit (exact, deterministic)."""
    gates = []
    for i in range(N):
        gates.append(HGate(i))
        for j in range(i + 1, min(i + 5, N)):  # limit to nearest 4 for speed
            angle = np.pi / (2 ** (j - i))
            cp = np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, np.exp(1j * angle)],
                ],
                dtype=np.complex128,
            )
            gates.append(TwoQubitGate(i, j, cp))
    return gates


def make_tfim(
    N,
    rng,
    layers=5,
    J=1.0,
    h=3.0,
    dt=0.3,
    disorder_strength_J=0.0,
    disorder_strength_h=0.0,
):
    return generate_tfim_circuit(
        N,
        layers,
        J=J,
        h=h,
        dt=dt,
        rng=rng,
        disorder_strength_J=disorder_strength_J,
        disorder_strength_h=disorder_strength_h,
    )


def make_rfim(N, rng, layers=5, J=1.0, h0=1.0, W=2.0, dt=0.2):
    """
    Quantum RFIM circuit generator.  Thin wrapper around generate_rfim_circuit.

    Uses a single disorder realisation per call (controlled by rng), so that
    ensemble averaging over many independent calls sweeps over different
    disorder realisations.

    Parameters match generate_rfim_circuit; see its docstring for physical
    interpretation and parameter guidance.

    Quick-reference phase map (J = h₀ = 1.0):
        W = 0.5   →  near-critical, large PRZ  →  BASS most helpful
        W = 2.0   →  crossover,    medium PRZ  →  BASS moderately helpful (default)
        W = 5.0   →  deep MBL,     small PRZ   →  BASS provides no benefit
    """
    return generate_rfim_circuit(
        num_qubits=N,
        num_layers=layers,
        J=J,
        h0=h0,
        W=W,
        dt=dt,
        rng=rng,
    )


# ─── Jordan-Wigner Pauli decompositions ────────────────────────────────────────
# Single excitation (i → a, i < a, i ∈ occ, a ∈ virt):
#   a†_a a_i - h.c. = (i/2)[X_i Z_{i+1}...Z_{a-1} Y_a - Y_i Z_{i+1}...Z_{a-1} X_a]
#
# Trotterized: exp(t A_{ia}) ≈ exp((it/2) P_1) · exp((-it/2) P_2)
#
# Double excitation (i < j < a < b, i,j ∈ occ, a,b ∈ virt):
#   a†_a a†_b a_j a_i - h.c.  has 8 Pauli-string terms.
#   The 8 Pauli combinations on {i,j,a,b} and their signs ε_k ∈ {±1} were
#   derived from the explicit JW matrix computation for the adjacent-qubit
#   case (0,1,2,3) and verified numerically:
#   a†_2 a†_3 a_1 a_0 - h.c. = (i/8) Σ_k ε_k P_k  (Ref. [1], Eq. 22)
#   where P_k is listed as Paulis on (i,j,a,b) with Z parity strings
#   between each adjacent pair, handled by the CNOT cascade.
#
# ε_k and Pauli-on-{i,j,a,b}:
_DOUBLE_EXC_TERMS = [
    ("X", "Y", "X", "X", +1),  # XYXX
    ("Y", "X", "X", "X", +1),  # YXXX
    ("X", "X", "X", "Y", -1),  # XXXY
    ("X", "X", "Y", "X", -1),  # XXYX
    ("X", "Y", "Y", "Y", -1),  # XYYY
    ("Y", "X", "Y", "Y", -1),  # YXYY
    ("Y", "Y", "X", "Y", +1),  # YYXY
    ("Y", "Y", "Y", "X", +1),  # YYYX
]
# Each has an odd number of Y operators, consistent with the anti-Hermitian
# structure of a†a†aa - h.c. under JW.


def make_uccsd(N, rng, n_electrons=None, n_trotter_steps=1):
    """
    Trotterised UCCSD circuit in Jordan-Wigner encoding.

    Implements exp(T - T†) where
        T = T₁ + T₂,
        T₁ = Σ_{i∈occ, a∈virt} θ_{ia} a†_a aᵢ,
        T₂ = Σ_{i<j∈occ, a<b∈virt} θ_{ijab} a†_a a†_b aⱼ aᵢ,
    via first-order Trotterisation of each excitation operator.

    Reference state
        |HF⟩ = X₀ X₁ ... X_{Ne-1} |0...0⟩  (half-filling by default).
        Qubits 0..Ne-1 are occupied; Ne..N-1 are virtual.
        This is a single computational-basis state → participation ratio PR ≈ 1 → BASS provides no benefit, confirming the paper's UCCSD benchmark claim.

    Circuit structure per Trotter step
        1. Singles: 2 Pauli-string circuits per (i, a) pair.
        2. Doubles: 8 Pauli-string circuits per (i, j, a, b) quadruple.

    Each Pauli-string circuit exp(i θ P) is implemented via the
    standard CNOT-cascade / basis-change method (Ref. [1]):
        - X qubit: H before and after.
        - Y qubit: RX(-π/4) before; RX(+π/4) after.
        gates.py:
            1) RXGate(q, θ) = exp(iθX);
            2) RX(-π/4) = exp(-iπX/4) diagonalises -> Y.
        - Z qubit: no basis change;
            participates in parity cascade.
        - CNOT(q_min → q_min+1 → ... → q_max):
            accumulates XOR parity of all active bits onto q_max.
        - RZGate(q_max, -2θ): implements exp(iθZ) on the parity qubit.
        gates.py:
            1) RZGate(q, θ) = exp(-iθZ/2);
            2) RZGate(q_max, -2θ) = exp(iθZ_{q_max}).
        - Reverse CNOT cascade restores bit values.

    Amplitude sampling
    ------------------
    Singles:  θ_{ia}    ~ U[-0.3,   0.3]   (CCSD t₁ amplitudes).
    Doubles:  θ_{ijab}  ~ U[-0.05, 0.05]   (CCSD t₂ amplitudes).

    For weakly correlated systems |HF⟩ dominates with amplitude ≈ 1
    and singly/doubly excited determinants appear with O(θ₁) / O(θ₂)
    amplitude.  The state therefore stays Z-sparse at these scales,
    which is why BASS reverts to fixed-basis behaviour on UCCSD.

    Gate count (one Trotter step)
    -----------------------------
    Singles (2 Pauli circuits per pair, each of depth O(a-i)):
        2 x Ne x Nv x O(N) ≈ O(N³)
    Doubles (8 Pauli circuits per quadruple, each of depth O(b-i)):
        8 x C(Ne,2) x C(Nv,2) x O(N) ≈ O(N⁵)

    Parameters
    ----------
    N : int
        Number of qubits = number of spin-orbitals.  Must satisfy N ≥ 4.
    rng : numpy.random.Generator
        Source of randomness for amplitude sampling.
    n_electrons : int, optional
        Number of electrons.  Defaults to N // 2 (half-filling).
    n_trotter_steps : int
        First-order Trotter steps.  Default 1 matches standard benchmarks.

    Returns
    -------
    list[Gate]
        Gate sequence implementing |HF⟩ preparation + Trotterised UCCSD.

    References
    ----------
    [1] Whitfield, Biamonte, Aspuru-Guzik, Mol. Phys. 109, 735 (2011).
    [2] Anand et al., Chem. Soc. Rev. 51, 1159 (2022).
    """
    if N < 4:
        raise ValueError(f"make_uccsd requires N >= 4 (got {N})")
    if n_electrons is None:
        n_electrons = N // 2
    Ne = n_electrons
    if not (1 <= Ne <= N - 1):
        raise ValueError(f"n_electrons must be in [1, N-1] (got {Ne})")

    occ = list(range(Ne))
    virt = list(range(Ne, N))
    n_occ, n_virt = len(occ), len(virt)

    # ── Sample CCSD-scale amplitudes ──────────────────────────────────────────
    theta_singles = rng.uniform(-0.30, 0.30, (n_occ, n_virt))

    n_occ_pairs = n_occ * (n_occ - 1) // 2
    n_virt_pairs = n_virt * (n_virt - 1) // 2
    if n_occ_pairs > 0 and n_virt_pairs > 0:
        theta_doubles = rng.uniform(-0.05, 0.05, (n_occ_pairs, n_virt_pairs))
    else:
        theta_doubles = None

    occ_pairs = [(occ[k], occ[l]) for k in range(n_occ) for l in range(k + 1, n_occ)]
    virt_pairs = [
        (virt[k], virt[l]) for k in range(n_virt) for l in range(k + 1, n_virt)
    ]

    gates = []

    # ── Step 1: Hartree-Fock reference |1^Ne 0^(N-Ne)⟩ ───────────────────────
    for q in occ:
        gates.append(XGate(q))

    # ── Pauli-string exponentiation helper ────────────────────────────────────
    def pauli_exp(active, theta):
        """
        Append gates for exp(i * theta * ⊗_j P_j).

        active : list of (qubit_index, pauli_char) for non-identity Paulis.
                Intermediate qubits (not listed) carry Z in the parity
                string and appear in the CNOT cascade automatically.
        theta  : rotation angle (the generator is iθP, so RZGate gets -2θ).

        Implements the standard basis-change + CNOT cascade circuit [1].
        """
        if abs(theta) < 1e-12 or not active:
            return

        # Fill in Z for intermediate qubits between q_min and q_max
        active_sorted = sorted(active, key=lambda x: x[0])
        q_min = active_sorted[0][0]
        q_max = active_sorted[-1][0]
        pauli_dict = {q: p for q, p in active_sorted}
        all_qubits = list(range(q_min, q_max + 1))
        all_paulis = [pauli_dict.get(q, "Z") for q in all_qubits]

        # ── Basis change (applied first to ket) ──
        for q, p in zip(all_qubits, all_paulis):
            if p == "X":
                gates.append(HGate(q))
            elif p == "Y":
                # RXGate(-π/4) = exp(-iπX/4) diagonalises Y:
                #   exp(-iπX/4) Y exp(+iπX/4) = Z  ✓
                gates.append(RXGate(q, -np.pi / 4))
            # Z: no basis change needed

        # ── CNOT cascade (q_min → q_min+1 → ... → q_max) ────────────────
        # Accumulates XOR parity of all qubit values onto q_max.
        for k in range(len(all_qubits) - 1):
            gates.append(CNOTGate(all_qubits[k], all_qubits[k + 1]))

        # ── Phase rotation ────────────────────────────────────────────────
        # RZGate(q, -2θ) = exp(-i(-2θ)Z/2) = exp(iθZ) ✓
        gates.append(RZGate(q_max, -2.0 * theta))

        # ── Reverse CNOT cascade (restores original bit values) ───────────
        for k in range(len(all_qubits) - 2, -1, -1):
            gates.append(CNOTGate(all_qubits[k], all_qubits[k + 1]))

        # ── Undo basis change (applied last to ket) ───────────────────────
        for q, p in zip(all_qubits, all_paulis):
            if p == "X":
                gates.append(HGate(q))
            elif p == "Y":
                gates.append(RXGate(q, +np.pi / 4))

    # ── Single excitation: exp(t_{ia}(a†_a aᵢ - h.c.)) ──────────────────────
    #
    #  JW generator: (i/2)[X_i Z...Z Y_a - Y_i Z...Z X_a]
    #  Trotterised:
    #    exp((it/2) X_i Z...Z Y_a) · exp((-it/2) Y_i Z...Z X_a)
    #
    def single_excitation(i, a, t):
        pauli_exp([(i, "X"), (a, "Y")], t / 2.0)
        pauli_exp([(i, "Y"), (a, "X")], -t / 2.0)

    # ── Double excitation: exp(t_{ijab}(a†_a a†_b aⱼ aᵢ - h.c.)) ─────────────
    #
    #  JW generator (i<j<a<b): (i/8) Σ_k ε_k P_k
    #  Trotterised: Π_k exp(i (ε_k t/8) P_k)
    #
    def double_excitation(i, j, a, b, t):
        # i < j < a < b is guaranteed by construction (occ < virt, occ_pairs sorted)
        for pi, pj, pa, pb, sign in _DOUBLE_EXC_TERMS:
            pauli_exp([(i, pi), (j, pj), (a, pa), (b, pb)], sign * t / 8.0)

    # ── Trotter layers ────────────────────────────────────────────────────────
    for _ in range(n_trotter_steps):

        # Singles (Ne x Nv pairs)
        for i_idx, i in enumerate(occ):
            for a_idx, a in enumerate(virt):
                t = theta_singles[i_idx, a_idx]
                single_excitation(i, a, t)

        # Doubles (C(Ne,2) x C(Nv,2) quadruples)
        if theta_doubles is not None:
            for ip, (oi, oj) in enumerate(occ_pairs):
                for ia, (va, vb) in enumerate(virt_pairs):
                    t = theta_doubles[ip, ia]
                    double_excitation(oi, oj, va, vb, t)

    return gates


def _random_3regular_graph(N, rng, max_tries=500):
    """
    Generate a uniformly random 3-regular graph on N vertices (N must be even)
    via the pairing model (Bollobás 1980).

    Returns a list of undirected edges (i, j) with i < j.
    Raises RuntimeError if a valid graph is not found within max_tries attempts.
    """
    if N % 2 != 0:
        raise ValueError(f"3-regular graph requires even N; got {N}")
    if N < 4:
        raise ValueError(f"3-regular graph requires N >= 4; got {N}")

    for _ in range(max_tries):
        stubs = list(range(N)) * 3  # each vertex appears 3 times (degree 3)
        rng.shuffle(stubs)
        edges = set()
        valid = True
        for k in range(0, len(stubs), 2):
            u, v = stubs[k], stubs[k + 1]
            if u == v or (min(u, v), max(u, v)) in edges:
                valid = False
                break
            edges.add((min(u, v), max(u, v)))
        if valid:
            return list(edges)

    raise RuntimeError(
        f"Failed to generate a 3-regular graph on N={N} vertices after "
        f"{max_tries} attempts.  This is extremely unlikely for N ≥ 6; "
        f"check that N is even and sufficiently large."
    )


def make_qaoa(N, rounds, rng):
    """
    QAOA MaxCut circuit on a single random 3-regular graph.

    Implements the standard p-level QAOA ansatz (Farhi, Goldstone, Gutmann 2014):
        |ψ_p⟩ = U_B(β_p) U_C(γ_p) ··· U_B(β_1) U_C(γ_1) |+⟩^N

    where the cost and mixer unitaries are:
        (a) U_C(γ_k) = exp(-i γ_k H_C),   H_C = Σ_{(i,j)∈E} (1 - Z_i Z_j) / 2
                ≈ Π_{(i,j)∈E} exp(+i γ_k/2  Z_i Z_j)   [global phase dropped]
                = Π_{(i,j)∈E} RZZGate(i, j,  γ_k / 2)

        (b) U_B(β_k) = exp(-i β_k H_B),   H_B = Σ_i X_i
                = Π_i exp(-i β_k X_i)
                = Π_i RXGate(i,  -β_k)

    Angle schedule
    --------------
    Near-optimal angles from the Fourier parameterisation of
    Zhou et al. (PRX Quantum 1, 020304, 2020):

        γ_k = u₁ sin((2k-1) π / (4p)),   k = 1, …, p
        β_k = v₁ cos((2k-1) π / (4p)),   k = 1, …, p

    with u₁ = v₁ = π/4 (the analytically motivated amplitude for the
    sinusoidal ramp that approximates the infinite-p optimal schedule for
    3-regular MaxCut; see Farhi & Harrow 2022 and Crooks 2018).

    This gives the physically correct pattern: γ increases monotonically
    across rounds (cost operator phase builds up), β decreases monotonically
    (mixer localises the state).  Fully random angles, as in the original,
    sample the QAOA landscape uniformly and bear no relation to near-optimal
    solutions; the result is closer to a Haar-random circuit than a
    variational one.

    A small per-instance perturbation ε ~ U(-δ, +δ) with δ = 0.05 is added
    to each angle, introducing circuit-to-circuit variation while staying
    near the optimum.

    The circuit is documented as a "QAOA-structured benchmark circuit" rather
    than an optimised VQA run, since angles are set analytically (not by
    energy minimisation on each instance).

    Parameters
    ----------
    N : int
        Number of qubits (must be even for a 3-regular graph to exist).
    rounds : int
        Number of QAOA layers p (typically 1–5).
    rng : numpy.random.Generator
        Controls graph generation and angle perturbations.

    Returns
    -------
    list[Gate]
        Circuit gates.  Initial |+⟩^N layer prepended.
        Gate count: N + rounds x (|E| + N) = N + p x (3N/2 + N) = N(1 + 5p/2).

    References
    ----------
    Farhi, Goldstone, Gutmann, arXiv:1411.4028 (2014) — original QAOA.
    Farhi, Harrow, arXiv:2202.00648 (2022) — QAOA performance on 3-reg graphs.
    Zhou et al., PRX Quantum 1, 020304 (2020) — Fourier angle parameterisation.
    Crooks, arXiv:1811.08419 (2018) — near-optimal angles for MaxCut.
    """
    p = rounds

    # ── One graph for the entire circuit (fixes regeneration bug) ────────────
    edges = _random_3regular_graph(N, rng)

    # ── Near-optimal Fourier angle schedule ──────────────────────────────────
    # u₁ = v₁ = π/4: quarter-period sinusoidal ramp (Crooks 2018; Farhi & Harrow 2022)
    # Gives: γ_1 < γ_2 < … < γ_p  (monotone increasing)
    #        β_1 > β_2 > … > β_p  (monotone decreasing)
    # Perturbation δ = 0.05 ≈ 8% of the maximum angle, small enough to stay
    # near the optimum but large enough to give meaningful instance variation.
    FOURIER_AMP = np.pi / 4.0
    PERTURB = 0.05

    gammas = [
        FOURIER_AMP * np.sin((2 * k - 1) * np.pi / (4 * p))
        + rng.uniform(-PERTURB, PERTURB)
        for k in range(1, p + 1)
    ]
    betas = [
        FOURIER_AMP * np.cos((2 * k - 1) * np.pi / (4 * p))
        + rng.uniform(-PERTURB, PERTURB)
        for k in range(1, p + 1)
    ]

    # ── Initial state: equal superposition |+⟩^N ─────────────────────────────
    circuit = [HGate(q) for q in range(N)]

    # ── p QAOA rounds, all on the SAME graph ─────────────────────────────────
    for k in range(p):
        gamma = gammas[k]
        beta = betas[k]

        # Cost unitary U_C(γ_k): one RZZ gate per edge
        # exp(-iγ (1-Z_iZ_j)/2) ≡ exp(+iγ/2 Z_iZ_j) x global_phase
        # → RZZGate(i, j, γ/2)  [global phase irrelevant]
        for i, j in edges:
            circuit.append(RZZGate(i, j, gamma / 2.0))

        # Mixer unitary U_B(β_k): one RX gate per qubit
        # exp(-iβ X_q) = RXGate(q, -β)
        for q in range(N):
            circuit.append(RXGate(q, -beta))

    return circuit


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
            circ = generate_random_circuit(N, depth)

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
):
    """
    Sweep k values, averaging over n_trials independent circuits.

    cache_exact=True (default): compute the exact statevector once per trial
    (reusing across k values) by pre-generating the circuit at seed 0.
    Set False for stochastic circuits where exact state depends on k.

    Returns
    -------
    dict with keys k_values, f_fixed, f_bass, pr_fixed, pr_bass
    Each value is a list of length len(k_values).
    """
    if bass_kwargs is None:
        bass_kwargs = {}

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
            seed = seed_base * 10**6 + t
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
            seed = seed_base * 10**6 + t
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

        results["f_fixed"].append(np.mean(ft_list))
        results["f_bass"].append(np.mean(fb_list))
        results["pr_fixed"].append(np.mean(prt_list))
        results["pr_bass"].append(np.mean(prb_list))

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
    n_boot=2000,
):
    """
    N-scaling benchmark.
    """

    if bass_kwargs is None:
        bass_kwargs = {}

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

            seed = seed_base * 10**6 + ni * 1000 + t

            circ = circuit_gen(
                N,
                np.random.default_rng(seed),
            )

            exact_svs.append(exact_statevector(N, circ))

        for t in range(n_trials):

            seed = seed_base * 10**6 + ni * 1000 + t

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
    n_boot=2000,
):
    """
    Sweep sparse budget k over n_trials independent circuits.
    """

    if bass_kwargs is None:
        bass_kwargs = {}

    k_values = np.asarray(k_values, dtype=int)

    nk = len(k_values)

    f_fixed = np.full((n_trials, nk), np.nan)
    f_bass = np.full((n_trials, nk), np.nan)

    pr_fixed = np.full((n_trials, nk), np.nan)
    pr_bass = np.full((n_trials, nk), np.nan)

    exact_svs = []

    for t in range(n_trials):

        seed = seed_base * 10**6 + t

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

            seed = seed_base * 10**6 + t

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
    Print publication-style benchmark table with mathematically
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
    Print publication-style N-scaling table.
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
    Prints a highly rigorous, publication-ready summary table.
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
