"""
Circuit family generators for BASS benchmarking.

This module is the single source of truth for circuit-family names and how
each family is generated. Every generator here has the signature
`(N, rng) -> list[Gate]` (or is wrapped to that shape in `FAMILIES` below)
and draws *all* of its randomness from the passed-in `rng`
(`numpy.random.Generator`, typically produced by
`src.core.seeding.ExperimentSeeder`) -- no generator in this file touches
the legacy global `np.random` state.


Kicked Ising is intentionally NOT registered in `FAMILIES`
------------------------------------------------------------
`make_kicked_ising` / `_make_kim_factory` are kept below (unchanged, still
usable, still tested) because they represent working code, but they are
not part of `FAMILIES`: Kicked Ising is not one of the five circuit
families defined in the current manuscript , and introducing a
new, previously-undefined family this close to submission was an explicit
scope decision. It is a natural next family for a follow-up "BASS vs. MPS
on volume-law circuits" paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

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
from src.utils.random_circuits import (
    generate_random_circuit,
    generate_tfim_circuit,
    generate_rfim_circuit,
)


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


def make_kicked_ising(
    N: int,
    num_periods: int,
    rng,
    J: float = 1.0,
    g: float = 0.9,
    h: float = 0.1,
    initial_superposition: bool = True,
) -> list:
    """
    Kicked Ising Model (KIM) circuit — chaotic 1D Floquet dynamics.

    Floquet unitary per period
    --------------------------
    U_F = [prod_{even bonds} exp(-i J Z_i Z_{i+1})]
        . [prod_{odd  bonds} exp(-i J Z_i Z_{i+1})]
        . [prod_i            exp(-i g X_i)         ]
        . [prod_i            exp(-i h Z_i)         ]

    Gate calls (BASS conventions):
        ZZ: RZZGate(i, i+1, -J)  = exp(+i*(-J)*ZZ) = exp(-i*J*ZZ)
        X:  RXGate(i, -g)        = exp(+i*(-g)*X)  = exp(-i*g*X)
        Z:  RZGate(i, 2*h)       = exp(-i*(2h/2)*Z) = exp(-i*h*Z)
            (factor of 2 compensates the 1/2 in RZGate's convention)

    CRITICAL — initial state:
        |0...0> is an eigenstate of every Z_i Z_{i+1}, so the first ZZ
        layer adds only a global phase at T=1 the output is a product
        state (chi=1).  Set initial_superposition=True (default) to prepend
        H^N, which breaks the eigenstate trap and generates entanglement
        from the first period.

    Entanglement growth (N=16, initial_superposition=True):
        T=1: S~0.60 nats, chi_exact~2
        T=3: S~1.40 nats, chi_exact~8
        T=6: volume-law, chi_exact~2^(N/2)

    References: Mi et al., Nature 604 (2022); Kim et al., Nature 618 (2023).

    Parameters
    ----------
    N : int
        Number of qubits (1D chain).
    num_periods : int
        Number of Floquet periods (3-6 recommended for MPS comparison).
    rng : np.random.Generator
        Accepted for API consistency; circuit is deterministic.
    J, g, h : float
        ZZ coupling, transverse (X) kick, longitudinal (Z) kick.
        At J=1, g=0.9, h=0.1 the system is in the chaotic phase.
    initial_superposition : bool
        Prepend H^N layer (default True).  Set False only for
        eigenstate / integrable-phase studies.
    """
    if num_periods < 1:
        raise ValueError(f"num_periods must be >= 1, got {num_periods}")
    if N < 2:
        raise ValueError(f"N must be >= 2 for a KIM chain, got {N}")

    gates: list = []

    if initial_superposition:
        for i in range(N):
            gates.append(HGate(i))

    for _ in range(num_periods):
        for i in range(0, N - 1, 2):
            gates.append(RZZGate(i, i + 1, -J))
        for i in range(1, N - 1, 2):
            gates.append(RZZGate(i, i + 1, -J))
        for i in range(N):
            gates.append(RXGate(i, -g))
        for i in range(N):
            gates.append(RZGate(i, 2 * h))
    return gates


def _make_kim_factory(
    num_periods: int = 3, J: float = 1.0, g: float = 0.9, h: float = 0.1
):
    """
    Return a circuit_gen(N, rng) factory for use in run_sweep_full(), etc.

    Example:
        circuit_gen = _make_kim_factory(num_periods=3)
        results = run_sweep_full(circuit_gen, N=20, k_values=[...], n_trials=100)
    """

    def factory(N: int, rng) -> list:
        return make_kicked_ising(
            N, num_periods, rng, J=J, g=g, h=h, initial_superposition=True
        )

    factory.__name__ = f"make_kim_T{num_periods}"
    factory.__qualname__ = f"make_kim_T{num_periods}"
    return factory


def make_haar(N, depth, rng):
    """Haar-random circuit (random 2-qubit gates on random pairs).
    """
    return generate_random_circuit(N, depth, rng=rng)


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


def _factor_grid(N: int) -> tuple:
    """
    Factor N qubits into a (rows, cols) grid for 2D brickwork, preferring
    the most nearly-square factorization -- with N=20 special-cased to
    (4, 5) to exactly match what every existing benchmark (Fig. 3,
    MPS_Comparison, PhaseDiagram, ...) already uses.
    """
    if N == 20:
        return 4, 5
    best = None
    for rows in range(1, int(N**0.5) + 1):
        if N % rows == 0:
            cols = N // rows
            if best is None or abs(rows - cols) < abs(best[0] - best[1]):
                best = (rows, cols)
    if best is None:
        raise ValueError(f"Cannot factor N={N} into a rows x cols grid")
    return best


@dataclass(frozen=True)
class CircuitFamily:
    """One registered circuit family.

    Attributes
    ----------
    name : str
        Human-readable family name. This is the ONLY place this string
        should be spelled out; everything else should read it from here
        (`FAMILIES[key].name` or simply iterate `FAMILIES.keys()`).
    generator : callable(N, rng) -> list[Gate]
        Fully-specified circuit generator: all parameters other than N and
        rng are already bound (via functools.partial or a lambda) to the
        values actually used in the manuscript's benchmarks.
    default_N : int
        The system size used for this family in the manuscript's main
        benchmark sweeps .
    description : str
        Short human-readable summary.
    """

    name: str
    generator: Callable[[int, np.random.Generator], list]
    default_N: int
    description: str = ""

    def generate(self, N: int, rng: np.random.Generator) -> list:
        return self.generator(N, rng)


FAMILIES: Dict[str, CircuitFamily] = {
    "1D Brickwork": CircuitFamily(
        name="1D Brickwork",
        generator=lambda N, rng: make_brickwork(N, depth=6, rng=rng),
        default_N=20,
        description="Alternating even/odd layers of Haar-random 2-qubit gates on a 1D chain.",
    ),
    "2D Brickwork": CircuitFamily(
        name="2D Brickwork",
        generator=lambda N, rng: make_brickwork_2d(*_factor_grid(N), depth=4, rng=rng),
        default_N=20,
        description="rows x cols 2D brickwork circuit (grid factored from N)"
    ),
    "RFIM (W=2)": CircuitFamily(
        name="RFIM (W=2)",
        generator=lambda N, rng: make_rfim(N, rng, layers=5, W=2.0),
        default_N=20,
        description="1D random-field Ising model, longitudinal disorder W=2, 5 Trotter layers.",
    ),
    "QAOA (p=3)": CircuitFamily(
        name="QAOA (p=3)",
        generator=lambda N, rng: make_qaoa(N, rounds=3, rng=rng),
        default_N=16,
        description="p=3 QAOA MaxCut on a random 3-regular graph.",
    ),
    "UCCSD": CircuitFamily(
        name="UCCSD",
        generator=lambda N, rng: make_uccsd(N, rng, n_trotter_steps=1),
        default_N=12, #20 is too expensive to run on a non-dedicated server
        description="First-order Trotterized UCCSD from the Hartree-Fock reference.",
    ),
    "Haar": CircuitFamily(
        name="Haar",
        generator=lambda N, rng: make_haar(N, depth=3, rng=rng),
        default_N=16,
        description="Haar-random 2-qubit gates on randomly paired qubits, depth 3.",
    ),
}


def get_family(name: str) -> CircuitFamily:
    """Look up a registered family by name, with a helpful error listing
    valid names (rather than a bare KeyError) if the name is mistyped."""
    try:
        return FAMILIES[name]
    except KeyError:
        raise KeyError(
            f"Unknown circuit family {name!r}. Registered families: "
            f"{sorted(FAMILIES.keys())}"
        ) from None
