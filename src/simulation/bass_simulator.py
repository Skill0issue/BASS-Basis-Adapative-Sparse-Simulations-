"""
BASS: Basis-Adaptive Sparse Simulator
====================================
A high-performance quantum circuit simulator for sparse-state vectors,
leveraging dynamically optimized local basis rotations to maximize state
concentration and amplitude retention under a tight state-budget constraint.

Designed for rigorous academic benchmarking and full code reproducibility,
this module implements the Core BASS engine alongside low-level, JIT-compiled
Numba acceleration kernels.

Mathematical Concept & Objectives
----------------------------------
Standard fixed-basis sparse simulators maintain a state representation of the form:
    |ψ⟩ ≈ Σ_{m=1}^k α_m |x_m⟩
in the predefined computational basis, where k is a hard upper bound on non-zero
amplitudes (nnz). This approach suffers severe fidelity degradation when multi-qubit
entanglement spreads the state vector uniformly across the Hilbert space.

BASS instead simulates the state in a dynamically updated, product-rotated basis:
    |ψ⟩_computational = ( ⊗_{j=0}^{N-1} U_j ) |ψ⟩_BASS
where each U_j ∈ SU(2) is a local single-qubit unitary. By identifying a localized
coordinate system, BASS minimizes the State Participation Ratio (PR), concentrating
quantum amplitudes into a compact set of basis states to maximize amplitude retention
during mandatory state truncations.

Key Algorithmic Functions
---------------------------
    1. Batched 1-Qubit Reduced Density Matrix (RDM) Optimization
        Instead of executing N independent state-sorting loops to find qubit-flip
        partners, BASS constructs an O(k) open-addressing hash table with linear probing.
        All N single-qubit RDMs are extracted concurrently in a single, tight pass
        scaling as O(N · nnz), lowering the analytical overhead from a loose
        O(N · nnz · log nnz) bound.

    2. Coordinate Descent Basis Shifts
        Using an analytical 2×2 Hermitian eigensolver (`eigh_2x2`), the single-qubit
        basis matrices {U_j} are updated sequentially via coordinate descent. A rotation
        is accepted if and only if it strictly decreases the normalized Participation Ratio
        (PR), adhering to a strict "do-no-harm" design philosophy.

    3. Deferred Truncation & Buffer Management
        Rather than truncating the state back to the budget k after every single gate,
        BASS allows an expanding intermediate wave-front up to a configurable hard cap
        (`max_nnz_factor * k`). This deferred truncation preserves vital interference terms
        during deeply layered cascades before selecting the top-k dominant amplitudes.

    4. Fast 2-Qubit Hash-Table Gate Application
        Non-diagonal 2-qubit gates are applied via a specialized deduplication kernel
        (`apply_2qubit_gate_ht`). Using the shared local hash table, overlapping output
        amplitudes are accumulated directly on the fly. This bypasses the traditional,
        expensive sort-and-merge step, ensuring O(nnz) time complexity.

    5. Basis-Aligned 2-Qubit Truncation Pass (Optional)
        When `use_2qubit_rotations` is active, BASS executes non-overlapping even/odd
        brick-wall passes. It projects pairs into their joint 4×4 2-qubit eigenbasis,
        truncates the weakest states in the rotated frame, and maps them back, capturing
        higher-order local correlations missed by single-qubit operations.

    6. Adaptive Optimization Triggering
        To isolate simulator execution runtime from excessive optimization loops, an
        adaptive trigger (`optimize_trigger`) skips the coordinate descent pass if the
        state's PR has not degraded beyond a predefined threshold (`pr_opt_threshold`)
        since the last optimization.

Reviewer & Reproducibility Metrics
-----------------------------------
    - `state.gamma`: Tracks the cumulative, mathematically rigorous norm reduction
    sustained throughout every state truncation. It serves as an active lower bound
    on state fidelity without requiring an exact, exponentially costly statevector inner product.

    - `to_statevector()`: Provides automated vector-matrix reconstruction by folding
    the active basis transforms {U_j} back into the dense state array via a vectorized
    `np.einsum` pipeline, permitting direct fidelity validation against exact simulators.

Memory & Spatial Footprint
--------------------------
    The internal open-address hash table allocates layout dimensions based on:
        ht_cap = next_power_of_2(8 * min(max_nnz_factor * k, 2^N))
    This approach maintains a strict load factor ≤ 50% to prevent probe chains from
    degenerating. For standard benchmark sizes (e.g., k=512), the table footprint
    remains bounded at ~384 KB, operating fully within high-speed CPU L2/L3 caches
    independent of the total system qubit size N.
"""

import numpy as np
from math import sqrt
from numba import njit
from src.core.sparse_state import SparseState, apply_2qubit_diagonal_kernel
from src.core.gates import SingleQubitGate, TwoQubitGate

# Compact O(k) hash-table kernels for partner lookup
# Open-addressing with linear probing.
# Sentinel: EMPTY = uint64(0xFFFFFFFFFFFFFFFF) — never a valid
# basis state for N <= 63 qubits (all-ones 64-bit word).
# Table capacity must be a power of 2 and >= 2*nnz to keep
# load factor ≤ 50% and avoid degenerate probe chains.


@njit(cache=True)
def ht_clear(ht_keys):
    """Reset all slots to the EMPTY sentinel.  O(ht_cap)."""
    empty = np.uint64(0xFFFFFFFFFFFFFFFF)
    for i in range(ht_keys.shape[0]):
        ht_keys[i] = empty


@njit(cache=True)
def ht_build(x, nnz, ht_keys, ht_vals):
    """
    Populate hash table from x[:nnz].
    ht_keys must already be cleared to EMPTY.
    ht_keys.shape[0] must be a power of 2.
    """
    empty = np.uint64(0xFFFFFFFFFFFFFFFF)
    cap = np.uint64(ht_keys.shape[0])
    mask = cap - np.uint64(1)
    for i in range(nnz):
        h = x[i] & mask
        while ht_keys[h] != empty:
            h = (h + np.uint64(1)) & mask
        ht_keys[h] = x[i]
        ht_vals[h] = np.int32(i)


@njit(cache=True)
def ht_lookup(ht_keys, ht_vals, key):
    """Return stored index for key, or -1 if absent."""
    empty = np.uint64(0xFFFFFFFFFFFFFFFF)
    cap = np.uint64(ht_keys.shape[0])
    mask = cap - np.uint64(1)
    h = key & mask
    while ht_keys[h] != empty:
        if ht_keys[h] == key:
            return np.int64(ht_vals[h])
        h = (h + np.uint64(1)) & mask
    return np.int64(-1)


@njit(cache=True)
def ht_insert(ht_keys, ht_vals, key, val):
    """Insert key→val.  If key already present, update its value."""
    empty = np.uint64(0xFFFFFFFFFFFFFFFF)
    cap = np.uint64(ht_keys.shape[0])
    mask = cap - np.uint64(1)
    h = key & mask
    while ht_keys[h] != empty:
        if ht_keys[h] == key:
            ht_vals[h] = np.int32(val)
            return
        h = (h + np.uint64(1)) & mask
    ht_keys[h] = key
    ht_vals[h] = np.int32(val)


@njit(cache=True)
def apply_1qubit_rotation_sparse(x, alpha, nnz, qubit, U):
    mask = np.uint64(1) << np.uint64(qubit)
    inv_mask = ~mask
    rest = np.empty(nnz, dtype=np.uint64)
    bits = np.empty(nnz, dtype=np.int64)
    for i in range(nnz):
        rest[i] = x[i] & inv_mask
        bits[i] = np.int64((x[i] >> np.uint64(qubit)) & np.uint64(1))
    order = np.argsort(rest[:nnz])
    new_x = np.empty(2 * nnz, dtype=np.uint64)
    new_alpha = np.empty(2 * nnz, dtype=np.complex128)
    new_nnz = 0
    i = 0
    while i < nnz:
        r = rest[order[i]]
        j = i + 1
        while j < nnz and rest[order[j]] == r:
            j += 1
        a0 = np.complex128(0.0)
        a1 = np.complex128(0.0)
        base_x = np.uint64(0)
        for idx in range(i, j):
            oi = order[idx]
            base_x = x[oi] & inv_mask
            if bits[oi] == 0:
                a0 = alpha[oi]
            else:
                a1 = alpha[oi]
        a0_new = U[0, 0] * a0 + U[0, 1] * a1
        a1_new = U[1, 0] * a0 + U[1, 1] * a1
        if abs(a0_new) > 1e-30:
            new_x[new_nnz] = base_x
            new_alpha[new_nnz] = a0_new
            new_nnz += 1
        if abs(a1_new) > 1e-30:
            new_x[new_nnz] = base_x | mask
            new_alpha[new_nnz] = a1_new
            new_nnz += 1
        i = j
    for i in range(new_nnz):
        x[i] = new_x[i]
        alpha[i] = new_alpha[i]
    return new_nnz


@njit(cache=True)
def compute_2qubit_rdm(x, alpha, nnz, q1, q2):
    """
    Compute 4×4 density matrix for qubit pair (q1, q2).
    Row/col index = (bit_q1 << 1) | bit_q2.
    """
    mask1 = np.uint64(1) << np.uint64(q1)
    mask2 = np.uint64(1) << np.uint64(q2)
    mask12 = mask1 | mask2
    inv12 = ~mask12
    idx2 = np.empty(nnz, dtype=np.int64)
    rest = np.empty(nnz, dtype=np.uint64)
    for i in range(nnz):
        b1 = np.int64((x[i] >> np.uint64(q1)) & np.uint64(1))
        b2 = np.int64((x[i] >> np.uint64(q2)) & np.uint64(1))
        idx2[i] = (b1 << 1) | b2
        rest[i] = x[i] & inv12
    order = np.argsort(rest[:nnz])
    rho = np.zeros((4, 4), dtype=np.complex128)
    i = 0
    while i < nnz:
        r = rest[order[i]]
        j = i + 1
        while j < nnz and rest[order[j]] == r:
            j += 1
        a = np.zeros(4, dtype=np.complex128)
        for idx in range(i, j):
            a[idx2[order[idx]]] = alpha[order[idx]]
        for p in range(4):
            for qq in range(4):
                rho[p, qq] += a[p] * np.conj(a[qq])
        i = j
    return rho


@njit(cache=True)
def apply_2qubit_rotation_sparse(x, alpha, nnz, q1, q2, U4):
    """
    Apply 4×4 unitary U4 to qubit pair (q1, q2) in sparse state.
    Max output nnz = 4 * input nnz.
    """
    mask1 = np.uint64(1) << np.uint64(q1)
    mask2 = np.uint64(1) << np.uint64(q2)
    mask12 = mask1 | mask2
    inv12 = ~mask12
    idx2 = np.empty(nnz, dtype=np.int64)
    rest = np.empty(nnz, dtype=np.uint64)
    for i in range(nnz):
        b1 = np.int64((x[i] >> np.uint64(q1)) & np.uint64(1))
        b2 = np.int64((x[i] >> np.uint64(q2)) & np.uint64(1))
        idx2[i] = (b1 << 1) | b2
        rest[i] = x[i] & inv12
    order = np.argsort(rest[:nnz])
    new_x = np.empty(4 * nnz, dtype=np.uint64)
    new_alpha = np.empty(4 * nnz, dtype=np.complex128)
    new_nnz = 0
    i = 0
    while i < nnz:
        r = rest[order[i]]
        j = i + 1
        while j < nnz and rest[order[j]] == r:
            j += 1
        a_in = np.zeros(4, dtype=np.complex128)
        base_x = np.uint64(0)
        for idx in range(i, j):
            oi = order[idx]
            base_x = x[oi] & inv12
            a_in[idx2[oi]] = alpha[oi]
        for out_idx in range(4):
            a_out = np.complex128(0.0)
            for in_idx in range(4):
                a_out += U4[out_idx, in_idx] * a_in[in_idx]
            if abs(a_out) > 1e-30:
                ob1 = np.int64((out_idx >> 1) & 1)
                ob2 = np.int64(out_idx & 1)
                xv = base_x
                if ob1:
                    xv |= mask1
                if ob2:
                    xv |= mask2
                new_x[new_nnz] = xv
                new_alpha[new_nnz] = a_out
                new_nnz += 1
        i = j
    for i in range(new_nnz):
        x[i] = new_x[i]
        alpha[i] = new_alpha[i]
    return new_nnz


@njit(cache=True)
def apply_2qubit_gate_ht(
    x_in, alpha_in, nnz_in, x_out, alpha_out, gate_matrix, q1, q2, ht_keys, ht_vals
):
    """
    Apply a 2-qubit gate with hash-table deduplication.  O(nnz), no sort.

    Each input state contributes up to 4 output states.  Duplicate output
    basis states (from different input states mapping to the same output)
    are accumulated directly into alpha_out via ht_lookup/ht_insert.

    The hash table must be EMPTY (all slots = EMPTY sentinel) on entry.
    It maps output basis state → index in x_out/alpha_out.
    Caller must ht_clear() after this call.

    Parameters
    ----------
    x_in, alpha_in  : input state (length >= nnz_in)
    x_out, alpha_out: output buffer (length >= 4*nnz_in)
    gate_matrix     : complex128[4,4]
    q1, q2          : qubit indices
    ht_keys, ht_vals: hash table (EMPTY on entry; capacity >= 8*nnz_in)

    Returns
    -------
    nnz_out
    """
    mask1 = np.uint64(1) << np.uint64(q1)
    mask2 = np.uint64(1) << np.uint64(q2)
    mask12 = mask1 | mask2
    nnz_out = 0

    for i in range(nnz_in):
        xi = x_in[i]
        a = alpha_in[i]
        b1 = np.int64((xi >> np.uint64(q1)) & np.uint64(1))
        b2 = np.int64((xi >> np.uint64(q2)) & np.uint64(1))
        in_idx = (b1 << np.int64(1)) | b2
        base = xi & ~mask12

        for out_idx in range(4):
            amp = gate_matrix[out_idx, in_idx] * a
            if amp.real == 0.0 and amp.imag == 0.0:
                continue

            ob1 = np.int64((out_idx >> 1) & 1)
            ob2 = np.int64(out_idx & 1)
            x_new = base
            if ob1:
                x_new |= mask1
            if ob2:
                x_new |= mask2

            p = ht_lookup(ht_keys, ht_vals, x_new)
            if p >= np.int64(0):
                alpha_out[p] += amp  # accumulate duplicate
            else:
                x_out[nnz_out] = x_new
                alpha_out[nnz_out] = amp
                ht_insert(ht_keys, ht_vals, x_new, nnz_out)
                nnz_out += 1

    return nnz_out


def _manual_kron_2x2(A, B):
    """
    Compute the Kronecker product A⊗B for 2×2 matrices A and B.
    Returns a 4×4 complex128 matrix built by manual block assembly.
    """
    C = np.empty((4, 4), dtype=np.complex128)
    # Top-left block: A[0,0]*B
    C[0, 0] = A[0, 0] * B[0, 0]
    C[0, 1] = A[0, 0] * B[0, 1]
    C[1, 0] = A[0, 0] * B[1, 0]
    C[1, 1] = A[0, 0] * B[1, 1]
    # Top-right block: A[0,1]*B
    C[0, 2] = A[0, 1] * B[0, 0]
    C[0, 3] = A[0, 1] * B[0, 1]
    C[1, 2] = A[0, 1] * B[1, 0]
    C[1, 3] = A[0, 1] * B[1, 1]
    # Bottom-left block: A[1,0]*B
    C[2, 0] = A[1, 0] * B[0, 0]
    C[2, 1] = A[1, 0] * B[0, 1]
    C[3, 0] = A[1, 0] * B[1, 0]
    C[3, 1] = A[1, 0] * B[1, 1]
    # Bottom-right block: A[1,1]*B
    C[2, 2] = A[1, 1] * B[0, 0]
    C[2, 3] = A[1, 1] * B[0, 1]
    C[3, 2] = A[1, 1] * B[1, 0]
    C[3, 3] = A[1, 1] * B[1, 1]
    return C


# Analytical 2×2 Hermitian eigensolver
@njit(cache=True)
def eigh_2x2(rho):
    """
    Analytical eigensolver for a 2×2 Hermitian matrix.

    Returns eigenvalues in DESCENDING order and corresponding
    eigenvectors as columns of V.

    Parameters
    ----------
    rho : complex128[2,2] — Hermitian matrix

    Returns
    -------
    (evals, V) : (float64[2], complex128[2,2])
        evals[0] >= evals[1]; V[:,k] is the k-th eigenvector.
    """
    a = rho[0, 0].real
    d = rho[1, 1].real
    b = rho[0, 1]
    b_sq = b.real**2 + b.imag**2

    tau = (d - a) * 0.5
    disc = sqrt(tau * tau + b_sq)

    lam0 = (a + d) * 0.5 + disc  # larger eigenvalue
    lam1 = (a + d) * 0.5 - disc  # smaller eigenvalue

    evals = np.array([lam0, lam1])
    V = np.zeros((2, 2), dtype=np.complex128)

    if b_sq < 1e-28:
        # Already diagonal — identity or swapped identity
        if a >= d:
            V[0, 0] = np.complex128(1.0)
            V[1, 1] = np.complex128(1.0)
        else:
            V[0, 1] = np.complex128(1.0)
            V[1, 0] = np.complex128(1.0)
    else:
        # v_max = [b, tau+disc],  v_min = [b, tau-disc]
        td_p = tau + disc
        td_m = tau - disc
        n0 = sqrt(b_sq + td_p * td_p)
        n1 = sqrt(b_sq + td_m * td_m)
        V[0, 0] = b / n0
        V[1, 0] = np.complex128(td_p) / n0
        V[0, 1] = b / n1
        V[1, 1] = np.complex128(td_m) / n1

    return evals, V


# BASS new kernels
@njit(cache=True)
def compute_all_1qubit_rdms(x, alpha, nnz, N, ht_keys, ht_vals):
    """
    Compute all N single-qubit RDMs in O(N*nnz) using hash-table partner lookup.

    For each state i and each qubit q:
    - Diagonal: accumulate |alpha_i|² into rho[q, bit, bit].
    - Off-diagonal (processed only when bit_q == 0 to avoid double-counting):
        look up the partner x_i | (1<<q) in O(1) via ht_lookup.

    ht_keys/ht_vals must already be populated with x[:nnz] (built before call).
    No argsort, no binary search — tight O(N*nnz) lower bound.
    """
    rho = np.zeros((N, 2, 2), dtype=np.complex128)
    for i in range(nnz):
        xi = x[i]
        ai = alpha[i]
        prob = ai.real * ai.real + ai.imag * ai.imag
        for q in range(N):
            bit = np.int64((xi >> np.uint64(q)) & np.uint64(1))
            rho[q, bit, bit] += prob
            # Off-diagonal: only when bit == 0 (partner has bit == 1 → larger x)
            if bit == np.int64(0):
                partner = xi | (np.uint64(1) << np.uint64(q))
                p = ht_lookup(ht_keys, ht_vals, partner)
                if p >= np.int64(0):
                    aj = alpha[p]
                    rho[q, 0, 1] += ai * np.conj(aj)
                    rho[q, 1, 0] += aj * np.conj(ai)
    return rho


def _is_gate_diagonal(G, tol=1e-12):
    """Return True if 4×4 matrix G is diagonal (off-diagonals all < tol)."""
    return np.max(np.abs(G - np.diag(np.diag(G)))) < tol


class BASS:
    """
    Three improvements over normal simulatoion.

    1. Batched 1-qubit RDM computation (compute_all_1qubit_rdms):
    Sort x once; binary-search for qubit-flip partners.
    N argsorts → 1 argsort + O(N·nnz·log nnz) binary searches.

    2. Deferred truncation (truncate_every > 1):
    Accumulate gates before enforcing top-k budget.
    Larger intermediate frontier → better amplitude selection.
    Hard cap at max_nnz_factor·k prevents OOM.

    3. Basis-aligned 2-qubit truncation (use_2qubit_rotations=True):
    After 1-qubit RDM optimisation, run two non-overlapping brick-wall
    passes (even pairs (0,1),(2,3),… then odd pairs (1,2),(3,4),…).
    For each pair: diagonalise the 2-qubit RDM → V, apply V† to stored
    state (concentrates amplitude), truncate top-k in rotated basis,
    undo rotation (apply V back), re-truncate.  Keep result only if
    participation ratio improves (do-no-harm).

    No V_pair tracking is needed: the rotation is applied and undone
    entirely within _optimize_basis. The gain comes from better top-k
    amplitude selection in the 2-qubit eigenbasis.
    """

    def __init__(
        self,
        num_qubits,
        k,
        optimize_every=5,
        buffer_factor=2,
        truncate_every=1,
        max_nnz_factor=8,
        use_2qubit_rotations=False,
        verbose=False,
        use_fast_eigh=True,
        use_diag_gate=True,
        use_kron_cache=True,
        pr_opt_threshold=0.90,
    ):
        self.N = num_qubits
        self.k = k

        self.optimize_every = optimize_every
        self.buffer_factor = buffer_factor
        self.verbose = verbose

        self.use_fast_eigh = use_fast_eigh
        self.use_diag_gate = use_diag_gate
        self.use_kron_cache = use_kron_cache

        self._kron_cache = {}
        self.U = None
        self._gamma_running = 1.0
        self.truncate_every = truncate_every
        self.max_nnz_factor = max_nnz_factor
        self.use_2qubit_rotations = use_2qubit_rotations
        self.pr_opt_threshold = pr_opt_threshold
        self._pr_at_last_opt = 0.0  # 0 → first call always fires

        # Compact O(k) hash table shared by two hot paths:
        #   1. _optimize_basis rotation loop  — max nnz ~2k (amplitude-gated)
        #   2. apply_2qubit_gate_ht           — max nnz ~4*hard_cap
        # Size = next power of 2 >= 8*hard_cap  → load ≤ 50% in worst case.
        # Memory: ~8*hard_cap*(8+4) bytes = 384 KB at k=512, independent of N.
        hard_cap = min(max_nnz_factor * k, 2**num_qubits)
        ht_cap = 1
        while ht_cap < 8 * hard_cap:
            ht_cap <<= 1
        self._ht_keys = np.full(ht_cap, np.iinfo(np.uint64).max, dtype=np.uint64)
        self._ht_vals = np.zeros(ht_cap, dtype=np.int32)

    def _renorm(self, state):
        norm = np.sqrt(np.sum(np.abs(state.alpha[: state.nnz]) ** 2))
        if norm > 1e-30:
            state.alpha[: state.nnz] /= norm

    def _truncate_to_k(self, bx, ba, cur_n):
        if cur_n <= self.k:
            return cur_n
        probs = np.abs(ba[:cur_n]) ** 2
        top_idx = np.argpartition(probs, -self.k)[-self.k :]
        bx[: self.k] = bx[top_idx].copy()
        ba[: self.k] = ba[top_idx].copy()
        return self.k

    def _truncate_state(self, state):
        if state.nnz <= self.k:
            return
        probs = np.abs(state.alpha[: state.nnz]) ** 2
        top_idx = np.argpartition(probs, -self.k)[-self.k :]
        state.x[: self.k] = state.x[top_idx].copy()
        state.alpha[: self.k] = state.alpha[top_idx].copy()
        state.nnz = self.k

    def _pr(self, state):
        if state.nnz == 0:
            return float("inf")
        denom = float(np.sum(np.abs(state.alpha[: state.nnz]) ** 4))
        return 1.0 / denom if denom > 1e-60 else float("inf")

    def _get_kron(self, q1, q2):
        """Return U[q1] ⊗ U[q2], optionally from cache."""
        if not self.use_kron_cache:
            return _manual_kron_2x2(self.U[q1], self.U[q2])
        key = (q1, q2)
        if key not in self._kron_cache:
            self._kron_cache[key] = _manual_kron_2x2(self.U[q1], self.U[q2])
        return self._kron_cache[key]

    def _invalidate_kron(self, qubit):
        """Remove all cache entries that involve the given qubit."""
        keys = [k for k in self._kron_cache if qubit in k]
        for k in keys:
            del self._kron_cache[k]

    def _rotate_qubit(self, state, qubit, U):
        needed = 2 * state.nnz
        if needed > len(state.x):
            old_x = state.x[: state.nnz].copy()
            old_a = state.alpha[: state.nnz].copy()
            state.x = np.empty(needed, dtype=np.uint64)
            state.alpha = np.empty(needed, dtype=np.complex128)
            state.x[: state.nnz] = old_x
            state.alpha[: state.nnz] = old_a
        state.nnz = apply_1qubit_rotation_sparse(
            state.x, state.alpha, state.nnz, qubit, U
        )

    def to_statevector(self, state):
        if self.N > 24:
            raise ValueError(f"N={self.N} too large for statevector")
        psi = np.zeros(2**self.N, dtype=np.complex128)
        for i in range(state.nnz):
            psi[int(state.x[i])] = state.alpha[i]
        # Vectorized basis rotation: reshape so qubit j is along axis 1,
        # apply Uj via einsum, reshape back. O(N·2^N) numpy ops, no Python loop.
        for j in range(self.N):
            Uj = self.U[j]
            psi3d = psi.reshape(-1, 2, 1 << j)  # (2^(N-j-1), 2, 2^j)
            psi3d = np.einsum("ij,kjl->kil", Uj, psi3d)  # apply Uj along axis 1
            psi = psi3d.reshape(-1)
        return psi

    # Basis optimisation (1-qubit batched + optional 2-qubit)
    def _ht_reset(self):
        """Clear the hash table to the EMPTY sentinel.  O(ht_cap)."""
        self._ht_keys.fill(np.iinfo(np.uint64).max)

        # 2-qubit helpers

    def _apply_2q_rotation(self, state, q1, q2, V4):
        """Apply 4×4 unitary V4 in-place, resizing state buffers if needed."""
        needed = 4 * state.nnz
        if needed > len(state.x):
            new_x = np.empty(needed, dtype=np.uint64)
            new_alpha = np.empty(needed, dtype=np.complex128)
            new_x[: state.nnz] = state.x[: state.nnz]
            new_alpha[: state.nnz] = state.alpha[: state.nnz]
            state.x = new_x
            state.alpha = new_alpha
        state.nnz = apply_2qubit_rotation_sparse(
            state.x, state.alpha, state.nnz, q1, q2, V4
        )

    def _apply_2q_gate_ht(self, state, q1, q2, G):
        """
        Apply a 2-qubit gate using hash-table deduplication.  O(nnz), no sort.
        The shared hash table must be in the RESET state on entry.
        """
        needed = 4 * state.nnz
        if needed > len(state._temp_x):
            state._temp_x = np.empty(needed, dtype=np.uint64)
            state._temp_alpha = np.empty(needed, dtype=np.complex128)

        state._temp_nnz = apply_2qubit_gate_ht(
            state.x,
            state.alpha,
            state.nnz,
            state._temp_x,
            state._temp_alpha,
            G,
            q1,
            q2,
            self._ht_keys,
            self._ht_vals,
        )

        self._ht_reset()  # restore RESET state for next use

        if state._temp_nnz > state.x.shape[0]:
            new_cap = max(state._temp_nnz, 2 * state.x.shape[0])
            state.x = np.empty(new_cap, dtype=np.uint64)
            state.alpha = np.empty(new_cap, dtype=np.complex128)
        state.x[: state._temp_nnz] = state._temp_x[: state._temp_nnz]
        state.alpha[: state._temp_nnz] = state._temp_alpha[: state._temp_nnz]
        state.nnz = state._temp_nnz

    # Brick-wall 2-qubit truncation-alignment pass

    def _optimize_2q_pass(self, state, pairs):
        """
        Basis-aligned truncation for non-overlapping 2-qubit pairs.

        For each pair (q1, q2):
        1. Diagonalise 2-qubit RDM → V_rot = V†
        2. Apply V_rot to stored state (concentrates amplitude)
        3. Truncate to top-k in rotated basis; track retained probability
        4. Undo V_rot (apply V back)
        5. Re-truncate; track retained probability again
        6. Keep if PR improved (and update _gamma_running); else revert.
        """
        for q1, q2 in pairs:
            rho = compute_2qubit_rdm(state.x, state.alpha, state.nnz, q1, q2)
            evals, evecs = np.linalg.eigh(rho)
            order = np.argsort(-evals)
            V = evecs[:, order]
            V_rot = V.conj().T

            x_save = state.x[: state.nnz].copy()
            a_save = state.alpha[: state.nnz].copy()
            nnz_save = state.nnz
            pr_before = self._pr(state)

            # 1. Rotate → 2q eigenbasis
            self._apply_2q_rotation(state, q1, q2, V_rot)

            # 2. Truncate in rotated basis; capture norm² before renorm
            self._truncate_state(state)
            frac1 = float(np.sum(np.abs(state.alpha[: state.nnz]) ** 2))
            self._renorm(state)

            # 3. Undo rotation
            self._apply_2q_rotation(state, q1, q2, V)

            # 4. Re-truncate
            self._truncate_state(state)
            frac2 = float(np.sum(np.abs(state.alpha[: state.nnz]) ** 2))
            self._renorm(state)

            pr_after = self._pr(state)

            if pr_after > pr_before:
                # no probability was actually lost
                state.x[:nnz_save] = x_save
                state.alpha[:nnz_save] = a_save
                state.nnz = nnz_save
            else:
                # track probability lost through both truncations
                self._gamma_running *= np.sqrt(frac1 * frac2)

    def _optimize_basis(self, state):
        if self._pr(state) < self.k / 4:
            return

        # 1-qubit pass: multi-pass coordinate descent
        # Each pass rebuilds RDMs from the current (possibly updated) state.
        # Stops when a full pass over all N qubits yields no improvement,
        # or after max_passes iterations.  Typical convergence: 1–2 passes.
        max_passes = 3
        for _pass in range(max_passes):
            # Hash table is only needed for RDM computation.
            # _rotate_qubit uses apply_1qubit_rotation_sparse (exact, no HT).
            self._ht_reset()
            ht_build(state.x, state.nnz, self._ht_keys, self._ht_vals)

            all_rho = compute_all_1qubit_rdms(
                state.x, state.alpha, state.nnz, self.N, self._ht_keys, self._ht_vals
            )

            # Clear hash table — not needed in the j-loop.
            self._ht_reset()

            pr = self._pr(state)
            improved_this_pass = False

            for j in range(self.N):
                rho = all_rho[j]
                if self.use_fast_eigh:
                    _, V = eigh_2x2(rho)
                else:
                    evals, V = np.linalg.eigh(rho)
                    V = V[:, np.argsort(-evals)]
                    if np.linalg.det(V).real < 0:
                        V[:, 1] *= -1
                if abs(V[0, 1]) + abs(V[1, 0]) < 1e-10:
                    continue

                x_save = state.x[: state.nnz].copy()
                a_save = state.alpha[: state.nnz].copy()
                nnz_save = state.nnz
                U_save = self.U[j].copy()

                # EXACT rotation — basis change is a coordinate transform,
                # not an approximation.  _rotate_qubit always creates both
                # branches regardless of amplitude, so U[j] and stored
                # amplitudes remain consistent.  _rotate_qubit_ht is only
                # appropriate for gate application (physical approximation).
                self._rotate_qubit(state, j, V.conj().T)

                # Truncate to k so PR comparison is always on k states.
                # Without this, nnz doubles (singletons expand to partners),
                # PR rises purely from state count, and every rotation is
                # rejected — causing BASS to do no optimization at large k.
                self._truncate_state(state)

                # Capture the retained norm before renormalizing the state
                frac_retained = float(np.sum(np.abs(state.alpha[: state.nnz]) ** 2))

                # Renorm after truncation so PR is comparable to pr (normalized).
                # Without renorm, truncated amplitudes have norm < 1; Σ|α|^4 is
                # smaller by r² where r = retained norm², making pr_new > pr even
                # when the rotation genuinely improved concentration.  This biases
                # every comparison toward rejection and kills large-k optimization.
                self._renorm(state)

                pr_new = self._pr(state)

                # Ensure strict inequality: reject if PR increases OR stays exactly the same
                if pr_new >= pr:
                    # Revert to saved state.
                    state.x[:nnz_save] = x_save
                    state.alpha[:nnz_save] = a_save
                    state.nnz = nnz_save
                    self.U[j] = U_save
                else:
                    self.U[j] = self.U[j] @ V
                    pr = pr_new

                    # Multiply the running gamma by the retained mass
                    self._gamma_running *= np.sqrt(
                        frac_retained if frac_retained > 1e-30 else 0
                    )

                    self._invalidate_kron(j)
                    improved_this_pass = True

            if not improved_this_pass:
                break  # converged — no further passes needed

        # --- 2-qubit brick-wall passes ---
        if self.use_2qubit_rotations and self.N >= 2:
            even_pairs = [(i, i + 1) for i in range(0, self.N - 1, 2)]
            odd_pairs = [(i, i + 1) for i in range(1, self.N - 1, 2)]
            self._optimize_2q_pass(state, even_pairs)
            self._optimize_2q_pass(state, odd_pairs)

        self._truncate_state(state)
        self._renorm(state)

    def optimize_trigger(self, state):
        """
        Adaptive trigger: only run _optimize_basis when PR has degraded
        significantly since the last call.

        PR measures sparsity (1/Σ|α|⁴).  Low PR → concentrated (good).
        After _optimize_basis, PR is at a local minimum.  We skip the next
        call if PR hasn't grown by more than (1 / pr_opt_threshold - 1),
        i.e., the basis is still nearly optimal.

        pr_opt_threshold=0.90 means: skip unless PR increased by >11%.
        pr_opt_threshold=1.00 means: never skip (original behaviour).
        """
        if self.pr_opt_threshold >= 1.0:
            return True
        current_pr = self._pr(state)
        return current_pr > self._pr_at_last_opt / self.pr_opt_threshold

    def simulate(self, circuit, initial_state=0):
        self.U = [np.eye(2, dtype=np.complex128) for _ in range(self.N)]
        self._kron_cache = {}
        self._pr_at_last_opt = 0.0  # 0 → first optimize_trigger always fires

        hard_cap = min(self.max_nnz_factor * self.k, 2**self.N)
        internal_k = min(max(max(self.buffer_factor, 2) * self.k, hard_cap), 2**self.N)
        state = SparseState(self.N, internal_k, initial_state)

        M = len(circuit)
        self._gamma_running = 1.0
        truncation_active = False
        gates_since_trunc = 0

        for i, gate in enumerate(circuit):
            if gate.n_qubits == 1:
                q = gate.qubits[0]
                G = self.U[q].conj().T @ gate.matrix @ self.U[q]
                tg = SingleQubitGate(q, G)
                state.apply_gate(tg, "top-k")
                gates_since_trunc += 1
            elif gate.n_qubits == 2:
                q1, q2 = gate.qubits
                Ut = self._get_kron(q1, q2)
                G = Ut.conj().T @ gate.matrix @ Ut
                if self.use_diag_gate and _is_gate_diagonal(G):
                    apply_2qubit_diagonal_kernel(
                        state.x,
                        state.alpha,
                        state.nnz,
                        q1,
                        q2,
                        np.diag(G).astype(np.complex128),
                    )
                    # nnz unchanged — skip generic apply and truncation block
                    if (
                        self.optimize_every > 0
                        and (i + 1) % self.optimize_every == 0
                        and i + 1 < M
                    ):
                        if truncation_active and self.optimize_trigger(state):
                            self._optimize_basis(state)
                            self._pr_at_last_opt = self._pr(state)
                        truncation_active = False
                    continue
                else:
                    # Non-diagonal 2q gate: hash-table dedup, no sort+merge
                    self._apply_2q_gate_ht(state, q1, q2, G)
                    gates_since_trunc += 1
            else:
                raise ValueError(f"{gate.n_qubits}-qubit gates unsupported")

            over_hard = state.nnz > hard_cap
            over_budget = state.nnz > self.k
            should_trunc = over_hard or (
                over_budget and gates_since_trunc >= self.truncate_every
            )

            if should_trunc:
                truncation_active = True
                probs = np.abs(state.alpha[: state.nnz]) ** 2
                total = np.sum(probs)
                top_idx = np.argpartition(probs, -self.k)[-self.k :]
                kept = np.sum(probs[top_idx])
                self._gamma_running *= kept / total if total > 1e-30 else 0
                state.x[: self.k] = state.x[top_idx].copy()
                state.alpha[: self.k] = state.alpha[top_idx].copy()
                state.nnz = self.k
                self._renorm(state)
                gates_since_trunc = 0

            if (
                self.optimize_every > 0
                and (i + 1) % self.optimize_every == 0
                and i + 1 < M
            ):
                if truncation_active and self.optimize_trigger(state):
                    self._optimize_basis(state)
                    self._pr_at_last_opt = self._pr(state)
                truncation_active = False

        state.gamma = np.sqrt(max(self._gamma_running, 0))
        return state
