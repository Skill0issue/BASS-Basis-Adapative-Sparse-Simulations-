"""
Simple MPS (Matrix Product State) Simulator
============================================
Bond-dimension-truncated simulation via sequential SVD truncation.

Used as a comparison baseline against BASS in the MPS_Comparison.ipynb
notebook. The implementation is intentionally simple and pedagogically clear
rather than maximally optimised.

MPS representation
------------------
The state is stored as N rank-3 tensors::

    tensors[i].shape == (D_left, 2, D_right)

where D_left, D_right ≤ chi.  The leftmost tensor has D_left = 1 and the
rightmost has D_right = 1.  The MPS is kept in *mixed canonical* form during
simulation: all tensors to the left of the most recently updated bond are
left-canonical (Frobenius-column-unitary), all tensors to the right are
not explicitly canonicalised (singular values absorbed into the right
tensor after each SVD step).

Gate application
----------------
* 1-qubit gates: direct einsum contraction, no truncation.
* 2-qubit gates on *adjacent* qubits (|q1 - q2| == 1):
contract the two-site tensor, apply gate, SVD, truncate to chi.
* 2-qubit gates on *non-adjacent* qubits:
SWAP-route the smaller qubit to be adjacent to the larger, apply gate,
SWAP-route back.  Each SWAP step is itself an adjacent 2-qubit gate and
introduces SVD truncation — this correctly captures the overhead that
non-local gates impose on MPS.

Fidelity
--------
``fidelity(exact_sv)`` contracts all tensors to a dense 2^N vector (feasible
only for N <= ~24) and returns |<exact|mps>|^2.  The exported statevector is
renormalized; norm-retention is tracked separately in ``gamma``.

Gamma / norm tracking
---------------------
``_gamma_running`` accumulates the *probability* fraction retained across all
SVD truncations (product of retained_||s||² / total_||s||²).
``gamma = sqrt(_gamma_running)`` uses the same bookkeeping convention as BASS
but is **NOT a rigorous lower bound on sqrt(fidelity)** for MPS.  The BASS
bound (γ² ≤ F) relies on amplitude-space truncation; MPS truncates in the
Schmidt/entanglement spectrum, and renormalization after each SVD can steer
the approximated state away from the exact state, so the product of per-step
retention fractions can exceed the true fidelity overlap.  Empirically we
observe gamma² > F violations up to ~5% for highly entangled local circuits.
Use gamma as a diagnostic of accumulated truncation loss, not as a rigorous
fidelity bound.

Memory
------

Total elements ≈ sum_i D_l[i]*2*D_r[i].  For a uniform chain this is
roughly N * chi² complex128 values.  The matched-memory comparison with BASS
(budget k) uses chi_matched = round(sqrt(k / N)).
"""

import numpy as np
from typing import List

_SWAP_MAT = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
)


class SimpleMPS:
    """
    MPS simulator with SVD bond-dimension truncation.

    Parameters
    ----------
    N : int
        Number of qubits.
    chi : int
        Maximum bond dimension (analogous to k in BASS / FixedBasisSimulator).

    Attributes
    ----------
    tensors : list of ndarray
        MPS tensors, tensors[i].shape == (D_l, 2, D_r).
    gamma : float
        sqrt of cumulative retained-probability product.
        Diagnostic only; unlike BASS top-k truncation, MPS SVD truncation
        plus renormalization does not make gamma² a rigorous fidelity bound.
    """

    def __init__(self, N: int, chi: int):
        self.N = N
        self.chi = chi
        self._gamma_running: float = 1.0
        self.tensors: List[np.ndarray] = []
        self._init_product_state()

    # Initialisation

    def _init_product_state(self):
        """Reset to |0...0⟩."""
        self._gamma_running = 1.0
        self.tensors = []
        for _ in range(self.N):
            T = np.zeros((1, 2, 1), dtype=np.complex128)
            T[0, 0, 0] = 1.0
            self.tensors.append(T)

    @property
    def gamma(self) -> float:
        return float(np.sqrt(max(self._gamma_running, 0.0)))

    # Gate application helpers 

    def _apply_1q(self, mat: np.ndarray, q: int):
        """Apply a 2×2 unitary to site q.  No bond truncation needed."""
        T = self.tensors[q]  # (D_l, 2, D_r)
        self.tensors[q] = np.einsum("ij,ljr->lir", mat, T)

    def _apply_2q_adjacent(self, mat: np.ndarray, q1: int, q2: int):
        """
        Apply a 4×4 gate to adjacent sites q1, q2 = q1 + 1.

        Steps
        -----
        1. Contract the two-site tensor theta = A ⊗ B.
        2. Apply gate.
        3. SVD + truncate to self.chi.
        4. Update tensors[q1] (left-canonical) and tensors[q2] (right factor).
        """
        assert q2 == q1 + 1, "qubits must be adjacent"

        A = self.tensors[q1]  # (D_l, 2, D_m)
        B = self.tensors[q2]  # (D_m, 2, D_r)
        D_l = A.shape[0]
        D_r = B.shape[2]

        # Contract: theta[D_l, 2, 2, D_r]
        theta = np.tensordot(
            A, B, axes=([2], [0])
        )  # (D_l, 2, D_m) x (D_m, 2, D_r) -> (D_l, 2, 2, D_r)

        # Apply gate: G[out_0, out_1, in_0, in_1]
        G = mat.reshape(2, 2, 2, 2)
        # theta_new[l, a, b, r] = sum_{i,j} G[a,b,i,j] * theta[l,i,j,r]
        theta_new = np.einsum("abij,lijr->labr", G, theta)  # (D_l, 2, 2, D_r)

        # SVD on the (D_l*2) × (2*D_r) matrix
        M = theta_new.reshape(D_l * 2, 2 * D_r)
        try:
            U, s, Vh = np.linalg.svd(M, full_matrices=False)
        except np.linalg.LinAlgError:
            # numpy's gesdd driver can fail on ill-conditioned matrices;
            # fall back to scipy's gesvd (slower but more robust).
            from scipy.linalg import svd as _scipy_svd

            U, s, Vh = _scipy_svd(M, full_matrices=False, lapack_driver="gesvd")

        # Track norm retention
        total_norm_sq = float(np.dot(s, s))
        chi_eff = min(self.chi, len(s))
        retained_norm_sq = float(np.dot(s[:chi_eff], s[:chi_eff]))
        frac = retained_norm_sq / total_norm_sq if total_norm_sq > 1e-300 else 1.0
        self._gamma_running *= frac

        # Renormalise so the MPS stays unit-norm
        scale = float(np.sqrt(retained_norm_sq)) if retained_norm_sq > 1e-300 else 1.0
        s_norm = s[:chi_eff] / scale

        # Restore tensors
        self.tensors[q1] = U[:, :chi_eff].reshape(D_l, 2, chi_eff)
        self.tensors[q2] = (np.diag(s_norm) @ Vh[:chi_eff, :]).reshape(chi_eff, 2, D_r)

    def _apply_swap(self, q1: int, q2: int):
        """Apply a physical SWAP gate on adjacent qubits q1, q2 = q1+1."""
        self._apply_2q_adjacent(_SWAP_MAT, q1, q2)

    # Non-adjacent gate via SWAP routing 

    def _apply_2q(self, mat: np.ndarray, q1: int, q2: int):
        """
        Apply a 4×4 gate to qubits q1 and q2 (possibly non-adjacent).

        For non-adjacent qubits (|q1-q2| > 1), bubbles q1 right to q2-1 via
        SWAP gates, applies the gate, then reverses the SWAPs.  Each SWAP is
        an SVD step and introduces truncation error — this faithfully models
        the MPS overhead for long-range gates.

        Gate qubit ordering convention: gate.matrix rows/cols are ordered with
        q1 as the *most significant* qubit (row index = s_q1 * 2 + s_q2).
        When q1 > q2 we permute the gate matrix so the smaller index is on the
        left in the MPS.
        """
        if q1 == q2:
            return

        # Ensure q_lo < q_hi; permute gate if needed
        if q1 < q2:
            q_lo, q_hi, mat_out = q1, q2, mat
        else:
            q_lo, q_hi = q2, q1
            # Swap the two qubit slots: mat_out = SWAP @ mat @ SWAP
            mat_out = _SWAP_MAT @ mat @ _SWAP_MAT

        if q_hi == q_lo + 1:
            self._apply_2q_adjacent(mat_out, q_lo, q_hi)
            return

        # Bubble q_lo to position q_hi - 1 via SWAPs
        for pos in range(q_lo, q_hi - 1):
            self._apply_swap(pos, pos + 1)
        # q_lo's state is now at position q_hi - 1
        self._apply_2q_adjacent(mat_out, q_hi - 1, q_hi)
        # Undo the SWAPs
        for pos in range(q_hi - 2, q_lo - 1, -1):
            self._apply_swap(pos, pos + 1)

    # Public interface 

    def simulate(self, circuit) -> "SimpleMPS":
        """
        Simulate ``circuit`` from |0...0⟩ and return self.

        Parameters
        ----------
        circuit : list[Gate]
            Gate sequence (same format as FixedBasisSimulator / BASS).
        """
        self._init_product_state()
        for gate in circuit:
            if gate.n_qubits == 1:
                self._apply_1q(gate.matrix, gate.qubits[0])
            elif gate.n_qubits == 2:
                self._apply_2q(gate.matrix, gate.qubits[0], gate.qubits[1])
            else:
                raise NotImplementedError(f"{gate.n_qubits}-qubit gates not supported")
        return self

    def to_statevector(self) -> np.ndarray:
        """
        Contract the MPS to a dense 2^N statevector in LSB qubit ordering.

        LSB ordering: index bit q = qubit q (qubit 0 = least significant bit).
        This matches the convention used by ExactSimulator and BASS.

        Only feasible for N <= ~24.  Returns a normalized vector.  The
        accumulated SVD retention diagnostic is available separately as
        ``gamma``.

        Returns
        -------
        np.ndarray, shape (2**N,), dtype complex128
        """
        # Contract MPS left-to-right; this naturally produces MSB ordering
        # (qubit 0 = most significant bit because tensor[0] contributes the
        # leading index in the reshaped product).
        T0 = self.tensors[0]  # (1, 2, D_r)
        psi = T0[0, :, :]  # (2, D_r)
        for i in range(1, self.N):
            T = self.tensors[i]  # (D_l, 2, D_r)
            psi = np.tensordot(psi, T, axes=([1], [0]))  # (..., 2, D_r)
            psi = psi.reshape(-1, T.shape[2])  # (2^(i+1), D_r)
        psi_msb = psi.reshape(-1)

        # Convert MSB → LSB: reshape to (2,)*N, reverse axis order, flatten.
        # After reversal axis k = old axis (N-1-k), so the new index encodes
        # bit q in position q (qubit 0 in the LSB position).
        cube = psi_msb.reshape([2] * self.N)
        cube = cube.transpose(list(range(self.N - 1, -1, -1)))
        out = np.ascontiguousarray(cube).reshape(-1)
        norm = float(np.linalg.norm(out))
        if norm > 1e-300:
            out = out / norm
        return out

    def fidelity(self, exact_sv: np.ndarray) -> float:
        """
        Compute |⟨exact|mps⟩|² against a dense exact statevector.

        Parameters
        ----------
        exact_sv : np.ndarray, shape (2**N,)
            Exact normalised statevector in the Z computational basis.

        Returns
        -------
        float in [0, 1].
        """
        approx = self.to_statevector()
        return float(abs(np.dot(exact_sv.conj(), approx)) ** 2)

    def memory_elements(self) -> int:
        """Total number of complex128 elements stored across all tensors."""
        return sum(T.size for T in self.tensors)

    def memory_bytes(self) -> int:
        """Peak memory estimate in bytes (complex128 = 16 bytes each)."""
        return self.memory_elements() * 16

    @staticmethod
    def chi_for_k(N: int, k: int) -> int:
        """
        Return the bond dimension whose memory matches BASS/Fixed budget k.

        Matched-memory criterion: N * chi^2 ≈ k  →  chi = round(sqrt(k/N)).
        Minimum chi = 1 (product-state MPS).
        """
        return max(1, round(float(k / N) ** 0.5))
