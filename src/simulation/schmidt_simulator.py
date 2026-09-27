"""
Schmidt-Weighted Truncation Simulator
======================================
Implements Schmidt-1cut and Schmidt-3cut truncation strategies as described in
Section IV of the BASS paper (arXiv:2605.27285).

These are competing methods against top-k truncation, used to demonstrate
Lemma 2: Schmidt-sector diversity provably lowers retained probability at
fixed k compared to the top-k strategy.

Bipartition convention (matches sparse_state.py bit ordering):
qubit q -> bit position q in integer x
qubit 0 = bit 0 (LSB), qubit N-1 = bit N-1 (MSB)

For bipartition at qubit `cut`:
LEFT  partition: qubits 0..cut-1   (lower bits, x & ((1<<cut)-1))
RIGHT partition: qubits cut..N-1   (upper bits, x >> cut)

Schmidt sector = RIGHT partition value = x >> cut.
C(x_i) = #{j : (x_j >> cut) == (x_i >> cut)} among stored states.

Schmidt-1cut score: s_i = |alpha_i|^2 / C_i          (cut at N//2)
Schmidt-3cut score: s_i = |alpha_i|^2 / geomean(C1, C2, C3)
                    cuts at N//4, N//2, 3N//4
"""

import numpy as np

from src.core.sparse_state import (
    apply_1qubit_gate_kernel_opt,
    apply_2qubit_gate_kernel_opt,
    merge_sorted_kernel,
    radix_sort_uint64,
)

_RADIX_THRESHOLD = 2000  # switch argsort -> radix_sort for large nnz


class SchmidtSimulator:
    """
    Sparse-state simulator with Schmidt-weighted truncation.

    Uses the same Numba gate-application kernels as FixedBasisSimulator,
    but replaces top-k selection with Schmidt-sector-diversified selection.

    Parameters
    ----------
    N : int
        Number of qubits.
    k : int
        Sparse budget (target number of retained basis states).
    cut_type : {"1cut", "3cut"}
        "1cut": Schmidt-1cut, single bipartition at N//2.
        "3cut": Schmidt-3cut, geometric mean of three bipartitions at
                N//4, N//2, and 3N//4.

    Notes
    -----
    This simulator is provided as a reproducibility artifact for Section IV
    of the BASS paper. The Schmidt strategies are suboptimal compared to top-k
    (by Lemma 2) and are included to quantify the penalty from enforcing
    Schmidt-sector diversity.
    """

    def __init__(self, N: int, k: int, cut_type: str = "1cut"):
        if cut_type not in ("1cut", "3cut"):
            raise ValueError(f"cut_type must be '1cut' or '3cut', got {cut_type!r}")
        self.N = N
        self.k = k
        self.cut_type = cut_type

        self._cuts_1 = [N // 2]
        self._cuts_3 = [N // 4, N // 2, 3 * N // 4]

        cap = 8 * k
        self.x = np.zeros(cap, dtype=np.uint64)
        self.alpha = np.zeros(cap, dtype=np.complex128)
        self._tx = np.zeros(4 * cap, dtype=np.uint64)
        self._ta = np.zeros(4 * cap, dtype=np.complex128)
        self.nnz = 0
        self.gamma = 1.0

    def _reset(self):
        self.x[:] = 0
        self.alpha[:] = 0.0
        self.x[0] = np.uint64(0)
        self.alpha[0] = 1.0 + 0j
        self.nnz = 1
        self.gamma = 1.0

    def _sort_merge(self, tnnz: int) -> int:
        if tnnz < _RADIX_THRESHOLD:
            order = np.argsort(self._tx[:tnnz])
            self._tx[:tnnz] = self._tx[order]
            self._ta[:tnnz] = self._ta[order]
        else:
            radix_sort_uint64(self._tx, self._ta, tnnz)
        return merge_sorted_kernel(self._tx, self._ta, tnnz)

    def _copy_back(self, tnnz: int):
        if tnnz > self.x.shape[0]:
            self.x = np.zeros(max(tnnz, 2 * self.x.shape[0]), dtype=np.uint64)
            self.alpha = np.zeros(self.x.shape[0], dtype=np.complex128)
        self.x[:tnnz] = self._tx[:tnnz]
        self.alpha[:tnnz] = self._ta[:tnnz]
        self.nnz = tnnz

    def _apply_1q(self, gate):
        needed = 2 * self.nnz
        if needed > self._tx.shape[0]:
            self._tx = np.zeros(needed * 2, dtype=np.uint64)
            self._ta = np.zeros(needed * 2, dtype=np.complex128)
        tnnz = apply_1qubit_gate_kernel_opt(
            self.x,
            self.alpha,
            self.nnz,
            self._tx,
            self._ta,
            gate.matrix,
            gate.qubits[0],
        )
        self._copy_back(self._sort_merge(tnnz))

    def _apply_2q(self, gate):
        needed = 4 * self.nnz
        if needed > self._tx.shape[0]:
            self._tx = np.zeros(needed * 2, dtype=np.uint64)
            self._ta = np.zeros(needed * 2, dtype=np.complex128)
        tnnz = apply_2qubit_gate_kernel_opt(
            self.x,
            self.alpha,
            self.nnz,
            self._tx,
            self._ta,
            gate.matrix,
            gate.qubits[0],
            gate.qubits[1],
        )
        if tnnz == -1:
            raise RuntimeError("Temp buffer overflow in 2-qubit gate")
        self._copy_back(self._sort_merge(tnnz))

    def _select_and_normalise(self, idx: np.ndarray):
        probs = np.abs(self.alpha[: self.nnz]) ** 2
        retained = float(np.sum(probs[idx]))
        scale = np.sqrt(max(retained, 1e-300))
        self.x[: self.k] = self.x[idx].copy()
        self.alpha[: self.k] = self.alpha[idx].copy() / scale
        self.nnz = self.k
        self.gamma *= scale

    def _truncate(self):
        cuts = self._cuts_1 if self.cut_type == "1cut" else self._cuts_3
        n = self.nnz
        xs = self.x[:n]
        probs = np.abs(self.alpha[:n]) ** 2

        log_count = np.zeros(n, dtype=float)
        for cut in cuts:
            sectors = (xs >> np.uint64(cut)).astype(np.int64)
            _, inverse = np.unique(sectors, return_inverse=True)
            counts = np.bincount(inverse)
            log_count += np.log(np.maximum(counts[inverse], 1).astype(float))

        scores = probs / np.exp(log_count / len(cuts))
        kth = n - self.k
        idx = np.argpartition(scores, kth)[kth:]
        self._select_and_normalise(idx)

    def simulate(self, circuit) -> "SchmidtSimulator":
        """
        Simulate ``circuit`` from |0...0> and return self.

        Parameters
        ----------
        circuit : list[Gate]
            Gate sequence (same format as used by FixedBasisSimulator / BASS).

        Returns
        -------
        self : SchmidtSimulator
            Simulator with updated x, alpha, nnz, gamma attributes.

        Notes
        -----
        Access the final sparse state via self.x[:self.nnz] and
        self.alpha[:self.nnz]. The gamma attribute holds the cumulative
        norm-retention factor. It is useful for diagnosing truncation loss,
        but Schmidt-weighted selection is not the top-k rule, so gamma is not
        advertised as a rigorous fidelity lower bound here.
        """
        self._reset()
        for gate in circuit:
            if gate.n_qubits == 1:
                self._apply_1q(gate)
            elif gate.n_qubits == 2:
                self._apply_2q(gate)
            else:
                raise NotImplementedError(f"{gate.n_qubits}-qubit gates not supported")
            if self.nnz > self.k:
                self._truncate()
        return self

    def fidelity(self, exact_sv: np.ndarray) -> float:
        """
        Compute |<exact|approx>|^2 with the current sparse state.

        Parameters
        ----------
        exact_sv : np.ndarray, shape (2**N,)
            Dense exact statevector in the Z computational basis.

        Returns
        -------
        float
            Fidelity in [0, 1].
        """
        xs = self.x[: self.nnz]
        a = self.alpha[: self.nnz]
        return float(abs(np.dot(exact_sv[xs].conj(), a)) ** 2)
