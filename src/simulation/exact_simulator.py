"""
ExactStatevectorSimulator
=========================
A high-performance, tensor-reshaping exact statevector simulator used as the
ground-truth reference baseline for FixedBasis and BASS algorithm validation.

To guarantee maximum execution speed and completely eliminate the overhead of
explicit index-array generation, this module routes all gate operations through
highly optimized NumPy tensor manipulations and vector-matrix tensor contractions
(BLAS-backed `np.einsum`).

Mathematical Engine & Implementation Mechanics
----------------------------------------------
The full quantum state vector is kept as a contiguous, dense complex array of
size 2^N. To enforce strict memory limits and safeguard against unintended allocation
cascades during large-scale automated sweeps, a hard safety ceiling is applied at
N = 24 qubits (~256 MB complex128 array capacity).

    1. Single-Qubit Tensor Contraction
        To apply an arbitrary 2×2 unitary operator U to qubit q, the flat array is
        dimensionally reshaped into a 3D tensor:
            Ψ_3d ∈ C^{ 2^(N-q-1) × 2 × 2^q }
        Here, axis 1 isolates the explicit target subspace of qubit q. The gate application
        is formulated as a matrix-tensor contraction:
            Ψ'_{k, i, l} = Σ_j U_{i, j} · Ψ_{k, j, l}
        This operation executes via vectorized BLAS pathways, bypassing Python loop structures
        entirely and restoring the state vector through an in-place flat reshape.

    2. Two-Qubit High-Performance Reshaping
        For arbitrary 2-qubit operations acting on spaces (q1, q2), the state vector
        is split along non-adjacent multi-index boundaries. The simulator strictly
        enforces the ordering convention q1 < q2. If a gate presents an inverted topology
        (q1 > q2), an algebraic index permutation [0, 2, 1, 3] is applied directly to the
        4×4 operator matrix to map the row/column basis indices from (b1, b2) to (b2, b1).

The state array is projected into a 5-dimensional tensor array structure:
    Ψ_5d ∈ C^{ A × 2 × B × 2 × C }

    Where the partitioned block dimensions correspond to:
        A = 2^(N - q2 - 1)     : Space above the upper target qubit q2
        axis 1 (size 2)        : Qubit subspace localized at q2 (in_b2)
        B = 2^(q2 - q1 - 1)    : Intermediate space isolated between q2 and q1
        axis 3 (size 2)        : Qubit subspace localized at q1 (in_b1)
        C = 2^q1               : Space below the lower target qubit q1

The 4×4 gate matrix is reshaped into an operator tensor M_{p, q, r, s} where
indices denote (out_b1, out_b2, in_b1, in_b2). The transformation is executed via
a generalized coordinate Einstein summation:
    Ψ'_{a, q, b, p, c} = Σ_{r, s} M_{p, q, r, s} · Ψ_{a, s, b, r, c}

Peer Review & Reproducibility Standard
--------------------------------------
- Execution Overhead Isolation: Avoids index mask lookups, bit-twiddling logic,
and explicit state-vector sorting in Python. This ensures that the exact baseline
is a pure, reproducible representation of classical statevector simulation scaling
exclusively at O(2^N).
- Verifiability: Provides the pristine state array against which non-parametric
statistics (Median/IQR for fidelity, and Geometric Mean/95% Bootstrap CI for
Participation Ratio compression) are rigorously calculated.
"""

import numpy as np
from tqdm import tqdm


class ExactSimulator:
    """
    Exact state vector simulation (only for small N!)
    """

    def __init__(self, num_qubits, verbose=True):
        if num_qubits > 24:
            raise ValueError(f"N={num_qubits} too large for exact simulation")

        self.N = num_qubits
        self.verbose = verbose

    def simulate(self, circuit, initial_state=None):
        """
        Simulate circuit exactly

        Returns:
            Dense state vector
        """
        # Initialize
        state = np.zeros(2**self.N, dtype=np.complex128)
        if initial_state is None:
            initial_state = 0
        state[initial_state] = 1.0

        # Apply gates
        iterator = tqdm(circuit) if self.verbose else circuit

        for gate in iterator:
            state = self.apply_gate_exact(state, gate)

        return state

    def apply_gate_exact(self, state, gate):
        """
        Apply gate to dense state vector
        """
        n_gate = gate.n_qubits
        matrix = gate.matrix
        qubits = gate.qubits

        if n_gate == 1:
            return self._apply_1qubit_dense(state, matrix, qubits[0])
        elif n_gate == 2:
            return self._apply_2qubit_dense(state, matrix, qubits[0], qubits[1])
        else:
            raise NotImplementedError()

    def _apply_1qubit_dense(self, state, matrix, q):
        """Apply single-qubit gate to dense state (reshape, no index arrays)."""
        # Reshape: (2^(N-q-1), 2, 2^q); axis 1 = bit q
        psi3d = state.reshape(-1, 2, 1 << q)
        result = np.einsum("ij,kjl->kil", matrix, psi3d)
        return result.reshape(-1)

    def _apply_2qubit_dense(self, state, matrix, q1, q2):
        """Apply two-qubit gate to dense state (reshape, no index arrays).

        Convention: in_idx = (b1 << 1) | b2 where b1=bit@q1, b2=bit@q2.
        Enforce q1 < q2; permute matrix if needed.
        """
        N = self.N
        if q1 > q2:
            q1, q2 = q2, q1
            perm = [0, 2, 1, 3]  # swap bit ordering: b1↔b2
            matrix = matrix[np.ix_(perm, perm)]

        # Reshape as (A, b2, B, b1, C):
        #   A = 2^(N-q2-1) : bits above q2
        #   b2 (size 2)    : bit at q2   → axis 1
        #   B = 2^(q2-q1-1): bits between
        #   b1 (size 2)    : bit at q1   → axis 3
        #   C = 2^q1       : bits below q1
        A, B, C = 1 << (N - q2 - 1), 1 << (q2 - q1 - 1), 1 << q1
        M = matrix.reshape(2, 2, 2, 2)  # (out_b1, out_b2, in_b1, in_b2)
        psi5 = state.reshape(A, 2, B, 2, C)  # (A, b2=s, B, b1=r, C)
        # result[a,out_b2,b,out_b1,c] = Σ_{r,s} M[p,q,r,s] * psi5[a,s,b,r,c]
        result = np.einsum("pqrs,asbrc->aqbpc", M, psi5)
        return result.reshape(-1)
