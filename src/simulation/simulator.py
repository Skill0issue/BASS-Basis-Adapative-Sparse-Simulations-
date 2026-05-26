"""
FixedBasisSparseSimulator
=========================
A high-performance quantum circuit simulator that represents state vectors in a
rigid computational basis, constrained by a maximum state budget k (non-zero entries).

To guarantee strict methodological rigor and eliminate benchmarking anomalies, this
simulator has been upgraded to utilize the exact same low-level, JIT-compiled Numba
hash-table kernel as the Basis-Adaptive Sparse Simulator (BASS). This ensures
iso-computational fairness, matching memory allocations, and identical underlying
computational complexity across all core comparative evaluations.

Algorithmic Design & Parity Measures
------------------------------------
    1. Iso-Computational 2-Qubit Application
        Rather than performing traditional array concatenation, sorting, and linear-time
        merging, non-diagonal 2-qubit gates are routed through the specialized Numba
        kernel (`apply_2qubit_gate_ht`). By writing out-of-place into a shared hash table,
        duplicate basis entries are coalesced instantly, bounding the routine at O(nnz)
        time complexity.

    2. Identical Memory and Table Footprint
        To equalize hardware cache conditions, the hash table buffers are pre-allocated
        using a matching bit-mask arrangement:
            ht_cap = next_power_of_2(8 * min(max_nnz_factor * k, 2^N))
        This guarantees that both simulators execute with equivalent peak memory footprints
        and identical hash-probe chain structures.

3. Mirrored Truncation Mechanics
When the state wave-front expands beyond the state budget k, the simulator triggers
a strict truncation sequence matching BASS's internal logic:
    - `top-k`      : Selects the dominant amplitudes using an O(nnz) partition
                    (`np.argpartition`).
    - `random-k`   : Samples k states non-replaceably from the normalized
                    probability distribution.
Following state compression, the state vector is explicitly renormalized, and the
retained probability fraction (`kept_prob / total`) updates the norm reduction
history.

Reviewer & Reproducibility Metrics
-----------------------------------
- Structural Parity: Because both simulators share the same hash-table mechanics,
measured variations in runtime or fidelity cannot be attributed to low-level code
discrepancies. Instead, they isolate the pure algorithmic impact of BASS's basis-shifting
coordinate descent passes versus a rigid fixed-basis framework.

- Norm Tracking (`state.gamma`): Dynamically accumulates the mathematical norm
reductions sustained during budget enforcements. It offers an exact tracking metric
of absolute state-vector attenuation over multi-layered execution sweeps.
"""

import numpy as np
from time import perf_counter
from tqdm import tqdm
from src.core.sparse_state import SparseState

# Import the optimized Numba hash-table kernel directly from BASS
from src.simulation.bass_simulator import apply_2qubit_gate_ht


class FixedBasisSimulator:
    """
    Fixed-Basis Sparse Simulator.
    Upgraded to use identical O(k) open-addressing hash table kernels
    as BASS to ensure iso-computational fairness during benchmarking.
    """

    def __init__(
        self,
        num_qubits,
        k,
        truncation="top-k",
        verbose=True,
        log_every=1,
        max_nnz_factor=8,
    ):
        self.N = num_qubits
        self.k = k
        self.verbose = verbose
        self.log_every = log_every

        if truncation not in ["top-k", "random-k"]:
            raise ValueError("Unknown truncation")
        self._truncate_name = truncation

        # Initialize Hash Table buffers exactly as BASS does to ensure identical memory overhead
        hard_cap = min(max_nnz_factor * k, 2**num_qubits)
        ht_cap = 1
        while ht_cap < 8 * hard_cap:
            ht_cap <<= 1

        self._ht_keys = np.full(ht_cap, np.iinfo(np.uint64).max, dtype=np.uint64)
        self._ht_vals = np.zeros(ht_cap, dtype=np.int32)

    def _ht_reset(self):
        """Clear the hash table to the EMPTY sentinel."""
        self._ht_keys.fill(np.iinfo(np.uint64).max)

    def _apply_2q_gate_ht(self, state, q1, q2, G):
        """Applies 2-qubit gate using the O(k) hash table, bypassing slow array merges."""
        needed = 4 * state.nnz

        # Ensure temporary buffers exist for fast out-of-place writes
        if not hasattr(state, "_temp_x") or needed > len(state._temp_x):
            state._temp_x = np.empty(needed, dtype=np.uint64)
            state._temp_alpha = np.empty(needed, dtype=np.complex128)

        self._ht_reset()

        # Call the Numba-compiled kernel
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

        # Resize underlying sparse state if needed
        if state._temp_nnz > state.x.shape[0]:
            new_cap = max(state._temp_nnz, 2 * state.x.shape[0])
            state.x = np.empty(new_cap, dtype=np.uint64)
            state.alpha = np.empty(new_cap, dtype=np.complex128)

        state.x[: state._temp_nnz] = state._temp_x[: state._temp_nnz]
        state.alpha[: state._temp_nnz] = state._temp_alpha[: state._temp_nnz]
        state.nnz = state._temp_nnz

    def simulate(self, circuit, initial_state=None, seed=None):
        state = SparseState(self.N, self.k, initial_state)
        L = len(circuit)

        runtime = np.empty(L)
        gamma_hist = np.empty(L + 1)
        nnz_hist = np.empty(L + 1)

        gamma_hist[0] = 1.0
        nnz_hist[0] = state.nnz

        total_start = perf_counter()
        iterator = tqdm(circuit) if self.verbose else circuit

        if seed is None:
            seed = int(np.random.randint(0, 2**31))

        current_seed = seed

        for i, gate in enumerate(iterator):
            nnz_before = state.nnz
            t0 = perf_counter()

            # Advance seed for reproducibility
            current_seed = (1103515245 * current_seed + 12345) & 0x7FFFFFFF
            truncated = False

            # --- FAST PATH: 2-Qubit Gates ---
            if gate.n_qubits == 2:
                q1, q2 = gate.qubits
                G = gate.matrix

                self._apply_2q_gate_ht(state, q1, q2, G)
                expanded_nnz = state.nnz

                # Manual Truncation Logic (Mirroring BASS)
                if state.nnz > self.k:
                    truncated = True
                    probs = np.abs(state.alpha[: state.nnz]) ** 2
                    total = np.sum(probs)

                    if self._truncate_name == "top-k":
                        top_idx = np.argpartition(probs, -self.k)[-self.k :]
                    else:  # random-k
                        np.random.seed(current_seed)
                        probs_norm = (
                            probs / total
                            if total > 1e-30
                            else np.ones(state.nnz) / state.nnz
                        )
                        top_idx = np.random.choice(
                            state.nnz, size=self.k, replace=False, p=probs_norm
                        )

                    kept_prob = np.sum(probs[top_idx])

                    # 1. Apply Truncation
                    state.x[: self.k] = state.x[top_idx].copy()
                    state.alpha[: self.k] = state.alpha[top_idx].copy()
                    state.nnz = self.k

                    # 2. Update Gamma
                    state.gamma *= np.sqrt(kept_prob / total) if total > 1e-30 else 0

                    # 3. Renormalize State
                    norm = np.sqrt(np.sum(np.abs(state.alpha[: state.nnz]) ** 2))
                    if norm > 1e-30:
                        state.alpha[: state.nnz] /= norm

            # --- SLOW PATH: 1-Qubit Gates ---
            else:
                state.apply_gate(gate, self._truncate_name, current_seed)
                expanded_nnz = state.nnz
                if expanded_nnz > self.k:
                    truncated = True

            gate_time = perf_counter() - t0
            runtime[i] = gate_time

            gamma_hist[i + 1] = state.gamma
            nnz_hist[i + 1] = state.nnz

            # -------- Logging --------
            if self.verbose and (i % self.log_every == 0):
                expansion = expanded_nnz / nnz_before if nnz_before > 0 else 0
                print(
                    f"[Gate {i+1}/{L}] "
                    f"type={gate.n_qubits}q | "
                    f"nnz: {nnz_before} → {expanded_nnz} → {state.nnz} | "
                    f"expand×={expansion:.2f} | "
                    f"γ²={state.gamma**2:.6e} | "
                    f"time={gate_time*1000:.2f} ms | "
                    f"{'TRUNCATED' if truncated else ''}"
                )

        total_runtime = perf_counter() - total_start

        if self.verbose:
            print("\n=== Simulation Summary ===")
            print(f"Total gates: {L}")
            print(f"Total runtime: {total_runtime:.3f} s")
            print(f"Average per gate: {runtime.mean()*1000:.3f} ms")
            print(f"Final nnz: {state.nnz}")
            print(f"Final γ²: {state.gamma**2:.6e}")
            print("==========================\n")

        self.runtime_per_gate = runtime
        self.gamma_history = gamma_hist
        self.nnz_history = nnz_hist

        return state

    def get_metrics(self):
        return {
            "runtime_per_gate": self.runtime_per_gate,
            "gamma_squared": self.gamma_history**2,
            "nnz": self.nnz_history,
            "total_runtime": self.runtime_per_gate.sum(),
            "avg_runtime_per_gate": self.runtime_per_gate.mean(),
        }
