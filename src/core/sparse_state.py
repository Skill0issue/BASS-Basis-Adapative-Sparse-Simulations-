import numpy as np
from numba import njit, uint64, int64, complex128
from .truncation import top_k_truncate_kernel_opt, random_k_truncate_kernel_opt


# Standalone Numba Kernels (NO self allowed)
@njit
def bitwise_sort_key(x_array, mask):
    """
    Compute sort key = free bits (bits not affected by gate)
    """
    return x_array & (~mask)


@njit(
    int64(
        uint64[:],
        complex128[:],
        int64,
        uint64[:],
        complex128[:],
        complex128[:, :],
        int64,
    ),
    fastmath=True,
    cache=True,
)
def apply_1qubit_gate_kernel_opt(
    x_in, alpha_in, nnz_in, x_out, alpha_out, gate_matrix, q
):

    nnz_out = 0
    mask = np.uint64(1) << q

    # Cache gate columns locally (avoids repeated indexing)
    g00 = gate_matrix[0, 0]
    g01 = gate_matrix[0, 1]
    g10 = gate_matrix[1, 0]
    g11 = gate_matrix[1, 1]

    for i in range(nnz_in):

        x = x_in[i]
        a = alpha_in[i]

        bit = (x >> q) & 1
        base = x & ~mask

        if bit == 0:
            out0 = g00 * a
            out1 = g10 * a
        else:
            out0 = g01 * a
            out1 = g11 * a

        # Avoid expensive np.abs
        if out0.real != 0.0 or out0.imag != 0.0:
            x_out[nnz_out] = base
            alpha_out[nnz_out] = out0
            nnz_out += 1

        if out1.real != 0.0 or out1.imag != 0.0:
            x_out[nnz_out] = base | mask
            alpha_out[nnz_out] = out1
            nnz_out += 1

    return nnz_out


@njit(
    int64(
        uint64[:],
        complex128[:],
        int64,
        uint64[:],
        complex128[:],
        complex128[:, :],
        int64,
        int64,
    ),
    fastmath=True,
    cache=True,
)
def apply_2qubit_gate_kernel_opt(
    x_in, alpha_in, nnz_in, x_out, alpha_out, gate_matrix, q1, q2
):
    nnz_out = 0
    mask1 = np.uint64(1) << q1
    mask2 = np.uint64(1) << q2

    for i in range(nnz_in):

        x = x_in[i]
        a = alpha_in[i]

        b1 = (x >> q1) & 1
        b2 = (x >> q2) & 1
        in_idx = (b1 << 1) | b2

        base = x & ~(mask1 | mask2)

        for out_idx in range(4):

            amp = gate_matrix[out_idx, in_idx] * a

            if amp.real != 0.0 or amp.imag != 0.0:

                out_b1 = (out_idx >> 1) & 1
                out_b2 = out_idx & 1

                x_new = base
                if out_b1:
                    x_new |= mask1
                if out_b2:
                    x_new |= mask2

                if nnz_out >= x_out.shape[0]:
                    return -1

                x_out[nnz_out] = x_new
                alpha_out[nnz_out] = amp
                nnz_out += 1

    return nnz_out


@njit(int64(uint64[:], complex128[:], int64), fastmath=True, cache=True)
def merge_sorted_kernel(x, alpha, nnz):
    """
    Assumes x is already sorted.
    Merges duplicates in-place.
    """
    if nnz == 0:
        return 0

    write = 0

    for read in range(1, nnz):

        if x[read] == x[write]:
            alpha[write] += alpha[read]
        else:
            write += 1
            x[write] = x[read]
            alpha[write] = alpha[read]

    return write + 1


@njit(cache=True)
def radix_sort_uint64(keys, values, n):
    """4-pass LSD radix sort, 16 bits per pass, covering all 64 bits."""
    tmp_k = np.empty(n, dtype=np.uint64)
    tmp_v = np.empty(n, dtype=np.complex128)

    for shift in (np.uint64(0), np.uint64(16), np.uint64(32), np.uint64(48)):
        # count
        counts = np.zeros(65536, dtype=np.int64)
        for i in range(n):
            bucket = int((keys[i] >> shift) & np.uint64(0xFFFF))
            counts[bucket] += 1
        # prefix sum
        for i in range(1, 65536):
            counts[i] += counts[i - 1]
        # scatter (reverse for stability)
        for i in range(n - 1, -1, -1):
            bucket = int((keys[i] >> shift) & np.uint64(0xFFFF))
            counts[bucket] -= 1
            tmp_k[counts[bucket]] = keys[i]
            tmp_v[counts[bucket]] = values[i]
        keys[:n] = tmp_k[:n]
        values[:n] = tmp_v[:n]


# SparseState Class (Pure Python wrapper)


@njit(fastmath=True, cache=True)
def apply_2qubit_diagonal_kernel(x, alpha, nnz, q1, q2, diag):
    """
    Apply a diagonal 2-qubit gate in-place (no nnz change).

    diag[k] is the k-th diagonal element where k = (bit_q1 << 1) | bit_q2.
    """
    mask1 = np.uint64(1) << np.uint64(q1)
    mask2 = np.uint64(1) << np.uint64(q2)
    for i in range(nnz):
        b1 = int((x[i] & mask1) >> np.uint64(q1))
        b2 = int((x[i] & mask2) >> np.uint64(q2))
        alpha[i] *= diag[(b1 << 1) | b2]


class SparseState:
    """
    Sparse quantum state representation:

        |φ⟩ = (1/γ) Σ α_i |x_i⟩

    Stores up to k basis states.
    """

    def __init__(self, num_qubits, k, init_state=None):
        self.N = num_qubits
        self.k = k
        self.RADIX_THRESHOLD = 2000  # Radix size for faster sorting at higher N

        # Main storage
        self.x = np.zeros(k, dtype=np.uint64)
        self.alpha = np.zeros(k, dtype=np.complex128)
        self.nnz = 0
        self.gamma = 1.0

        # Initialize |0...0>
        if init_state is None:
            init_state = 0

        self.x[0] = np.uint64(init_state)
        self.alpha[0] = 1.0 + 0.0j
        self.nnz = 1

        # Temporary buffers (expand factor 4)
        # Need to work on space efficiency exploring n qubit states before merging producing more than 4k entries need to merge efficiently to maintain memory
        # self._temp_x = np.zeros(8 * k, dtype=np.uint64)
        # self._temp_alpha = np.zeros(8 * k, dtype=np.complex128)
        self._temp_x = np.zeros(4 * k, dtype=np.uint64)
        self._temp_alpha = np.zeros(4 * k, dtype=np.complex128)
        self._temp_nnz = 0

    def __repr__(self):
        return f"SparseState(N={self.N}, k={self.k}, nnz={self.nnz}, γ²={self.gamma**2:.6f})"

    def copy(self):
        new_state = SparseState(self.N, self.k)
        new_state.x[: self.nnz] = self.x[: self.nnz]
        new_state.alpha[: self.nnz] = self.alpha[: self.nnz]
        new_state.nnz = self.nnz
        new_state.gamma = self.gamma
        return new_state

    def to_dense(self):
        """
        Convert sparse state to dense vector.

        Since truncation normalizes alpha (Σ|α_i|² = 1), the stored alpha_i
        values ARE the actual state amplitudes. Do NOT divide by gamma here.
        gamma is for tracking cumulative probability retained, not for
        re-normalizing the state vector.
        """
        if self.N > 20:
            raise ValueError("Dense conversion only allowed for N <= 20")

        dense = np.zeros(2**self.N, dtype=np.complex128)
        for i in range(self.nnz):
            dense[self.x[i]] = self.alpha[i]  # <-- no /self.gamma
        return dense

    def probability_distribution(self):
        probs = np.abs(self.alpha[: self.nnz]) ** 2 / (self.gamma**2)
        return probs

    def sample(self, num_samples=1):
        probs = self.probability_distribution()
        probs /= probs.sum()
        idx = np.random.choice(self.nnz, size=num_samples, p=probs)
        return self.x[idx]

    # Gate Application
    def apply_single_qubit_gate(self, gate, method, seed):
        q = gate.qubits[0]

        self._temp_nnz = apply_1qubit_gate_kernel_opt(
            self.x, self.alpha, self.nnz, self._temp_x, self._temp_alpha, gate.matrix, q
        )

        # Sort + merge duplicates (kernel can produce |x⟩ and |x⊕2^q⟩ duplicates)
        if self._temp_nnz < self.RADIX_THRESHOLD:
            order = np.argsort(self._temp_x[: self._temp_nnz])
            self._temp_x[: self._temp_nnz] = self._temp_x[order]
            self._temp_alpha[: self._temp_nnz] = self._temp_alpha[order]
        else:
            radix_sort_uint64(self._temp_x, self._temp_alpha, self._temp_nnz)

        self._temp_nnz = merge_sorted_kernel(
            self._temp_x, self._temp_alpha, self._temp_nnz
        )

        # Grow main buffer if needed
        if self._temp_nnz > self.x.shape[0]:
            new_cap = max(self._temp_nnz, 2 * self.x.shape[0])
            self.x = np.zeros(new_cap, dtype=np.uint64)
            self.alpha = np.zeros(new_cap, dtype=np.complex128)

        self.x[: self._temp_nnz] = self._temp_x[: self._temp_nnz]
        self.alpha[: self._temp_nnz] = self._temp_alpha[: self._temp_nnz]
        self.nnz = self._temp_nnz

        if self.nnz > self.k:
            if method == "top-k":
                self.truncate_top_k()
            else:
                self.truncate_random_k(seed)

    def apply_two_qubit_gate(self, gate, method, seed):
        q1, q2 = gate.qubits

        # Ensure temp buffer large enough
        required = 4 * self.nnz
        if required > self._temp_x.shape[0]:
            self._temp_x = np.zeros(required, dtype=np.uint64)
            self._temp_alpha = np.zeros(required, dtype=np.complex128)

        self._temp_nnz = apply_2qubit_gate_kernel_opt(
            self.x,
            self.alpha,
            self.nnz,
            self._temp_x,
            self._temp_alpha,
            gate.matrix,
            q1,
            q2,
        )

        if self._temp_nnz == -1:
            raise RuntimeError("Temporary buffer overflow")

        # SORT by x
        # {TODO: radix sort alternative }
        # order = np.argsort(self._temp_x[:self._temp_nnz]) #O(nlogn) its slower at low n making the log-log nk 0.8 instead of 1
        # self._temp_x[:self._temp_nnz] = self._temp_x[order]
        # self._temp_alpha[:self._temp_nnz] = self._temp_alpha[order]

        # radix_sort_uint64(self._temp_x,self._temp_alpha,self._temp_nnz)

        if self._temp_nnz < self.RADIX_THRESHOLD:
            order = np.argsort(self._temp_x[: self._temp_nnz])
            self._temp_x[: self._temp_nnz] = self._temp_x[order]
            self._temp_alpha[: self._temp_nnz] = self._temp_alpha[order]
        else:
            radix_sort_uint64(self._temp_x, self._temp_alpha, self._temp_nnz)

        # MERGE duplicates
        self._temp_nnz = merge_sorted_kernel(
            self._temp_x, self._temp_alpha, self._temp_nnz
        )

        # if self._temp_nnz > self.x.shape[0]:
        #     new_capacity = max(self._temp_nnz, 2 * self.x.shape[0])
        #     new_x = np.zeros(new_capacity, dtype=np.uint64)
        #     new_alpha = np.zeros(new_capacity, dtype=np.complex128)

        #     new_x[:self.nnz] = self.x[:self.nnz]
        #     new_alpha[:self.nnz] = self.alpha[:self.nnz]

        #     self.x = new_x
        #     self.alpha = new_alpha

        # # Copy back
        # self.x[:self._temp_nnz] = self._temp_x[:self._temp_nnz]
        # self.alpha[:self._temp_nnz] = self._temp_alpha[:self._temp_nnz]
        # self.nnz = self._temp_nnz

        if self._temp_nnz > self.x.shape[0]:
            new_capacity = max(self._temp_nnz, 2 * self.x.shape[0])
            self.x = np.zeros(new_capacity, dtype=np.uint64)
            self.alpha = np.zeros(new_capacity, dtype=np.complex128)

        # Copy back
        self.x[: self._temp_nnz] = self._temp_x[: self._temp_nnz]
        self.alpha[: self._temp_nnz] = self._temp_alpha[: self._temp_nnz]
        self.nnz = self._temp_nnz

        # Truncate if needed
        if self.nnz > self.k:
            if method == "top-k":
                self.truncate_top_k()
            else:
                self.truncate_random_k(seed)

    def apply_gate(self, gate, method="top-k", seed="None"):
        if gate.n_qubits == 1:
            self.apply_single_qubit_gate(gate, method, seed)
        elif gate.n_qubits == 2:
            self.apply_two_qubit_gate(gate, method, seed)
        else:
            raise NotImplementedError("Only 1- and 2-qubit gates supported")

    # Truncation
    def truncate_top_k(self):
        if self.nnz <= self.k:
            return

        self.nnz, gamma_sq_new = top_k_truncate_kernel_opt(
            self.x, self.alpha, self.nnz, self.k
        )

        gamma_layer = np.sqrt(gamma_sq_new)

        self.alpha[: self.nnz] /= gamma_layer

        self.gamma *= gamma_layer

    def truncate_random_k(self, seed):
        if self.nnz <= self.k:
            return

        if seed is None:
            seed = int(np.random.randint(0, 2**31))

        self.nnz, gamma_sq_new = random_k_truncate_kernel_opt(
            self.x, self.alpha, self.nnz, self.k, seed
        )

        gamma_layer = np.sqrt(gamma_sq_new)
        self.alpha[: self.nnz] /= gamma_layer
        self.gamma *= gamma_layer
