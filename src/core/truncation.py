import numpy as np
from numba import njit, int64, uint64, complex128, float64, types


@njit(
    types.Tuple((int64, float64))(
        uint64[:], complex128[:], int64, int64  # x  # alpha  # nnz  # k
    ),
    fastmath=True,
    cache=True,
)
def top_k_truncate_kernel_opt(x, alpha, nnz, k):

    if nnz <= k:
        gamma_sq = 0.0
        for i in range(nnz):
            gamma_sq += alpha[i].real * alpha[i].real + alpha[i].imag * alpha[i].imag
        return nnz, gamma_sq

    # Compute magnitudes
    mags = np.empty(nnz, dtype=np.float64)
    for i in range(nnz):
        mags[i] = alpha[i].real * alpha[i].real + alpha[i].imag * alpha[i].imag

    # Indices of top-k largest magnitudes
    kth = nnz - k
    idx = np.argpartition(mags, kth)[kth:]

    # Compute gamma_sq from pre-computed mags (unaffected by any writes)
    gamma_sq = 0.0
    for j in range(k):
        gamma_sq += mags[idx[j]]

    # --- FIX: use temp buffers to avoid aliasing ---
    # In the code, doing x[j] = x[idx[j]] in-place can overwrite
    # a source element before it's been read (when idx[j] < j), corrupting
    # amplitudes and breaking norm conservation across gates.
    temp_x = np.empty(k, dtype=np.uint64)
    temp_alpha = np.empty(k, dtype=np.complex128)

    for j in range(k):
        i_sel = idx[j]
        temp_x[j] = x[i_sel]
        temp_alpha[j] = alpha[i_sel]

    for j in range(k):
        x[j] = temp_x[j]
        alpha[j] = temp_alpha[j]

    return k, gamma_sq


@njit(
    types.Tuple((int64, float64))(
        uint64[:], complex128[:], int64, int64, int64  # x  # alpha  # nnz  # k  # seed
    ),
    fastmath=True,
    cache=True,
)
def random_k_truncate_kernel_opt(x, alpha, nnz, k, seed):

    if nnz <= k:
        gamma_sq = 0.0
        for i in range(nnz):
            gamma_sq += alpha[i].real * alpha[i].real + alpha[i].imag * alpha[i].imag
        return nnz, gamma_sq

    perm = np.arange(nnz)
    rng = seed

    # Partial Fisher-Yates shuffle (only first k needed)
    for i in range(k):
        rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
        r = i + (rng % (nnz - i))
        tmp = perm[i]
        perm[i] = perm[r]
        perm[r] = tmp

    # --- FIX: same aliasing issue, use temp buffers ---
    temp_x = np.empty(k, dtype=np.uint64)
    temp_alpha = np.empty(k, dtype=np.complex128)

    gamma_sq = 0.0
    for j in range(k):
        idx = perm[j]
        temp_x[j] = x[idx]
        temp_alpha[j] = alpha[idx]
        gamma_sq += (
            alpha[idx].real * alpha[idx].real + alpha[idx].imag * alpha[idx].imag
        )

    for j in range(k):
        x[j] = temp_x[j]
        alpha[j] = temp_alpha[j]

    return k, gamma_sq
