"""
QuantumFidelityMetrics
======================
Rigorous statistical analysis tools and validation metrics designed to evaluate state 
reconstruction precision, state sparsity, and tracking bounds for sparse simulation frameworks.

This module provides the numerical tracking backend that evaluates the absolute fidelity 
of approximate sparse vectors against dense statevector baselines, completely bypassing 
historical normalization bugs.

Mathematical Formulations & Metrics
-----------------------------------
    1. Pristine State Vector Fidelity (`compute_fidelity`)
        Computes the absolute squared overlap between an exact reference state |ψ_exact⟩ 
        and a truncated, sparse-state wave-front |φ_sparse⟩:
            f = | ⟨ψ_exact | φ_sparse⟩ |²
        *Methodological Note:* Amplitudes (alpha) are strictly normalized at each internal 
        truncation boundary (Convention B). The stored array coefficients represent the true physical 
        amplitudes of the normalized sparse state. This completely avoids historical inflation bugs 
        (dividing by γ), ensuring that measured values are mathematically bounded at f ≤ 1.0.

    2. Deterministic Fidelity Lower Bounds (`compute_fidelity_bounds`)
        Extracts active lower bounds without running an exponentially costly exact statevector inner product:
            f_lower = γ²
        Where γ² represents the cumulative, running probability mass preserved throughout the entire 
        sequence of simulator budget enforcements.

    3. Inverse Participation Ratio / Sparsity Measure (`compute_participation_ratio`)
        Quantifies the computational basis localization and state compression profile via the 
        2nd-order Renyi entropy footprint:
            PR = 1 / ( Σ_{i=1}^nnz |α_i|⁴ )
        A localized computational basis state yields PR = 1.0, while a perfectly uniform superposition 
        over k states scales as PR = k. This metric acts as the vital cost function for BASS coordinate 
        descent sweeps.

    4. Linear Cross-Entropy Benchmarking (`estimate_cross_entropy_fidelity`)
        Simulates experimental cross-entropy benchmarking (XEB) sequences via importance sampling:
            F_XEB = 2^N · ⟨ P_exact(x) ⟩_samples - 1
        Draws physical bitstring configurations directly from the sparse state's active basis distribution 
        to reconstruct an unbiased statistical estimator of system fidelity.
"""



import numpy as np


def compute_fidelity(sparse_state, exact_state):
    """
    Compute f = |<ψ_exact | φ_sparse>|²

    Convention: truncation normalizes alpha so Σ|α_i|² = 1 after each
    truncation step (Convention B). gamma² tracks the cumulative probability
    retained. The stored alpha_i values ARE the actual state amplitudes of
    the normalized sparse state — no division by gamma is needed here.

    Dividing by gamma (the old bug) was treating normalized alphas as if they
    were raw unnormalized amplitudes, inflating fidelity by 1/gamma² and
    producing values that could exceed 1 or be wildly inconsistent with gamma².
    """
    xs = sparse_state.x[: sparse_state.nnz]
    alphas = sparse_state.alpha[: sparse_state.nnz]
    overlap = np.dot(exact_state[xs].conj(), alphas)
    fidelity = float(np.abs(overlap) ** 2)
    return fidelity


def compute_fidelity_bounds(sparse_state):
    """
    Compute fidelity bounds without exact state.
    gamma² is a lower bound on fidelity (Eq. 8 of paper: f_bar >= gamma²).
    """
    gamma_sq = sparse_state.gamma**2
    f_lower = gamma_sq  # probability retained is a lower bound
    f_upper = 1.0
    return f_lower, f_upper


def compute_participation_ratio(sparse_state):
    """
    Participation ratio PR = 1 / Σ|α_i|^4.

    PR measures sparsity: PR = 1 for a single-basis state, PR = k for a
    uniform superposition over k states. Lower PR → more concentrated → better
    compression. Used in Figure 3.
    """
    alphas = sparse_state.alpha[: sparse_state.nnz]
    denom = float(np.sum(np.abs(alphas) ** 4))
    return 1.0 / denom if denom > 1e-60 else float("inf")


def estimate_cross_entropy_fidelity(sparse_state, exact_state, num_samples=1000):
    """
    Estimate cross-entropy fidelity (XEB).
    XEB = 2^N * <P(x)> - 1 ≈ fidelity
    """
    N = sparse_state.N
    samples = sparse_state.sample(num_samples)
    probs_exact = np.abs(exact_state[samples]) ** 2
    mean_prob = np.mean(probs_exact)
    XEB = 2**N * mean_prob - 1
    return XEB
