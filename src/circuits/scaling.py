"""
Single source of truth for the empirical Z-basis participation-ratio
scaling exponent, PRZ ~ C * 2^(alpha * N), used to classify
"undersampled" (k < PRZ) vs. "oversampled" (k >= PRZ) regimes.

"""

# The exponent currently used for undersampled/oversampled regime labeling.
PRZ_SCALING_ALPHA = 0.891 # family prx test for brickwork at l = 6 not the pooled mean (specific for runtimeoverhead tests)



def prz_estimate(N: int, alpha: float = PRZ_SCALING_ALPHA, C: float = 1.0) -> float:
    """PRZ ~ C * 2^(alpha * N). Single implementation used by every
    notebook that needs to classify (N, k) as undersampled/oversampled."""
    return C * 2.0 ** (alpha * N)


def regime_label(
    k: int, N: int, alpha: float = PRZ_SCALING_ALPHA, C: float = 1.0
) -> str:
    """'under' if k < PRZ_estimate(N), else 'over'. Matches the convention
    already used in runtimeOverhead.ipynb / Table VIII."""
    return "under" if k < prz_estimate(N, alpha=alpha, C=C) else "over"
