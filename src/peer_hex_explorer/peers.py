"""Nearest cells to a target in the scaled feature matrix.

Brute force over the whole matrix: ~221k rows x 13 columns is a few milliseconds per query, so an
index would buy nothing but staleness when the user toggles a feature.
"""

import numpy as np


def nearest(
    M: np.ndarray, target_idx: int, cols: np.ndarray, k: int, candidate_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of the `k` rows of `M` nearest the target (Euclidean, over `cols`), nearest first.

    Only rows where `candidate_mask` is True are eligible, and the target never is. Returns fewer
    than `k` when there are fewer eligible rows. Ties are broken by row index, so the result is
    deterministic.
    """
    d = np.linalg.norm(M[:, cols] - M[target_idx, cols], axis=1)
    d[~candidate_mask] = np.inf
    d[target_idx] = np.inf
    k = min(k, int(np.isfinite(d).sum()))
    if k <= 0:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=d.dtype)
    top = np.argpartition(d, k - 1)[:k]
    # lexsort's last key is the primary one: distance, then row index for ties
    order = np.lexsort((top, d[top]))
    return top[order], d[top][order]


def contributions(M: np.ndarray, target_idx: int, peer_idx: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Each feature's share of each peer's squared distance to the target: (n_peers, len(cols)).

    Rows sum to 1, so a row reads as "where this peer's distance comes from" -- a small share on
    a feature means the peer matches the target on it. A peer identical to the target over `cols`
    has no distance to apportion and gets a row of zeros.
    """
    sq = (M[np.ix_(peer_idx, cols)] - M[target_idx, cols]) ** 2
    total = sq.sum(axis=1, keepdims=True)
    return np.divide(sq, total, out=np.zeros_like(sq), where=total > 0)
