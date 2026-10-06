"""Nearest cells to a target in the scaled feature matrix.

Brute force over the whole matrix: ~221k rows x 13 columns is a few milliseconds per query, so an
index would buy nothing but staleness when the user toggles a feature.
"""

from typing import Literal

import numpy as np

Metric = Literal["euclidean", "cosine"]


def _directions(X: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length. A zero row has no direction and comes back as NaN."""
    norm = np.linalg.norm(X, axis=-1, keepdims=True)
    return np.divide(X, norm, out=np.full_like(X, np.nan), where=norm > 0)


def nearest(
    M: np.ndarray,
    target_idx: int,
    cols: np.ndarray,
    k: int,
    candidate_mask: np.ndarray,
    metric: Metric = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of the `k` rows of `M` nearest the target over `cols`, nearest first, and their distances.

    "euclidean" is the straight-line distance. "cosine" is 1 - cosine similarity, 0 to 2: it compares
    directions from the origin and ignores how far along them a row is. A row that is all zeros over
    `cols` has no direction, so under "cosine" it is never a peer, and a zero target has none.

    Only rows where `candidate_mask` is True are eligible, and the target never is. Returns fewer
    than `k` when there are fewer eligible rows. Ties are broken by row index, so the result is
    deterministic.
    """
    X = M[:, cols]
    if metric == "cosine":
        X = _directions(X)
    d = np.linalg.norm(X - X[target_idx], axis=1)
    if metric == "cosine":
        # equals 1 - cos for unit vectors, and unlike 1 - dot it can't round to just below zero
        d = d**2 / 2
    d[~candidate_mask | np.isnan(d)] = np.inf
    d[target_idx] = np.inf
    k = min(k, int(np.isfinite(d).sum()))
    if k <= 0:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=d.dtype)
    top = np.argpartition(d, k - 1)[:k]
    # lexsort's last key is the primary one: distance, then row index for ties
    order = np.lexsort((top, d[top]))
    return top[order], d[top][order]


def contributions(
    M: np.ndarray, target_idx: int, peer_idx: np.ndarray, cols: np.ndarray, metric: Metric = "euclidean"
) -> np.ndarray:
    """Each feature's share of each peer's distance to the target: (n_peers, len(cols)).

    Shares of the squared difference, of the rows themselves for "euclidean" and of their unit
    vectors for "cosine" (whose distance is half that sum, so the shares are exact either way).

    Rows sum to 1, so a row reads as "where this peer's distance comes from" -- a small share on
    a feature means the peer matches the target on it. A peer identical to the target over `cols`
    (in direction, for "cosine") has no distance to apportion and gets a row of zeros.
    """
    peers, target = M[np.ix_(peer_idx, cols)], M[target_idx, cols]
    if metric == "cosine":
        peers, target = _directions(peers), _directions(target)
    sq = (peers - target) ** 2
    total = sq.sum(axis=1, keepdims=True)
    return np.divide(sq, total, out=np.zeros_like(sq), where=total > 0)
