import numpy as np
import pytest

from peer_hex_explorer.peers import contributions, nearest


@pytest.fixture
def M() -> np.ndarray:
    # row 0 is the target; row 2 is nearest on columns 0-1, row 3 is nearest if column 2 counts
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 5.0, 0.0],
            [1.0, 0.0, 9.0],
            [0.0, 2.0, 0.0],
            [3.0, 3.0, 3.0],
        ]
    )


ALL = np.ones(5, dtype=bool)


def test_known_nearest(M):
    idx, d = nearest(M, 0, np.array([0, 1]), 2, ALL)
    assert idx.tolist() == [2, 3]
    np.testing.assert_allclose(d, [1.0, 2.0])


def test_column_selection_changes_answer(M):
    idx, _ = nearest(M, 0, np.array([0, 1, 2]), 1, ALL)
    assert idx.tolist() == [3]


def test_deselected_columns_have_no_effect(M):
    cols = np.array([0, 1])
    before, d_before = nearest(M, 0, cols, 3, ALL)
    M2 = M.copy()
    M2[:, 2] = np.random.default_rng(0).normal(size=5) * 100
    after, d_after = nearest(M2, 0, cols, 3, ALL)
    assert before.tolist() == after.tolist()
    np.testing.assert_array_equal(d_before, d_after)


def test_target_is_never_its_own_peer(M):
    # a duplicate of the target is a legitimate peer at distance 0; the target itself is not
    M2 = np.vstack([M, M[0]])
    idx, d = nearest(M2, 0, np.array([0, 1, 2]), 10, np.ones(6, dtype=bool))
    assert 0 not in idx
    assert idx[0] == 5 and d[0] == 0.0
    assert len(idx) == 5


def test_candidate_mask(M):
    # e.g. within-force: only rows 0, 1 and 4 share the target's force
    in_force = np.array([True, True, False, False, True])
    idx, _ = nearest(M, 0, np.array([0, 1]), 5, in_force)
    assert idx.tolist() == [4, 1]


def test_no_candidates(M):
    idx, d = nearest(M, 0, np.array([0, 1]), 3, np.zeros(5, dtype=bool))
    assert idx.size == 0 and d.size == 0


def test_contributions_sum_to_one(M):
    cols = np.array([0, 1, 2])
    peers = np.array([1, 2, 3, 4])
    c = contributions(M, 0, peers, cols)
    assert c.shape == (4, 3)
    np.testing.assert_allclose(c.sum(axis=1), 1.0)
    # row 2 differs from the target almost entirely on column 2
    np.testing.assert_allclose(c[1], [1 / 82, 0, 81 / 82])


def test_contributions_identical_peer_is_zero(M):
    M2 = np.vstack([M, M[0]])
    c = contributions(M2, 0, np.array([5]), np.array([0, 1, 2]))
    np.testing.assert_array_equal(c, 0.0)


def test_cosine_ignores_magnitude(M):
    # from the target at [1, 1, 0], row 1 is the same direction 5x further out; row 3 is nearer but off-direction
    M2 = M.copy()
    M2[0] = [1.0, 1.0, 0.0]
    cols = np.array([0, 1, 2])
    idx, d = nearest(M2, 0, cols, 2, ALL, "cosine")
    assert idx.tolist() == [1, 4]
    np.testing.assert_allclose(d, [0.0, 1 - 2 / np.sqrt(6)], atol=1e-12)
    assert nearest(M2, 0, cols, 1, ALL)[0].tolist() == [3]


def test_cosine_distance_is_one_minus_similarity():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(50, 4))
    idx, d = nearest(M, 0, np.arange(4), 49, np.ones(50, dtype=bool), "cosine")
    unit = M / np.linalg.norm(M, axis=1, keepdims=True)
    np.testing.assert_allclose(d, 1 - unit[idx] @ unit[0], atol=1e-12)
    assert d.min() >= 0 and d.max() <= 2


def test_cosine_skips_rows_with_no_direction(M):
    # row 0 is all zeros: as a candidate it is skipped, as the target nothing can be compared with it
    M2 = np.vstack([[1.0, 1.0, 1.0], M])
    idx, _ = nearest(M2, 0, np.array([0, 1, 2]), 10, np.ones(6, dtype=bool), "cosine")
    assert 1 not in idx and len(idx) == 4
    idx, d = nearest(M, 0, np.array([0, 1, 2]), 3, ALL, "cosine")
    assert idx.size == 0 and d.size == 0


def test_cosine_contributions_are_shares_of_the_unit_vector_difference(M):
    M2 = M.copy()
    M2[0] = [1.0, 2.0, 0.5]
    cols = np.array([0, 1, 2])
    peers = np.array([1, 2, 3, 4])
    c = contributions(M2, 0, peers, cols, "cosine")
    np.testing.assert_allclose(c.sum(axis=1), 1.0)
    unit = M2 / np.linalg.norm(M2, axis=1, keepdims=True)
    np.testing.assert_allclose(c, (unit[peers] - unit[0]) ** 2 / ((unit[peers] - unit[0]) ** 2).sum(axis=1)[:, None])
