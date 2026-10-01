"""The copied pipeline against the original's output on a fixed sample.

The fixture was made once with safer-streets-eda's own hex_features, in that repo's environment:
a 2,000-row sample (`random_state=20261001`, sorted by index) of `select_features` applied to
beahiv202-characterisation.parquet saved as parity_input.parquet, and `clean_features` ->
`ilr_features` -> `robust_scale` of it saved as parity_expected.parquet. Redo that whenever eda's
pipeline changes on purpose.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peer_hex_explorer.features import SHORT_LABELS, percentiles, scale_features

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def parity() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_parquet(FIXTURES / "parity_input.parquet"), pd.read_parquet(FIXTURES / "parity_expected.parquet")


def test_matches_hex_features(parity):
    raw, expected = parity
    scaled, _ = scale_features(raw)
    assert list(scaled.columns) == list(expected.columns)
    assert scaled.index.equals(expected.index)
    np.testing.assert_allclose(scaled.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-9)


def test_structural_zero_fill_is_idempotent(parity):
    # data.py fixes n_stops at source with COALESCE; that must not change anything downstream
    raw, expected = parity
    scaled, _ = scale_features(raw.assign(n_stops=raw.n_stops.fillna(0)))
    np.testing.assert_allclose(scaled.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-9)


def test_every_column_has_a_short_label(parity):
    _, expected = parity
    assert set(expected.columns) <= set(SHORT_LABELS)


def test_percentiles_bounded(parity):
    _, expected = parity
    p = percentiles(expected)
    assert p.to_numpy().min() >= -100 and p.to_numpy().max() <= 100
