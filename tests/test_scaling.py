"""The population fit in data.py, on the parity fixture: no Azure needed."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peer_hex_explorer.data import scale_to_population
from peer_hex_explorer.features import clean_features, ilr_features, robust_scale, scale_features

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / "parity_input.parquet")


@pytest.fixture(scope="module")
def in_population(raw) -> np.ndarray:
    return np.random.default_rng(0).random(len(raw)) < 0.2


def test_population_rows_match_robust_scale(raw, in_population):
    # data.py repeats robust_scale's log1p to apply the fit to every row; on the rows it was fitted on, the two agree
    scaled, _ = scale_to_population(raw, in_population)
    features, _ = clean_features(raw)
    ilr, _ = ilr_features(features)
    expected, _ = robust_scale(ilr[in_population])
    np.testing.assert_allclose(scaled[in_population].to_numpy(), expected.to_numpy(), rtol=0, atol=1e-9)


def test_whole_population_is_the_national_fit(raw):
    scaled, _ = scale_to_population(raw, np.ones(len(raw), dtype=bool))
    expected, _ = scale_features(raw)
    np.testing.assert_allclose(scaled.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-9)
