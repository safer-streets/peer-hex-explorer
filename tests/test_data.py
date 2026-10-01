"""The startup build and loaders against the live Azure store. Skipped without credentials.

Credentials come from AZURE_STORAGE_CONNSTR or, failing that, .streamlit/secrets.toml. The
reference characterisation is safer-streets-eda's notebook output, overridable with
BEAHIV_CHARACTERISATION_PARQUET.
"""

import os
import tomllib
from pathlib import Path

import beahiv
import numpy as np
import pandas as pd
import pytest

from peer_hex_explorer.data import build_characterisation, query_characterisation, query_crime_counts
from peer_hex_explorer.database import SOURCE, duckdb_connector

ROOT = Path(__file__).parents[1]
REFERENCE = Path(
    os.environ.get(
        "BEAHIV_CHARACTERISATION_PARQUET", ROOT.parent / "safer-streets-eda" / "beahiv202-characterisation.parquet"
    )
)


def _connstr() -> str | None:
    if connstr := os.environ.get("AZURE_STORAGE_CONNSTR"):
        return connstr
    secrets = ROOT / ".streamlit" / "secrets.toml"
    if secrets.exists():
        return tomllib.loads(secrets.read_text()).get("azure_storage_connstr")
    return None


pytestmark = [pytest.mark.azure, pytest.mark.skipif(_connstr() is None, reason="no Azure credentials")]


@pytest.fixture(scope="module")
def con():
    connstr = _connstr()
    assert connstr is not None  # the module is skipped otherwise
    con = duckdb_connector(connstr)
    yield con
    con.close()


@pytest.fixture(scope="module")
def raw(con) -> pd.DataFrame:
    return query_characterisation(con.cursor())


@pytest.mark.skipif(not REFERENCE.exists(), reason=f"no reference parquet at {REFERENCE}")
def test_matches_notebook_parquet(raw):
    reference = pd.read_parquet(REFERENCE)
    reference["n_stops"] = reference["n_stops"].fillna(0)
    common = raw.index.intersection(reference.index)
    # The notebook kept NI, which the build drops. Beyond that the sets differ only at the margin:
    # geogs was rebuilt after the parquet was written, and ~1k edge cells (0.5%) moved in or out
    # of E&W. The values on the cells in both must still match exactly.
    assert len(common) / len(raw) > 0.99, (len(common), len(raw))
    features = [c for c in raw.columns if c != "pfa24cd"]
    assert set(features) <= set(reference.columns)
    for c in features:
        pd.testing.assert_series_equal(
            raw.loc[common, c], reference.loc[common, c].astype("float64"), check_names=False, rtol=1e-9
        )


def test_no_northern_ireland(con, raw):
    ni = con.sql(
        f"SELECT spatial_id FROM read_parquet('{SOURCE}/transform/beahiv202_geogs.parquet') WHERE msoa21cd IS NULL"
    ).fetchnumpy()["spatial_id"]
    assert len(ni) > 10_000  # i.e. the filter really is removing NI, not nothing
    assert not np.isin(raw.index.to_numpy(), ni).any()


def test_bundle(con):
    bundle = build_characterisation(con.cursor())
    n = len(bundle.spatial_id)
    assert bundle.scaled.shape == bundle.percentile.shape == (n, len(bundle.columns))
    assert np.all(np.diff(bundle.spatial_id) > 0)
    assert np.isfinite(bundle.scaled).all()
    assert bundle.rows_of(bundle.spatial_id[[0, 5, n - 1]]).tolist() == [0, 5, n - 1]
    assert bundle.rows_of([1]).tolist() == [-1]
    assert sum(p is None for p in bundle.pfa24cd) <= 1


def test_hotspot_ids_decode(con):
    counts = query_crime_counts(con.cursor(), "Robbery", ["2026-05", "2026-06", "2026-07"])
    assert len(counts) > 0
    assert counts.national_rank.tolist() == list(range(1, len(counts) + 1))
    assert {beahiv.decode(int(i)).side_length for i in counts.index} == {202}
