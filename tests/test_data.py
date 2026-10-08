"""The startup build and loaders against the live Azure store. Skipped without credentials.

Credentials come from AZURE_STORAGE_CONNSTR or, failing that, .streamlit/secrets.toml. The
reference characterisation is safer-streets-eda's notebook output, overridable with
BEAHIV_CHARACTERISATION_PARQUET.
"""

import os
import tomllib
from pathlib import Path
from typing import get_args

import beahiv
import numpy as np
import pandas as pd
import pytest

from peer_hex_explorer.data import (
    SHOP_CATEGORIES,
    build_characterisation,
    coverage_gaps,
    query_characterisation,
    query_coverage,
    query_crime_counts,
    query_hotspot_population,
)
from peer_hex_explorer.database import SOURCE, duckdb_connector
from peer_hex_explorer.utils import Force

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
    # the building counts were added here before the notebook has them
    assert set(features) - set(reference.columns) <= {"n_res_buildings", "n_nonres_buildings"}
    for c in set(features) & set(reference.columns):
        pd.testing.assert_series_equal(
            raw.loc[common, c], reference.loc[common, c].astype("float64"), check_names=False, rtol=1e-9
        )


def test_no_northern_ireland(con, raw):
    ni = con.sql(
        f"SELECT spatial_id FROM read_parquet('{SOURCE}/transform/beahiv202_geogs.parquet') WHERE msoa21cd IS NULL"
    ).fetchnumpy()["spatial_id"]
    assert len(ni) > 10_000  # i.e. the filter really is removing NI, not nothing
    assert not np.isin(raw.index.to_numpy(), ni).any()


def test_shops(con, raw):
    shops = raw["n_shops"]
    assert (shops >= 0).all()  # and so no NaN: nothing is left for the median fill
    shop_cells = con.execute(
        f"SELECT beahiv202_id FROM read_parquet('{SOURCE}/extract/poi.parquet') WHERE basic_category IN ?",
        (list(SHOP_CATEGORIES),),
    ).fetchnumpy()["beahiv202_id"]
    # the extract's bbox reaches into Scotland, so only shops in an E&W cell are expected in the counts
    in_matrix = np.isin(shop_cells, raw.index.to_numpy())
    assert in_matrix.sum() > 100_000  # i.e. the POI extract has been rerun with the shop categories
    assert shops.sum() == in_matrix.sum()


def test_buildings(con, raw):
    for c in ("n_res_buildings", "n_nonres_buildings"):
        assert (raw[c] >= 0).all()  # and so no NaN: nothing is left for the median fill
    totals = con.execute(
        f"""
        SELECT map_simple_use, SUM(building_count)
        FROM read_parquet('{SOURCE}/transform/beahiv202_building_counts.parquet')
        WHERE spatial_id IN ?
        GROUP BY map_simple_use
        """,
        (raw.index.tolist(),),
    ).fetchall()
    by_use = dict(totals)
    assert set(by_use) == {"Residential", "Non Residential", "Mixed Use"}
    # mixed use counts once in each
    assert raw["n_res_buildings"].sum() == by_use["Residential"] + by_use["Mixed Use"]
    assert raw["n_nonres_buildings"].sum() == by_use["Non Residential"] + by_use["Mixed Use"]


def test_bundle(con):
    bundle = build_characterisation(con.cursor())
    n = len(bundle.spatial_id)
    assert bundle.scaled.shape == (n, len(bundle.columns))
    assert np.all(np.diff(bundle.spatial_id) > 0)
    assert np.isfinite(bundle.scaled).all()
    assert bundle.rows_of(bundle.spatial_id[[0, 5, n - 1]]).tolist() == [0, 5, n - 1]
    assert bundle.rows_of([1]).tolist() == [-1]
    assert sum(p is None for p in bundle.pfa24cd) <= 1
    assert 10_000 < bundle.in_population.sum() < 20_000


def test_hotspot_ids_decode(con):
    counts = query_crime_counts(con.cursor(), "Robbery", ["2026-05", "2026-06", "2026-07"])
    assert len(counts) > 0
    assert counts.national_rank.tolist() == list(range(1, len(counts) + 1))
    assert {beahiv.decode(int(i)).side_length for i in counts.index} == {202}


def test_coverage_resolves_every_force(con):
    months = ("2026-05", "2026-06", "2026-07")
    coverage = query_coverage(con.cursor(), "Robbery", list(months))
    assert set(coverage.index.get_level_values("force")) == set(get_args(Force))
    gaps = coverage_gaps(coverage, months)
    assert gaps["Greater Manchester"] == list(months)  # GMP hasn't published to police.uk since 2019
    assert "West Yorkshire" not in gaps


def test_hotspot_population(con, raw):
    population = query_hotspot_population(con.cursor())
    assert np.all(np.diff(population) > 0)
    assert 10_000 < len(population) < 20_000  # 13,163 when written
    assert np.isin(population, raw.index.to_numpy()).mean() > 0.99
