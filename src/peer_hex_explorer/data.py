"""Everything the app reads, all of it through DuckDB straight onto the Azure parquet store.

The feature matrix is built once per process at startup (`characterisation`) and shared by
every session; the crime counts and place names are small cached queries keyed on what the user
picked. Nothing here is read from local files.
"""

import resource
import time
from dataclasses import dataclass
from typing import get_args

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import shapely
import streamlit as st
from beahiv import cell_polygons, k_ring
from duckdb import list_type
from duckdb.func import PythonUDFType
from duckdb.sqltypes import BIGINT, INTEGER
from pyproj import Transformer

from peer_hex_explorer.database import SOURCE, duckdb_connector
from peer_hex_explorer.features import LOG1P_COLUMNS, clean_features, ilr_features, robust_scale
from peer_hex_explorer.utils import CrimeType, Force, fix_force_name

# deck.gl gets the geometry as JSON over the websocket on every rerun, so trim it at source. 1e-6
# degrees is ~0.1m, far below anything visible.
COORD_DECIMALS = 6

_TO_WGS84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)


@st.cache_resource
def _base_con() -> duckdb.DuckDBPyConnection:
    return duckdb_connector(azure_connstr=st.secrets["azure_storage_connstr"])


def get_con() -> duckdb.DuckDBPyConnection:
    """
    Per-script-run handle on the shared in-memory DuckDB instance.

    st.cache_resource shares one connection across all sessions, but Streamlit runs each session's script in its
    own thread and a DuckDBPyConnection is not safe for concurrent use across threads: simultaneous users could
    otherwise interleave queries on it. Cursors are cheap to create and share the base connection's extensions
    and GLOBAL settings (including the Azure credentials) while giving each rerun its own thread-safe session.
    """
    return _base_con().cursor()


# --- the feature matrix ---------------------------------------------------------------------------


# Count hospital POI in the cell's k-ring:
# cell plus its 6 neighbours - the grid-native analogue of the H3 "hospital in the parent cell"
# rule, and much cheaper than a spatial join (no geometry is touched at all, just id arithmetic).
# DuckDB has no BEAHIV function, so the ring comes from a vectorised (type="arrow") UDF over
# beahiv.k_ring, returning BIGINT[]. k is a parameter so the neighbourhood can be widened to k=2
# (19 cells) without touching the UDF.
def _k_ring(spatial_id: pa.ChunkedArray, k: pa.ChunkedArray) -> pa.Array:
    return pa.array(
        [k_ring(cell, ring) for cell, ring in zip(spatial_id.to_pylist(), k.to_pylist(), strict=True)],
        type=pa.list_(pa.int64()),
    )


def _register_k_ring(con: duckdb.DuckDBPyConnection) -> None:
    # re-registering raises CatalogException once the UDF has executed, so skip it if the catalog
    # already has the name - the function is pure, so an existing registration is the same function
    (registered,) = con.execute(
        "SELECT COUNT(*) FROM duckdb_functions() WHERE function_name = 'beahiv_k_ring'"
    ).fetchone() or (0,)
    if not registered:
        con.create_function("beahiv_k_ring", _k_ring, [BIGINT, INTEGER], list_type(BIGINT), type=PythonUDFType.ARROW)


# beahiv-characterisation.ipynb's query, minus the IMD columns, oa21cd and lsoa21cd (context, not
# features: hex_features.NON_FEATURE_COLUMNS) and with three changes:
# - n_stops is COALESCEd: a NULL there is a structural zero (see features.clean_features), so it is
#   fixed at source rather than downstream
# - Northern Ireland is excluded. The BEAHIV grid covers it, but NI matches no E&W boundary so every
#   geography column is NULL (12,136 cells, all with msoa21cd, lsoa21cd and pfa24cd NULL together).
#   It has to go *before* scaling, or NI cells move every median, IQR and percentile.
# - pfa24cd comes along as context (one per cell: max overlap) for the within-force scope
# bh_hosp_kring is a CTE here rather than the notebook's view, so the build leaves no catalog state.
CHARACTERISATION_QUERY = f"""
WITH bh_hosp_kring AS (
    -- Expand the ~8.6k hospital POI over their rings rather than expanding the hexes over theirs:
    -- hex distance is symmetric, so "hospitals in this cell's k-ring" is the same count either way,
    -- and this one unnests thousands of rows instead of hundreds of thousands. Cells with no hospital
    -- nearby are simply absent, hence the COALESCE below.
    -- note that "hospital" includes various clinics, so places like Harley St have an extremely high count
    WITH hosp AS (
        SELECT beahiv202_id
        FROM read_parquet('{SOURCE}/extract/poi.parquet')
        WHERE basic_category IN ('hospital', 'emergency_department')
    )
    SELECT ring.cell_id AS spatial_id, COUNT(*) AS n_hospital
    FROM hosp, UNNEST(beahiv_k_ring(hosp.beahiv202_id, 1)) AS ring(cell_id)
    GROUP BY ring.cell_id
)
SELECT
    hex.spatial_id,
    hex.pfa24cd,
    hex.retail_centre_distance,
    COALESCE(hex.urban_overlap_area / hex.cell_area, 0) AS prop_urban,
    COALESCE(hex.suburban_overlap_area / hex.cell_area, 0) AS prop_suburban,
    COALESCE(hex.road_overlap_length, 0) AS road_overlap_length,
    COALESCE(junctions.road_intersection_count, 0) AS road_intersections,
    COALESCE(hex.greenspace_overlap_area / hex.cell_area, 0) AS prop_greenspace,
    -- we can't reliably distinguish 11+ schools in Wales as the age data is mostly missing
    COALESCE(schools.sum_overlap_area / hex.cell_area, 0) AS school_isochrone_depth,
    COALESCE(poi.n_alcohol, 0) AS n_alcohol,
    COALESCE(poi.n_food, 0) AS n_food,
    COALESCE(hosp.n_hospital, 0) AS n_hospital,
    COALESCE(naptan.n_stops, 0) AS n_stops,
    COALESCE(food_outlets.n_takeaways, 0) AS n_takeaways,
    population.residential_population,
    population.workplace_population
FROM read_parquet('{SOURCE}/transform/beahiv202_geogs.parquet') hex
LEFT JOIN (
    SELECT
        poi.beahiv202_id AS spatial_id,
        COUNT(*) FILTER (WHERE basic_category IN ('bar', 'alcoholic_beverage_venue', 'lounge', 'inn')) AS n_alcohol,
        COUNT(*) FILTER (WHERE basic_category IN ('casual_eatery', 'fast_food_restaurant', 'food_service')) AS n_food, -- removed 'restaurant'
    FROM read_parquet('{SOURCE}/extract/poi.parquet') poi
    GROUP BY poi.beahiv202_id
) poi ON hex.spatial_id = poi.spatial_id
-- With H3 hexes hospitals were counted in the PARENT cell, so siblings sharing a parent were
-- captured too. These hexes have no parent, so bh_hosp_kring counts hospital POI in the cell's
-- k-ring of 1 (the cell and its 6 neighbours) instead - a cell centre lies within ~350m, i.e. a
-- tighter catchment than the 500m buffer bh_hosp uses.
LEFT JOIN bh_hosp_kring hosp ON hex.spatial_id = hosp.spatial_id
-- naptan public transport stops per hex
LEFT JOIN (
    SELECT beahiv202_id AS spatial_id, COUNT(*) AS n_stops
    FROM read_parquet('{SOURCE}/extract/naptan.parquet')
    GROUP BY beahiv202_id
) naptan ON hex.spatial_id = naptan.spatial_id
LEFT JOIN (
    SELECT beahiv202_id AS spatial_id, COUNT(*) AS n_takeaways
    FROM read_parquet('{SOURCE}/extract/food_outlets.parquet')
    WHERE business_type = 'Takeaway/sandwich shop'
    GROUP BY beahiv202_id
) food_outlets ON hex.spatial_id = food_outlets.spatial_id
LEFT JOIN (
    SELECT spatial_id, SUM(overlap_area) AS sum_overlap_area
    FROM read_parquet('{SOURCE}/transform/beahiv202_schools_lookup.parquet')
    GROUP BY spatial_id
) schools ON hex.spatial_id = schools.spatial_id
LEFT JOIN read_parquet('{SOURCE}/transform/beahiv202_road_intersection_counts.parquet') junctions ON hex.spatial_id = junctions.spatial_id
LEFT JOIN read_parquet('{SOURCE}/transform/beahiv202_population_counts.parquet') population ON hex.spatial_id = population.spatial_id
WHERE hex.msoa21cd IS NOT NULL  -- NI has no MSOA codes
ORDER BY hex.spatial_id
"""

CONTEXT_COLUMNS = ["spatial_id", "pfa24cd"]

# A cell is in the reference population if it is among the fewest cells accounting for this share of any one crime
# type's crime, over every month: safer-streets-eda's hex_features.hotspot_cells rule, on this grid and all types.
HOTSPOT_SHARE = 0.25


def query_hotspot_population(con: duckdb.DuckDBPyConnection) -> np.ndarray:
    """The spatial_ids, ascending, the scaling and the radar percentiles are fitted over.

    NI is excluded inside the counts, as in hotspot_cells: it would otherwise be in each type's 25% denominator and
    take slots in its ranking. `spatial_id` breaks count ties so the cut lands the same way every time.
    """
    rows = con.execute(
        f"""
        WITH counts AS (
            SELECT crime.crime_type, crime.spatial_id, SUM(crime.count) AS n
            FROM read_parquet('{SOURCE}/transform/beahiv202_crime_counts.parquet') crime
            SEMI JOIN (
                SELECT spatial_id
                FROM read_parquet('{SOURCE}/transform/beahiv202_geogs.parquet')
                WHERE msoa21cd IS NOT NULL  -- NI has no MSOA codes
            ) ew ON crime.spatial_id = ew.spatial_id
            WHERE crime.crime_type IN ?
            GROUP BY crime.crime_type, crime.spatial_id
        ),
        ranked AS (
            SELECT spatial_id, SUM(n) OVER running / SUM(n) OVER (PARTITION BY crime_type) AS cum_share
            FROM counts
            WINDOW running AS (
                PARTITION BY crime_type ORDER BY n DESC, spatial_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )
        )
        SELECT DISTINCT spatial_id FROM ranked WHERE cum_share <= ? ORDER BY spatial_id
        """,
        (list(get_args(CrimeType)), HOTSPOT_SHARE),
    ).fetchnumpy()["spatial_id"]
    return np.asarray(rows, dtype=np.int64)


def scale_to_population(raw: pd.DataFrame, in_population: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """features.scale_features, except that the median and IQR are fitted on the `in_population` rows and then applied
    to every row. Cleaning (median imputation) stays national: it fills gaps, it doesn't set feature weights.

    robust_scale only returns the scaled population, so its log1p step is repeated here to apply the same fit to the
    rest; test_data checks the population rows come out identical to robust_scale's own.
    """
    features, summary = clean_features(raw)
    ilr, _ = ilr_features(features)
    _, diagnostics = robust_scale(ilr[in_population])
    logged = ilr.copy()
    for c in LOG1P_COLUMNS:
        if c in logged.columns:
            logged[c] = np.log1p(logged[c].clip(lower=0))
    return (logged - logged[in_population].median()) / diagnostics["divisor"], summary


def population_percentiles(scaled: pd.DataFrame, in_population: np.ndarray) -> pd.DataFrame:
    """Each cell's percentile on each feature *within the population*, -100..+100: features.percentiles' transform
    with the population as the reference, so a cell outside it still gets the position it would have among it. Ties
    take the midpoint, as rank(pct=True) does."""
    values = scaled.to_numpy()
    reference = np.sort(values[in_population], axis=0)
    pct = np.empty_like(values)
    for j in range(values.shape[1]):
        below = np.searchsorted(reference[:, j], values[:, j], side="left")
        at_or_below = np.searchsorted(reference[:, j], values[:, j], side="right")
        pct[:, j] = (below + at_or_below) / (2 * len(reference))
    return pd.DataFrame((pct - 0.5) * 200, index=scaled.index, columns=scaled.columns)


def query_characterisation(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """The raw (unscaled) characterisation, indexed by spatial_id in ascending order."""
    _register_k_ring(con)
    columns = con.sql(CHARACTERISATION_QUERY).fetchnumpy()
    spatial_id = np.asarray(columns.pop("spatial_id"), dtype=np.int64)
    if len(np.unique(spatial_id)) != len(spatial_id):
        raise ValueError("characterisation has duplicate spatial_ids: a join upstream has fanned out")
    # not np.ma.filled(..., None): a fill value of None means "the default", which for strings is "?"
    raw_pfa = np.ma.asarray(columns.pop("pfa24cd"))
    pfa24cd = np.asarray(raw_pfa, dtype=object).copy()
    pfa24cd[np.ma.getmaskarray(raw_pfa)] = None
    # NULLs arrive as masked entries; the feature pipeline wants them as NaN
    features = {name: np.ma.filled(np.ma.asarray(values, dtype="float64"), np.nan) for name, values in columns.items()}
    return pd.DataFrame({"pfa24cd": pfa24cd, **features}, index=pd.Index(spatial_id, name="spatial_id"))


@dataclass(frozen=True, eq=False)
class Characterisation:
    """The scaled national feature matrix and what the app needs alongside it.

    Rows are in ascending spatial_id order, which is what makes `rows_of` a binary search.
    """

    spatial_id: np.ndarray  # int64 (n,)
    pfa24cd: np.ndarray  # object (n,): the cell's force code, None for the one (Scottish border) cell with no force
    in_population: np.ndarray  # bool (n,): the hotspot cells the scaling and percentiles are fitted over
    columns: tuple[str, ...]
    scaled: np.ndarray  # float32 (n, p): robust-scaled on the population, what distances are measured in
    percentile: np.ndarray  # float32 (n, p): percentile within the population per column, -100..+100, for the radars
    imputed_pct: dict[str, float]  # share of each raw column that was median-filled
    build_seconds: float
    peak_rss_mb: float

    def rows_of(self, ids) -> np.ndarray:
        """Row index for each id, -1 where an id isn't in the matrix."""
        ids = np.asarray(ids, dtype=np.int64)
        pos = np.searchsorted(self.spatial_id, ids).clip(max=len(self.spatial_id) - 1)
        return np.where(self.spatial_id[pos] == ids, pos, -1)


def build_characterisation(con: duckdb.DuckDBPyConnection) -> Characterisation:
    start = time.perf_counter()
    raw = query_characterisation(con)
    in_population = raw.index.isin(query_hotspot_population(con))
    scaled, summary = scale_to_population(raw.drop(columns="pfa24cd"), in_population)
    pct = population_percentiles(scaled, in_population)
    bundle = Characterisation(
        spatial_id=raw.index.to_numpy(),
        pfa24cd=raw.pfa24cd.to_numpy(),
        in_population=in_population,
        columns=tuple(scaled.columns),
        scaled=scaled.to_numpy(dtype=np.float32),
        percentile=pct.to_numpy(dtype=np.float32),
        imputed_pct={k: float(v) for k, v in summary["median_imputed_%"].items() if v > 0},
        build_seconds=time.perf_counter() - start,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,  # KiB on Linux
    )
    # shared by every session: make accidental in-place edits fail loudly
    for array in (bundle.spatial_id, bundle.pfa24cd, bundle.in_population, bundle.scaled, bundle.percentile):
        array.flags.writeable = False
    return bundle


@st.cache_resource(show_spinner="Building cell characterisation…")
def characterisation() -> Characterisation:
    return build_characterisation(get_con())


# --- crime counts and lookups ---------------------------------------------------------------------


@st.cache_data(ttl="1d")
def all_months() -> tuple[str, ...]:
    raw = (
        get_con()
        .sql(f"SELECT DISTINCT month FROM read_parquet('{SOURCE}/transform/beahiv202_crime_counts.parquet') ORDER BY 1")
        .fetchall()
    )
    return tuple(m[0] for m in raw)


def query_crime_counts(con: duckdb.DuckDBPyConnection, crime_type: CrimeType, months: list[str]) -> pd.DataFrame:
    """Crimes per E&W cell over `months`, every cell with at least one, ranked nationally.

    One query serves the hotspot list (filter by force, take the top n), the "≥1 crime" candidate
    pool (its index) and the peers' counts and ranks (look up): each would otherwise be its own scan
    of the crime table over Azure. Rank order is `n DESC, spatial_id`, the tie-break
    hex_features.crime_rank uses, so equal counts always rank the same way.

    NI is dropped by the join to geogs (`msoa21cd IS NOT NULL`), not afterwards, because NI cells
    would otherwise take rank slots from E&W ones.
    """
    counts = (
        con.sql(
            f"""
            SELECT crime.spatial_id, ANY_VALUE(ew.pfa24cd) AS pfa24cd, SUM(crime.count)::BIGINT AS n
            FROM read_parquet('{SOURCE}/transform/beahiv202_crime_counts.parquet') crime
            JOIN (
                SELECT spatial_id, pfa24cd
                FROM read_parquet('{SOURCE}/transform/beahiv202_geogs.parquet')
                WHERE msoa21cd IS NOT NULL  -- NI has no MSOA codes
            ) ew ON crime.spatial_id = ew.spatial_id
            WHERE crime.crime_type = ? AND crime.month IN ?
            GROUP BY crime.spatial_id
            HAVING SUM(crime.count) > 0
            ORDER BY n DESC, crime.spatial_id
            """,
            params=(crime_type, months),
        )
        .df()
        .set_index("spatial_id")
    )
    counts["national_rank"] = np.arange(1, len(counts) + 1)
    return counts


@st.cache_data(max_entries=64)
def crime_counts(crime_type: CrimeType, months: tuple[str, ...]) -> pd.DataFrame:
    return query_crime_counts(get_con(), crime_type, list(months))


@st.cache_resource
def force_codes() -> dict[str, str]:
    """App force name -> pfa24cd. Every name in `Force` resolves, or this raises."""
    rows = get_con().sql(f"SELECT pfa24nm, spatial_id FROM read_parquet('{SOURCE}/extract/police_force_areas.parquet')")
    by_name = dict(rows.fetchall())
    return {force: by_name[fix_force_name(force)] for force in get_args(Force)}


# police.uk's force names, once the suffix is stripped, are the app's (`Force`) except for these
_POLICE_UK_NAMES = {"Devon & Cornwall": "Devon and Cornwall", "Dyfed-Powys": "Dyfed Powys"}


def app_force_name(police_uk_name: str) -> str:
    """e.g. "Metropolitan Police Service" -> "Metropolitan", "Dyfed-Powys Police" -> "Dyfed Powys"."""
    for suffix in (" Police Service", " Constabulary", " Police"):
        police_uk_name = police_uk_name.removesuffix(suffix)
    return _POLICE_UK_NAMES.get(police_uk_name, police_uk_name)


def query_coverage(con: duckdb.DuckDBPyConnection, crime_type: CrimeType, months: list[str]) -> pd.Series:
    """police.uk's own count of `crime_type` per force and month, indexed by (app force name, month). Forces outside
    `Force` (British Transport Police, PSNI) are dropped."""
    coverage = con.sql(
        f"""
        SELECT force, month, n_crimes
        FROM read_parquet('{SOURCE}/extract/crime_coverage.parquet')
        WHERE crime_type = ? AND month IN ?
        """,
        params=(crime_type, months),
    ).df()
    coverage["force"] = coverage["force"].map(app_force_name)
    coverage = coverage[coverage["force"].isin(get_args(Force))]
    return coverage.set_index(["force", "month"])["n_crimes"]


@st.cache_data(max_entries=64)
def coverage(crime_type: CrimeType, months: tuple[str, ...]) -> pd.Series:
    return query_coverage(get_con(), crime_type, list(months))


def coverage_gaps(coverage: pd.Series, months: tuple[str, ...]) -> dict[str, list[str]]:
    """App force name -> the months in `months` it recorded none of the crime type in, for forces with any such month.
    A missing row counts as none."""
    n = coverage.to_dict()
    gaps = {force: [m for m in months if n.get((force, m), 0) <= 0] for force in get_args(Force)}
    return {force: missing for force, missing in gaps.items() if missing}


FORCE_TOLERANCE_M = 50


@st.cache_resource
def force_outlines() -> dict[str, list[list[list[list[float]]]]]:
    """pfa24cd -> the force area as pydeck polygons, each [exterior, *holes] with rings of [lon, lat].

    Simplified to within FORCE_TOLERANCE_M of the true line: a quarter of a cell's 202m side, so at worst a border
    cell straddles the "within force" outline rather than sitting clearly on the wrong side of it, while cutting 4.1M
    vertices to ~50k (the largest force ~5k, ~110 KB of JSON per rerun). Each force is simplified on its own, so
    neighbours' shared edges can differ by up to twice that.
    """
    rows = get_con().sql(
        f"""
        SELECT spatial_id, ST_AsWKB(ST_SimplifyPreserveTopology(geom, {FORCE_TOLERANCE_M}))
        FROM read_parquet('{SOURCE}/extract/police_force_areas.parquet')
        """
    )

    def rings(polygon: shapely.Polygon) -> list[list[list[float]]]:
        return [
            np.round(np.column_stack(_TO_WGS84.transform(*np.asarray(ring.coords).T[:2])), COORD_DECIMALS).tolist()
            for ring in (polygon.exterior, *polygon.interiors)
        ]

    return {code: [rings(p) for p in shapely.get_parts(shapely.from_wkb(bytes(wkb)))] for code, wkb in rows.fetchall()}


@st.cache_data(max_entries=256)
def descriptions(ids: tuple[int, ...]) -> pd.Series:
    """`description` for the cells on screen, indexed by spatial_id: a sentence on the cell's character, road,
    retail centre and MSOA, e.g. "Urban, on Briggate; in Albion Street & Briggate, Leeds (regional centre). Leeds 111."
    """
    return (
        get_con()
        .sql(
            f"""
            SELECT spatial_id, description
            FROM read_parquet('{SOURCE}/transform/beahiv202_descriptions.parquet')
            WHERE spatial_id IN ?
            """,
            params=(list(ids),),
        )
        .df()
        .set_index("spatial_id")
        .description
    )


def cell_outlines(ids) -> list[list[list[float]]]:
    """Each cell's closed outline as [[lon, lat], ...] in WGS84, ready for a pydeck PolygonLayer.

    Straight from the cell id via beahiv, so no boundary table is read.
    """
    ids = np.asarray(ids, dtype=np.int64)
    if not len(ids):
        return []
    xy = shapely.get_coordinates(cell_polygons(ids))  # closed rings: 7 vertices per cell
    lon, lat = _TO_WGS84.transform(xy[:, 0], xy[:, 1])
    return np.round(np.column_stack([lon, lat]), COORD_DECIMALS).reshape(len(ids), -1, 2).tolist()
