# Plan: BEAHIV peer-hex explorer

A Streamlit app in a new repo. It follows the pattern of `prototype-streamlit-app`: standalone, DuckDB straight onto
Azure parquet, no API. The workflow is the one in `ho_top_kc.py`, with the HO 350m grid and offence counts replaced by
the **BEAHIV 202m grid** and **public police.uk crime counts**.

Dependencies: `beahiv`, yes. `safer-streets-core`, **no**. Nothing is imported from `safer-streets-eda` either; the
pieces that are needed get copied in.

## What it does

1. The user picks a **crime type**, a **force** (or all of England & Wales) and a **time window**. The app lists the
   top hotspot cells.
2. The user picks **one hotspot**. The app shows its **N nearest peer cells** by Euclidean distance in the scaled
   feature space, on a map, in a table and as radar charts.
3. The user can **switch features on and off**, and the peers are recomputed straight away.

## Data access

Every read goes through DuckDB, as `con.sql(... read_parquet('az://phase2/...'))`. The connection pattern is the
prototype's: one in-memory connection with `azure` and `spatial` loaded, and the credentials set with `SET GLOBAL` from
`st.secrets`. Each rerun gets its own `.cursor()`. There is no `pd.read_parquet`, no fsspec/adlfs and no local files.

### Feature matrix: built at startup, cached

`data.py: build_characterisation(con)`, wrapped in `@st.cache_resource` so it runs once per process and every session
shares the result. It reproduces `beahiv-characterisation.ipynb`:

1. **Register the `beahiv_k_ring` UDF** (notebook cell 7). It is an Arrow UDF over `beahiv.k_ring` returning `BIGINT[]`.
   Guard the registration with a check against `duckdb_functions()`, because registering twice raises
   `CatalogException`.
2. **Create the `bh_hosp_kring` view.** It counts hospital and emergency-department POIs (`extract/poi.parquet`) in
   each cell's 1-ring. It does this by unnesting the ~8.6k POIs over their rings, not by unnesting the hexes.
3. **Run the characterisation query** (notebook cell 9), keeping the notebook's sources: `transform/beahiv202_geogs`,
   `extract/poi`, `extract/naptan`, `extract/food_outlets`, `transform/beahiv202_schools_lookup`,
   `transform/beahiv202_road_intersection_counts` and `transform/beahiv202_population_counts`. It differs from the
   notebook in three ways:
   - It drops `oa21cd`, `lsoa21cd`, the IMD columns and the IMD join. These are context, not features
     (`NON_FEATURE_COLUMNS`).
   - It uses `COALESCE(naptan.n_stops, 0)`. That NULL is a structural zero, so it is fixed at source.
   - The source is fixed to `az://phase2`, with no `data_dir()`.
4. **Fetch the result** with `.fetchnumpy()`, then assert that `spatial_id` is unique (221,428 rows at the time of
   writing).
5. **Run the numpy feature pipeline** once (see below).
6. **Return a frozen bundle:**
   - the `float32` scaled matrix (~221k × 13, ~11 MB)
   - the `spatial_id` array
   - the column names
   - a per-column percentile matrix for the radars

**Cold start:** the query scans several extracts over Azure, so the first run will be slow. Show
`st.spinner("Building cell characterisation…")`, then measure the time and peak memory (build step 6). Only if the
measurement calls for it, also write the bundle to a local parquet in the app's cache directory, keyed on the
timestamps in Azure's `index.parquet`.

### Other loaders

All of these are cached and all of them push their filtering and aggregation into SQL.

| Loader | Query | Cache |
| --- | --- | --- |
| Months | `SELECT DISTINCT month FROM beahiv202_crime_counts` | `cache_data(ttl="1d")` |
| Hotspots | `beahiv202_crime_counts` filtered on `crime_type = ?` and `month IN ?`, semi-joined to `beahiv202_geogs` (`msoa21cd IS NOT NULL` to drop NI, `pfa24cd = ?` for the force). `SUM(count)` per cell, ordered `n DESC, spatial_id`, with `LIMIT ?`. The tie-break matches `hex_features.crime_rank`. | `cache_data` |
| Candidate pool | The hotspot query without the `LIMIT`, returning only `spatial_id`s of cells with ≥1 crime | `cache_data` |
| Crime counts and ranks for peers | The same aggregation, `WHERE spatial_id IN ?` | `cache_data` |
| Force membership | `SELECT DISTINCT spatial_id, pfa24cd FROM beahiv202_geogs` | `cache_resource` |
| Place names | `beahiv202_descriptions`, `short_location`, `WHERE spatial_id IN ?`, only for the cells on screen | `cache_data` |

Cell geometry comes from `beahiv.cell_polygons(ids)`, so no boundary table is needed. It is reprojected to WGS84 and
the coordinates are thinned to 1e-6°, as in the prototype.

## Feature pipeline (`features.py`)

These pieces are copied from `hex_features.py` and `clusterability_audit.py`, comments included:

- the constants: `STRUCTURAL_ZERO`, `LOG1P_COLUMNS`, `DEGENERATE_IQR_RATIO`, `COMPOSITIONAL_PARTS`,
  `ZERO_REPLACEMENT`, `SHORT_LABELS`
- `clean_features`: structural zeros set to 0, the median filled in for the rest. That covers the population columns
  (9.4% NULL) and `retail_centre_distance` (32% NULL).
- `ilr_transform` and `compositional_features` (`add_residual=False`). These turn the three land-cover shares into two
  ILR coordinates.
- `robust_scale`: log1p on the skewed counts, then median/IQR scaling, with the near-constant → sd and no-IQR → unit
  fallbacks.

Scaling is fitted **once, over all cells nationally**. It works column by column, so switching a feature off leaves
every other column's scaling unchanged. Distances therefore stay comparable as features are toggled, and they don't
change with the crime type.

## Peer search (`peers.py`, pure numpy)

```python
def nearest(M, target_idx, cols, k, candidate_mask) -> tuple[np.ndarray, np.ndarray]:
    d = np.linalg.norm(M[:, cols] - M[target_idx, cols], axis=1)
    d[~candidate_mask] = np.inf
    d[target_idx] = np.inf
    top = np.argpartition(d, k)[:k]
    order = np.argsort(d[top])
    return top[order], d[top][order]
```

- **Metric:** Euclidean only.
- **Brute force:** over ~221k rows it takes milliseconds, so there is no index and no vss.
- **Candidate mask:** the scope (national, or within the target's force, as in `ho_top_kc`'s two PDFs) combined with
  the pool (all cells, or only cells with ≥1 crime of the selected type in the window).
- **`contributions(M, target_idx, peer_idx, cols)`:** each feature's share of the squared distance. The table uses it
  to show *why* a peer is close.

## App flow (`main.py`)

### Sidebar

- Crime type (`CrimeType`, from the prototype)
- Force (`Force`, plus "England & Wales")
- Lookback in months, ending at the latest month
- Number of hotspots (default 20)
- Number of peers (default 5, max 20)
- Scope: national or within force
- Pool: all cells, or cells with ≥1 crime
- Features: a multiselect with everything on by default, except `retail_centre_distance`, which is off by default. The
  two ILR coordinates appear as a single "land cover" option.

### Main panel

1. **Hotspot table**, built with `st.dataframe(on_select="rerun", selection_mode="single-row")`. Columns: rank, crime
   count, `short_location`, force, `spatial_id`. Cells are equal-area, so ranking by count is the same as ranking by
   density. Hotspots that are missing from the feature matrix are listed but marked, and can't be selected.
2. **Map** (pydeck): every hotspot in a faint colour, the selected hotspot in the reference colour, and the peers
   numbered. The tooltip shows rank, count and place.
3. **Peer table**: #, distance, crime count, crime rank, `short_location`, force, and the top 3 contributing features.
4. **Radars** (plotly `Scatterpolar`, small multiples): the target as a tinted backdrop on each, a percentile radial
   scale from −100 to +100 across the full national matrix, a fixed spoke order and `SHORT_LABELS`. The visual rules
   are the ones in `hex_features.radar_panels`.

### State

`st.session_state` holds the selected hotspot. Changing the crime type, force or window clears it. Changing the
features, k, scope or pool keeps it and recomputes the peers.

## Repo layout

This follows the existing uv scaffold (`src/` layout, `uv_build`, Python 3.14):

```text
peer-hex-explorer/
  pyproject.toml   python >=3.14; streamlit, duckdb, pydeck, plotly, pandas, numpy, pyproj, python-dotenv,
                   beahiv[arrow] @ git+ssh://git@github.com/safer-streets/beahiv@v1.0.1
  app.py           thin Streamlit entry point (`streamlit run app.py`), calls peer_hex_explorer.main.main()
  src/peer_hex_explorer/
    main.py        UI
    database.py    from the prototype, minus the h3 extension
    data.py        startup characterisation build + cached loaders
    features.py    copied feature pipeline
    peers.py       Euclidean search + contributions
    utils.py       CrimeType / Force / Month / fix_force_name (from the prototype)
    assets/
  .streamlit/secrets.toml   (gitignored; azure_storage_connstr)
  tests/
```

The scaffold's `[project.scripts]` entry (`peer_hex_explorer:main`) doesn't fit a Streamlit app, so either remove it or
make it a wrapper that runs `streamlit run`.

If the app is deployed to Streamlit Cloud, beahiv has to be reachable without SSH: either a public repo over https or
a PyPI release.

## Tests

- `test_peers.py`:
  - a toy matrix with a known nearest neighbour
  - deselected columns have no effect on the result
  - the target is never its own peer
  - the within-force mask works
  - the contributions sum to 1
- `test_features.py`: a parity fixture. Run `hex_features` (in eda) on a fixed sample of ~2k cells, save its output,
  and check the copied pipeline matches it to 1e-9. This catches drift between the two copies.
- `test_data.py`, marked as needing Azure and skipped without credentials:
  - The startup build matches the existing `beahiv202-characterisation.parquet` column by column on the feature
    columns, with `n_stops` compared after filling with 0.
  - There are no NI cells.
  - Every hotspot id decodes with `beahiv.decode` at side length 202.

## Build order

1. Scaffold the repo: `pyproject.toml`, `database.py`, `utils.py`, secrets. Confirm the Azure connection works.
2. The startup characterisation build, plus the parity check against the notebook's parquet.
3. `features.py` and `peers.py`, with their tests.
4. The hotspot list and map, with no peers yet.
5. Hotspot selection, then the peer map and table, then the radars and contributions.
6. Measure cold-start time and peak memory. Add the local parquet cache only if needed.

## Open decisions

- **Population to scale over.** The plan uses all cells nationally. `ho_top_kc` scaled over the HO hotspot set, which
  moves the medians and percentiles. The alternative, scaling over the selected category's hotspots, would mean
  rescaling every time the category changes.
- **`retail_centre_distance`.** It is offered but off by default: 32% of its values are NULL and are filled with the
  median, and `ho_top_kc` dropped it.
- **Cells on force borders.** Under within-force scope, a cell counts as in every force it overlaps (from
  `beahiv202_geogs`).
- **Population imputation.** The 9.4% of cells with a median-filled population carry over `hex_features`' current
  choice. Flag this in the app's "More info" panel.
