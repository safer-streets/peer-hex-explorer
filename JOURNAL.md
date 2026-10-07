# Journal — `peer-hex-explorer`

The task/design log for this repo, newest first. Every task gets an entry recording **Why**, **What**, **Design
decisions**, and **Follow-ups** — see [Task & Design Summaries](AGENTS.md#task--design-summaries) in
[AGENTS.md](AGENTS.md) for the rules. Write the entry as part of the change, not after the fact.

<!-- New entries go directly below this line. -->

## Building counts as features

- **Why** — Requested: the Verisk building counts (`transform/beahiv202_building_counts`) as features.
- **What**
  - `data.py`: `CHARACTERISATION_QUERY` joins the building counts and adds `n_res_buildings` and
    `n_nonres_buildings`, COALESCEd to 0 (no row means no building of that use).
  - `features.py` (and eda's `hex_features.py`, in step): both columns in `LOG1P_COLUMNS` and `SHORT_LABELS`.
  - `main.py`: More info lists them; `feature_groups` makes them one sidebar pill, "buildings", like land cover.
    Tests: `test_buildings`; the notebook-parity check allows these two extra columns.
- **Design decisions**
  - **Mixed Use counts in both**, as in eda's `buildings.ipynb`: two features, not three.
  - **One on/off pill for both.** They overlap (Mixed Use) and describe one building stock; the radars and
    contributions still show them separately.
  - **log1p, then the usual robust scale.** Over all E&W cells: residential median 55, max 895, 7% zero;
    non-residential median 5, max 401, 12% zero. Fitted on the hotspot population, both IQRs are usable (divisors
    1.13 and 1.46, no fallback).
- **Follow-ups**
  - `n_res_buildings` has Spearman 0.94 with `residential_population` (the population is assigned via buildings), so
    the two together roughly double the weight on residents. Non-residential vs workers: 0.73.
  - eda's `beahiv-characterisation.ipynb` and its parquet don't have the columns yet. The parity fixture doesn't
    either, so `test_features.py` doesn't cover them.

## Cosine distance; consistent coverage notes

- **Why** — Requested: a switch between Euclidean and cosine similarity. And the greyed-out force notes read
  inconsistently: "no robbery recorded in the window" for the whole window, but "none recorded in 2025-03" (no crime
  type) for part of it.
- **What**
  - `peers.py`: `Metric` (`"euclidean"` | `"cosine"`); `nearest` and `contributions` take `metric`, default Euclidean.
    `_directions` normalises rows to unit length.
  - `main.py`: "Distance" segmented control in the Peers sidebar (session key `metric`, not in the table key, so the
    selected hotspot survives a switch). Distances shown to 3 dp under cosine (`DISTANCE_DIGITS`). The peer caption
    names the metric. Warning when the target is all zeros over the selected features. More info: the two metrics,
    "differs most on" and the radars under cosine. Coverage note is now "no {crime} recorded in {months}" both ways.
  - Tests: four cosine cases in `test_peers.py`; `test_app.py::test_cosine_peers`. AGENTS.md's peers.py row and
    UI-state checklist item.
- **Design decisions**
  - **Cosine on the scaled matrix as it is.** Centring is on the median hotspot, so cosine compares the direction a
    cell departs from the median hotspot in, ignoring how far. Scaling is unchanged, so the fixed-population property
    holds for both metrics.
  - **Computed as ||û − v̂||² / 2**, equal to 1 − cos for unit vectors: never slightly negative from rounding, and it
    makes "differs most on" exact (each feature's share of the squared unit-vector difference).
  - **Zero rows have no direction**: excluded as candidates, and a zero target gets a warning rather than arbitrary
    peers. Only likely with very few features selected (e.g. one outlet count, where the median hotspot has none).
  - **Radars unchanged**: they still show scaled values, so a cosine peer can be a bigger or smaller copy of the
    target's shape. More info says to compare shapes.
- **Follow-ups**
  - `AppTest` drops an injected table selection on *any* rerun, so the "selection survives a metric switch" property
    isn't testable there; check it in a browser.
  - With one or two features, cosine ranks are mostly ties (broken by row index). Left as is; More info says so.

## Shops replace retail centre distance

- **Why** — `retail_centre_distance` is only looked up within 2km (tooling's `RETAIL_RADIUS`) and is NULL beyond:
  31% of cells, median-filled, so the remotest cells read as a typical distance from a centre. Requested: drop it
  and count shops instead.
- **What**
  - safer-streets-tooling (branch `poi-shops`): 27 Overture shop categories added to the POI extract's config. The
    extract has to be rerun before `n_shops` is anything but 0.
  - `data.py`: `SHOP_CATEGORIES`; the characterisation query drops `retail_centre_distance` and adds `n_shops`
    (`COUNT(*) FILTER` over `poi`, like `n_alcohol`/`n_food`).
  - `main.py`: `DEFAULT_OFF` removed, so every feature starts on. More info's feature list and caveats.
  - `features.py`: `n_shops` in `LOG1P_COLUMNS` and `SHORT_LABELS` (`"shops"`), copied from eda's `hex_features.py`,
    which made the same change; `retail_centre_distance`'s label is gone. `test_shops` checks the counts against
    the extract exactly, and notebook parity now covers `n_shops` too.
  - Test references regenerated (2026-10-05): eda's `beahiv202-characterisation.parquet` re-run (cells 2, 3, 7, 9 and
    11 of `beahiv-characterisation.ipynb`, on the re-extracted POI), and `tests/fixtures/parity_*.parquet`
    re-sampled from it by `test_features.py`'s recipe, so both carry `n_shops`.
- **Design decisions**
  - **Changed in eda first, then copied.** It began as a fork here (`data.LOG1P`, a local `SHORT_LABELS` entry),
    but eda's characterisations dropped the distance too, once the geogs stopped carrying it. Making the change
    upstream keeps `features.py` an exact copy, and the local workarounds are gone.
  - **log1p, like the other outlet counts**: shop counts have the same skewed, zero-heavy shape.
  - **On by default.** Retail was off because it was a third median-filled; a count has no gaps to fill.
  - **Retail proximity was tried first** (1 in a centre, decaying to 0 at 2km) and dropped. Two findings: linear
    (1 - d/2km) gave a z of -9.4 beyond 2km and ~23% of hotspot-to-hotspot squared distance, because 56% of
    hotspots are in a centre (d = 0) and that sets a tiny IQR; exp(-d/500m) brought it to 7%. And no curve of
    distance tells cells *inside* a centre apart, which a count does.
- **Follow-ups**
  - Shop weight in the distance hasn't been measured (no data yet). Check its share once the extract is in.

## Radars: scaled values, not percentiles

- **Why** — Cells in a retail centre didn't reach the rim of the retail spoke. Percentile ties take their midpoint:
  55.5% of hotspots have proximity exactly 1, so they all plotted at 72. Mirror image on the outlet counts: hotspots
  with none (51-61%) plotted at 26-31, not the centre. More generally, percentiles stretched the scale unevenly, so a
  gap on a spoke didn't match that feature's part of the distance.
- **What**
  - `main.py`: `radar` draws `bundle.scaled` clipped to ±`RADAR_LIMIT` (3), ring at 0 (median hotspot); hover shows
    the unclipped value in IQRs. More info and the caption rewritten.
  - `data.py`: `population_percentiles` and `Characterisation.percentile` removed. `tests/test_scaling.py` loses
    `test_population_percentiles`; `features.percentiles` stays as the parity copy. AGENTS.md's scaling and NI rules.
- **Design decisions**
  - **Same units as the distance.** The gap on a spoke is that feature's difference, comparable across spokes.
  - **±3, chosen by the user over ±1 and ±2.** Every feature's population divisor is its IQR (no sd/unit fallback),
    so "IQR" is exact. 0-4% of hotspot values per feature fall outside ±3 (residents 4.2%, hospital 3.1%); ±1 would
    have clipped 92% of hotspots on at least one spoke.
  - **One-sided spokes are left as they are.** Retail and the outlet counts sit on the ring at their typical value
    (in a centre; none) and only extend one way. That's what the distance sees; More info says so.


## CI

- **Why** — There was none; the gates only ran in pre-commit.
- **What**
  - `.github/workflows/lint-test.yml`: on push to `main` and on PRs, `uv sync --locked`, then `ruff check`,
    `ruff format --check`, `ty check`, `pytest -rs`. Ubuntu, Python 3.14; action versions as in beahiv's workflow.
  - `tests/test_app.py`: hands the connection string to `AppTest` via `at.secrets`, since the app reads `st.secrets`
    and CI has only the `AZURE_STORAGE_CONNSTR` env var, no `secrets.toml`.
- **Design decisions**
  - **Azure tests run in CI only if the `AZURE_STORAGE_CONNSTR` repo secret is set**, and never for PRs from forks.
    Without it they skip; `-rs` lists the skips in the log so a green run without them is visible as such.
  - Checked locally: with no `secrets.toml` and no env var, 17 pass and 10 skip; with only the env var, 27 pass.
- **Follow-ups**
  - Someone with admin rights must add the `AZURE_STORAGE_CONNSTR` secret.

## Scale on a hotspot population; outline the force under "Within force"

- **Why** — Radars put every hotspot near the rim on almost every feature, and the user asked how the PDFs differed.
  `ho_top_kc.py` used the same pipeline but fitted it (and ranked the radar percentiles) over the 4,433 HO hotspot
  hexes; this app fitted over all 209k cells. Separately: show the force when peers are limited to it.
- **What**
  - `data.py`: `HOTSPOT_SHARE`, `query_hotspot_population` (eda's `hotspot_cells` rule on BEAHIV, all 14 types,
    all months, NI excluded inside: 13,163 cells), `scale_to_population`, `population_percentiles`;
    `Characterisation.in_population`. `build_characterisation` uses them in place of `scale_features`/`percentiles`.
  - `main.py`: More info and the radar caption describe the population; `cell_map(scope_force=...)` draws the
    target's force boundary (unfilled, not pickable) under "Within force".
  - `cell_map`: every listed hotspot is drawn, peers included (before, a peer that was also a hotspot was drawn only
    as a peer). Such a peer is drawn in the same blue at lower alpha, so the hotspot shows through, and its tooltip
    says "peer 3 · hotspot 7". Alpha rather than a new colour, at the user's request. A colour key caption sits under
    the map once a target is picked. Marker numbers now come from `#`, not the label.
  - Radars show plain 0-100 percentiles (median ring at 50), not eda's -100..+100: "percentile −100" read as
    nonsense. `population_percentiles` returns 0-100; `features.percentiles` is left as the parity copy.
  - `force_outlines`: simplification tolerance 200m -> 50m (`FORCE_TOLERANCE_M`). 200m is most of a 202m cell,
    enough to put a border cell on the wrong side of the outline; 20m was tried, 2x the vertices for no visible gain.
  - `features.py` untouched. Tests: `tests/test_scaling.py` (parity fixture, no Azure), population tests in
    `test_data.py`. AGENTS.md's scaling rule rewritten.
- **Design decisions**
  - **Why it matters for distances, not just radars.** Nationally alcohol, food, hospital and takeaways are 86-90%
    zero, so `robust_scale` leaves them in log1p units (no IQR). Among the top 4,433 Robbery cells they then made up
    51% of the mean squared distance; roads + junctions + workers 8%. Over the population no feature exceeds 20%
    (residents) and those four are ~5% each.
  - **Fixed population, not the selection's candidates.** Fitting per crime type or window would make distances
    depend on the selection; one fixed set keeps every property the national fit had.
  - **Union of per-type cuts**, not one cut on total crime, so cells hot for a low-volume type (robbery, weapons) are
    represented rather than drowned by violence and ASB.
  - **features.py not forked.** `robust_scale` only returns the scaled input, so `scale_to_population` repeats its
    log1p to apply the fit to the rest; `test_scaling.py` pins that it matches `robust_scale` on the population rows
    and reproduces the national fit when the population is every cell.
  - **Imputation stays national.** It fills gaps; it doesn't set weights.
- **Follow-ups**
  - Top hotspots still sit at the 90th-98th population percentile on the outlet features; that's now a real
    statement ("busier than most hotspots"), not a saturation artefact.

## Grey out forces with no data

- **Why** — GMP publishes nothing to police.uk, so its area looked crime-free on the map; other forces miss months.
- **What**
  - `data.py`: `query_coverage`/`coverage` read `extract/crime_coverage.parquet` (police.uk's per-force, per-month,
    per-crime-type counts) for the selected crime type and window; `coverage_gaps` gives force -> months with zero
    (or no row); `app_force_name` maps police.uk names ("Metropolitan Police Service") to `Force`;
    `force_outlines` reads `extract/police_force_areas.parquet`, simplified to 200m, as pydeck polygons.
  - `main.py`: `cell_map` draws a grey layer under everything: solid for forces with none of the crime type in any
    month of the window, light for some months; tooltip and a caption name them. More info caveat added.
  - Tests: `tests/test_coverage.py` (pure), `test_coverage_resolves_every_force` in `test_data.py`.
- **Design decisions**
  - **Coverage table, not our own counts.** GMP's border cells still get ~50 crimes/month in
    `beahiv202_crime_counts` (neighbours' crimes located across the boundary), so "zero in our counts" misses it.
    police.uk's per-force count is exactly zero.
  - **Per crime type, every month must be > 0.** As requested. So a genuine zero in a small force can grey it
    lightly (City of London, Possession of weapons, 2025-08).
  - **Force code via name, not the boundary table's join.** The coverage table has names only; `force_codes`
    already resolves app names to `pfa24cd`.
- **Follow-ups**
  - Only greys the map: hotspots and peers in gap forces are still listed, unflagged.

## Initial build

- **Why** — Make `ho_top_kc.py`'s hotspot → nearest-cells PDFs interactive, on the BEAHIV 202m grid with public
  police.uk counts instead of the HO 350m grid and offence data. Design in [PLAN.md](PLAN.md).
- **What**
  - `data.py`: startup build of the national feature matrix (beahiv-characterisation.ipynb's query, minus IMD/OA/LSOA,
    `n_stops` COALESCEd, NI excluded), scaled and frozen into a `Characterisation` bundle; one cached crime-count query
    per crime type + window; `descriptions`; cell outlines via `beahiv.cell_polygons`.
  - `features.py`: the pipeline copied from eda, pinned by `test_features.py` to 1e-9.
  - `peers.py`: brute-force Euclidean `nearest` and `contributions`.
  - `main.py`: overview map above the top-N E&W hotspot table; select a hotspot for its peers (table in an expander)
    and one map + radar panel per cell.
  - `basemap.py`: server-side OSM close-ups for the panels.
- **Design decisions**
  - **NI excluded at source.** The BEAHIV tables carry 12,136 NI cells with every geography NULL and ~415k crimes.
    Dropping them before scaling also cut median-imputed population from the plan's 9.4% to 4.1%.
  - **One crime query, not three.** The plan had separate loaders for hotspots, the candidate pool and peer
    counts/ranks; each would scan the crime table over Azure. One cached aggregate serves all three.
  - **No force-membership loader.** `beahiv202_geogs` has exactly one `pfa24cd` per cell (max overlap), so the
    plan's "border cells count in every force" question is moot; the force rides along in the bundle.
  - **No force dropdown, no pool option** (user feedback after the first cut). Hotspots are national; "Within force"
    means the selected hotspot's force; peers are always cells with ≥1 crime, documented on the page.
  - **Panel maps are images, not pydeck.** 21 panels + the overview would exceed the browser's ~16 WebGL contexts.
    Carto raster tiles now need an API key, so they're OSM (z17).
  - **Radars use national percentiles.** Same transform as `radar_panels(radial="percentile")`, but over all 209k
    E&W cells; the PDFs ranked within the 4,433 HO hotspot hexes, so city-centre cells sit nearer the rim here.
  - **No local parquet cache.** Measured cold start: 4.1s, 870 MB peak RSS.
  - **beahiv pinned to `v0.0.2` over https**, not the plan's `v1.0.1` over ssh: confirmed correct, and https means
    no SSH key is needed to install it (e.g. on Streamlit Cloud).
- **Follow-ups**
  - Radar percentiles could be ranked over the crime-affected candidate cells instead, to read like the PDFs —
    offered, not done.
  - OSM tiles are fine for a prototype; a deployment with real traffic needs a commercial or self-hosted tile source.
  - The pydeck overview map has never been seen rendered by the agent (no WebGL headless); checked by the user only.
  - `test_data.py`'s notebook-parity check allows 1% id mismatch: geogs was rebuilt after the notebook parquet was
    written and ~1k edge cells moved. Values on shared cells match exactly.
  - `.pre-commit-config.yaml`'s pytest comment says the suite is "well under a second"; with Azure credentials it's
    ~20s.
  - The script that made the parity fixture was temporary and isn't committed; the recipe is in
    `test_features.py`'s docstring.
