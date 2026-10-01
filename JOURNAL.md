# Journal — `peer-hex-explorer`

The task/design log for this repo, newest first. Every task gets an entry recording **Why**, **What**, **Design
decisions**, and **Follow-ups** — see [Task & Design Summaries](AGENTS.md#task--design-summaries) in
[AGENTS.md](AGENTS.md) for the rules. Write the entry as part of the change, not after the fact.

<!-- New entries go directly below this line. -->

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
