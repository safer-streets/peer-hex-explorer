# Agent Guidelines for `peer-hex-explorer`

This file instructs AI agents acting as developer, reviewer, and QA for this repository.

## Project Overview

A Streamlit app: pick a crime hotspot on the [BEAHIV](https://github.com/safer-streets/beahiv) 202m grid, see the
cells that most resemble it in feature space. It is the workflow of safer-streets-eda's `ho_top_kc.py` (which made one
PDF page per Home Office knife-crime hotspot) made interactive, on the BEAHIV grid with public police.uk crime counts.
[PLAN.md](PLAN.md) is the original design; [JOURNAL.md](JOURNAL.md) records where the build departed from it and why.

It follows `prototype-streamlit-app`'s pattern: standalone, no API, DuckDB reading the Azure parquet store directly.
It depends on `beahiv`, and deliberately **not** on `safer-streets-core` or `safer-streets-eda`: the pieces it needs
are copied in (see Developer Rules).

| File | Role |
| ---- | ---- |
| [app.py](app.py) | Entry point (`uv run streamlit run app.py`); calls `main.main()` |
| [main.py](src/peer_hex_explorer/main.py) | The UI: sidebar, overview map, hotspot/peer tables, map + radar panels |
| [data.py](src/peer_hex_explorer/data.py) | Connection, the startup feature-matrix build (`characterisation`), cached loaders, cell outlines |
| [features.py](src/peer_hex_explorer/features.py) | The feature pipeline, **copied** from eda's `hex_features.py` / `clusterability_audit.py` |
| [peers.py](src/peer_hex_explorer/peers.py) | Pure numpy: brute-force Euclidean or cosine `nearest`, per-feature `contributions` |
| [basemap.py](src/peer_hex_explorer/basemap.py) | Server-side close-up map images (OSM tiles + hex outline) for the panels |
| [database.py](src/peer_hex_explorer/database.py) | DuckDB connector (spatial + azure), `SOURCE = "az://phase2"` |
| [utils.py](src/peer_hex_explorer/utils.py) | `CrimeType`, `Force`, `Month`, `fix_force_name` (from the prototype) |

## Toolchain

| Tool | Command |
| ---- | ------- |
| Package manager | `uv` |
| Run the app | `uv run streamlit run app.py` |
| Linter / formatter | `ruff` (`uv run ruff check`, `uv run ruff format`) |
| Type checker | `ty` (`uv run ty check`) |
| Tests | `uv run pytest` |

[.pre-commit-config.yaml](.pre-commit-config.yaml) runs `uv-lock`, `ruff check --fix`, `ruff format`, `ty` and
`pytest`. The ruff hooks *write*, so a commit can contain content you didn't stage.

## Quality Gates

All of the following must pass before any change is considered complete:

```sh
uv run ruff check .          # zero lint errors
uv run ruff format --check . # zero formatting issues
uv run ty check              # zero type errors
uv run pytest                # all tests pass
```

`tests/test_data.py` and `tests/test_app.py` run against Azure and **skip silently without credentials**
(`AZURE_STORAGE_CONNSTR`, or `.streamlit/secrets.toml`). A green run without credentials has not exercised the data
build or the app; say so rather than reporting it as fully tested. With credentials the suite takes ~20s and needs the
network, so a one-off failure there is worth a rerun before debugging.

## Developer Rules

- **NO SECRET VALUE MAY EVER ENTER YOUR PROMPT OR CONTEXT WINDOW.** Never open, `cat`, `grep` or otherwise read
  `.streamlit/secrets.toml`, `.env*`, or any other credential store, and don't run anything that prints one
  (`env`, `printenv`, a script that echoes the connection string). Code may *load* the secret (`st.secrets`,
  `tomllib` in tests) as long as nothing prints it. Never write one into a file, command line, commit message or
  [JOURNAL.md](JOURNAL.md). If a credential problem needs debugging, name the variable and let the human look. If a
  secret reaches the context anyway, say so, don't repeat it, and recommend rotating it.
- **Every read goes through DuckDB onto `az://phase2`.** No `pd.read_parquet`, no fsspec/adlfs, no local data files.
  User-supplied values reach SQL only as `?` parameters; table and column names are only ever interpolated from fixed
  strings in code.
- **One base connection, a cursor per rerun.** `data._base_con()` is a `cache_resource` shared by every session;
  queries go through `get_con()`, which returns a fresh `.cursor()`. A DuckDBPyConnection is not safe across
  Streamlit's per-session threads, and cursors inherit the `SET GLOBAL` Azure settings.
- **`features.py` is a copy, pinned by a parity test.** Don't "improve" it in place: a change there must either be
  made in eda's `hex_features.py` first and copied over, or be a deliberate fork recorded in the journal. When eda's
  pipeline changes on purpose, regenerate the fixture in eda's environment (recipe in `test_features.py`'s
  docstring); `test_features.py` failing is how drift shows up.
- **Scaling is fitted once, on a fixed hotspot population, column by column, and applied to every cell.** The
  population (`data.query_hotspot_population`) is the union over crime types of the fewest cells accounting for 25%
  of that type's crime, all months: eda's `hotspot_cells` rule. The radars draw these same scaled values (clipped at
  ±3 IQRs), so they show what distances measure. Being fixed and per-column is what keeps distances unchanged when a feature is switched off, and
  independent of the crime type and window. Anything that refits scaling per selection (per crime type, per force, per
  feature set) breaks that property and needs a journal entry saying why. Imputation (`clean_features`) stays national.
- **Northern Ireland is excluded before scaling and before ranking.** The BEAHIV tables include NI cells with every
  geography column NULL; both the characterisation query and the crime-count query filter `msoa21cd IS NOT NULL`
  *inside* the query. Filtering afterwards would leave NI in the medians, IQRs and rank slots.
- **Peers come only from cells with ≥1 crime of the selected type in the window.** There is no "all cells" option by
  design, and the page says so (More info, and the caption under the peer heading).
- **No per-panel WebGL.** The overview is the only pydeck map. Panel maps are server-side images from
  [basemap.py](src/peer_hex_explorer/basemap.py), because browsers cap WebGL contexts at ~16 per page. Keep tile use
  polite (identifying User-Agent, cached, ≤2 connections); OSM's tile policy does not cover heavy traffic.
- **pydeck tooltips escape field values.** Markup goes in the tooltip *template*, never in the data. 19-digit
  `spatial_id`s exceed a JS number's precision, so they only reach the browser as strings.
- **Light theme only** ([.streamlit/config.toml](.streamlit/config.toml)): the OSM panels and radar colours assume it.
- **No comments explaining *what* the code does.** Only the *why*, when it's non-obvious. Match the existing style.

## Reviewer Checklist

1. **Secrets** — nothing in the diff, the tests or the journal contains or prints a credential.
2. **Data access** — new reads go through `get_con()` and DuckDB, with parameters for anything user-chosen.
3. **Pipeline drift** — `features.py` unchanged, or changed in step with eda and the fixture regenerated.
4. **Scaling and ranking populations** — scaling still fitted on the fixed hotspot population, everything still
   NI-free, still ranked `n DESC, spatial_id`.
5. **Caching** — expensive loaders are cached with a `max_entries`/`ttl` bound; cheap filtering stays outside.
6. **UI state** — the hotspot selection lives in the table's widget state, keyed on crime type, lookback and number
   of hotspots; changing features, k, scope or distance metric must keep it.
7. **Docs** — the More info panel and [JOURNAL.md](JOURNAL.md) match the behaviour.

## QA Rules

- Run the full gate suite before declaring a task done, with Azure credentials available when the change touches
  `data.py` or `main.py`.
- `AppTest` can't click a dataframe row: `tests/test_app.py` injects the selection into session state instead.
  Rendering (the pydeck map, the panels) still needs a look in a real browser; headless Chrome here has no WebGL.
- Streamlit doesn't hot-reload changes to modules under `src/`, only to `app.py`: restart the server after editing.

## Task & Design Summaries

Every development task — feature, fix, refactor, or non-trivial investigation — gets an entry in
[JOURNAL.md](JOURNAL.md), written as part of the change. Newest first, directly below the
`<!-- New entries go directly below this line. -->` marker. Each entry records:

- **Why** — the motivating request or problem.
- **What** — the change: files touched, functions added/renamed/removed.
- **Design decisions** — any non-obvious tradeoff and the reasoning. Omit for mechanical changes.
- **Follow-ups** — anything deferred or deliberately left inconsistent.

Keep entries terse: bullets, not paragraphs.

## Repository Layout

```text
app.py                      # streamlit entry point
src/peer_hex_explorer/
  main.py  data.py  features.py  peers.py  basemap.py  database.py  utils.py
  assets/                   # logo
tests/
  test_peers.py             # pure numpy, always runs
  test_coverage.py          # force coverage gaps, always runs
  test_scaling.py           # the hotspot-population fit, always runs
  test_features.py          # parity with eda's hex_features, always runs
  test_data.py  test_app.py # Azure; skipped without credentials
  fixtures/                 # parity fixture (input sample + hex_features output)
.streamlit/config.toml      # light theme (secrets.toml alongside it is gitignored)
PLAN.md                     # original design
AGENTS.md
JOURNAL.md
```
