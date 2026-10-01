# peer-hex-explorer

A Streamlit app for finding the places that most resemble a crime hotspot.

Pick a crime type and time window to list the top hotspot cells in England & Wales, on the
[BEAHIV](https://github.com/safer-streets/beahiv) 202m equal-area hexagonal grid, from public
[police.uk](https://data.police.uk) data. Select one to see its nearest cells in feature space (roads, junctions,
amenities, transit, population, land cover) on a map, in a table, and as map + radar panels. Features can be switched
on and off, and peers limited to the hotspot's own force.

## Running

```sh
uv sync
uv run streamlit run app.py
```

Data is read directly from the project's Azure store, so `.streamlit/secrets.toml` needs an
`azure_storage_connstr` entry (it is gitignored).

## Tests

```sh
uv run pytest
```

The data and app tests need Azure credentials and are skipped without them.

See [AGENTS.md](AGENTS.md) for development rules and [JOURNAL.md](JOURNAL.md) for design history.
