"""
Pick a crime hotspot, find the cells that most resemble it in feature space.

Standalone (no safer-streets-core dependency), no API: DuckDB reads the Azure parquet store directly. The workflow is
safer-streets-eda's ho_top_kc.py on the BEAHIV 202m grid with public police.uk crime counts.
"""

from pathlib import Path
from typing import cast, get_args

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from pydeck.data_utils import compute_view

from peer_hex_explorer.basemap import cell_map_image
from peer_hex_explorer.data import (
    Characterisation,
    all_months,
    cell_outlines,
    characterisation,
    coverage,
    coverage_gaps,
    crime_counts,
    descriptions,
    force_codes,
    force_outlines,
)
from peer_hex_explorer.features import SHORT_LABELS
from peer_hex_explorer.peers import contributions, nearest
from peer_hex_explorer.utils import CrimeType

CATEGORIES = get_args(CrimeType)
NATIONAL, WITHIN_FORCE = "National", "Within force"
LAND_COVER = "land cover"
DEFAULT_OFF = {"retail"}  # retail_centre_distance: 31% median-filled, and ho_top_kc dropped it

CELL_COLOUR = "#356285"  # hex_features.CELL_COLOUR
CELL_OUTLINE = "#1d3a52"
REFERENCE_COLOUR = "#b3402a"  # the cell every other panel is compared against
REFERENCE_OUTLINE = "#6e2116"
REFERENCE_TINT = "#d19a90"  # the reference drawn as a backdrop: a tint of REFERENCE_COLOUR, so it reads as it
HOTSPOT_COLOUR = "#f76707"  # strong orange: reads on the light basemap, and apart from the target red and peer blue
HOTSPOT_OUTLINE = "#7a2e00"
FORCE_OUTLINE = "#333333"  # the "within force" boundary: dark and unfilled, so it frames without hiding anything
NO_DATA_COLOUR = "#6b6b6b"  # forces without police.uk data: neutral, so it reads as absence rather than a category

N_TOP_CONTRIBUTIONS = 3


def _rgba(hex_colour: str, alpha: float) -> list[int]:
    h = hex_colour.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)] + [round(alpha * 255)]


def _css_rgba(hex_colour: str, alpha: float) -> str:
    r, g, b, _ = _rgba(hex_colour, alpha)
    return f"rgba({r},{g},{b},{alpha})"


def feature_groups(columns: tuple[str, ...]) -> dict[str, list[int]]:
    """Sidebar option -> column indices. The two ILR coordinates are one option: switching off half of a
    composition would leave a log-ratio whose meaning depends on the part that's gone."""
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(columns):
        label = LAND_COVER if c.startswith("ilr_") else SHORT_LABELS.get(c, c)
        groups.setdefault(label, []).append(i)
    return groups


def force_names() -> dict[str, str]:
    """pfa24cd -> app force name"""
    return {code: name for name, code in force_codes().items()}


def init(months: tuple[str, ...]) -> None:
    st.session_state.setdefault("category", "Robbery")
    st.session_state.setdefault("lookback", min(12, len(months)))
    st.session_state.setdefault("n_hotspots", 20)
    st.session_state.setdefault("n_peers", 5)
    st.session_state.setdefault("scope", NATIONAL)


def peer_frame(
    bundle: Characterisation,
    counts: pd.DataFrame,
    names: dict[str, str],
    target_row: int,
    peer_rows: np.ndarray,
    distances: np.ndarray,
    cols: np.ndarray,
) -> pd.DataFrame:
    """One row per peer, nearest first, indexed by spatial_id. `description` is added by the caller."""
    share = contributions(bundle.scaled, target_row, peer_rows, cols)
    labels = [SHORT_LABELS.get(bundle.columns[c], bundle.columns[c]) for c in cols]
    peer_ids = bundle.spatial_id[peer_rows]
    return pd.DataFrame(
        {
            "#": np.arange(1, len(peer_rows) + 1),
            "distance": distances,
            # peers are only ever drawn from cells with crime, so every one has a count and a rank
            "crimes": counts["n"].loc[peer_ids].to_numpy(),
            "rank": counts["national_rank"].loc[peer_ids].to_numpy(),
            "force": [names.get(p, "") for p in bundle.pfa24cd[peer_rows]],
            "differs most on": [
                ", ".join(f"{labels[j]} {s[j]:.0%}" for j in np.argsort(-s)[:N_TOP_CONTRIBUTIONS] if s[j] > 0)
                for s in share
            ],
            "spatial_id": peer_ids.astype(str),
        },
        index=pd.Index(peer_ids, name="spatial_id"),
    )


def more_info(bundle: Characterisation) -> None:
    imputed = ", ".join(f"`{c}` {pct:.1f}%" for c, pct in bundle.imputed_pct.items())
    st.markdown(
        f"""
Uses [police.uk](https://data.police.uk) public crime data, counted on the
[BEAHIV](https://github.com/safer-streets/beahiv) 202m-side equal-area hexagonal grid. Cells are equal-area, so ranking
by count is ranking by density.

1. Pick a crime type and time window: the table lists the top hotspot cells in England & Wales.
2. Select one: its nearest cells in feature space are shown on the map, in the peer table and as map + radar panels.
3. Switch features on and off to see which ones the match depends on.

**Which cells can be peers.** Only cells that recorded **at least one crime of the selected type in the window**: the
question is which *crime-affected* places resemble the hotspot, not which quiet ones do. So the candidate set changes
with the crime type and window, even though the distances don't. Under "Within force", peers are further limited to
the force the selected hotspot lies in.

**How "near" is measured.** Each cell is described by {len(bundle.columns)} features (road length, junctions, school
catchment depth, food/alcohol/takeaway outlets, hospitals, transit stops, residents, workers, distance to a retail
centre, and land cover as two log-ratio coordinates). Skewed counts are log1p-transformed, then every feature is
median-centred and IQR-scaled, fitted **once, over a fixed population of {bundle.in_population.sum():,} hotspot
cells**: those among the fewest cells accounting for 25% of any one crime type's crime, over all months. The fit is
then applied to all {len(bundle.spatial_id):,} cells in England & Wales. Distance is Euclidean over the selected
features. Scaling is per feature, so switching a feature off leaves every other feature's scaling, and so the
distances, unchanged; and it doesn't depend on the crime type or window.

Why hotspots and not every cell: most cells are rural or suburban, and 86-90% have no food, alcohol, takeaway or
hospital at all. Fitted over every cell, those four have no spread to scale by, stay in raw log units, and end up
carrying half the distance between two urban cells. Fitted over hotspots, every feature's spread is set by places like
the ones being compared, so the weight is shared out much more evenly.

**Peer table.** "Differs most on" is each feature's share of that peer's squared distance to the target: the features
where the match is weakest.

**Radars.** Each spoke is the cell's percentile on that feature *among the hotspot population*, from −100 (lowest)
through 0 (the typical hotspot, dotted) to +100 (highest). The target is the tinted shape behind every panel.

**Caveats.**
- Median-imputed values: {imputed}. These cells read as typical on that feature, whatever they really are; retail
  distance is off by default for this reason.
- Northern Ireland is excluded throughout: police.uk publishes its crime, but it has no matching geography or features.
- Each cell belongs to one force (by greatest overlap), which is what "within force" uses.
- Some forces publish nothing to police.uk (Greater Manchester, since 2019), and others miss months. The map greys
  out a force that recorded none of the selected crime type in any month of the window, and shades one with only some
  such months more lightly (police.uk's own per-force counts); their cells are under-counted and rarely appear as
  hotspots or peers. Border cells of a greyed-out force can still show crimes: those are a neighbour's, located
  across the boundary.
"""
    )
    st.caption(
        f"Feature matrix built in {bundle.build_seconds:.1f}s; peak process memory {bundle.peak_rss_mb:,.0f} MB."
    )


def selected_hotspot(hotspots: pd.DataFrame, key: str) -> int | None:
    """The spatial_id selected in the hotspot table keyed `key`, if any.

    Read from session_state rather than from the widget's return value so it is known before the table is drawn: the
    peers' place names then come back in the same query as the hotspots'.
    """
    state = st.session_state.get(key)
    rows = state["selection"]["rows"] if state is not None else []
    return int(hotspots.index[rows[0]]) if rows and rows[0] < len(hotspots) else None


def hotspot_table(hotspots: pd.DataFrame, key: str) -> None:
    display = pd.DataFrame(
        {
            "rank": hotspots["rank"],
            "crimes": hotspots["n"],
            "description": hotspots["description"],
            "force": hotspots["force"],
            "spatial_id": hotspots.index.astype(str),
        }
    )
    if not hotspots["in_matrix"].all():
        display["note"] = np.where(hotspots["in_matrix"], "", "no features: can't be compared")
    st.dataframe(display, hide_index=True, on_select="rerun", selection_mode="single-row", key=key, width="stretch")


def peer_table(peers: pd.DataFrame) -> None:
    st.dataframe(
        peers[["#", "distance", "crimes", "rank", "description", "force", "differs most on", "spatial_id"]],
        hide_index=True,
        width="stretch",
        column_config={
            "distance": st.column_config.NumberColumn(format="%.2f"),
            "crimes": st.column_config.NumberColumn(format="%d"),
            "rank": st.column_config.NumberColumn(format="%d", help="National rank by crime count"),
        },
    )


def cell_map(
    hotspots: pd.DataFrame,
    target: int | None,
    peers: pd.DataFrame | None,
    gaps: dict[str, list[str]],
    window: tuple[str, ...],
    category: CrimeType,
    scope_force: str | None = None,
) -> None:
    """Hotspots strong until a target is picked, then faint; the target in the reference colour, peers numbered. Frames
    the target and its peers when there are any, otherwise the hotspots. Forces with months of `window` without
    `category` (`gaps`: force name -> those months) are greyed out underneath: darker if it's the whole window.
    `scope_force` (pfa24cd) is outlined when peers are limited to that force."""

    def records(cells: pd.DataFrame, labels: list[str], details: list[str]) -> list[dict]:
        # Markup goes in the tooltip template, not here: field values are HTML-escaped when substituted, so tags in
        # the data would show up as text
        return [
            {
                "polygon": outline,
                "centre": np.mean(outline[:-1], axis=0).tolist(),
                "label": label,
                "description": description,
                "detail": detail,
            }
            for outline, label, description, detail in zip(
                cell_outlines(cells.index), labels, cells["description"], details, strict=True
            )
        ]

    def hotspot_records(cells: pd.DataFrame, prefix: str = "") -> list[dict]:
        return records(cells, [f"{prefix}hotspot {r}" for r in cells["rank"]], [f"{n:,} crimes" for n in cells["n"]])

    # peers stay in: a peer that is also a hotspot shows the hotspot through its (lighter) peer colour
    background = hotspot_records(hotspots.drop(index=[target] if target is not None else []))

    # Every set of cells gets a polygon layer plus a fixed-size marker at its centre: at the national zoom a cell is a
    # pixel or two across, so without the marker it would vanish until zoomed right in. The marker carries the
    # contrast, so the polygon fill stays light enough to read the streets underneath when zoomed in.
    def polygons(data: list[dict], colour: str, outline: str, fill_alpha: float, line_px: float) -> pdk.Layer:
        return pdk.Layer(
            "PolygonLayer",
            data,
            get_polygon="polygon",
            get_fill_color=_rgba(colour, fill_alpha),
            get_line_color=_rgba(outline, 1.0),
            line_width_min_pixels=line_px,
            pickable=True,
        )

    def markers(data: list[dict], colour: str, outline: str, radius_px: float, alpha: float = 0.95) -> pdk.Layer:
        return pdk.Layer(
            "ScatterplotLayer",
            data,
            get_position="centre",
            get_radius=1,
            radius_min_pixels=radius_px,
            get_fill_color=_rgba(colour, alpha),
            get_line_color=_rgba(outline, alpha),
            stroked=True,
            line_width_min_pixels=1.5,
            pickable=True,
        )

    outlines, codes = force_outlines(), force_codes()
    no_data: list[dict] = []
    part_data: list[dict] = []
    notes = {}
    for force, missing in gaps.items():
        whole = len(missing) == len(window)
        notes[force] = note = (
            f"no {category.lower()} recorded in the window" if whole else f"none recorded in {', '.join(missing)}"
        )
        (no_data if whole else part_data).extend(
            {"polygon": polygon, "label": force, "description": note, "detail": ""}
            for polygon in outlines.get(codes[force], [])
        )
    force_layers = [
        polygons(no_data, NO_DATA_COLOUR, NO_DATA_COLOUR, 0.45, 1),
        polygons(part_data, NO_DATA_COLOUR, NO_DATA_COLOUR, 0.15, 1),
    ]
    if scope_force is not None:
        # not pickable: the stroke would otherwise take the tooltip from cells lying on the boundary
        force_layers.append(
            pdk.Layer(
                "PolygonLayer",
                [{"polygon": polygon} for polygon in outlines.get(scope_force, [])],
                get_polygon="polygon",
                filled=False,
                get_line_color=_rgba(FORCE_OUTLINE, 0.9),
                line_width_min_pixels=2,
            )
        )

    if target is None or peers is None:
        # nothing selected (or nothing to compare): the hotspots are the subject, so draw them strongly
        layers = [
            *force_layers,
            polygons(background, HOTSPOT_COLOUR, HOTSPOT_OUTLINE, 0.35, 2),
            markers(background, HOTSPOT_COLOUR, HOTSPOT_OUTLINE, 6),
        ]
        focus = background
    else:
        # a target is selected: the other hotspots drop back to context, the target and its peers take the contrast
        # a peer that is also a listed hotspot is drawn lighter, so the hotspot under it shows, and its label says so
        hotspot_rank = hotspots["rank"].reindex(peers.index)
        peer_records = records(
            peers,
            [
                f"peer {i}" + (f" · hotspot {r:.0f}" if pd.notna(r) else "")
                for i, r in zip(peers["#"], hotspot_rank, strict=True)
            ],
            [f"distance {d:.2f} · {n:,} crimes" for d, n in zip(peers["distance"], peers["crimes"], strict=True)],
        )
        peer_only = [r for r, rank in zip(peer_records, hotspot_rank, strict=True) if pd.isna(rank)]
        peer_hot = [r for r, rank in zip(peer_records, hotspot_rank, strict=True) if pd.notna(rank)]
        target_records = hotspot_records(hotspots.loc[[target]], prefix="target: ")
        layers = [
            *force_layers,
            polygons(background, HOTSPOT_COLOUR, HOTSPOT_COLOUR, 0.12, 1),
            markers(background, HOTSPOT_COLOUR, HOTSPOT_COLOUR, 3, alpha=0.5),
            polygons(peer_only, CELL_COLOUR, CELL_OUTLINE, 0.3, 2),
            polygons(peer_hot, CELL_COLOUR, CELL_OUTLINE, 0.15, 2),
            polygons(target_records, REFERENCE_COLOUR, REFERENCE_OUTLINE, 0.35, 3),
            markers(peer_only, CELL_COLOUR, CELL_OUTLINE, 10),
            markers(peer_hot, CELL_COLOUR, CELL_OUTLINE, 10, alpha=0.55),
            markers(target_records, REFERENCE_COLOUR, REFERENCE_OUTLINE, 11),
            # peer numbers sit on their markers, so they read against the marker rather than the basemap
            pdk.Layer(
                "TextLayer",
                [{**r, "text": str(i)} for r, i in zip(peer_records, peers["#"], strict=True)],
                get_position="centre",
                get_text="text",
                get_size=12,
                get_color=[255, 255, 255, 255],
                get_text_anchor="'middle'",
                get_alignment_baseline="'center'",
                font_weight=700,
            ),
        ]
        focus = target_records + peer_records

    view_state = compute_view([r["centre"] for r in focus], view_proportion=1)
    view_state.zoom = min(view_state.zoom, 15)
    st.pydeck_chart(
        pdk.Deck(
            map_style="light",  # the app is light-only (.streamlit/config.toml)
            layers=layers,
            initial_view_state=view_state,
            tooltip={"html": "<b>{label}</b><br/>{description}<br/>{detail}"},
        ),
        height=520,
    )
    if target is not None and peers is not None:
        # streamlit's named colours, the nearest to the map's
        st.caption(":red[■] target · :blue[■] peer (lighter where it is also a listed hotspot) · :orange[■] hotspot")
    if notes:
        st.caption(
            "Greyed out (no data on police.uk): "
            + " · ".join(f"{force}, {note}" for force, note in notes.items())
            + ". Their crimes are under-counted, so their cells rarely appear as hotspots or peers."
        )


def radar(values: np.ndarray, reference: np.ndarray, labels: list[str], is_reference: bool) -> go.Figure:
    """One panel, after hex_features.radar_panels(radial="percentile"): fixed spoke order (the shape *is* the
    identity, so a reordered axis would be a different chart), the reference tinted behind, 0 dotted."""

    def closed(a):
        return [*a, a[0]]

    theta = closed(labels)
    colour = REFERENCE_COLOUR if is_reference else CELL_COLOUR
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=[0] * len(theta),
            theta=theta,
            mode="lines",
            line={"color": "#888888", "width": 1, "dash": "dot"},
            hoverinfo="skip",
        )
    )
    if not is_reference:
        fig.add_trace(
            go.Scatterpolar(
                r=closed(reference),
                theta=theta,
                fill="toself",
                fillcolor=_css_rgba(REFERENCE_TINT, 0.3),
                line={"color": REFERENCE_TINT, "width": 1.2},
                name="target",
                hovertemplate="%{theta}: %{r:.0f}",
            )
        )
    fig.add_trace(
        go.Scatterpolar(
            r=closed(values),
            theta=theta,
            fill="toself",
            fillcolor=_css_rgba(colour, 0.18),
            line={"color": colour, "width": 2},
            name="this cell",
            hovertemplate="%{theta}: %{r:.0f}",
        )
    )
    fig.update_layout(
        showlegend=False,
        height=480,
        margin={"l": 80, "r": 80, "t": 40, "b": 40},
        polar={
            # no tick labels: they sit on a spoke and collide with its label; the caption gives the scale
            "radialaxis": {"range": [-100, 100], "tickvals": [-100, 0, 100], "showticklabels": False},
            "angularaxis": {"tickfont": {"size": 12}},
        },
    )
    return fig


@st.cache_data(max_entries=512, show_spinner=False)
def cell_image(spatial_id: int, colour: str) -> bytes:
    return cell_map_image(cell_outlines([spatial_id])[0], colour)


def cell_panels(bundle: Characterisation, rows: list[int], panels: pd.DataFrame, cols: np.ndarray) -> None:
    """One panel per cell, one per row, the target (rows[0]) first: a close-up map of the hex beside its radar, as in
    the PDFs of ho_top_kc.py. `panels` is one row per cell, in the same order, with `heading` and `subheading`."""
    labels = [SHORT_LABELS.get(bundle.columns[c], bundle.columns[c]) for c in cols]
    reference = bundle.percentile[rows[0], cols]
    for row, spatial_id, heading, subheading in zip(
        rows, panels.index, panels["heading"], panels["subheading"], strict=True
    ):
        is_reference = row == rows[0]
        with st.container(border=True):
            st.markdown(f"**{heading}**")
            st.caption(subheading)
            map_col, radar_col = st.columns([1.2, 1], vertical_alignment="center")
            map_col.image(
                cell_image(int(spatial_id), REFERENCE_COLOUR if is_reference else CELL_COLOUR), width="stretch"
            )
            radar_col.plotly_chart(
                radar(bundle.percentile[row, cols], reference, labels, is_reference),
                width="stretch",
                config={"displayModeBar": False},
                key=f"radar-{spatial_id}",
            )


def main() -> None:
    st.set_page_config(layout="wide", page_title="Peer hex explorer", page_icon=":material/hexagon:")
    st.logo(str(Path(__file__).parent / "assets" / "safer-streets-small.png"), size="large")

    # Fast UI first: the title and sidebar render while the (first-run-only) feature build is in progress. The
    # pieces that need the built matrix get their slots reserved here, in page order, and are filled afterwards.
    st.title("Peer hex explorer")
    st.subheader("Which places look like this crime hotspot?")
    info_slot = st.container()

    months = all_months()
    init(months)

    # --- sidebar ---
    st.sidebar.header(":material/local_fire_department: Hotspots")
    category = cast(CrimeType, st.sidebar.selectbox("Crime type", CATEGORIES, key="category"))
    lookback = st.sidebar.slider(
        "Lookback (months)",
        1,
        len(months),
        key="lookback",
        help=f"Months of crime to count, ending at the latest available ({months[-1]})",
    )
    n_hotspots = st.sidebar.slider("Number of hotspots", 5, 100, step=5, key="n_hotspots")

    st.sidebar.header(":material/hub: Peers")
    n_peers = st.sidebar.slider("Number of peers", 1, 20, key="n_peers")
    scope = st.sidebar.segmented_control(
        "Scope",
        [NATIONAL, WITHIN_FORCE],
        key="scope",
        required=True,
        help="Where peers may be drawn from: anywhere in England & Wales, or only the selected hotspot's force",
    )
    features_slot = st.sidebar.container()

    bundle = characterisation()  # spinner comes from its cache_resource
    with info_slot.expander("More info", icon=":material/info:"):
        more_info(bundle)

    groups = feature_groups(bundle.columns)
    # pills rather than a multiselect: every feature stays visible, on or off, so switching one back on is one click
    selected = features_slot.pills(
        "Features",
        list(groups),
        selection_mode="multi",
        default=[g for g in groups if g not in DEFAULT_OFF],
        key="features",
        help="Distances are measured over the highlighted features. Land cover is the two log-ratio coordinates of "
        "urban / suburban / greenspace share, taken together.",
        width="stretch",
    )

    # --- hotspots ---
    window = months[-lookback:]
    period = f"{window[0]} to {window[-1]}" if lookback > 1 else window[0]
    with st.spinner("Counting crimes..."):
        counts = crime_counts(category, window)

    st.markdown(f"### {category} in England & Wales, {period}")
    if counts.empty:
        st.info(f"No {category} was recorded in {period}.")
        return

    names = force_names()
    hotspots = counts.head(n_hotspots).copy()
    hotspots["rank"] = hotspots["national_rank"]
    hotspots["force"] = hotspots["pfa24cd"].map(names)
    hotspots["in_matrix"] = bundle.rows_of(hotspots.index) >= 0
    st.markdown(
        f"{counts['n'].sum():,} crimes in {len(counts):,} cells. "
        f"The top {len(hotspots)} account for {hotspots['n'].sum() / counts['n'].sum():.1%}. "
        "**Select a hotspot in the table to find its peers.**"
    )

    # The selected hotspot lives in the table's session state. The key carries everything that changes the table's
    # rows, so a new crime type or window starts with nothing selected; features, k and scope aren't in it, so
    # changing those keeps the target and just recomputes its peers.
    table_key = f"hotspots|{category}|{lookback}|{n_hotspots}"
    target = selected_hotspot(hotspots, table_key)

    # --- peers ---
    cols = np.array(sorted(i for g in selected for i in groups[g]), dtype=np.intp)
    peers = None
    message = None
    if target is not None:
        target_row = int(bundle.rows_of([target])[0])
        if target_row < 0:
            message = "That cell has no features, so it can't be compared. Pick another."
        elif not len(cols):
            message = "Select at least one feature."
        else:
            # peers come only from cells with at least one crime of this type in the window (see More info)
            mask = np.isin(bundle.spatial_id, counts.index.to_numpy())
            if scope == WITHIN_FORCE:
                mask &= bundle.pfa24cd == bundle.pfa24cd[target_row]
            peer_rows, distances = nearest(bundle.scaled, target_row, cols, n_peers, mask)
            if len(peer_rows):
                peers = peer_frame(bundle, counts, names, target_row, peer_rows, distances, cols)
            else:
                message = "No other cell in the force recorded this crime in the window."

    on_screen = hotspots.index if peers is None else hotspots.index.union(peers.index)
    described = descriptions(tuple(sorted(on_screen.tolist())))
    hotspots["description"] = described.reindex(hotspots.index).fillna("").to_numpy()
    if peers is not None:
        peers["description"] = described.reindex(peers.index).fillna("").to_numpy()

    gaps = coverage_gaps(coverage(category, window), window)
    scope_force = None
    if scope == WITHIN_FORCE and peers is not None:
        scope_force = bundle.pfa24cd[bundle.rows_of([target])[0]]
    cell_map(hotspots, target if peers is not None else None, peers, gaps, window, category, scope_force)
    hotspot_table(hotspots, table_key)

    if message is not None:
        st.warning(message)
    if peers is None:
        return

    target_info = hotspots.loc[target]
    where = f"within {target_info['force']}" if scope == WITHIN_FORCE else "in England & Wales"
    st.markdown(f"### {target_info['description']} and its {len(peers)} nearest cells {where}")
    st.caption(
        f"Hotspot {target_info['rank']} · {target_info['n']:,} crimes · spatial_id {target} · "
        f"distances over {len(cols)} of {len(bundle.columns)} feature columns · "
        f"candidates are the {len(counts):,} cells with at least one {category.lower()} in the window"
    )
    with st.expander("Peer table", icon=":material/table:"):
        peer_table(peers)

    st.markdown("#### Cells and profiles")
    st.caption(
        "Radars show features as percentiles among hotspot cells (centre = lowest, dotted = median, rim = highest). "
        "The target is the tinted shape behind each peer's radar."
    )
    panels = pd.DataFrame(
        {
            "heading": [
                f"Target: {target_info['description']}",
                *(f"#{n} {loc}" for n, loc in zip(peers["#"], peers["description"], strict=True)),
            ],
            "subheading": [
                f"hotspot {target_info['rank']} · {target_info['n']:,} crimes · {target_info['force']}",
                *(
                    f"distance {d:.2f} · rank {r:,} · {n:,} crimes · {f}"
                    for d, r, n, f in zip(
                        peers["distance"], peers["rank"], peers["crimes"], peers["force"], strict=True
                    )
                ),
            ],
        },
        index=[target, *peers.index],
    )
    cell_panels(bundle, [int(bundle.rows_of([target])[0]), *bundle.rows_of(peers.index).tolist()], panels, cols)
