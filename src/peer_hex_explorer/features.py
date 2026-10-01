"""The feature pipeline, copied from safer-streets-eda.

`clean_features`, `robust_scale` and the constants come from `hex_features.py`;
`ilr_transform` and `compositional_features` from `clusterability_audit.py`. Comments are carried
over with the code. This is a copy, not a dependency, so tests/test_features.py pins it to the
original's output on a fixed sample: if either copy changes, that test is what notices.

`scale_features` is the composition `ho_top_kc.load_scaled_features` makes (clean -> ILR ->
robust scale), except that `retail_centre_distance` is kept: the app scales it with everything
else and lets the user switch it off. Scaling is column by column, so a column being present
changes nothing about how any other column is scaled.
"""

import numpy as np
import pandas as pd

STRUCTURAL_ZERO = ["n_stops", "n_hospital"]

# Counts and lengths whose shape, not just their units, is the problem: skew 1.9-18.3 with up to
# 91% zeros. RobustScaler recentres and rescales but cannot change shape, so without log1p these
# arrive at any distance-based method as a spike with a very long tail.
LOG1P_COLUMNS = [
    # skew 1.76 raw, -0.08 after log1p, and IQR/sd lands at 1.39 -- almost exactly the 1.35 a
    # normal distribution gives, so this column needs no degeneracy handling at all once uncapped
    "school_isochrone_depth",
    "n_alcohol",
    "n_food",
    "n_hospital",
    "n_stops",
    "n_takeaways",
    "road_intersections",
    "road_overlap_length",
    "residential_population",
    "workplace_population",
]

# A column whose middle 50% is near-constant but whose tail is not has a tiny IQR, and dividing by
# it makes that column's scaled variance explode and swamp every distance. Below this ratio of IQR
# to standard deviation, use the standard deviation instead.
DEGENERATE_IQR_RATIO = 0.50

COMPOSITIONAL_PARTS = ["prop_urban", "prop_suburban", "prop_greenspace"]
ZERO_REPLACEMENT = 0.001  # matches the audits' default

# Radars need short labels -- the full names are unreadable at 12 spokes. The one set for every
# radar in both notebooks, so the same spoke never carries two names on charts meant to be read
# side by side.
SHORT_LABELS = {
    "road_overlap_length": "road length",
    "road_intersections": "junctions",
    "school_isochrone_depth": "school depth",
    "n_alcohol": "alcohol",
    "n_food": "food",
    "n_hospital": "hospital",
    "n_stops": "transit stops",
    "n_takeaways": "takeaways",
    "residential_population": "residents",
    "workplace_population": "workers",
    "retail_centre_distance": "retail",
    "ilr_urban_vs_suburban": "urban-suburban",
    "ilr_urban_suburban_vs_greenspace": "builtup-greenspace",
}


def clean_features(hex_characterisation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Float64 features with the NULLs resolved, plus a per-column summary.

    The LEFT JOINs upstream produce NULLs with two different meanings, and they need different
    fixes.

    Structural zeros. `n_stops` and `n_hospital` come from `COUNT(*) ... GROUP BY`, so the smallest
    value a matched row can carry is 1 and NULL is the only available encoding for "none here".
    Confirmed against the per-force null rate: it varies smoothly (26-65% for stops) instead of any
    force being wholly absent, so this is absence of stops, not absence of data. 53.7% of hexes
    have no NaPTAN stop and 37.1% have no POI at all in the parent cell.

    Real missingness. The population columns are the one case that is genuinely ambiguous, and this
    departs from the earlier feature-vector cell, which filled them with 0 as "a genuine count of
    zero". That table stores 127,725 explicit zero rows out of 714,618, so it does distinguish "no
    population here" from "no row" -- which makes absence look like coverage the build never
    emitted rather than a known zero. Median imputation is the conservative reading and it invents
    a value for 9.3% of rows; `.fillna(0.0)` is the other defensible call. Worth settling before
    these feed anything, because 9.3% of rows is not a rounding error.
    """
    features = hex_characterisation.astype("float64")
    structural = [c for c in STRUCTURAL_ZERO if c in features.columns]
    features[structural] = features[structural].fillna(0.0)

    imputed_share = features.isna().mean()
    features = features.fillna(features.median())

    summary = pd.DataFrame(
        {
            "zero_%": (features == 0).mean() * 100,
            "skew": features.skew(),
            "median": features.median(),
            "iqr": features.quantile(0.75) - features.quantile(0.25),
            "sd": features.std(),
            "max": features.max(),
            "median_imputed_%": imputed_share * 100,
        }
    ).round(3)
    return features, summary


def robust_scale(df, log1p_columns=LOG1P_COLUMNS, degenerate="unit"):
    """Median-centre and IQR-scale, after log1p on the skewed counts.

    Returns (scaled, diagnostics). Written out rather than using RobustScaler directly because
    sklearn silently substitutes 1.0 for an IQR of exactly zero, which four of these columns have
    (n_alcohol, n_food, n_hospital, n_takeaways are 87-91% zero). That substitution is a real
    decision about feature weighting and it should be visible, not a default nobody chose.

    `degenerate` picks what happens to a column with no usable IQR:

    - "unit" (default) divides by 1.0, leaving it in log1p units. The column keeps a small scaled
      sd (~0.33) so it contributes little, and only genuinely extreme hexes stand out.
    - "sd" divides by the standard deviation, giving every column equal variance. Cleaner on
      paper, but it hands a 91%-zero column the same weight as `prop_suburban`, and k-means will
      take that split: see the retail_centre_distance episode in clusterability_audit.py, where an
      effectively binary feature scored a perfect, perfectly meaningless prediction strength.
    """
    if degenerate not in ("unit", "sd"):
        raise ValueError("degenerate must be 'unit' or 'sd'")

    df = df.copy()
    for c in log1p_columns:
        if c in df.columns:
            df[c] = np.log1p(df[c].clip(lower=0))

    centre = df.median()
    iqr = df.quantile(0.75) - df.quantile(0.25)
    sd = df.std()

    # Near-constant middle always falls back to sd; an IQR of exactly zero follows the policy.
    near_constant = (iqr > 0) & (iqr < DEGENERATE_IQR_RATIO * sd)
    no_iqr = iqr <= 0
    scale = iqr.where(~near_constant, sd)
    scale = scale.where(~no_iqr, sd if degenerate == "sd" else 1.0)
    scale = scale.where(scale > 0, 1.0)  # a wholly constant column stays at zero, not NaN

    scaled = (df - centre) / scale
    diagnostics = pd.DataFrame(
        {
            "iqr": iqr,
            "sd": sd,
            "divisor": scale,
            "fallback": np.where(no_iqr, f"no IQR -> {degenerate}", np.where(near_constant, "near-constant -> sd", "")),
            "scaled_sd": scaled.std(),
            "max_abs_z": scaled.abs().max(),
        }
    )
    return scaled, diagnostics


def ilr_transform(parts: np.ndarray, delta: float):
    """Isometric log-ratio coordinates for a matrix of compositional parts.

    `parts` is (n, D) of non-negative shares. A residual part is *not* added here --
    the caller decides what the parts are. Returns (n, D-1) coordinates under the
    default sequential binary partition, in which coordinate i contrasts the geometric
    mean of parts 0..i against part i+1.
    """
    x = np.asarray(parts, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    x = np.clip(x, 0.0, None)

    # Multiplicative simple replacement: zeros become delta, non-zeros shrink by the
    # amount given away, so every row still sums to its original total.
    n_zero = (x == 0).sum(axis=1, keepdims=True)
    total = x.sum(axis=1, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    shrink = 1.0 - (n_zero * delta) / total
    x = np.where(x == 0, delta * np.ones_like(x), x * shrink)
    x = np.clip(x, delta * 1e-6, None)

    x = x / x.sum(axis=1, keepdims=True)  # close to the simplex
    logx = np.log(x)
    d = x.shape[1]
    coords = np.empty((x.shape[0], d - 1), dtype=float)
    for i in range(d - 1):
        coords[:, i] = np.sqrt((i + 1) / (i + 2)) * (logx[:, : i + 1].mean(axis=1) - logx[:, i + 1])
    return coords


def compositional_features(df, cols, parts, delta, add_residual=False):
    """Replace a group of compositional columns with their ILR coordinates.

    Returns (values, names, info). The parts are taken in the order given and closed to
    sum to one, giving D-1 coordinates. Columns not in `parts` pass through untouched, and
    the ILR coordinates are appended last. `add_residual` restores the pre-2026-09-07
    behaviour of appending an "other" part first, which yields one extra coordinate -- see
    the comment in the else-branch for why that coordinate was mostly an artefact.
    """
    present = [c for c in parts if c in cols]
    if len(present) < 2:
        return df[cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float), list(cols), None

    passthrough = [c for c in cols if c not in present]
    raw_parts = df[present].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    raw_parts = np.clip(raw_parts, 0.0, None)

    if add_residual:
        residual = np.clip(1.0 - raw_parts.sum(axis=1), 0.0, None)
        full = np.column_stack([raw_parts, residual])
        labels = [*present, "other"]
    else:
        # D parts constrained to a whole carry D-1 degrees of freedom, so three land-cover
        # shares give exactly two coordinates. The residual "other" part that used to be
        # appended is *exactly zero* for 68-87% of rows (87% on Robbery), which means the
        # third coordinate it bought was, for most rows, a restatement of the arbitrary
        # `delta` below rather than anything measured -- the same manufactured-atom problem
        # as `retail_centre_distance`, but self-inflicted.
        #
        # Measurement says dropping it is free: best-k prediction strength moves by at most
        # 0.005 on every dataset, and sweeping `delta` over 1e-2..1e-4 moves it by at most
        # 0.017 either way. So this is a simplification that removes a column and a free
        # parameter without changing an answer, not an improvement in its own right.
        total = raw_parts.sum(axis=1, keepdims=True)
        full = raw_parts / np.where(total > 0, total, 1.0)
        labels = list(present)

    coords = ilr_transform(full, delta)
    names = [
        f"ilr_{'_'.join(s.replace('prop_', '') for s in labels[: i + 1])}_vs_{labels[i + 1].replace('prop_', '')}"
        for i in range(len(labels) - 1)
    ]
    values = np.column_stack(
        [df[passthrough].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float), coords] if passthrough else [coords]
    )
    info = {
        "parts": present,
        "residual_part": "other" if add_residual else None,
        "coordinates": names,
        "zero_replacement": delta,
        "zero_share_by_part": {c: float((raw_parts[:, i] == 0).mean()) for i, c in enumerate(present)},
        "closure_median": float(np.median(raw_parts.sum(axis=1))),
        "residual_zero_share": (
            float(np.mean(np.clip(1.0 - raw_parts.sum(axis=1), 0.0, None) <= 0)) if add_residual else None
        ),
    }
    return values, [*passthrough, *names], info


def ilr_features(
    features: pd.DataFrame,
    parts: list[str] = COMPOSITIONAL_PARTS,
    delta: float = ZERO_REPLACEMENT,
) -> tuple[pd.DataFrame, dict]:
    """Replace the land-cover shares with their ILR coordinates.

    ILR (isometric log-ratio) coordinates replace the three parts with the two independent
    contrasts they actually carry. Reusing clusterability_audit.py's implementation rather than
    rewriting it: both audits have encoded these columns this way since 2026-08-27, so the
    notebooks and the audits stay on the same footing.
    """
    values, names, info = compositional_features(features, list(features.columns), parts, delta)
    return pd.DataFrame(values, index=features.index, columns=names), info


def scale_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw characterisation (feature columns only) -> robust-scaled matrix with ILR land cover.

    Returns (scaled, summary), where `summary` is `clean_features`' per-column table: the app
    quotes its `median_imputed_%` so the imputation is visible rather than buried.
    """
    features, summary = clean_features(raw)
    features_ilr, _ = ilr_features(features)
    scaled, _ = robust_scale(features_ilr)
    return scaled, summary


def percentiles(scaled: pd.DataFrame) -> pd.DataFrame:
    """Each feature's percentile within `scaled`, rescaled to -100..+100.

    The radial transform of `hex_features.radar_panels(radial="percentile")`, so a cell draws the
    same polygon here as in the notebooks -- provided `scaled` is the whole national matrix.
    """
    return (scaled.rank(pct=True) - 0.5) * 200
