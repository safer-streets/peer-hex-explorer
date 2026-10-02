from typing import get_args

import pandas as pd

from peer_hex_explorer.data import app_force_name, coverage_gaps
from peer_hex_explorer.utils import Force


def test_app_force_name():
    assert app_force_name("Metropolitan Police Service") == "Metropolitan"
    assert app_force_name("Avon and Somerset Constabulary") == "Avon and Somerset"
    assert app_force_name("Devon & Cornwall Police") == "Devon and Cornwall"
    assert app_force_name("Dyfed-Powys Police") == "Dyfed Powys"
    assert app_force_name("City of London Police") == "City of London"


def test_coverage_gaps():
    months = ("2026-01", "2026-02")
    coverage = pd.Series({(force, month): 10 for force in get_args(Force) for month in months}, name="n_crimes").drop(
        [("Gloucestershire", "2026-01"), ("Gloucestershire", "2026-02")]
    )
    coverage[("Greater Manchester", "2026-01")] = 0
    coverage[("Greater Manchester", "2026-02")] = 0
    coverage[("North Yorkshire", "2026-02")] = 0
    # a missing row counts as no crimes, the same as a zero
    assert coverage_gaps(coverage, months) == {
        "Gloucestershire": ["2026-01", "2026-02"],
        "Greater Manchester": ["2026-01", "2026-02"],
        "North Yorkshire": ["2026-02"],
    }
