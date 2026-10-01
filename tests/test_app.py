"""Headless runs of the whole app against Azure. Skipped without credentials.

AppTest can't click a row in st.dataframe, so the hotspot selection is injected into the table's session state
before each run -- the same place a real click puts it.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from test_data import _connstr

pytestmark = [pytest.mark.azure, pytest.mark.skipif(_connstr() is None, reason="no Azure credentials")]

APP = str(Path(__file__).parents[1] / "app.py")
TABLE_KEY = "hotspots|Robbery|12|20"  # the defaults set in main.init


def select(at: AppTest, row: int) -> AppTest:
    at.session_state[TABLE_KEY] = {"selection": {"rows": [row], "columns": [], "cells": []}}
    return at.run()


@pytest.fixture
def at() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception
    return at


def test_hotspots_listed_without_peers(at):
    assert len(at.dataframe) == 1
    assert len(at.dataframe[0].value) == 20
    assert not at.get("plotly_chart")


def test_selecting_a_hotspot_shows_peers(at):
    select(at, 1)
    assert not at.exception
    peers = at.dataframe[1].value
    assert peers["#"].tolist() == [1, 2, 3, 4, 5]
    assert peers["distance"].is_monotonic_increasing
    assert (peers["crimes"] > 0).all()  # peers only come from cells with crime
    # one map + radar panel per cell: target + 5 peers
    assert len(at.get("plotly_chart")) == 6
    assert len(at.image) == 6


def test_within_force_uses_the_hotspots_force(at):
    at.slider(key="n_peers").set_value(3)
    at.button_group(key="scope").set_value("Within force")
    select(at, 1)
    target_force = at.dataframe[0].value["force"].iloc[1]
    peers = at.dataframe[1].value
    assert len(peers) == 3
    assert (peers["force"] == target_force).all()


def test_no_features_warns(at):
    at.button_group(key="features").set_value([])
    select(at, 1)
    assert at.warning[0].value == "Select at least one feature."
    assert len(at.dataframe) == 1
