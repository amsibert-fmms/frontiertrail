"""Tests for the balancing sweep report helper script."""

import os
import sys

# Ensure the project root is on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.balancing_sweep import build_rows


def test_build_rows_returns_requested_number_of_rows() -> None:
    """ELI5: if we ask for N runs, we should get N spreadsheet rows."""
    rows = build_rows(start_seed=1, runs=3, party_size=10)
    assert len(rows) == 3


def test_build_rows_running_reach_rate_stays_in_valid_range() -> None:
    """ELI5: reach-rate is a percentage, so it must stay between 0 and 1."""
    rows = build_rows(start_seed=10, runs=4, party_size=10)
    for row in rows:
        assert 0.0 <= row.reach_rate_to_date <= 1.0
