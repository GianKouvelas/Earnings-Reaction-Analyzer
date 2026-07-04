"""
test_calculations.py — Unit tests for the pure calculation functions.

Run with:
    pytest tests/test_calculations.py
"""

import pytest
from data_pipeline.calculations import calculate_surprise_pct, calculate_pct_move


# ---- Tests for calculate_surprise_pct ----

def test_surprise_pct_beat_estimate():
    result = calculate_surprise_pct(eps_estimate=1.00, eps_actual=1.10)
    assert result == pytest.approx(10.0)


def test_surprise_pct_missed_estimate():
    result = calculate_surprise_pct(eps_estimate=1.00, eps_actual=0.90)
    assert result == pytest.approx(-10.0)


def test_surprise_pct_exact_match():
    result = calculate_surprise_pct(eps_estimate=1.00, eps_actual=1.00)
    assert result == pytest.approx(0.0)


def test_surprise_pct_negative_estimate():
    # Edge case: negative estimate (company expected to lose money),
    # but still beat it. abs() in the formula should keep the sign meaningful.
    result = calculate_surprise_pct(eps_estimate=-0.10, eps_actual=0.05)
    assert result == pytest.approx(150.0)


def test_surprise_pct_missing_estimate_returns_none():
    result = calculate_surprise_pct(eps_estimate=None, eps_actual=1.10)
    assert result is None


def test_surprise_pct_missing_actual_returns_none():
    result = calculate_surprise_pct(eps_estimate=1.00, eps_actual=None)
    assert result is None


def test_surprise_pct_zero_estimate_returns_none():
    result = calculate_surprise_pct(eps_estimate=0.0, eps_actual=0.05)
    assert result is None


# ---- Tests for calculate_pct_move ----

def test_pct_move_price_increase():
    result = calculate_pct_move(price_before=100.0, price_after=110.0)
    assert result == pytest.approx(10.0)


def test_pct_move_price_decrease():
    result = calculate_pct_move(price_before=100.0, price_after=90.0)
    assert result == pytest.approx(-10.0)


def test_pct_move_no_change():
    result = calculate_pct_move(price_before=100.0, price_after=100.0)
    assert result == pytest.approx(0.0)


def test_pct_move_zero_price_before_raises_error():
    with pytest.raises(ValueError):
        calculate_pct_move(price_before=0.0, price_after=10.0)