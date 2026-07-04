"""
calculations.py — Pure calculation functions used across the data pipeline.

"""

from typing import Optional


def calculate_surprise_pct(eps_estimate: Optional[float], eps_actual: Optional[float]) -> Optional[float]:
    """
    Calculates the earnings surprise percentage:
    how much the actual EPS beat (or missed) the estimate, in %.

    Returns None if we don't have both values, or if estimate is 0
    (division by zero would be meaningless here).
    """
    if eps_estimate is None or eps_actual is None:
        return None
    if eps_estimate == 0:
        return None

    return ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100


def calculate_pct_move(price_before: float, price_after: float) -> float:
    """
    Calculates the percentage price move between two prices.
    Positive = price went up, negative = price went down.
    """
    if price_before == 0:
        raise ValueError("price_before cannot be zero")

    return ((price_after - price_before) / price_before) * 100