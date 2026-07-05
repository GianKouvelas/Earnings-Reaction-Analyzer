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

def calculate_implied_move_proxy(previous_pct_moves: list) -> Optional[float]:
    """
    Proxy for "implied move" using historical data only (no options data available
    for past earnings via yfinance).

    Takes a list of *previous* earnings' pct_move values (most recent last, order
    doesn't actually matter for the average) and returns the average absolute move.

    IMPORTANT: the caller must only pass in moves from earnings *before* the one
    being evaluated — never include the current/future event, or the model would
    be trained on information it wouldn't have had at prediction time (lookahead bias).

    Returns None if there isn't enough history yet (empty list).
    """
    if not previous_pct_moves:
        return None

    abs_moves = [abs(m) for m in previous_pct_moves]
    return sum(abs_moves) / len(abs_moves)