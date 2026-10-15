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


def calculate_surprise_streak(previous_surprises: list) -> int:
    """
    Counts consecutive beats/misses leading up to (but not including) the
    current event, using only *previous* surprise_pct values (most recent last).

    Positive streak = N consecutive beats (surprise_pct > 0) in a row.
    Negative streak = N consecutive misses (surprise_pct < 0) in a row.
    0 = no history, or the most recent event was exactly in-line (surprise_pct == 0).

    Example: previous_surprises = [-2, 5, 3, 7] -> last three were positive beats
    in a row (5, 3, 7) -> streak = 3.
    """
    if not previous_surprises:
        return 0

    last_sign = None
    streak = 0
    for value in reversed(previous_surprises):
        sign = 1 if value > 0 else (-1 if value < 0 else 0)
        if sign == 0:
            break
        if last_sign is None:
            last_sign = sign
            streak = 1
        elif sign == last_sign:
            streak += 1
        else:
            break

    return streak * (last_sign or 0)


def calculate_implied_move_trend(previous_pct_moves: list, recent_window: int = 2) -> Optional[float]:
    """
    Measures whether historical volatility is rising or falling, by comparing
    the average |pct_move| of the most recent `recent_window` events against
    the average of the events before that.

    Returns a positive number if volatility is increasing recently, negative
    if decreasing, or None if there isn't enough history (need at least
    2 * recent_window previous events).
    """
    if len(previous_pct_moves) < recent_window * 2:
        return None

    recent = previous_pct_moves[-recent_window:]
    older = previous_pct_moves[-recent_window * 2:-recent_window]

    recent_avg = sum(abs(m) for m in recent) / len(recent)
    older_avg = sum(abs(m) for m in older) / len(older)

    return recent_avg - older_avg