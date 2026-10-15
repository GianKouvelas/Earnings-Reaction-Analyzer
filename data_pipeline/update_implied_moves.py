"""
update_implied_moves.py — Computes historical-volatility-based features for each
earnings event and stores them in price_reactions: implied_move, surprise_streak,
implied_move_trend.

Logic: for each ticker, ordered by report_date, every feature for event N is
computed using only *previous* events — never event N itself or anything after it
(avoiding lookahead bias).

Run directly:
    python3 -m data_pipeline.update_implied_moves
"""

from sqlalchemy import text
from data_pipeline.db import get_session
from data_pipeline.calculations import (
    calculate_implied_move_proxy,
    calculate_surprise_streak,
    calculate_implied_move_trend,
)

LOOKBACK = 4  # how many previous earnings events to average over for implied_move
TREND_WINDOW = 2  # recent vs older window size for the trend feature


def get_price_reactions_by_ticker(session):
    result = session.execute(
        text("""
            SELECT pr.id, t.symbol, ee.report_date, pr.pct_move, ee.surprise_pct
            FROM price_reactions pr
            JOIN earnings_events ee ON pr.earnings_event_id = ee.id
            JOIN tickers t ON ee.ticker_id = t.id
            ORDER BY t.symbol, ee.report_date ASC
        """)
    )
    return result.fetchall()


def update_features(session, price_reaction_id: int, implied_move, surprise_streak, implied_move_trend):
    session.execute(
        text("""
            UPDATE price_reactions
            SET implied_move = :implied_move,
                surprise_streak = :surprise_streak,
                implied_move_trend = :implied_move_trend
            WHERE id = :id
        """),
        {
            "id": price_reaction_id,
            "implied_move": implied_move,
            "surprise_streak": surprise_streak,
            "implied_move_trend": implied_move_trend,
        },
    )


def main():
    session = get_session()
    rows = get_price_reactions_by_ticker(session)
    print(f"Processing {len(rows)} price reactions across all tickers.\n")

    updated = 0
    current_symbol = None
    move_history = []
    surprise_history = []

    for pr_id, symbol, report_date, pct_move, surprise_pct in rows:
        if symbol != current_symbol:
            current_symbol = symbol
            move_history = []
            surprise_history = []

        previous_moves = move_history[-LOOKBACK:]
        implied_move = calculate_implied_move_proxy(previous_moves)
        surprise_streak = calculate_surprise_streak(surprise_history)
        implied_move_trend = calculate_implied_move_trend(move_history, recent_window=TREND_WINDOW)

        update_features(session, pr_id, implied_move, surprise_streak, implied_move_trend)
        updated += 1

        move_history.append(pct_move)
        if surprise_pct is not None:
            surprise_history.append(surprise_pct)

    session.commit()
    session.close()

    print(f"Updated features for {updated} price reactions.")


if __name__ == "__main__":
    main()