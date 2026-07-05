"""
update_implied_moves.py — Computes a historical-volatility proxy for "implied move"
for each earnings event and stores it in price_reactions.implied_move.

Logic: for each ticker, ordered by report_date, the implied_move for event N
is the average absolute pct_move of the *previous* LOOKBACK events (never
including event N itself — that would be lookahead bias).

Run directly:
    python3 -m data_pipeline.update_implied_moves
"""

from sqlalchemy import text
from data_pipeline.db import get_session
from data_pipeline.calculations import calculate_implied_move_proxy

LOOKBACK = 4  # how many previous earnings events to average over


def get_price_reactions_by_ticker(session):
    """
    Returns all price_reactions joined with their ticker + report_date,
    ordered by ticker then chronologically — so we can walk through each
    ticker's history in order and compute a rolling average.
    """
    result = session.execute(
        text("""
            SELECT pr.id, t.symbol, ee.report_date, pr.pct_move
            FROM price_reactions pr
            JOIN earnings_events ee ON pr.earnings_event_id = ee.id
            JOIN tickers t ON ee.ticker_id = t.id
            ORDER BY t.symbol, ee.report_date ASC
        """)
    )
    return result.fetchall()


def update_implied_move(session, price_reaction_id: int, implied_move: float):
    session.execute(
        text("""
            UPDATE price_reactions
            SET implied_move = :implied_move
            WHERE id = :id
        """),
        {"id": price_reaction_id, "implied_move": implied_move},
    )


def main():
    session = get_session()
    rows = get_price_reactions_by_ticker(session)
    print(f"Processing {len(rows)} price reactions across all tickers.\n")

    updated = 0
    skipped_no_history = 0
    current_symbol = None
    history = []

    for pr_id, symbol, report_date, pct_move in rows:
        if symbol != current_symbol:
            current_symbol = symbol
            history = []

        previous_moves = history[-LOOKBACK:]
        implied_move = calculate_implied_move_proxy(previous_moves)

        if implied_move is None:
            skipped_no_history += 1
        else:
            update_implied_move(session, pr_id, implied_move)
            updated += 1

        history.append(pct_move)

    session.commit()
    session.close()

    print(f"Updated implied_move for {updated} price reactions.")
    print(f"Skipped {skipped_no_history} (not enough history yet, e.g. first {LOOKBACK} events per ticker).")


if __name__ == "__main__":
    main()