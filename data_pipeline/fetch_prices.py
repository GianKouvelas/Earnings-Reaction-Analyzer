"""
fetch_prices.py — Fetches closing prices before/after each earnings event
and stores the price reaction in PostgreSQL (price_reactions table).

Logic: for each earnings_event with a known eps_actual (i.e. already reported),
we take the closing price on the trading day before report_date, and the
closing price on the trading day after report_date, and compute % move.

Run directly:
    python3 -m data_pipeline.fetch_prices
"""

import yfinance as yf
import pandas as pd
from datetime import timedelta
from sqlalchemy import text
from data_pipeline.db import get_session


def get_reported_earnings_events(session):
    """
    Returns all earnings events that have already happened (eps_actual is not null)
    and don't yet have a price_reaction stored.
    """
    result = session.execute(
        text("""
            SELECT ee.id, t.symbol, ee.report_date
            FROM earnings_events ee
            JOIN tickers t ON ee.ticker_id = t.id
            LEFT JOIN price_reactions pr ON pr.earnings_event_id = ee.id
            WHERE ee.eps_actual IS NOT NULL
              AND pr.id IS NULL
            ORDER BY t.symbol, ee.report_date
        """)
    )
    return result.fetchall()


def get_price_window(symbol: str, report_date):
    """
    Fetches a small window of historical daily closes around report_date
    and returns (price_before, price_after) as native floats, or (None, None)
    if not enough data is available.
    """
    start = report_date - timedelta(days=7)
    end = report_date + timedelta(days=7)

    hist = yf.Ticker(symbol).history(start=start, end=end, interval="1d")

    if hist is None or hist.empty:
        return None, None

    hist.index = hist.index.date
    trading_days = sorted(hist.index)

    before_days = [d for d in trading_days if d <= report_date]
    after_days = [d for d in trading_days if d > report_date]

    if not before_days or not after_days:
        return None, None

    price_before = hist.loc[before_days[-1]]["Close"]
    price_after = hist.loc[after_days[0]]["Close"]

    if isinstance(price_before, pd.Series):
        price_before = price_before.iloc[0]
    if isinstance(price_after, pd.Series):
        price_after = price_after.iloc[0]

    return float(price_before), float(price_after)


def store_price_reaction(session, earnings_event_id: int, price_before: float, price_after: float):
    pct_move = ((price_after - price_before) / price_before) * 100

    session.execute(
        text("""
            INSERT INTO price_reactions
                (earnings_event_id, price_before, price_after, pct_move)
            VALUES
                (:earnings_event_id, :price_before, :price_after, :pct_move)
        """),
        {
            "earnings_event_id": earnings_event_id,
            "price_before": price_before,
            "price_after": price_after,
            "pct_move": pct_move,
        },
    )
    session.commit()


def main():
    session = get_session()
    events = get_reported_earnings_events(session)
    print(f"Found {len(events)} earnings events needing a price reaction.\n")

    stored = 0
    for event_id, symbol, report_date in events:
        try:
            price_before, price_after = get_price_window(symbol, report_date)

            if price_before is None or price_after is None:
                print(f"  ⚠️  {symbol} {report_date}: not enough price data, skipping")
                continue

            store_price_reaction(session, event_id, price_before, price_after)
            pct = ((price_after - price_before) / price_before) * 100
            print(f"  ✅ {symbol} {report_date}: {price_before:.2f} -> {price_after:.2f} ({pct:+.2f}%)")
            stored += 1

        except Exception as e:
            session.rollback()
            print(f"  ⚠️  Skipping {symbol} {report_date} (error: {e})")
            continue

    session.close()
    print(f"\nDone. Stored {stored} price reactions.")


if __name__ == "__main__":
    main()