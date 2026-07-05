"""
fetch_earnings.py — Pulls ticker info and historical earnings data via yahoo finance
and stores it in PostgreSQL (tickers + earnings_events tables).

Run directly:
    python3 data_pipeline/fetch_earnings.py
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from sqlalchemy import text
from data_pipeline.db import get_session
from data_pipeline.calculations import calculate_surprise_pct

# Our initial universe of tickers to track
TICKERS = [
    "META", "AMZN", "SOFI", "MELI", "NOW", "NVDA", "AMD",
    "PLTR", "MA", "MSFT", "NFLX", "ORCL", "AVGO", "ASML",
    "TSM", "MU", "TSLA",
]


def upsert_ticker(session, symbol: str) -> int | None:
    """
    Fetches company info for a ticker and inserts/updates it in the `tickers` table.
    Returns the ticker's DB id, or None if the fetch failed.
    """
    try:
        info = yf.Ticker(symbol).info

        company_name = info.get("longName") or info.get("shortName") or symbol
        sector = info.get("sector")
        market_cap = info.get("marketCap")

        # Upsert: insert if new, update if it already exists (based on unique symbol)
        result = session.execute(
            text("""
                INSERT INTO tickers (symbol, company_name, sector, market_cap, updated_at)
                VALUES (:symbol, :company_name, :sector, :market_cap, NOW())
                ON CONFLICT (symbol)
                DO UPDATE SET
                    company_name = EXCLUDED.company_name,
                    sector = EXCLUDED.sector,
                    market_cap = EXCLUDED.market_cap,
                    updated_at = NOW()
                RETURNING id
            """),
            {
                "symbol": symbol,
                "company_name": company_name,
                "sector": sector,
                "market_cap": market_cap,
            },
        )
        session.commit()
        ticker_id = result.scalar()
        print(f"  ✅ {symbol}: {company_name} (sector={sector}) -> id={ticker_id}")
        return ticker_id

    except Exception as e:
        session.rollback()
        print(f"  ⚠️  Skipping {symbol} (ticker info fetch failed): {e}")
        return None


def upsert_earnings_events(session, symbol: str, ticker_id: int) -> int:
    """
    Fetches historical earnings dates + EPS estimate/actual for a ticker
    and inserts them into the `earnings_events` table.
    Returns the number of events successfully stored.
    """
    stored_count = 0
    try:
        earnings_history = yf.Ticker(symbol).earnings_dates

        if earnings_history is None or earnings_history.empty:
            print(f"  ⚠️  No earnings history found for {symbol}")
            return 0

        for report_date, row in earnings_history.iterrows():
            try:
                raw_estimate = row.get("EPS Estimate")
                raw_actual = row.get("Reported EPS")

                # yfinance/pandas returns numpy.float64, which psycopg2 can't adapt
                # directly to SQL. Convert to native Python float (or None if NaN/missing).
                eps_estimate = float(raw_estimate) if raw_estimate is not None and not pd.isna(raw_estimate) else None
                eps_actual = float(raw_actual) if raw_actual is not None and not pd.isna(raw_actual) else None

                # Skip rows where both values are missing (e.g. future/unreported earnings)
                if eps_estimate is None and eps_actual is None:
                    continue

                surprise_pct = calculate_surprise_pct(eps_estimate, eps_actual)

                session.execute(
                    text("""
                        INSERT INTO earnings_events
                            (ticker_id, report_date, eps_estimate, eps_actual, surprise_pct)
                        VALUES
                            (:ticker_id, :report_date, :eps_estimate, :eps_actual, :surprise_pct)
                        ON CONFLICT (ticker_id, report_date)
                        DO UPDATE SET
                            eps_estimate = EXCLUDED.eps_estimate,
                            eps_actual = EXCLUDED.eps_actual,
                            surprise_pct = EXCLUDED.surprise_pct
                    """),
                    {
                        "ticker_id": ticker_id,
                        "report_date": report_date.date(),
                        "eps_estimate": eps_estimate,
                        "eps_actual": eps_actual,
                        "surprise_pct": surprise_pct,
                    },
                )
                stored_count += 1

            except Exception as row_err:
                # One bad row shouldn't kill the whole ticker's data.
                # IMPORTANT: once a statement fails, Postgres blocks the rest of the
                # transaction until we rollback — otherwise every subsequent insert
                # in this loop fails too, even valid ones.
                session.rollback()
                print(f"    ⚠️  Skipping one earnings row for {symbol} ({report_date}): {row_err}")
                continue

        session.commit()
        print(f"  ✅ {symbol}: stored {stored_count} earnings events")
        return stored_count

    except Exception as e:
        session.rollback()
        print(f"  ⚠️  Could not fetch earnings history for {symbol}: {e}")
        return 0


def main():
    session = get_session()
    print(f"Starting fetch for {len(TICKERS)} tickers — {datetime.now().isoformat()}\n")

    total_events = 0
    for symbol in TICKERS:
        print(f"Processing {symbol}...")
        ticker_id = upsert_ticker(session, symbol)

        if ticker_id is None:
            continue

        total_events += upsert_earnings_events(session, symbol, ticker_id)

    session.close()
    print(f"\nDone. Stored/updated earnings events: {total_events}")


if __name__ == "__main__":
    main()