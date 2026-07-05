"""
build_features.py — Builds the ML-ready feature DataFrame from raw database data.

This does NOT persist anything to the database — it's meant to be called
by the training scrip to get a fresh, in-memory pandas DataFrame
each time, built directly from the current state of the raw tables.

Usage (as a library):
    from data_pipeline.build_features import build_feature_dataframe
    df = build_feature_dataframe()

Run directly for a quick preview:
    python3 -m data_pipeline.build_features
"""

import pandas as pd
from sqlalchemy import text
from data_pipeline.db import get_session


def build_feature_dataframe() -> pd.DataFrame:
    """
    Queries the database and returns a pandas DataFrame with one row per
    earnings event that has complete data (both pct_move and implied_move
    are available), ready to feed into an ML model.

    Columns:
        symbol             - ticker symbol (kept for reference, not a model input)
        report_date        - date of the earnings event (kept for reference)
        sector             - company sector (categorical feature)
        market_cap         - company market cap at time of last update (numeric feature)
        surprise_pct       - EPS surprise % (numeric feature)
        implied_move       - historical volatility proxy (numeric feature)
        pct_move           - actual price move % (kept for reference/debugging)
        exceeded_implied_move - target: True if |pct_move| > implied_move
    """
    session = get_session()

    result = session.execute(
        text("""
            SELECT
                t.symbol,
                ee.report_date,
                t.sector,
                t.market_cap,
                ee.surprise_pct,
                pr.implied_move,
                pr.pct_move
            FROM price_reactions pr
            JOIN earnings_events ee ON pr.earnings_event_id = ee.id
            JOIN tickers t ON ee.ticker_id = t.id
            WHERE pr.implied_move IS NOT NULL
              AND pr.pct_move IS NOT NULL
              AND ee.surprise_pct IS NOT NULL
            ORDER BY t.symbol, ee.report_date
        """)
    )

    rows = result.fetchall()
    session.close()

    df = pd.DataFrame(
        rows,
        columns=["symbol", "report_date", "sector", "market_cap",
                 "surprise_pct", "implied_move", "pct_move"],
    )

    # Express market cap in billions for readability (e.g. 100000000000$ -> 100b$)
    df["market_cap"] = df["market_cap"] / 1000000000
    
    # The target: did the actual move exceed what history would have "implied"?
    df["exceeded_implied_move"] = df["pct_move"].abs() > df["implied_move"]

    return df


if __name__ == "__main__":
    df = build_feature_dataframe()
    print(f"Built feature DataFrame with {len(df)} rows.\n")
    print(df.head(10))
    print(f"\nTarget balance:\n{df['exceeded_implied_move'].value_counts(normalize=True)}")