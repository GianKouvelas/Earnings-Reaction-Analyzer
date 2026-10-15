"""
build_features.py — Builds the ML-ready feature DataFrame from raw database data.

This does NOT persist anything to the database — it's meant to be called
by the training script to get a fresh, in-memory pandas DataFrame each time,
built directly from the current state of the raw tables.

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
    earnings event that has complete data, ready to feed into an ML model.

    Columns:
        symbol                 - ticker symbol (reference only)
        report_date            - date of the earnings event (reference only)
        sector                 - company sector (categorical feature)
        market_cap             - company market cap in billions (numeric feature)
        surprise_pct           - EPS surprise % (numeric feature)
        implied_move           - historical volatility proxy (numeric feature)
        surprise_streak        - consecutive beats(+)/misses(-) streak (numeric feature)
        implied_move_trend     - recent vs older volatility trend (numeric feature)
        pct_move               - actual price move % (reference/debugging)
        exceeded_implied_move  - target: True if |pct_move| > implied_move
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
                pr.surprise_streak,
                pr.implied_move_trend,
                pr.pct_move
            FROM price_reactions pr
            JOIN earnings_events ee ON pr.earnings_event_id = ee.id
            JOIN tickers t ON ee.ticker_id = t.id
            WHERE pr.implied_move IS NOT NULL
              AND pr.implied_move_trend IS NOT NULL
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
                 "surprise_pct", "implied_move", "surprise_streak",
                 "implied_move_trend", "pct_move"],
    )

    numeric_columns = ["market_cap", "surprise_pct", "implied_move", "implied_move_trend", "pct_move"]
    for col in numeric_columns:
        df[col] = df[col].astype(float)

    df["market_cap"] = df["market_cap"] / 1e9

    df["exceeded_implied_move"] = df["pct_move"].abs() > df["implied_move"]

    return df


if __name__ == "__main__":
    df = build_feature_dataframe()
    print(f"Built feature DataFrame with {len(df)} rows.\n")

    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print(df.head(10))
    print(f"\nTarget balance:\n{df['exceeded_implied_move'].value_counts(normalize=True)}")