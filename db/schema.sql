-- Earnings Reaction Analyzer — Database Schema

CREATE TABLE tickers (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) UNIQUE NOT NULL,
    company_name VARCHAR(150) NOT NULL,
    sector VARCHAR(100),
    market_cap NUMERIC,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE earnings_events (
    id SERIAL PRIMARY KEY,
    ticker_id INTEGER NOT NULL REFERENCES tickers(id) ON DELETE CASCADE,
    report_date DATE NOT NULL,
    eps_estimate NUMERIC,
    eps_actual NUMERIC,
    surprise_pct NUMERIC,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (ticker_id, report_date)
);

CREATE TABLE price_reactions (
    id SERIAL PRIMARY KEY,
    earnings_event_id INTEGER NOT NULL REFERENCES earnings_events(id) ON DELETE CASCADE,
    price_before NUMERIC,
    price_after NUMERIC,
    pct_move NUMERIC,
    implied_move NUMERIC,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_earnings_events_ticker_id ON earnings_events(ticker_id);
CREATE INDEX idx_price_reactions_earnings_event_id ON price_reactions(earnings_event_id);