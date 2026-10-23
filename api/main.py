"""
main.py: FastAPI backend for the Earnings Reaction Analyzer.

Endpoints:
    GET /health                  - liveness check + model info
    GET /tickers                 - list of all the companies
    GET /history/{ticker}        - past earnings reactions from the database
    GET /predict/{ticker}        - probability that the next earnings move will
                                   exceed the implied move, for a given surprise_pct

Run locally:
    uvicorn api.main:app --reload
Interactive docs: http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager
from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text

from data_pipeline.calculations import calculate_implied_move_proxy
from data_pipeline.db import get_session
from model.prediction import load_artifact, predict

LOOKBACK = 4  # must match the lookback used when building the implied_move feature

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not on every request
    state["artifact"] = load_artifact()
    yield
    state.clear()


app = FastAPI(title="Earnings Reaction Analyzer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------- Response models ----------

class HistoryItem(BaseModel):
    report_date: date
    eps_estimate: Optional[float]
    eps_actual: Optional[float]
    surprise_pct: Optional[float]
    pct_move: Optional[float]
    implied_move: Optional[float]


class HistoryResponse(BaseModel):
    symbol: str
    company_name: str
    events: list[HistoryItem]


class PredictionResponse(BaseModel):
    symbol: str
    surprise_pct: float
    implied_move: float
    probability_exceeds_implied_move: float
    prediction: bool
    

class TickerItem(BaseModel):
    symbol: str
    company_name: str
    sector: Optional[str]


# ---------- Helpers ----------

def _to_float(value) -> Optional[float]:
    # PostgreSQL NUMERIC comes back as Decimal
    return float(value) if value is not None else None


def _get_ticker(session, symbol: str):
    row = session.execute(
        text("SELECT id, symbol, company_name, sector, market_cap FROM tickers WHERE symbol = :s"),
        {"s": symbol.upper()},
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Ticker '{symbol.upper()}' not found")
    return row


# ---------- Endpoints ----------

@app.get("/health")
def health():
    artifact = state["artifact"]
    return {
        "status": "ok",
        "model_trained_at": artifact["trained_at"],
        "model_n_samples": artifact["n_samples"],
    }


@app.get("/tickers", response_model=list[TickerItem])
def list_tickers():
    session = get_session()
    try:
        rows = session.execute(
            text("SELECT symbol, company_name, sector FROM tickers ORDER BY symbol")
        ).fetchall()
    finally:
        session.close()

    return [TickerItem(symbol=r.symbol, company_name=r.company_name, sector=r.sector) for r in rows]


@app.get("/history/{symbol}", response_model=HistoryResponse)
def history(symbol: str, limit: int = Query(12, ge=1, le=50)):
    session = get_session()
    try:
        ticker = _get_ticker(session, symbol)
        rows = session.execute(
            text("""
                SELECT ee.report_date, ee.eps_estimate, ee.eps_actual, ee.surprise_pct,
                       pr.pct_move, pr.implied_move
                FROM earnings_events ee
                LEFT JOIN price_reactions pr ON pr.earnings_event_id = ee.id
                WHERE ee.ticker_id = :tid
                ORDER BY ee.report_date DESC
                LIMIT :limit
            """),
            {"tid": ticker.id, "limit": limit},
        ).fetchall()
    finally:
        session.close()

    events = [
        HistoryItem(
            report_date=r.report_date,
            eps_estimate=_to_float(r.eps_estimate),
            eps_actual=_to_float(r.eps_actual),
            surprise_pct=_to_float(r.surprise_pct),
            pct_move=_to_float(r.pct_move),
            implied_move=_to_float(r.implied_move),
        )
        for r in rows
    ]
    return HistoryResponse(symbol=ticker.symbol, company_name=ticker.company_name, events=events)


@app.get("/predict/{symbol}", response_model=PredictionResponse)
def predict_endpoint(
    symbol: str,
    surprise_pct: float = Query(..., ge=-100, le=500, description="Assumed EPS surprise % for the upcoming report"),
):
    session = get_session()
    try:
        ticker = _get_ticker(session, symbol)

        recent_moves = [
            float(r.pct_move)
            for r in session.execute(
                text("""
                    SELECT pr.pct_move
                    FROM price_reactions pr
                    JOIN earnings_events ee ON pr.earnings_event_id = ee.id
                    WHERE ee.ticker_id = :tid AND pr.pct_move IS NOT NULL
                    ORDER BY ee.report_date DESC
                    LIMIT :n
                """),
                {"tid": ticker.id, "n": LOOKBACK},
            ).fetchall()
        ]
    finally:
        session.close()

    if len(recent_moves) < LOOKBACK:
        raise HTTPException(status_code=422, detail=f"Not enough earnings history for {ticker.symbol}")
    if ticker.market_cap is None:
        raise HTTPException(status_code=422, detail=f"No market cap on file for {ticker.symbol}")

    implied_move = calculate_implied_move_proxy(recent_moves)

    result = predict(
        state["artifact"],
        surprise_pct=surprise_pct,
        implied_move=implied_move,
        market_cap=float(ticker.market_cap) / 1e9,  # training data uses billions
        sector=ticker.sector or "",
    )

    return PredictionResponse(
        symbol=ticker.symbol,
        surprise_pct=surprise_pct,
        implied_move=round(implied_move, 2),
        probability_exceeds_implied_move=result["probability_exceeds_implied_move"],
        prediction=result["prediction"],
    )