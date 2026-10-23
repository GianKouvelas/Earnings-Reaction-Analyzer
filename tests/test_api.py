"""
test_api.py: Integration tests for the FastAPI endpoints.

These run against the real local database and the saved model artifact, so the
Postgres container must be running (docker compose up -d) and
model/artifacts/earnings_model.joblib must exist (python3 -m model.train_final).

Run with:
    pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager triggers the lifespan,
    # i.e. the model is loaded once for all tests in this module
    with TestClient(app) as c:
        yield c


# ---- /health ----


def test_health_returns_ok_and_model_info(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_n_samples"] > 0


# ---- /history ----


def test_history_known_ticker(client):
    response = client.get("/history/NVDA")
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "NVDA"
    assert len(body["events"]) > 0


def test_history_is_case_insensitive(client):
    response = client.get("/history/nvda")
    assert response.status_code == 200
    assert response.json()["symbol"] == "NVDA"


def test_history_respects_limit(client):
    response = client.get("/history/NVDA?limit=3")
    assert response.status_code == 200
    assert len(response.json()["events"]) <= 3


def test_history_unknown_ticker_returns_404(client):
    response = client.get("/history/FAKE")
    assert response.status_code == 404


def test_history_invalid_limit_returns_422(client):
    response = client.get("/history/NVDA?limit=0")
    assert response.status_code == 422


# ---- /predict ----


def test_predict_returns_valid_probability(client):
    response = client.get("/predict/NVDA?surprise_pct=5")
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "NVDA"
    assert 0.0 <= body["probability_exceeds_implied_move"] <= 1.0
    assert isinstance(body["prediction"], bool)


def test_predict_unknown_ticker_returns_404(client):
    response = client.get("/predict/FAKE?surprise_pct=5")
    assert response.status_code == 404


def test_predict_missing_surprise_pct_returns_422(client):
    response = client.get("/predict/NVDA")
    assert response.status_code == 422


def test_predict_out_of_range_surprise_pct_returns_422(client):
    response = client.get("/predict/NVDA?surprise_pct=9999")
    assert response.status_code == 422


def test_predict_prediction_matches_probability(client):
    body = client.get("/predict/NVDA?surprise_pct=5").json()
    assert body["prediction"] == (body["probability_exceeds_implied_move"] >= 0.5)
    
    
# ---- /tickers ----

def test_tickers_lists_all_symbols(client):
    response = client.get("/tickers")
    assert response.status_code == 200
    symbols = [t["symbol"] for t in response.json()]
    assert "NVDA" in symbols
    assert symbols == sorted(symbols)