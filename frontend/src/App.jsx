import { useEffect, useState } from "react";
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

async function fetchJson(path, signal) {
  const response = await fetch(`${API_URL}${path}`, { signal });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed (${response.status})`);
  }
  return response.json();
}

export default function App() {
  const [tickers, setTickers] = useState([]);
  const [symbol, setSymbol] = useState("");
  const [surprise, setSurprise] = useState(5);
  const [history, setHistory] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [predictionError, setPredictionError] = useState(null);

  // Load the ticker list once
  useEffect(() => {
    const controller = new AbortController();
    fetchJson("/tickers", controller.signal)
      .then((data) => {
        setTickers(data);
        if (data.length > 0) setSymbol(data[0].symbol);
      })
      .catch((err) => {
        if (err.name !== "AbortError") setError(err.message);
      });
    return () => controller.abort();
  }, []);

  // Load history whenever the ticker changes
  useEffect(() => {
    if (!symbol) return;
    const controller = new AbortController();
    fetchJson(`/history/${symbol}?limit=16`, controller.signal)
      .then(setHistory)
      .catch((err) => {
        if (err.name !== "AbortError") setError(err.message);
      });
    return () => controller.abort();
  }, [symbol]);

  // Refresh the prediction when ticker or slider changes (debounced)
  useEffect(() => {
    if (!symbol) return;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      fetchJson(`/predict/${symbol}?surprise_pct=${surprise}`, controller.signal)
        .then((data) => {
          setPrediction(data);
          setPredictionError(null);
        })
        .catch((err) => {
          if (err.name === "AbortError") return;
          setPrediction(null);
          setPredictionError(err.message);
        });
    }, 250);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [symbol, surprise]);

  // Ignore data that belongs to a previously selected ticker
  const currentHistory = history?.symbol === symbol ? history : null;
  const currentPrediction = prediction?.symbol === symbol ? prediction : null;

  const chartData = currentHistory
    ? currentHistory.events
        .filter((e) => e.pct_move !== null)
        .slice()
        .reverse() // API returns newest first, chart needs oldest first
        .map((e) => ({
          date: e.report_date,
          move: Math.abs(e.pct_move),
          implied: e.implied_move,
        }))
    : [];

  const selected = tickers.find((t) => t.symbol === symbol);
  const probability = currentPrediction
    ? currentPrediction.probability_exceeds_implied_move * 100
    : null;

  return (
    <main className="app">
      <header>
        <h1>Earnings Reaction Analyzer</h1>
        <p className="subtitle">
          Will the post-earnings move exceed what history says to expect?
        </p>
      </header>

      {error && <div className="error">Could not reach the API: {error}</div>}

      <section className="card controls">
        <label>
          Company
          <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
            {tickers.map((t) => (
              <option key={t.symbol} value={t.symbol}>
                {t.symbol} — {t.company_name}
              </option>
            ))}
          </select>
        </label>

        <label>
          Assumed EPS surprise: <strong>{surprise}%</strong>
          <input
            type="range"
            min="-30"
            max="50"
            step="1"
            value={surprise}
            onChange={(e) => setSurprise(Number(e.target.value))}
          />
        </label>
        {selected?.sector && <p className="muted">Sector: {selected.sector}</p>}
      </section>

      <section className="card">
        <h2>Model estimate</h2>
        {predictionError && <p className="muted">{predictionError}</p>}
        {currentPrediction && (
          <>
            <p className="big">{probability.toFixed(1)}%</p>
            <div className="bar">
              <div className="bar-fill" style={{ width: `${probability}%` }} />
            </div>
            <p className="muted">
              Probability that the move exceeds the expected{" "}
              {currentPrediction.implied_move}% (average of the last 4 earnings moves).
            </p>
          </>
        )}
        {!currentPrediction && !predictionError && <p className="muted">Loading…</p>}
      </section>

      <section className="card">
        <h2>History: actual move vs. expected move</h2>
        {chartData.length === 0 ? (
          <p className="muted">Loading…</p>
        ) : (
          <ResponsiveContainer width="100%" height={320}>
            <ComposedChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis unit="%" />
              <Tooltip formatter={(v) => `${Number(v).toFixed(2)}%`} />
              <Legend />
              <Bar dataKey="move" name="Actual move (absolute)" fill="#6366f1" />
              <Line
                dataKey="implied"
                name="Expected move"
                stroke="#f59e0b"
                strokeWidth={2}
                dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </section>

      <footer className="muted">
        Educational project, not investment advice. The model has only a modest edge
        over a naive baseline.
      </footer>
    </main>
  );
}