import { useEffect, useState } from "react";
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  Cell,
  ReferenceLine,
  Tooltip,
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

function MoveTooltip({ active, payload }) {
  if (!active || !payload || payload.length === 0) return null;
  const point = payload[0].payload;
  const sign = point.move > 0 ? "+" : "";
  const hasExpected = point.implied !== null && point.implied !== undefined;

  return (
    <div className="tooltip">
      <strong>{point.date}</strong>
      <div>
        Move: {sign}
        {point.move.toFixed(2)}%
      </div>
      {hasExpected && <div>Expected: ±{point.implied.toFixed(2)}%</div>}
      {hasExpected && (
        <div>
          {Math.abs(point.move) > point.implied
            ? "Exceeded expectation"
            : "Within expectation"}
        </div>
      )}
    </div>
  );
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
          move: e.pct_move, // signed: positive = up, negative = down
          implied: e.implied_move,
          impliedNeg: e.implied_move !== null ? -e.implied_move : null,
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
        <h2>History: actual move vs. expected range</h2>
        {chartData.length === 0 ? (
          <p className="muted">Loading…</p>
        ) : (
          <>
            <ResponsiveContainer width="100%" height={340}>
              <ComposedChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={24} />
                <YAxis unit="%" />
                <Tooltip content={<MoveTooltip />} />
                <ReferenceLine y={0} stroke="#9ca3af" />
                <Bar dataKey="move" name="Actual move">
                  {chartData.map((d) => (
                    <Cell key={d.date} fill={d.move >= 0 ? "#16a34a" : "#dc2626"} />
                  ))}
                </Bar>
                <Line
                  dataKey="implied"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  strokeDasharray="5 3"
                  dot={false}
                />
                <Line
                  dataKey="impliedNeg"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  strokeDasharray="5 3"
                  dot={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
            <p className="muted">
              Green = stock rose, red = stock fell. Dashed orange lines = expected range
              (±). A bar reaching beyond the dashed lines means the move exceeded
              expectations.
            </p>
          </>
        )}
      </section>

      <footer className="muted">
        Educational project, not investment advice. The model has only a modest edge
        over a naive baseline.
      </footer>
    </main>
  );
}