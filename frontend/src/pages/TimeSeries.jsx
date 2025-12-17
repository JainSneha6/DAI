import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

const API_URL = "http://localhost:5000";
const INDEX_NAME = "sales_forecasts";

export default function TimeSeriesDashboard() {
  const [series, setSeries] = useState([]);
  const [loading, setLoading] = useState(false);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [status, setStatus] = useState("");

  // -----------------------------------
  // Load latest forecast on mount
  // -----------------------------------
  useEffect(() => {
    fetchLatestForecast();
  }, []);

  async function fetchLatestForecast() {
    try {
      setLoading(true);
      const res = await fetch(
        `${API_URL}/timeseries/forecast/latest?index_name=${INDEX_NAME}`
      );
      const j = await res.json();
      setSeries(j.series || []);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  // -----------------------------------
  // Run Forecast
  // -----------------------------------
  async function runForecast() {
    setStatus("Running forecast...");
    setAnswer("");
    try {
      const res = await fetch(`${API_URL}/timeseries/forecast/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          index_name: INDEX_NAME,
          horizon: 30,
        }),
      });
      const j = await res.json();
      if (j.success) {
        setStatus("Forecast generated successfully");
        fetchLatestForecast();
      } else {
        setStatus("Failed to run forecast");
      }
    } catch (e) {
      setStatus("Error running forecast");
    }
  }

  // -----------------------------------
  // Ask NLP Question
  // -----------------------------------
  async function askQuestion() {
    if (!question.trim()) return;
    setAnswer("Thinking...");
    try {
      const res = await fetch(`${API_URL}/timeseries/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          index_name: INDEX_NAME,
          question,
        }),
      });
      const j = await res.json();
      setAnswer(j.answer || j.explanation || JSON.stringify(j));
    } catch (e) {
      setAnswer("Error processing question");
    }
    setQuestion("");
  }

  // -----------------------------------
  // Explain Forecast
  // -----------------------------------
  async function explainForecast() {
    setAnswer("Generating explanation...");
    try {
      const res = await fetch(`${API_URL}/timeseries/forecast/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ index_name: INDEX_NAME }),
      });
      const j = await res.json();
      setAnswer(j.explanation);
    } catch {
      setAnswer("Failed to explain forecast");
    }
  }

  // -----------------------------------
  // Compare Forecasts
  // -----------------------------------
  async function compareForecasts() {
    setAnswer("Comparing forecasts...");
    try {
      const res = await fetch(`${API_URL}/timeseries/forecast/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ index_name: INDEX_NAME }),
      });
      const j = await res.json();
      setAnswer(j.summary);
    } catch {
      setAnswer("Comparison failed");
    }
  }

  return (
    <div className="min-h-screen bg-linear-to-br from-[#1b1f30] via-[#1a2238] to-[#111827] p-10 text-slate-100">
      {/* Header */}
      <motion.h1
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-4xl font-semibold text-center mb-8"
      >
        Forecast Intelligence Dashboard
      </motion.h1>

      {/* Controls */}
      <div className="flex justify-center gap-4 mb-6">
        <button
          onClick={runForecast}
          className="px-6 py-3 rounded-xl bg-linear-to-r from-[#4c6ef5] to-[#7c3aed] shadow-lg hover:opacity-90"
        >
          🔮 Predict the Future
        </button>

        <button
          onClick={explainForecast}
          className="px-6 py-3 rounded-xl bg-slate-700/40 border border-white/10 hover:bg-slate-600/40"
        >
          🧠 Explain
        </button>

        <button
          onClick={compareForecasts}
          className="px-6 py-3 rounded-xl bg-slate-700/40 border border-white/10 hover:bg-slate-600/40"
        >
          📊 Compare
        </button>
      </div>

      {status && (
        <div className="text-center text-sm text-slate-300 mb-4">
          {status}
        </div>
      )}

      {/* Chart */}
      <div className="bg-slate-900/40 rounded-2xl p-6 border border-white/10 mb-8">
        <h2 className="text-xl mb-4">Latest Forecast</h2>
        {loading ? (
          <div className="text-slate-400">Loading forecast…</div>
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={series}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="timestamp" hide />
              <YAxis />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#7c3aed"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* NLP Question */}
      <div className="bg-slate-900/40 rounded-2xl p-6 border border-white/10">
        <h2 className="text-xl mb-4">Ask a Question</h2>

        <div className="flex gap-3 mb-4">
          <input
            className="flex-1 p-3 rounded-xl bg-slate-800/50 border border-white/10 text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-600"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && askQuestion()}
            placeholder="What will demand look like next month?"
          />
          <button
            onClick={askQuestion}
            className="px-6 py-3 rounded-xl bg-linear-to-r from-[#4c6ef5] to-[#7c3aed]"
          >
            Ask
          </button>
        </div>

        {answer && (
          <div className="mt-3 p-4 rounded-xl bg-slate-800/50 border border-white/10 text-slate-200">
            {answer}
          </div>
        )}
      </div>
    </div>
  );
}
