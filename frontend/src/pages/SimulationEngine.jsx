import React, { useState } from "react";

export default function SimulationEngine() {
  const [file, setFile] = useState(null);
  const [channel, setChannel] = useState("");
  const [change, setChange] = useState(-15);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);

    if (!file) {
      setError("Please upload a CSV file to run the simulation.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("change_pct", change);
    if (channel) formData.append("channel", channel);

    setLoading(true);
    try {
      const resp = await fetch("/simulate/marketing", {
        method: "POST",
        body: formData,
      });

      const data = await resp.json();
      if (!resp.ok) {
        setError(data.error || "Simulation failed");
      } else if (!data.success) {
        setError(data.error || "Simulation failed");
      } else {
        setResult(data.result);
      }
    } catch (err) {
      setError(err.message || "Unexpected error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h2 className="text-2xl font-semibold mb-4">Scenario Simulation Engine ⚙️</h2>

      <div className="bg-white/5 p-6 rounded-lg shadow-sm mb-6">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-200 mb-1">Dataset (CSV)</label>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files && e.target.files[0])}
              className="text-sm text-slate-200"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-200 mb-1">Channel (optional)</label>
              <input
                value={channel}
                onChange={(e) => setChannel(e.target.value)}
                placeholder="E.g., Google Search or search"
                className="w-full px-3 py-2 rounded bg-neutral-900 text-slate-200 text-sm"
              />
              <p className="text-xs text-slate-500 mt-1">Leave empty to apply change across all spend channels</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-200 mb-1">Change (%)</label>
              <input
                type="number"
                step="0.5"
                value={change}
                onChange={(e) => setChange(Number(e.target.value))}
                className="w-full px-3 py-2 rounded bg-neutral-900 text-slate-200 text-sm"
              />
              <p className="text-xs text-slate-500 mt-1">Use negative values to reduce budget (e.g., -15)</p>
            </div>
          </div>

          <div className="flex gap-3">
            <button
              type="submit"
              disabled={loading}
              className="px-4 py-2 rounded bg-blue-500 hover:bg-blue-600 text-white"
            >
              {loading ? "Running..." : "Run Simulation"}
            </button>

            <button
              type="button"
              onClick={() => { setFile(null); setChannel(""); setChange(-15); setResult(null); setError(null) }}
              className="px-4 py-2 rounded bg-white/5 hover:bg-white/10 text-slate-200"
            >
              Reset
            </button>
          </div>
        </form>
      </div>

      {error && (
        <div className="bg-red-900/30 border border-red-700 p-4 rounded text-red-200 mb-4">{error}</div>
      )}

      {result && (
        <div className="bg-white/5 p-6 rounded-lg shadow-sm">
          <h3 className="text-lg font-semibold mb-2">Simulation Results ✅</h3>

          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <div className="text-sm text-slate-400">Target column</div>
              <div className="font-medium">{result.target_column}</div>
            </div>
            <div>
              <div className="text-sm text-slate-400">Spend columns detected</div>
              <div className="font-medium">{result.spend_columns.join(", ")}</div>
            </div>

            <div>
              <div className="text-sm text-slate-400">Baseline total ({result.target_column})</div>
              <div className="font-medium">{Number(result.baseline_total).toFixed(2)}</div>
            </div>

            <div>
              <div className="text-sm text-slate-400">Projected total</div>
              <div className="font-medium">{Number(result.sim_total).toFixed(2)}</div>
            </div>
          </div>

          <div className="mb-3">
            <div className="text-sm text-slate-400">Delta</div>
            <div className="font-medium">{result.delta_pct !== null ? `${result.delta_pct.toFixed(2)}%` : "N/A"} ({result.delta_abs.toFixed(2)})</div>
          </div>

          <div className="mb-4">
            <div className="text-sm text-slate-400 mb-2">Top channel recommendations</div>
            <div className="text-sm">{result.recommendation}</div>
          </div>

          <div>
            <div className="text-sm text-slate-400 mb-2">Per-channel summary</div>
            <div className="space-y-2">
              {Object.entries(result.per_channel).map(([ch, info]) => (
                <div key={ch} className="p-2 rounded bg-white/3 flex justify-between">
                  <div>
                    <div className="font-medium">{ch}</div>
                    <div className="text-xs text-slate-400">Orig spend: {Number(info.orig_spend).toFixed(2)}</div>
                  </div>
                  <div className="text-right">
                    <div className="font-semibold">New spend: {Number(info.new_spend).toFixed(2)}</div>
                    <div className="text-xs text-slate-400">ROI: {info.approx_roi !== null ? Number(info.approx_roi).toFixed(2) : "N/A"}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
