// src/pages/DecisivAIUploadAndResults.jsx
import React, { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import { Download, RefreshCw } from "lucide-react";

export default function TimeSeries() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchModels = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch("http://localhost:5000/api/models");
      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(`${resp.status} ${txt}`);
      }
      const data = await resp.json();
      if (!data.success) throw new Error(data.error || "Failed to fetch models");
      setModels(data.models || []);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const PrettyJSON = ({ data }) => (
    <pre className="whitespace-pre-wrap text-xs lg:text-sm text-slate-200 bg-white/3 p-3 rounded-md overflow-auto">
      {JSON.stringify(data, null, 2)}
    </pre>
  );

  return (
    <div className="min-h-screen flex items-center justify-center bg-linear-to-br from-[#1b1f30] via-[#1a2238] to-[#111827] p-8">
      <div className="w-full max-w-6xl rounded-3xl shadow-[0_0_60px_-20px_rgba(0,0,0,0.6)] overflow-hidden relative border border-white/10 bg-linear-to-br from-[#0d111a]/60 to-[#0e1320]/40 backdrop-blur-xl p-8">
        <header className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl lg:text-3xl font-semibold text-slate-100">DecisivAI — Results</h1>
            <p className="text-sm text-slate-300 mt-1">
              All saved models and metadata. Upload happens on <span className="font-medium">/input</span>.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={fetchModels}
              disabled={loading}
              className="inline-flex items-center gap-2 rounded-full px-4 py-2 bg-linear-to-r from-[#4c6ef5] to-[#7c3aed] text-white font-medium shadow hover:opacity-95 disabled:opacity-50"
            >
              <RefreshCw size={16} /> {loading ? "Refreshing..." : "Refresh"}
            </button>
          </div>
        </header>

        <main className="grid grid-cols-1 gap-6">
          <section>
            <div className="rounded-xl bg-white/3 p-4 border border-white/6">
              <h3 className="text-slate-100 font-semibold mb-3">Saved models</h3>

              {error && (
                <div className="mb-4 text-sm text-rose-200 bg-rose-700/10 p-3 rounded">
                  Error: {error}
                </div>
              )}

              {!models.length && !loading ? (
                <div className="text-slate-300 text-sm">No saved models found. Upload from <span className="font-medium">/input</span>.</div>
              ) : (
                <div className="space-y-4 max-h-[70vh] overflow-auto pr-2">
                  {models.map((m, idx) => {
                    const meta = m.metadata || {};
                    const createdAt = meta.created_at || meta.train_end || "";
                    return (
                      <div key={idx} className="bg-white/4 p-3 rounded-md border border-white/6">
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <div className="font-medium text-slate-100">{meta.model_name || "Model"}</div>
                            <div className="text-xs text-slate-300 mt-1">{m.meta_filename} • {createdAt}</div>
                          </div>

                          <div className="flex items-center gap-2">
                            {m.model_url ? (
                              <a
                                className="inline-flex items-center gap-2 px-3 py-1 rounded-md bg-white/5 text-xs text-slate-100 hover:bg-white/6"
                                href={m.model_url}
                              >
                                <Download size={14} /> Download Model
                              </a>
                            ) : (
                              <div className="text-xs text-slate-400 px-3 py-1 rounded-md">No artifact</div>
                            )}
                            {m.meta_url && (
                              <a
                                className="inline-flex items-center gap-2 px-3 py-1 rounded-md bg-white/5 text-xs text-slate-100 hover:bg-white/6"
                                href={m.meta_url}
                              >
                                Metadata
                              </a>
                            )}
                          </div>
                        </div>

                        <div className="mt-3 grid grid-cols-1 gap-3">
                          <div>
                            <div className="text-xs text-slate-300 mb-1">Metadata</div>
                            <PrettyJSON data={meta} />
                          </div>

                          {/* If metadata contains future_forecast, show a compact preview */}
                          {meta.future_forecast && Array.isArray(meta.future_forecast) && (
                            <div className="mt-2 p-2 rounded-md bg-white/4 border border-white/6">
                              <div className="text-sm font-semibold text-slate-100">Future forecast (preview)</div>
                              <div className="mt-2 text-xs text-slate-200">
                                {meta.future_forecast.slice(0, 20).map((v, i) => (
                                  <span key={i} className="inline-block mr-2">{Number(v).toFixed(2)}</span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              {loading && <div className="mt-4 text-sm text-slate-300">Loading models...</div>}
            </div>
          </section>
        </main>

        <footer className="mt-6 text-center text-xs text-slate-400">
          Models are stored on the server and can be downloaded from the links above.
        </footer>
      </div>
    </div>
  );
}
