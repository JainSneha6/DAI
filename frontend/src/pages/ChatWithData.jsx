// Updated React component with centered heading, robot on left, chat on right
import React, { useState } from "react";
import { motion } from "framer-motion";

const API_URL = "http://localhost:5000";

export default function Chatbot() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  async function sendQuery() {
    if (!query.trim()) return;
    const userMsg = { role: "user", text: query };
    setMessages((m) => [...m, userMsg]);
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 6 }),
      });
      const j = await res.json();

      if (j && j.success) {
        const assistantMsg = {
          role: "assistant",
          text: j.answer || j.raw_response || "No answer",
          sources: j.sources || [],
        };
        setMessages((m) => [...m, assistantMsg]);
      } else {
        const errMsg = { role: "assistant", text: `Error: ${j?.error || "unknown"}` };
        setMessages((m) => [...m, errMsg]);
      }
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", text: "Network error: " + e }]);
    } finally {
      setLoading(false);
      setQuery("");
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-start bg-linear-to-br from-[#1b1f30] via-[#1a2238] to-[#111827] p-10 relative overflow-hidden">
      {/* Glow Accents */}
      <div className="absolute top-20 left-32 w-72 h-72 bg-blue-600/10 rounded-full blur-3xl" />
      <div className="absolute bottom-10 right-40 w-80 h-80 bg-violet-600/10 rounded-full blur-3xl" />

      {/* Centered Page Heading */}
      <motion.h1
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-5xl font-semibold text-slate-100 tracking-tight mb-10 text-center"
      >
        Virtual Consultant
      </motion.h1>

      {/* Main Layout: Robot Left, Chat Right */}
      <div className="w-full max-w-6xl flex items-start justify-between gap-16 relative z-10 mt-5 ml-72">
        {/* Robot */}
        <motion.img
          src="/robot.png"
          alt="Robot"
          className="w-80 h-auto drop-shadow-2xl fixed left-4 top-40 z-20"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        />

        {/* Chat Area */}
        <div className="flex-1 flex flex-col">
          <p className="text-slate-200/80 mb-6 text-lg text-center">
            Ask questions about uploaded CSVs — sourced directly from your documents.
          </p>

          {/* Messages */}
          <div className="space-y-4 mb-6 h-auto  pr-2 custom-scrollbar">
            {messages.map((m, i) => (
              <div key={i} className={m.role === "user" ? "text-right" : "text-left"}>
                <div
                  className={`inline-block p-3 rounded-xl max-w-[80%] backdrop-blur-lg border border-white/10 ${
                    m.role === "user"
                      ? "bg-blue-600/20 text-blue-100"
                      : "bg-slate-700/20 text-slate-200"
                  }`}
                >
                  <div style={{ whiteSpace: "pre-wrap" }}>{m.text}</div>

                  {m.sources?.length > 0 && (
                    <div className="mt-2 text-xs text-slate-400">
                      Sources:
                      <ul className="list-disc list-inside">
                        {m.sources.map((s, idx) => (
                          <li key={idx}>
                            Doc {s.doc_index}: {s.metadata?.filename || s.metadata?.source || "unknown"} — {s.text_snippet}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Input */}
          <div className="flex gap-3">
            <input
              className="flex-1 p-3 rounded-xl bg-slate-900/40 border border-white/10 text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-600"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendQuery()}
              placeholder="Ask about your data…"
            />

            <button
              className="px-6 py-3 rounded-xl bg-linear-to-r from-[#4c6ef5] to-[#7c3aed] text-white font-medium shadow-lg hover:opacity-90"
              onClick={sendQuery}
              disabled={loading}
            >
              {loading ? "Thinking…" : "Send"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
