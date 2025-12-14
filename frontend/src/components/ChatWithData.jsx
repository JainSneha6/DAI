// src/components/Chatbot.jsx
import React, { useState } from "react";

const API_URL = "http://localhost:5000";

export default function Chatbot() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]); // {role, text, sources?}
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
        const errMsg = { role: "assistant", text: `Error: ${j && j.error ? j.error : "unknown"}` };
        setMessages((m) => [...m, errMsg]);
      }
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", text: "Network error: " + String(e) }]);
    } finally {
      setLoading(false);
      setQuery("");
    }
  }

  return (
    <div className="max-w-2xl mx-auto p-4">
      <div className="mb-4">
        <h2 className="text-xl font-bold">Enterprise Chatbot</h2>
        <p className="text-sm text-slate-400">Ask questions about uploaded CSVs (answers are sourced from uploaded documents).</p>
      </div>

      <div className="space-y-3 mb-4">
        {messages.map((m, i) => (
          <div key={i} className={m.role === "user" ? "text-right" : ""}>
            <div className={`inline-block p-3 rounded ${m.role === "user" ? "bg-blue-100" : "bg-gray-100"}`}>
              <div style={{ whiteSpace: "pre-wrap" }}>{m.text}</div>
              {m.sources && m.sources.length > 0 && (
                <div className="mt-2 text-xs text-slate-500">
                  Sources:
                  <ul className="list-disc list-inside">
                    {m.sources.map((s, idx) => (
                      <li key={idx}>
                        Doc {s.doc_index}: {s.metadata && s.metadata.filename ? s.metadata.filename : s.metadata?.source || "unknown"} — {s.text_snippet}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="flex gap-2">
        <input
          className="flex-1 p-2 border rounded"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") sendQuery(); }}
          placeholder="Ask about uploaded data (e.g. 'What were top-selling SKUs last month?')"
        />
        <button className="px-4 py-2 bg-blue-600 text-white rounded" onClick={sendQuery} disabled={loading}>
          {loading ? "Thinking…" : "Send"}
        </button>
      </div>
    </div>
  );
}
