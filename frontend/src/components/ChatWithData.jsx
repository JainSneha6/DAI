// src/components/ChatWithData.jsx
import React, { useState } from "react";

export default function ChatWithData() {
    const [question, setQuestion] = useState("");
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);

    const send = async () => {
        if (!question.trim()) return;
        const userMsg = { role: "user", text: question };
        setMessages((m) => [...m, userMsg]);
        setLoading(true);
        try {
            const res = await fetch("http://localhost:5000/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question }),
            });
            const j = await res.json();
            if (j.success) {
                const assistant = { role: "assistant", text: j.answer, sources: j.sources || [] };
                setMessages((m) => [...m, assistant]);
            } else {
                setMessages((m) => [...m, { role: "assistant", text: "Error: " + (j.error || "unknown") }]);
            }
        } catch (e) {
            setMessages((m) => [...m, { role: "assistant", text: "Request failed: " + String(e) }]);
        } finally {
            setLoading(false);
            setQuestion("");
        }
    };

    return (
        <div className="p-4 bg-white/3 rounded-lg">
            <div className="space-y-3 mb-4">
                <div className="h-64 overflow-auto p-2 bg-white/5 rounded">
                    {messages.length === 0 ? (
                        <div className="text-slate-400">Ask questions about trends, anomalies or summaries — e.g. "show me last 30 day sales trend"</div>
                    ) : (
                        messages.map((m, i) => (
                            <div key={i} className={`mb-2 ${m.role === "user" ? "text-right" : ""}`}>
                                <div className={`inline-block p-2 rounded ${m.role === "user" ? "bg-slate-700" : "bg-slate-800"}`}>
                                    <div className="whitespace-pre-wrap">{m.text}</div>
                                    {m.sources && m.sources.length > 0 && (
                                        <div className="text-xs text-slate-400 mt-2">
                                            Sources: {m.sources.map(s => s.filename || s.category).join(", ")}
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>

            <div className="flex gap-2">
                <input
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Ask about your data..."
                    className="flex-1 p-2 rounded bg-slate-900 text-white"
                />
                <button onClick={send} disabled={loading} className="px-4 py-2 rounded bg-indigo-600">
                    {loading ? "Thinking..." : "Ask"}
                </button>
            </div>
        </div>
    );
}
