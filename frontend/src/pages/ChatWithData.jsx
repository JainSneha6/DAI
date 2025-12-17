import React, { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { Send, Loader2, FileText, Sparkles } from "lucide-react";

const API_URL = "http://localhost:5000";

export default function ChatWithData() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  async function sendQuery() {
    if (!query.trim()) return;
    const userMsg = { role: "user", text: query };
    setMessages((m) => [...m, userMsg]);
    setLoading(true);
    setQuery("");

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
      if (inputRef.current) {
        inputRef.current.focus();
      }
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#1b1f30] via-[#1a2238] to-[#111827] relative overflow-hidden">

      {/* Ambient glow effects */}
      <div className="absolute top-20 left-32 w-96 h-96 bg-blue-600/15 rounded-full blur-[120px]" />
      <div className="absolute bottom-10 right-40 w-[28rem] h-[28rem] bg-violet-600/15 rounded-full blur-[120px]" />

      {/* Grid overlay */}
      <div className="absolute inset-0 opacity-[0.02]">
        <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="chat-grid" width="60" height="60" patternUnits="userSpaceOnUse">
              <path d="M 60 0 L 0 0 0 60" fill="none" stroke="white" strokeWidth="0.5" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#chat-grid)" />
        </svg>
      </div>

      <div className="relative z-10 p-4 md:p-8 h-screen flex flex-col">

        {/* Progress indicator */}
        <div className="max-w-7xl mx-auto mb-6 w-full">
          <div className="flex items-center justify-center gap-2 text-sm">
            <div className="flex items-center gap-2 text-emerald-400">
              <div className="w-8 h-8 rounded-full bg-emerald-500/20 border border-emerald-400/50 flex items-center justify-center font-medium text-xs">
                ✓
              </div>
              <span className="hidden sm:inline">Upload Data</span>
            </div>
            <div className="w-12 sm:w-16 h-px bg-emerald-400/50" />
            <div className="flex items-center gap-2 text-emerald-400">
              <div className="w-8 h-8 rounded-full bg-emerald-500/20 border border-emerald-400/50 flex items-center justify-center font-medium text-xs">
                ✓
              </div>
              <span className="hidden sm:inline">Dashboard</span>
            </div>
            <div className="w-12 sm:w-16 h-px bg-blue-400/50" />
            <div className="flex items-center gap-2 text-blue-400">
              <div className="w-8 h-8 rounded-full bg-blue-500/20 border border-blue-400/50 flex items-center justify-center font-medium text-xs">
                3
              </div>
              <span className="hidden sm:inline">Analyze</span>
            </div>
          </div>
        </div>

        <div className="w-full max-w-6xl mx-auto flex-1 flex flex-col min-h-0">
          <div className="rounded-[2rem] shadow-[0_20px_80px_-20px_rgba(0,0,0,0.7)] overflow-hidden relative border border-white/10 bg-gradient-to-br from-[#0d111a]/80 to-[#0e1320]/60 backdrop-blur-2xl flex flex-col h-full">

            {/* Top accent */}
            <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-blue-400/40 to-transparent" />

            {/* Header */}
            <div className="p-6 md:p-8 border-b border-white/10">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-400/20 text-blue-400 text-sm font-medium mb-4">
                  <div className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
                  Virtual Consultant Active
                </div>

                <h1 className="text-3xl md:text-4xl font-bold text-slate-50 tracking-tight mb-2 leading-tight">
                  AI Analysis Chat
                </h1>
                <p className="text-slate-300 text-sm md:text-base font-light">
                  Ask questions about your data — sourced directly from your documents.
                </p>
              </motion.div>
            </div>

            {/* Chat Container */}
            <div className="flex-1 flex overflow-hidden">

              {/* Robot Illustration - Hidden on mobile */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
                className="hidden lg:flex w-80 flex-shrink-0 items-center justify-center p-8"
              >
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-br from-blue-500/20 to-violet-500/20 rounded-full blur-3xl" />
                  <img
                    src="/robot.png"
                    alt="AI Assistant"
                    className="w-full h-auto drop-shadow-2xl relative z-10"
                  />
                </div>
              </motion.div>

              {/* Messages Area */}
              <div className="flex-1 flex flex-col min-w-0">

                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-white/10">

                  {messages.length === 0 && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.6, delay: 0.3 }}
                      className="flex flex-col items-center justify-center h-full text-center"
                    >
                      <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-blue-500/20 to-violet-500/20 border border-white/10 flex items-center justify-center mb-6">
                        <Sparkles className="w-10 h-10 text-blue-400" strokeWidth={1.5} />
                      </div>
                      <h3 className="text-xl font-semibold text-slate-100 mb-3">Start Your Analysis</h3>
                      <p className="text-slate-400 max-w-md mb-8">
                        Ask any question about your uploaded data. I'll analyze it and provide insights based on your documents.
                      </p>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl">
                        <button
                          onClick={() => setQuery("What are the main trends in the data?")}
                          className="p-4 rounded-xl bg-white/5 border border-white/10 text-left hover:bg-white/10 transition-colors group"
                        >
                          <div className="text-sm font-medium text-slate-200 mb-1 group-hover:text-blue-400 transition-colors">
                            Identify Trends
                          </div>
                          <div className="text-xs text-slate-400">
                            What patterns emerge from the data?
                          </div>
                        </button>
                        <button
                          onClick={() => setQuery("Summarize the key findings")}
                          className="p-4 rounded-xl bg-white/5 border border-white/10 text-left hover:bg-white/10 transition-colors group"
                        >
                          <div className="text-sm font-medium text-slate-200 mb-1 group-hover:text-blue-400 transition-colors">
                            Get Summary
                          </div>
                          <div className="text-xs text-slate-400">
                            Overview of important insights
                          </div>
                        </button>
                        <button
                          onClick={() => setQuery("What anomalies exist in the data?")}
                          className="p-4 rounded-xl bg-white/5 border border-white/10 text-left hover:bg-white/10 transition-colors group"
                        >
                          <div className="text-sm font-medium text-slate-200 mb-1 group-hover:text-blue-400 transition-colors">
                            Find Anomalies
                          </div>
                          <div className="text-xs text-slate-400">
                            Detect unusual patterns or outliers
                          </div>
                        </button>
                        <button
                          onClick={() => setQuery("What recommendations can you provide?")}
                          className="p-4 rounded-xl bg-white/5 border border-white/10 text-left hover:bg-white/10 transition-colors group"
                        >
                          <div className="text-sm font-medium text-slate-200 mb-1 group-hover:text-blue-400 transition-colors">
                            Get Recommendations
                          </div>
                          <div className="text-xs text-slate-400">
                            Actionable insights from analysis
                          </div>
                        </button>
                      </div>
                    </motion.div>
                  )}

                  {messages.map((m, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3 }}
                      className={m.role === "user" ? "flex justify-end" : "flex justify-start"}
                    >
                      <div
                        className={`max-w-[85%] md:max-w-[75%] rounded-2xl p-4 backdrop-blur-xl border ${m.role === "user"
                            ? "bg-gradient-to-br from-blue-600/20 to-violet-600/20 border-blue-400/30 text-blue-50"
                            : "bg-gradient-to-br from-slate-700/20 to-slate-800/20 border-slate-600/30 text-slate-100"
                          }`}
                      >
                        <div className="whitespace-pre-wrap leading-relaxed">{m.text}</div>

                        {m.sources?.length > 0 && (
                          <div className="mt-4 pt-4 border-t border-white/10">
                            <div className="flex items-center gap-2 text-xs font-medium text-slate-400 mb-3">
                              <FileText size={14} />
                              <span>Sources ({m.sources.length})</span>
                            </div>
                            <ul className="space-y-2">
                              {m.sources.map((s, idx) => (
                                <li
                                  key={idx}
                                  className="text-xs p-3 rounded-lg bg-white/5 border border-white/10"
                                >
                                  <div className="font-medium text-slate-300 mb-1">
                                    {s.metadata?.filename || s.metadata?.source || `Document ${s.doc_index}`}
                                  </div>
                                  <div className="text-slate-500 line-clamp-2">
                                    {s.text_snippet}
                                  </div>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  ))}

                  {loading && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="flex justify-start"
                    >
                      <div className="max-w-[75%] rounded-2xl p-4 backdrop-blur-xl border bg-gradient-to-br from-slate-700/20 to-slate-800/20 border-slate-600/30">
                        <div className="flex items-center gap-3 text-slate-400">
                          <Loader2 className="w-5 h-5 animate-spin" />
                          <span>Analyzing your data...</span>
                        </div>
                      </div>
                    </motion.div>
                  )}

                  <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                <div className="p-6 md:p-8 border-t border-white/10">
                  <div className="flex gap-3">
                    <input
                      ref={inputRef}
                      className="flex-1 p-4 rounded-xl bg-slate-900/40 border border-white/10 text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-400/50 transition-all"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendQuery()}
                      placeholder="Ask about your data…"
                      disabled={loading}
                    />

                    <button
                      className="px-6 py-4 rounded-xl bg-gradient-to-r from-[#4c6ef5] to-[#7c3aed] text-white font-medium shadow-[0_0_30px_-10px_rgba(124,58,237,0.5)] hover:shadow-[0_0_40px_-10px_rgba(124,58,237,0.7)] disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:scale-[1.02] focus:outline-none focus:ring-4 focus:ring-violet-700/40 flex items-center gap-2"
                      onClick={sendQuery}
                      disabled={loading || !query.trim()}
                    >
                      <span className="hidden sm:inline">{loading ? "Sending..." : "Send"}</span>
                      <Send size={18} />
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Bottom accent */}
            <div className="absolute bottom-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-violet-400/40 to-transparent" />
          </div>
        </div>
      </div>
    </div>
  );
}