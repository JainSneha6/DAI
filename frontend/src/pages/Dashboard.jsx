import React from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { MessageSquare, TrendingUp, BarChart3 } from "lucide-react";
import FilesList from "../components/FilesList";

export default function Dashboard() {
  const navigate = useNavigate();

  return (
    // ✅ Removed duplicate background - Layout component provides it
    <div className="min-h-screen relative overflow-hidden">

      {/* Background effects */}
      <div className="absolute top-20 left-32 w-96 h-96 bg-blue-600/15 rounded-full blur-[120px] animate-pulse" style={{ animationDuration: '4s' }} />
      <div className="absolute bottom-10 right-40 w-[28rem] h-[28rem] bg-violet-600/15 rounded-full blur-[120px] animate-pulse" style={{ animationDuration: '5s', animationDelay: '1s' }} />

      <div className="absolute inset-0 opacity-[0.03]">
        <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="grid" width="60" height="60" patternUnits="userSpaceOnUse">
              <path d="M 60 0 L 0 0 0 60" fill="none" stroke="white" strokeWidth="0.5" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>
      </div>

      <div className="relative z-10 p-8">

        {/* Progress Indicator */}
        <div className="max-w-7xl mx-auto mb-8">
          <div className="flex items-center justify-center gap-4">
            {[
              { step: 1, label: "Upload", status: "complete" },
              { step: 2, label: "Dashboard", status: "active" },
              { step: 3, label: "Analyze", status: "inactive" }
            ].map((item, idx) => (
              <React.Fragment key={item.step}>
                <div className="flex flex-col items-center gap-2">
                  <div
                    className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold border-2 transition-all ${item.status === "complete"
                        ? "bg-emerald-500/20 border-emerald-400/50 text-emerald-400"
                        : item.status === "active"
                          ? "bg-blue-500/20 border-blue-400/50 text-blue-400"
                          : "bg-slate-800/50 border-slate-700 text-slate-600"
                      }`}
                  >
                    {item.status === "complete" ? "✓" : item.step}
                  </div>
                  <span className={`text-xs font-medium ${item.status === "active" ? "text-blue-400" : "text-slate-500"
                    }`}>
                    {item.label}
                  </span>
                </div>
                {idx < 2 && (
                  <div className={`w-16 h-0.5 ${idx === 0 ? "bg-emerald-400/50" : "bg-slate-700"
                    }`} />
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Main Content */}
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="mb-8">
              <span className="inline-block px-3 py-1 rounded-full text-xs font-medium bg-blue-500/10 border border-blue-400/30 text-blue-400 mb-4">
                Step 2 of 3
              </span>
              <h1 className="text-4xl font-bold text-slate-50 mb-3">Your Dashboard</h1>
              <p className="text-lg text-slate-400">
                Review your uploaded files and start analyzing
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

              {/* Files List - Takes 2 columns */}
              <div className="lg:col-span-2">
                <FilesList />
              </div>

              {/* Action Cards */}
              <div className="space-y-6">

                {/* Quick Stats */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.1 }}
                  className="rounded-2xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 backdrop-blur-xl p-6"
                >
                  <h3 className="text-lg font-semibold text-slate-100 mb-4">Quick Stats</h3>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-3 rounded-lg bg-white/5">
                      <span className="text-sm text-slate-400">Files Processed</span>
                      <span className="text-lg font-bold text-emerald-400">Ready</span>
                    </div>
                    <div className="flex items-center justify-between p-3 rounded-lg bg-white/5">
                      <span className="text-sm text-slate-400">AI Status</span>
                      <span className="text-lg font-bold text-blue-400">Active</span>
                    </div>
                    <div className="flex items-center justify-between p-3 rounded-lg bg-white/5">
                      <span className="text-sm text-slate-400">Categories</span>
                      <span className="text-lg font-bold text-violet-400">Multiple</span>
                    </div>
                  </div>
                </motion.div>

                {/* CTA Card */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                  className="rounded-2xl bg-gradient-to-br from-blue-500/10 to-violet-500/10 border border-blue-400/30 backdrop-blur-xl p-6"
                >
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500/20 to-violet-500/20 border border-blue-400/30 flex items-center justify-center mb-4">
                    <MessageSquare size={24} className="text-blue-400" />
                  </div>
                  <h3 className="text-lg font-semibold text-slate-100 mb-2">
                    Ready to Analyze?
                  </h3>
                  <p className="text-sm text-slate-400 mb-4">
                    Start chatting with your data using AI-powered insights
                  </p>
                  <button
                    onClick={() => navigate("/chat")}
                    className="w-full px-6 py-3 rounded-full bg-gradient-to-r from-blue-500 to-violet-500 text-white font-medium shadow-[0_0_30px_-10px_rgba(124,58,237,0.5)] hover:scale-105 hover:shadow-[0_0_40px_-10px_rgba(124,58,237,0.7)] transition-all"
                  >
                    Start Analyzing
                  </button>
                </motion.div>

                {/* Info Card */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.3 }}
                  className="rounded-2xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 backdrop-blur-xl p-6"
                >
                  <div className="flex items-start gap-3">
                    <div className="w-10 h-10 rounded-lg bg-cyan-500/10 border border-cyan-400/30 flex items-center justify-center flex-shrink-0">
                      <BarChart3 size={20} className="text-cyan-400" />
                    </div>
                    <div>
                      <h4 className="text-sm font-semibold text-slate-200 mb-1">
                        Deep Analysis Available
                      </h4>
                      <p className="text-xs text-slate-400">
                        Click any file's "Analyze" button to view detailed statistics and visualizations
                      </p>
                    </div>
                  </div>
                </motion.div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}