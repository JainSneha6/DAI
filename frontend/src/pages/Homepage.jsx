import React from "react";
import { motion } from "framer-motion";
import { ArrowRight, Sparkles, Brain, TrendingUp } from "lucide-react";

export default function DecisivAIHomepage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-[#1b1f30] via-[#1a2238] to-[#111827] p-8 relative overflow-hidden">
      {/* Enhanced ambient glow effects */}
      <div className="absolute top-20 left-32 w-96 h-96 bg-blue-600/15 rounded-full blur-[120px] animate-pulse" style={{ animationDuration: '4s' }} />
      <div className="absolute bottom-10 right-40 w-[28rem] h-[28rem] bg-violet-600/15 rounded-full blur-[120px] animate-pulse" style={{ animationDuration: '5s', animationDelay: '1s' }} />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[32rem] h-[32rem] bg-cyan-500/8 rounded-full blur-[140px]" />

      {/* Refined geometric grid overlay */}
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

      <div className="w-full max-w-6xl relative z-10">
        <div className="rounded-[2rem] shadow-[0_20px_80px_-20px_rgba(0,0,0,0.7)] overflow-hidden relative border border-white/10 bg-gradient-to-br from-[#0d111a]/80 to-[#0e1320]/60 backdrop-blur-2xl">

          {/* Top accent line */}
          <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-blue-400/40 to-transparent" />

          <div className="p-16 lg:p-20">
            <main className="flex flex-col items-center text-center relative">

              {/* Logo/Brand mark */}
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.6, ease: "easeOut" }}
                className="mb-8 relative"
              >
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-blue-500/20 to-violet-600/20 border border-white/10 flex items-center justify-center backdrop-blur-xl">
                  <Brain className="w-10 h-10 text-blue-400" strokeWidth={1.5} />
                </div>
                <div className="absolute -top-1 -right-1 w-6 h-6 bg-gradient-to-br from-violet-400 to-blue-500 rounded-full flex items-center justify-center">
                  <Sparkles className="w-3 h-3 text-white" />
                </div>
              </motion.div>

              <motion.h1
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.1 }}
                className="text-5xl lg:text-6xl font-bold text-slate-50 tracking-tight mb-6 leading-tight"
              >
                Welcome to{" "}
                <span className="bg-gradient-to-r from-blue-400 via-violet-400 to-blue-400 bg-clip-text text-transparent">
                  DecisivAI
                </span>
              </motion.h1>

              <motion.p
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
                className="max-w-3xl text-lg lg:text-xl text-slate-300 leading-relaxed mb-12 font-light"
              >
                Empowering businesses with AI-driven insights for smarter, faster, and more informed
                decision-making. DecisivAI acts as your virtual consultant—simulating scenarios,
                visualizing decisions, and analyzing insights to keep your business ahead.
              </motion.p>

              {/* Feature highlights */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.3 }}
                className="flex flex-wrap items-center justify-center gap-6 mb-12"
              >
                <div className="flex items-center gap-2 text-slate-400 text-sm">
                  <div className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                  AI-Powered Analysis
                </div>
                <div className="flex items-center gap-2 text-slate-400 text-sm">
                  <div className="w-1.5 h-1.5 rounded-full bg-violet-400" />
                  Smart Visualizations
                </div>
                <div className="flex items-center gap-2 text-slate-400 text-sm">
                  <div className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                  Real-time Insights
                </div>
              </motion.div>

              <motion.a
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.4 }}
                href="/upload"
                className="group relative inline-flex items-center gap-3 rounded-full px-10 py-4 bg-gradient-to-r from-[#4c6ef5] to-[#7c3aed] text-white text-lg font-medium shadow-[0_0_40px_-10px_rgba(124,58,237,0.5)] hover:shadow-[0_0_50px_-10px_rgba(124,58,237,0.7)] transition-all duration-300 hover:scale-105 focus:outline-none focus:ring-4 focus:ring-violet-700/40"
              >
                Get Started
                <ArrowRight
                  size={20}
                  className="transition-transform duration-300 group-hover:translate-x-1"
                />
              </motion.a>


            </main>
          </div>

          {/* Bottom accent line */}
          <div className="absolute bottom-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-violet-400/40 to-transparent" />
        </div>
      </div>
    </div>
  );
}