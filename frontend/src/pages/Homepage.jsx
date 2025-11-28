import React from "react";
import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";

export default function DecisivAIHomepage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-[#1b1f30] via-[#1a2238] to-[#111827] p-8 relative overflow-hidden">
      {/* Soft enterprise glow accents */}
      <div className="absolute top-20 left-32 w-72 h-72 bg-blue-600/10 rounded-full blur-3xl" />
      <div className="absolute bottom-10 right-40 w-80 h-80 bg-violet-600/10 rounded-full blur-3xl" />

      {/* Minimalist geometric lines */}
      <div className="absolute inset-0 opacity-[0.08]">
        <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
          <line x1="0" y1="0" x2="100%" y2="100%" stroke="white" strokeWidth="1" />
          <line x1="100%" y1="0" x2="0" y2="100%" stroke="white" strokeWidth="1" />
        </svg>
      </div>

      <div className="w-full max-w-5xl rounded-3xl shadow-[0_0_60px_-20px_rgba(0,0,0,0.6)] overflow-hidden relative border border-white/10 bg-gradient-to-br from-[#0d111a]/60 to-[#0e1320]/40 backdrop-blur-xl p-16">
        <main className="flex flex-col items-center text-center relative z-10">
          <motion.h1
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-4xl lg:text-5xl font-semibold text-slate-100 tracking-tight mb-6"
          >
            Welcome to DecisivAI
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.15 }}
            className="max-w-3xl text-lg lg:text-xl text-slate-200/85 leading-relaxed mb-10"
          >
            Empowering businesses with AI-driven insights for smarter, faster, and more informed decision-making. DecisivAI acts as your virtual consultant—simulating scenarios, visualizing decisions, and analyzing insights to keep your business ahead.
          </motion.p>

          <motion.a
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            href="#get-started"
            className="inline-flex items-center gap-3 rounded-full px-8 py-3 bg-gradient-to-r from-[#4c6ef5] to-[#7c3aed] text-white text-lg font-medium shadow-xl hover:opacity-95 focus:outline-none focus:ring-4 focus:ring-violet-700/40"
          >
            Get Started
            <ArrowRight size={18} />
          </motion.a>
        </main>

        {/* Subtle bottom divider */}
        <div className="absolute bottom-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
      </div>
    </div>
  );
}
