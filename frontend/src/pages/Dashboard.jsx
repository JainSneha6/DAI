import React, { useState, useRef, useCallback, useEffect } from "react";
import { motion } from "framer-motion";
import FilesList from "../components/FilesList.jsx";
import ChatWithData from "./ChatWithData.jsx";


export default function Dashboard() {

  return (
    <div className="min-h-screen bg-linear-to-br from-[#1b1f30] via-[#1a2238] to-[#111827] p-8">
      <div className="w-full max-w-7xl mx-auto rounded-3xl shadow-[0_0_60px_-20px_rgba(0,0,0,0.6)] overflow-hidden relative border border-white/10 bg-linear-to-br from-[#0d111a]/60 to-[#0e1320]/40 backdrop-blur-xl p-8">
        <div className="flex flex-col lg:flex-row gap-8">

            <div className="sticky top-6">
              <div className="mb-4">
                <h2 className="text-xl font-semibold text-slate-100 mb-2">Files & Categories</h2>
                <p className="text-sm text-slate-300">Browse uploaded files and their inferred categories.</p>
              </div>
              <FilesList />
            </div>
        </div>

        <div className="absolute bottom-0 left-0 w-full h-px bg-linear-to-r from-transparent via-white/20 to-transparent" />
      </div>
    </div>
  );
}
