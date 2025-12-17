import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
    Home,
    UploadCloud,
    LayoutDashboard,
    MessageSquare,
    ChevronLeft,
    ChevronRight,
    Brain,
    Sparkles,
    FileText
} from "lucide-react";

export default function Sidebar() {
    const [collapsed, setCollapsed] = useState(false);
    const navigate = useNavigate();
    const location = useLocation();

    const navItems = [
        {
            path: "/",
            icon: Home,
            label: "Home",
            description: "Welcome page"
        },
        {
            path: "/upload",
            icon: UploadCloud,
            label: "Upload",
            description: "Upload data files"
        },
        {
            path: "/dashboard",
            icon: LayoutDashboard,
            label: "Dashboard",
            description: "View your files"
        },
        {
            path: "/chat",
            icon: MessageSquare,
            label: "Chat",
            description: "Analyze with AI"
        }
    ];

    const isActive = (path) => {
        if (path === "/" && location.pathname === "/") return true;
        if (path !== "/" && location.pathname.startsWith(path)) return true;
        return false;
    };

    return (
        <>
            {/* Sidebar */}
            <motion.div
                initial={false}
                animate={{ width: collapsed ? "80px" : "280px" }}
                transition={{ duration: 0.3, ease: "easeInOut" }}
                className="fixed left-0 top-0 h-screen bg-gradient-to-b from-[#0d111a] to-[#0e1320] border-r border-white/10 z-50 flex flex-col"
            >
                {/* Logo Section */}
                <div className="p-6 border-b border-white/10">
                    <div className="flex items-center justify-between">
                        <AnimatePresence mode="wait">
                            {!collapsed ? (
                                <motion.div
                                    key="full-logo"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    exit={{ opacity: 0 }}
                                    transition={{ duration: 0.2 }}
                                    className="flex items-center gap-3"
                                >
                                    <div className="relative">
                                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-violet-500 flex items-center justify-center">
                                            <Brain size={24} className="text-white" />
                                        </div>
                                        <Sparkles size={14} className="absolute -top-1 -right-1 text-violet-400" />
                                    </div>
                                    <div>
                                        <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-violet-400 bg-clip-text text-transparent">
                                            DecisivAI
                                        </h1>
                                        <p className="text-xs text-slate-500">Data Analytics</p>
                                    </div>
                                </motion.div>
                            ) : (
                                <motion.div
                                    key="collapsed-logo"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    exit={{ opacity: 0 }}
                                    transition={{ duration: 0.2 }}
                                    className="relative mx-auto"
                                >
                                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-violet-500 flex items-center justify-center">
                                        <Brain size={24} className="text-white" />
                                    </div>
                                    <Sparkles size={12} className="absolute -top-1 -right-1 text-violet-400" />
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </div>

                {/* Navigation Items */}
                <nav className="flex-1 p-4 space-y-2 overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-white/10">
                    {navItems.map((item) => {
                        const Icon = item.icon;
                        const active = isActive(item.path);

                        return (
                            <motion.button
                                key={item.path}
                                onClick={() => navigate(item.path)}
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${active
                                        ? "bg-gradient-to-r from-blue-500/20 to-violet-500/20 border border-blue-400/30 text-blue-400"
                                        : "hover:bg-white/5 text-slate-400 hover:text-slate-200"
                                    }`}
                            >
                                <Icon size={20} className="flex-shrink-0" />
                                <AnimatePresence mode="wait">
                                    {!collapsed && (
                                        <motion.div
                                            key="nav-text"
                                            initial={{ opacity: 0, x: -10 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            exit={{ opacity: 0, x: -10 }}
                                            transition={{ duration: 0.2 }}
                                            className="flex-1 text-left"
                                        >
                                            <div className="font-medium text-sm">{item.label}</div>
                                            {active && (
                                                <div className="text-xs opacity-70">{item.description}</div>
                                            )}
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                                {active && (
                                    <motion.div
                                        layoutId="active-indicator"
                                        className="w-1 h-6 rounded-full bg-gradient-to-b from-blue-400 to-violet-400"
                                    />
                                )}
                            </motion.button>
                        );
                    })}
                </nav>

                {/* Analysis Files Section */}
                {!collapsed && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.2 }}
                        className="p-4 border-t border-white/10"
                    >
                        <div className="text-xs font-semibold text-slate-500 mb-2 px-4">RECENT ANALYSIS</div>
                        <div className="space-y-1">
                            {/* Placeholder for recent files - can be populated dynamically */}
                            <button
                                onClick={() => navigate("/dashboard")}
                                className="w-full flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-slate-200 transition-colors text-left"
                            >
                                <FileText size={16} />
                                <span className="text-xs truncate">View all files</span>
                            </button>
                        </div>
                    </motion.div>
                )}

                {/* Toggle Button */}
                <div className="p-4 border-t border-white/10">
                    <button
                        onClick={() => setCollapsed(!collapsed)}
                        className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-slate-400 hover:text-slate-200 transition-all"
                    >
                        {collapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
                        {!collapsed && <span className="text-sm">Collapse</span>}
                    </button>
                </div>
            </motion.div>

            {/* Spacer for main content */}
            <motion.div
                initial={false}
                animate={{ width: collapsed ? "80px" : "280px" }}
                transition={{ duration: 0.3, ease: "easeInOut" }}
                className="flex-shrink-0"
            />
        </>
    );
}