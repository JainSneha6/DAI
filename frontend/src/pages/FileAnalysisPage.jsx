import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { useParams, useNavigate } from "react-router-dom";
import {
    ArrowLeft,
    Download,
    TrendingUp,
    BarChart3,
    PieChart,
    Activity,
    Loader2,
    AlertCircle,
    Database,
    Table,
    Grid3x3,
    Layers
} from "lucide-react";

// Import tab components
import OverviewTab from "../components/OverviewTab";
import NumericalTab from "../components/NumericalTab";
import CategoricalTab from "../components/CategoricalTab";
import { CorrelationsTab, TimeSeriesTab, GroupedTab } from "../components/AnalysisTabs";

const API_URL = "http://localhost:5000";

export default function FileAnalysisPage() {
    const { filename } = useParams();
    const navigate = useNavigate();
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [analysisData, setAnalysisData] = useState(null);
    const [activeTab, setActiveTab] = useState("overview");

    useEffect(() => {
        if (!filename) return;

        fetchAnalysis();
    }, [filename]);

    const fetchAnalysis = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${API_URL}/api/files/${encodeURIComponent(filename)}/analysis`);
            const data = await response.json();
            console.log("Fetched analysis data:", data);

            if (data.success) {
                setAnalysisData(data);
            } else {
                setError(data.error || "Failed to load analysis");
            }
        } catch (err) {
            setError("Network error: " + err.message);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-[#1b1f30] via-[#1a2238] to-[#111827] flex items-center justify-center">
                <div className="text-center">
                    <Loader2 className="w-12 h-12 text-blue-400 animate-spin mx-auto mb-4" />
                    <p className="text-slate-300">Loading analysis...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-[#1b1f30] via-[#1a2238] to-[#111827] flex items-center justify-center p-8">
                <div className="max-w-md w-full">
                    <div className="rounded-2xl bg-gradient-to-br from-red-500/10 to-red-600/10 border border-red-400/30 p-8 text-center">
                        <AlertCircle className="w-12 h-12 text-red-400 mx-auto mb-4" />
                        <h3 className="text-xl font-bold text-slate-100 mb-2">Analysis Failed</h3>
                        <p className="text-slate-300 mb-6">{error}</p>
                        <button
                            onClick={() => navigate("/dashboard")}
                            className="px-6 py-3 rounded-full bg-gradient-to-r from-blue-500 to-violet-500 text-white font-medium"
                        >
                            Back to Dashboard
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    const tabs = [
        { id: "overview", label: "Overview", icon: Grid3x3 },
        { id: "numerical", label: "Numerical Analysis", icon: BarChart3 },
        { id: "categorical", label: "Categorical Analysis", icon: PieChart },
        { id: "correlations", label: "Correlations", icon: Activity },
        { id: "timeseries", label: "Time Series", icon: TrendingUp },
        { id: "grouped", label: "Grouped Analysis", icon: Layers }
    ];

    return (
        <div className="min-h-screen bg-gradient-to-br from-[#1b1f30] via-[#1a2238] to-[#111827] relative overflow-hidden">

            {/* Background effects */}
            <div className="absolute top-20 left-32 w-96 h-96 bg-blue-600/15 rounded-full blur-[120px]" />
            <div className="absolute bottom-10 right-40 w-[28rem] h-[28rem] bg-violet-600/15 rounded-full blur-[120px]" />

            <div className="absolute inset-0 opacity-[0.02]">
                <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <pattern id="analysis-grid" width="60" height="60" patternUnits="userSpaceOnUse">
                            <path d="M 60 0 L 0 0 0 60" fill="none" stroke="white" strokeWidth="0.5" />
                        </pattern>
                    </defs>
                    <rect width="100%" height="100%" fill="url(#analysis-grid)" />
                </svg>
            </div>

            <div className="relative z-10 p-8">

                {/* Header */}
                <div className="max-w-7xl mx-auto mb-8">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5 }}
                    >
                        <button
                            onClick={() => navigate("/dashboard")}
                            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 text-slate-300 hover:bg-white/10 transition-all mb-6"
                        >
                            <ArrowLeft size={18} />
                            Back to Dashboard
                        </button>

                        <div className="flex items-start justify-between gap-6 mb-8">
                            <div>
                                <h1 className="text-4xl font-bold text-slate-50 mb-3">{filename}</h1>
                                <div className="flex items-center gap-4 text-sm text-slate-400">
                                    <div className="flex items-center gap-2">
                                        <Database size={16} />
                                        <span>{analysisData?.basic_info?.row_count?.toLocaleString()} rows</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <Table size={16} />
                                        <span>{analysisData?.basic_info?.column_count} columns</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <Activity size={16} />
                                        <span>{analysisData?.basic_info?.memory_usage}</span>
                                    </div>
                                </div>
                            </div>

                            <a
                                href={`${API_URL}/models/${encodeURIComponent(filename)}`}
                                className="inline-flex items-center gap-2 px-6 py-3 rounded-full bg-gradient-to-r from-emerald-500 to-cyan-500 text-white font-medium hover:scale-105 transition-transform"
                            >
                                <Download size={18} />
                                Download
                            </a>
                        </div>

                        {/* Stats Cards */}
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
                            <StatCard
                                label="Numeric Columns"
                                value={analysisData?.basic_info?.numeric_columns}
                                icon={BarChart3}
                                color="blue"
                            />
                            <StatCard
                                label="Categorical Columns"
                                value={analysisData?.basic_info?.categorical_columns}
                                icon={PieChart}
                                color="violet"
                            />
                            <StatCard
                                label="Time Columns"
                                value={analysisData?.basic_info?.time_columns}
                                icon={TrendingUp}
                                color="emerald"
                            />
                            <StatCard
                                label="Total Records"
                                value={analysisData?.basic_info?.row_count?.toLocaleString()}
                                icon={Database}
                                color="cyan"
                            />
                        </div>
                    </motion.div>
                </div>

                {/* Tabs */}
                <div className="max-w-7xl mx-auto">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: 0.1 }}
                        className="mb-8"
                    >
                        <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-white/10">
                            {tabs.map((tab) => {
                                const Icon = tab.icon;
                                return (
                                    <button
                                        key={tab.id}
                                        onClick={() => setActiveTab(tab.id)}
                                        className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all whitespace-nowrap ${activeTab === tab.id
                                            ? "bg-gradient-to-r from-blue-500 to-violet-500 text-white shadow-lg"
                                            : "bg-white/5 border border-white/10 text-slate-300 hover:bg-white/10"
                                            }`}
                                    >
                                        <Icon size={18} />
                                        {tab.label}
                                    </button>
                                );
                            })}
                        </div>
                    </motion.div>

                    {/* Tab Content */}
                    <motion.div
                        key={activeTab}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.4 }}
                    >
                        {activeTab === "overview" && <OverviewTab data={analysisData} />}
                        {activeTab === "numerical" && <NumericalTab data={analysisData} />}
                        {activeTab === "categorical" && <CategoricalTab data={analysisData} />}
                        {activeTab === "correlations" && <CorrelationsTab data={analysisData} />}
                        {activeTab === "timeseries" && <TimeSeriesTab data={analysisData} />}
                        {activeTab === "grouped" && <GroupedTab data={analysisData} />}
                    </motion.div>
                </div>
            </div>
        </div>
    );
}

// Stat Card Component
function StatCard({ label, value, icon: Icon, color }) {
    const colorClasses = {
        blue: "from-blue-500/10 to-blue-600/10 border-blue-400/30 text-blue-400",
        violet: "from-violet-500/10 to-violet-600/10 border-violet-400/30 text-violet-400",
        emerald: "from-emerald-500/10 to-emerald-600/10 border-emerald-400/30 text-emerald-400",
        cyan: "from-cyan-500/10 to-cyan-600/10 border-cyan-400/30 text-cyan-400"
    };

    return (
        <div className={`rounded-xl bg-gradient-to-br ${colorClasses[color]} border backdrop-blur-xl p-6`}>
            <Icon size={24} className="mb-3" />
            <div className="text-2xl font-bold text-slate-100 mb-1">{value || 0}</div>
            <div className="text-sm text-slate-400">{label}</div>
        </div>
    );
}

// Chart Card Wrapper
export function ChartCard({ title, icon: Icon, children, height = "auto" }) {
    return (
        <div className="rounded-2xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 backdrop-blur-xl p-6">
            <div className="flex items-center gap-3 mb-6">
                {Icon && <Icon size={24} className="text-blue-400" />}
                <h3 className="text-xl font-semibold text-slate-100">{title}</h3>
            </div>
            <div style={{ height }}>{children}</div>
        </div>
    );
}

export { StatCard };