import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    TrendingUp,
    TrendingDown,
    DollarSign,
    AlertTriangle,
    Sparkles,
    ChevronRight,
    Play,
    Activity,
    Package,
    Shuffle,
    Heart,
    Shield,
    Loader2,
    LineChart,
    Zap,
    Target,
    Lightbulb,
    Info,
} from "lucide-react";

const API_URL = "http://localhost:5000";

export default function ScenarioSimulator() {
    const [selectedScenario, setSelectedScenario] = useState(null);
    const [scenarioParams, setScenarioParams] = useState({
        // Marketing Budget
        budgetChange: -15,
        affectedChannels: [],

        // Demand Forecast
        forecastPeriods: 12,

        // Inventory
        skuList: [],
        holdingCostChange: 0,
        orderingCostChange: 0,
        leadTimeChange: 0,

        // Channel Mix
        sourceChannels: [],
        targetChannels: [],
        shiftPercentage: 10,

        // Customer Retention
        discountPercentage: 5,
        targetSegment: 'high_value',
        targetPercentile: 0.2,

        // Supplier Risk
        highRiskThreshold: 0.7,
        alternativeCostPremium: 0.05,
    });

    const [availableChannels, setAvailableChannels] = useState([]);
    const [simulationResult, setSimulationResult] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        loadAvailableModels();
    }, []);

    async function loadAvailableModels() {
        try {
            const res = await fetch(`${API_URL}/api/scenario/available-models`);
            const data = await res.json();
            if (data.success && data.models.marketing) {
                setAvailableChannels(data.models.marketing.channels || []);
            }
        } catch (e) {
            console.error("Failed to load models:", e);
        }
    }

    async function runSimulation() {
        if (!selectedScenario) return;

        setLoading(true);
        setSimulationResult(null);

        try {
            let endpoint = "";
            let payload = {};

            if (selectedScenario === "marketing") {
                endpoint = `${API_URL}/api/scenario/marketing-budget`;
                payload = {
                    budget_change_pct: scenarioParams.budgetChange,
                    affected_channels: scenarioParams.affectedChannels.length > 0
                        ? scenarioParams.affectedChannels
                        : null,
                };
            } else if (selectedScenario === "forecast") {
                endpoint = `${API_URL}/api/scenario/demand-forecast`;
                payload = {
                    periods_ahead: scenarioParams.forecastPeriods,
                };
            } else if (selectedScenario === "inventory") {
                endpoint = `${API_URL}/api/scenario/inventory-replenishment`;
                payload = {
                    sku_list: scenarioParams.skuList.length > 0 ? scenarioParams.skuList : null,
                    holding_cost_change_pct: scenarioParams.holdingCostChange,
                    ordering_cost_change_pct: scenarioParams.orderingCostChange,
                    lead_time_change_days: scenarioParams.leadTimeChange,
                };
            } else if (selectedScenario === "channel-mix") {
                endpoint = `${API_URL}/api/scenario/channel-mix`;
                payload = {
                    source_channels: scenarioParams.sourceChannels,
                    target_channels: scenarioParams.targetChannels,
                    shift_percentage: scenarioParams.shiftPercentage,
                };
            } else if (selectedScenario === "retention") {
                endpoint = `${API_URL}/api/scenario/customer-retention`;
                payload = {
                    discount_percentage: scenarioParams.discountPercentage,
                    target_segment: scenarioParams.targetSegment,
                    target_percentile: scenarioParams.targetPercentile,
                };
            } else if (selectedScenario === "supplier") {
                endpoint = `${API_URL}/api/scenario/supplier-risk`;
                payload = {
                    high_risk_threshold: scenarioParams.highRiskThreshold,
                    alternative_supplier_cost_premium: scenarioParams.alternativeCostPremium,
                };
            }

            const res = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const data = await res.json();
            setSimulationResult(data);
        } catch (e) {
            console.error("Simulation failed:", e);
            setSimulationResult({
                success: false,
                error: "Network error: " + e.message,
            });
        } finally {
            setLoading(false);
        }
    }

    const scenarioTypes = [
        {
            id: "marketing",
            title: "Marketing Budget Shift",
            description: "Simulate budget changes across marketing channels",
            icon: DollarSign,
            color: "from-amber-500 to-orange-600",
        },
        {
            id: "channel-mix",
            title: "Channel Mix Optimization",
            description: "Shift ad spend between marketing channels",
            icon: Shuffle,
            color: "from-purple-500 to-pink-600",
        },
        {
            id: "retention",
            title: "Customer Retention",
            description: "Loyalty discount programs for customer segments",
            icon: Heart,
            color: "from-rose-500 to-red-600",
        },
        {
            id: "inventory",
            title: "Inventory Replenishment",
            description: "Optimize inventory costs and order quantities",
            icon: Package,
            color: "from-blue-500 to-cyan-600",
        },
        {
            id: "supplier",
            title: "Supplier Risk Mitigation",
            description: "Diversify away from high-risk suppliers",
            icon: Shield,
            color: "from-green-500 to-emerald-600",
        },
        {
            id: "forecast",
            title: "Demand Forecast",
            description: "Project future sales and demand trends",
            icon: LineChart,
            color: "from-indigo-500 to-violet-600",
        },
    ];

    const getRecommendationIcon = (type) => {
        switch (type) {
            case "warning":
                return AlertTriangle;
            case "opportunity":
                return Zap;
            case "action":
                return Target;
            case "insight":
                return Lightbulb;
            default:
                return Info;
        }
    };

    const getRecommendationColor = (priority) => {
        switch (priority) {
            case "high":
                return "from-red-500/20 to-orange-500/20 border-red-400/40";
            case "medium":
                return "from-yellow-500/20 to-amber-500/20 border-yellow-400/40";
            default:
                return "from-blue-500/20 to-cyan-500/20 border-blue-400/40";
        }
    };

    const toggleChannelSelection = (channel, arrayName) => {
        setScenarioParams(prev => ({
            ...prev,
            [arrayName]: prev[arrayName].includes(channel)
                ? prev[arrayName].filter(c => c !== channel)
                : [...prev[arrayName], channel]
        }));
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 relative overflow-hidden">
            {/* Background effects */}
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-violet-900/20 via-transparent to-transparent" />
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom_left,_var(--tw-gradient-stops))] from-cyan-900/20 via-transparent to-transparent" />

            {/* Animated grid */}
            <div className="absolute inset-0 opacity-[0.03]">
                <div
                    className="absolute inset-0"
                    style={{
                        backgroundImage: `
              linear-gradient(to right, white 1px, transparent 1px),
              linear-gradient(to bottom, white 1px, transparent 1px)
            `,
                        backgroundSize: "60px 60px",
                    }}
                />
            </div>

            {/* Floating orbs */}
            <motion.div
                className="absolute top-20 left-20 w-96 h-96 bg-violet-600/20 rounded-full blur-[120px]"
                animate={{
                    x: [0, 30, 0],
                    y: [0, -30, 0],
                    scale: [1, 1.1, 1],
                }}
                transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
            />
            <motion.div
                className="absolute bottom-32 right-32 w-[32rem] h-[32rem] bg-cyan-600/20 rounded-full blur-[120px]"
                animate={{
                    x: [0, -40, 0],
                    y: [0, 40, 0],
                    scale: [1, 1.15, 1],
                }}
                transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
            />

            <div className="relative z-10 p-4 md:p-8">
                {/* Progress indicator */}
                <div className="max-w-7xl mx-auto mb-6">
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
                        <div className="w-12 sm:w-16 h-px bg-emerald-400/50" />
                        <div className="flex items-center gap-2 text-emerald-400">
                            <div className="w-8 h-8 rounded-full bg-emerald-500/20 border border-emerald-400/50 flex items-center justify-center font-medium text-xs">
                                ✓
                            </div>
                            <span className="hidden sm:inline">Analyze</span>
                        </div>
                        <div className="w-12 sm:w-16 h-px bg-violet-400/50" />
                        <div className="flex items-center gap-2 text-violet-400">
                            <div className="w-8 h-8 rounded-full bg-violet-500/20 border border-violet-400/50 flex items-center justify-center font-medium text-xs">
                                4
                            </div>
                            <span className="hidden sm:inline">Simulate</span>
                        </div>
                    </div>
                </div>

                <div className="max-w-7xl mx-auto">
                    {/* Header */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6 }}
                        className="text-center mb-12"
                    >
                        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-violet-500/10 border border-violet-400/20 text-violet-400 text-sm font-medium mb-6">
                            <Activity className="w-4 h-4" />
                            <span>Scenario Simulation Engine</span>
                        </div>

                        <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-white via-violet-100 to-cyan-100 bg-clip-text text-transparent mb-4 tracking-tight">
                            What-If Analysis
                        </h1>
                        <p className="text-lg text-slate-300 max-w-2xl mx-auto">
                            Model future outcomes across marketing, inventory, customers, and suppliers with AI-powered insights
                        </p>
                    </motion.div>

                    {/* Scenario Selection */}
                    {!selectedScenario && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.6, delay: 0.1 }}
                            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12"
                        >
                            {scenarioTypes.map((scenario) => {
                                const Icon = scenario.icon;
                                return (
                                    <motion.button
                                        key={scenario.id}
                                        onClick={() => setSelectedScenario(scenario.id)}
                                        className="group relative p-8 rounded-3xl bg-gradient-to-br from-slate-800/50 to-slate-900/50 border border-white/10 hover:border-white/20 backdrop-blur-xl transition-all text-left overflow-hidden"
                                        whileHover={{ y: -4, scale: 1.02 }}
                                        transition={{ duration: 0.2 }}
                                    >
                                        <div className={`absolute inset-0 bg-gradient-to-br opacity-0 group-hover:opacity-100 transition-opacity duration-300 ${scenario.color}`} />

                                        <div className="relative z-10">
                                            <div
                                                className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${scenario.color} flex items-center justify-center mb-6 shadow-lg`}
                                            >
                                                <Icon className="w-8 h-8 text-white" strokeWidth={1.5} />
                                            </div>

                                            <h3 className="text-2xl font-bold text-white mb-3">
                                                {scenario.title}
                                            </h3>
                                            <p className="text-slate-400 mb-6">
                                                {scenario.description}
                                            </p>

                                            <div className="flex items-center gap-2 text-violet-400 font-medium">
                                                <span>Configure Scenario</span>
                                                <ChevronRight className="w-5 h-5" />
                                            </div>
                                        </div>
                                    </motion.button>
                                );
                            })}
                        </motion.div>
                    )}

                    {/* Configuration Panel */}
                    {selectedScenario && !simulationResult && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.6 }}
                            className="max-w-4xl mx-auto"
                        >
                            <div className="rounded-3xl bg-gradient-to-br from-slate-800/50 to-slate-900/50 border border-white/10 backdrop-blur-xl p-8 md:p-12">
                                <button
                                    onClick={() => {
                                        setSelectedScenario(null);
                                        setSimulationResult(null);
                                    }}
                                    className="mb-6 text-slate-400 hover:text-white transition-colors flex items-center gap-2"
                                >
                                    <ChevronRight className="w-4 h-4 rotate-180" />
                                    <span>Back to scenarios</span>
                                </button>

                                <h2 className="text-3xl font-bold text-white mb-2">
                                    {scenarioTypes.find((s) => s.id === selectedScenario)?.title}
                                </h2>
                                <p className="text-slate-400 mb-8">
                                    Configure your simulation parameters
                                </p>

                                {/* Marketing Budget Parameters */}
                                {selectedScenario === "marketing" && (
                                    <div className="mb-8 space-y-6">
                                        <div>
                                            <label className="block text-sm font-medium text-slate-300 mb-3">
                                                Budget Change (%)
                                            </label>
                                            <div className="flex items-center gap-4">
                                                <input
                                                    type="range"
                                                    min="-50"
                                                    max="50"
                                                    step="5"
                                                    value={scenarioParams.budgetChange}
                                                    onChange={(e) =>
                                                        setScenarioParams({
                                                            ...scenarioParams,
                                                            budgetChange: parseFloat(e.target.value),
                                                        })
                                                    }
                                                    className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-violet-500"
                                                />
                                                <div className="w-24 text-right">
                                                    <span
                                                        className={`text-2xl font-bold ${scenarioParams.budgetChange < 0
                                                                ? "text-red-400"
                                                                : "text-emerald-400"
                                                            }`}
                                                    >
                                                        {scenarioParams.budgetChange > 0 ? "+" : ""}
                                                        {scenarioParams.budgetChange}%
                                                    </span>
                                                </div>
                                            </div>
                                        </div>

                                        {availableChannels.length > 0 && (
                                            <div>
                                                <label className="block text-sm font-medium text-slate-300 mb-3">
                                                    Affected Channels (optional - leave empty for all)
                                                </label>
                                                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                                                    {availableChannels.map((channel) => {
                                                        const isSelected = scenarioParams.affectedChannels.includes(channel);
                                                        return (
                                                            <button
                                                                key={channel}
                                                                onClick={() => toggleChannelSelection(channel, 'affectedChannels')}
                                                                className={`p-4 rounded-xl border transition-all ${isSelected
                                                                        ? "bg-violet-500/20 border-violet-400/50 text-violet-300"
                                                                        : "bg-slate-800/50 border-slate-700 text-slate-400 hover:border-slate-600"
                                                                    }`}
                                                            >
                                                                <span className="font-medium capitalize">
                                                                    {channel}
                                                                </span>
                                                            </button>
                                                        );
                                                    })}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}

                                {/* Channel Mix Parameters */}
                                {selectedScenario === "channel-mix" && (
                                    <div className="mb-8 space-y-6">
                                        <div>
                                            <label className="block text-sm font-medium text-slate-300 mb-3">
                                                Source Channels (reduce spend from)
                                            </label>
                                            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                                                {availableChannels.map((channel) => {
                                                    const isSelected = scenarioParams.sourceChannels.includes(channel);
                                                    return (
                                                        <button
                                                            key={channel}
                                                            onClick={() => toggleChannelSelection(channel, 'sourceChannels')}
                                                            className={`p-4 rounded-xl border transition-all ${isSelected
                                                                    ? "bg-red-500/20 border-red-400/50 text-red-300"
                                                                    : "bg-slate-800/50 border-slate-700 text-slate-400 hover:border-slate-600"
                                                                }`}
                                                        >
                                                            <span className="font-medium capitalize">{channel}</span>
                                                        </button>
                                                    );
                                                })}
                                            </div>
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-slate-300 mb-3">
                                                Target Channels (increase spend to)
                                            </label>
                                            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                                                {availableChannels.map((channel) => {
                                                    const isSelected = scenarioParams.targetChannels.includes(channel);
                                                    return (
                                                        <button
                                                            key={channel}
                                                            onClick={() => toggleChannelSelection(channel, 'targetChannels')}
                                                            className={`p-4 rounded-xl border transition-all ${isSelected
                                                                    ? "bg-emerald-500/20 border-emerald-400/50 text-emerald-300"
                                                                    : "bg-slate-800/50 border-slate-700 text-slate-400 hover:border-slate-600"
                                                                }`}
                                                        >
                                                            <span className="font-medium capitalize">{channel}</span>
                                                        </button>
                                                    );
                                                })}
                                            </div>
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-slate-300 mb-3">
                                                Shift Percentage: {scenarioParams.shiftPercentage}%
                                            </label>
                                            <input
                                                type="range"
                                                min="5"
                                                max="50"
                                                step="5"
                                                value={scenarioParams.shiftPercentage}
                                                onChange={(e) =>
                                                    setScenarioParams({
                                                        ...scenarioParams,
                                                        shiftPercentage: parseFloat(e.target.value),
                                                    })
                                                }
                                                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                                            />
                                        </div>
                                    </div>
                                )}

                                {/* Customer Retention Parameters */}
                                {selectedScenario === "retention" && (
                                    <div className="mb-8 space-y-6">
                                        <div>
                                            <label className="block text-sm font-medium text-slate-300 mb-3">
                                                Discount Percentage: {scenarioParams.discountPercentage}%
                                            </label>
                                            <input
                                                type="range"
                                                min="1"
                                                max="20"
                                                step="1"
                                                value={scenarioParams.discountPercentage}
                                                onChange={(e) =>
                                                    setScenarioParams({
                                                        ...scenarioParams,
                                                        discountPercentage: parseFloat(e.target.value),
                                                    })
                                                }
                                                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-500"
                                            />
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-slate-300 mb-3">
                                                Target Segment
                                            </label>
                                            <div className="grid grid-cols-3 gap-3">
                                                {['high_value', 'at_risk', 'all'].map((segment) => (
                                                    <button
                                                        key={segment}
                                                        onClick={() =>
                                                            setScenarioParams({
                                                                ...scenarioParams,
                                                                targetSegment: segment,
                                                            })
                                                        }
                                                        className={`p-4 rounded-xl border transition-all ${scenarioParams.targetSegment === segment
                                                                ? "bg-rose-500/20 border-rose-400/50 text-rose-300"
                                                                : "bg-slate-800/50 border-slate-700 text-slate-400 hover:border-slate-600"
                                                            }`}
                                                    >
                                                        <span className="font-medium capitalize">
                                                            {segment.replace('_', ' ')}
                                                        </span>
                                                    </button>
                                                ))}
                                            </div>
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-slate-300 mb-3">
                                                Target Percentile: Top {(scenarioParams.targetPercentile * 100).toFixed(0)}%
                                            </label>
                                            <input
                                                type="range"
                                                min="0.05"
                                                max="0.5"
                                                step="0.05"
                                                value={scenarioParams.targetPercentile}
                                                onChange={(e) =>
                                                    setScenarioParams({
                                                        ...scenarioParams,
                                                        targetPercentile: parseFloat(e.target.value),
                                                    })
                                                }
                                                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-500"
                                            />
                                        </div>
                                    </div>
                                )}

                                {/* Inventory Parameters */}
                                {selectedScenario === "inventory" && (
                                    <div className="mb-8 space-y-6">
                                        <div>
                                            <label className="block text-sm font-medium text-slate-300 mb-3">
                                                Holding Cost Change: {scenarioParams.holdingCostChange > 0 ? '+' : ''}{scenarioParams.holdingCostChange}%
                                            </label>
                                            <input
                                                type="range"
                                                min="-30"
                                                max="30"
                                                step="5"
                                                value={scenarioParams.holdingCostChange}
                                                onChange={(e) =>
                                                    setScenarioParams({
                                                        ...scenarioParams,
                                                        holdingCostChange: parseFloat(e.target.value),
                                                    })
                                                }
                                                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                                            />
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-slate-300 mb-3">
                                                Ordering Cost Change: {scenarioParams.orderingCostChange > 0 ? '+' : ''}{scenarioParams.orderingCostChange}%
                                            </label>
                                            <input
                                                type="range"
                                                min="-30"
                                                max="30"
                                                step="5"
                                                value={scenarioParams.orderingCostChange}
                                                onChange={(e) =>
                                                    setScenarioParams({
                                                        ...scenarioParams,
                                                        orderingCostChange: parseFloat(e.target.value),
                                                    })
                                                }
                                                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                                            />
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-slate-300 mb-3">
                                                Lead Time Change: {scenarioParams.leadTimeChange > 0 ? '+' : ''}{scenarioParams.leadTimeChange} days
                                            </label>
                                            <input
                                                type="range"
                                                min="-5"
                                                max="10"
                                                step="1"
                                                value={scenarioParams.leadTimeChange}
                                                onChange={(e) =>
                                                    setScenarioParams({
                                                        ...scenarioParams,
                                                        leadTimeChange: parseInt(e.target.value),
                                                    })
                                                }
                                                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                                            />
                                        </div>
                                    </div>
                                )}

                                {/* Supplier Risk Parameters */}
                                {selectedScenario === "supplier" && (
                                    <div className="mb-8 space-y-6">
                                        <div>
                                            <label className="block text-sm font-medium text-slate-300 mb-3">
                                                High-Risk Threshold: {scenarioParams.highRiskThreshold.toFixed(2)}
                                            </label>
                                            <input
                                                type="range"
                                                min="0.5"
                                                max="0.9"
                                                step="0.05"
                                                value={scenarioParams.highRiskThreshold}
                                                onChange={(e) =>
                                                    setScenarioParams({
                                                        ...scenarioParams,
                                                        highRiskThreshold: parseFloat(e.target.value),
                                                    })
                                                }
                                                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-green-500"
                                            />
                                            <p className="text-xs text-slate-500 mt-2">
                                                Suppliers with risk score above this threshold will be considered high-risk
                                            </p>
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-slate-300 mb-3">
                                                Alternative Supplier Cost Premium: {(scenarioParams.alternativeCostPremium * 100).toFixed(0)}%
                                            </label>
                                            <input
                                                type="range"
                                                min="0"
                                                max="0.25"
                                                step="0.01"
                                                value={scenarioParams.alternativeCostPremium}
                                                onChange={(e) =>
                                                    setScenarioParams({
                                                        ...scenarioParams,
                                                        alternativeCostPremium: parseFloat(e.target.value),
                                                    })
                                                }
                                                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-green-500"
                                            />
                                            <p className="text-xs text-slate-500 mt-2">
                                                Extra cost expected when switching to alternative suppliers
                                            </p>
                                        </div>
                                    </div>
                                )}

                                {/* Forecast Parameters */}
                                {selectedScenario === "forecast" && (
                                    <div className="mb-8">
                                        <label className="block text-sm font-medium text-slate-300 mb-3">
                                            Forecast Periods (days)
                                        </label>
                                        <input
                                            type="number"
                                            min="1"
                                            max="365"
                                            value={scenarioParams.forecastPeriods}
                                            onChange={(e) =>
                                                setScenarioParams({
                                                    ...scenarioParams,
                                                    forecastPeriods: parseInt(e.target.value) || 12,
                                                })
                                            }
                                            className="w-full p-4 rounded-xl bg-slate-800/50 border border-slate-700 text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50 focus:border-violet-400/50"
                                        />
                                    </div>
                                )}

                                {/* Run Button */}
                                <motion.button
                                    onClick={runSimulation}
                                    disabled={loading}
                                    className="w-full py-5 rounded-xl bg-gradient-to-r from-violet-600 to-purple-600 text-white font-semibold text-lg shadow-[0_0_40px_-10px_rgba(139,92,246,0.5)] hover:shadow-[0_0_50px_-10px_rgba(139,92,246,0.7)] disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-3"
                                    whileHover={!loading ? { scale: 1.02 } : {}}
                                    whileTap={!loading ? { scale: 0.98 } : {}}
                                >
                                    {loading ? (
                                        <>
                                            <Loader2 className="w-6 h-6 animate-spin" />
                                            <span>Running Simulation...</span>
                                        </>
                                    ) : (
                                        <>
                                            <Play className="w-6 h-6" />
                                            <span>Run Simulation</span>
                                        </>
                                    )}
                                </motion.button>
                            </div>
                        </motion.div>
                    )}

                    {/* Results Display */}
                    {simulationResult && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.6 }}
                            className="space-y-6"
                        >
                            <div className="flex items-center justify-between mb-6">
                                <h2 className="text-3xl font-bold text-white">
                                    Simulation Results
                                </h2>
                                <button
                                    onClick={() => {
                                        setSimulationResult(null);
                                        setSelectedScenario(null);
                                    }}
                                    className="px-6 py-3 rounded-xl bg-slate-800/50 border border-slate-700 text-slate-300 hover:text-white hover:border-slate-600 transition-all"
                                >
                                    New Simulation
                                </button>
                            </div>

                            {!simulationResult.success ? (
                                <div className="rounded-3xl bg-gradient-to-br from-red-900/20 to-orange-900/20 border border-red-500/30 backdrop-blur-xl p-8">
                                    <div className="flex items-start gap-4 text-red-400">
                                        <AlertTriangle className="w-8 h-8 flex-shrink-0 mt-1" />
                                        <div className="flex-1">
                                            <h3 className="text-xl font-bold mb-2">
                                                Simulation Error
                                            </h3>
                                            <p className="text-red-300 mb-3">{simulationResult.error}</p>
                                            {simulationResult.hint && (
                                                <div className="mt-4 p-4 rounded-xl bg-yellow-500/10 border border-yellow-500/30">
                                                    <div className="flex items-start gap-3">
                                                        <Lightbulb className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                                                        <div>
                                                            <p className="text-sm font-medium text-yellow-300 mb-1">Hint:</p>
                                                            <p className="text-sm text-yellow-200">{simulationResult.hint}</p>
                                                        </div>
                                                    </div>
                                                </div>
                                            )}
                                            {simulationResult.details && (
                                                <p className="text-sm text-red-200 mt-3">{simulationResult.details}</p>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <>
                                    {/* Impact Summary Card */}
                                    <div className="rounded-3xl bg-gradient-to-br from-slate-800/50 to-slate-900/50 border border-white/10 backdrop-blur-xl p-8">
                                        <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
                                            <TrendingUp className="w-6 h-6 text-violet-400" />
                                            Impact Summary
                                        </h3>

                                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                            {simulationResult.impact && Object.entries(simulationResult.impact).slice(0, 6).map(([key, value], idx) => (
                                                <div
                                                    key={idx}
                                                    className="p-4 rounded-xl bg-slate-800/50 border border-slate-700"
                                                >
                                                    <div className="text-sm text-slate-400 mb-2 capitalize">
                                                        {key.replace(/_/g, ' ')}
                                                    </div>
                                                    <div className="flex items-baseline gap-2">
                                                        <span
                                                            className={`text-2xl font-bold ${typeof value === 'number' && value >= 0
                                                                    ? "text-emerald-400"
                                                                    : "text-red-400"
                                                                }`}
                                                        >
                                                            {typeof value === 'number'
                                                                ? (value >= 0 ? '+' : '') + (key.includes('pct') ? value.toFixed(1) + '%' : value.toLocaleString(undefined, { maximumFractionDigits: 2 }))
                                                                : String(value)
                                                            }
                                                        </span>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Recommendations */}
                                    {simulationResult.recommendations?.length > 0 && (
                                        <div className="rounded-3xl bg-gradient-to-br from-slate-800/50 to-slate-900/50 border border-white/10 backdrop-blur-xl p-8">
                                            <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
                                                <Sparkles className="w-6 h-6 text-yellow-400" />
                                                AI Recommendations
                                            </h3>

                                            <div className="space-y-4">
                                                {simulationResult.recommendations.map((rec, idx) => {
                                                    const Icon = getRecommendationIcon(rec.type);
                                                    return (
                                                        <motion.div
                                                            key={idx}
                                                            initial={{ opacity: 0, x: -20 }}
                                                            animate={{ opacity: 1, x: 0 }}
                                                            transition={{ duration: 0.3, delay: idx * 0.1 }}
                                                            className={`p-6 rounded-2xl bg-gradient-to-br border backdrop-blur-xl ${getRecommendationColor(
                                                                rec.priority
                                                            )}`}
                                                        >
                                                            <div className="flex items-start gap-4">
                                                                <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-white/10 flex items-center justify-center">
                                                                    <Icon className="w-5 h-5 text-white" />
                                                                </div>
                                                                <div className="flex-1">
                                                                    <div className="flex items-center gap-3 mb-2">
                                                                        <h4 className="font-semibold text-white">
                                                                            {rec.title}
                                                                        </h4>
                                                                        <span
                                                                            className={`px-2 py-0.5 rounded-full text-xs font-medium ${rec.priority === "high"
                                                                                    ? "bg-red-500/20 text-red-300"
                                                                                    : rec.priority === "medium"
                                                                                        ? "bg-yellow-500/20 text-yellow-300"
                                                                                        : "bg-blue-500/20 text-blue-300"
                                                                                }`}
                                                                        >
                                                                            {rec.priority}
                                                                        </span>
                                                                    </div>
                                                                    <p className="text-slate-300 text-sm leading-relaxed">
                                                                        {rec.description}
                                                                    </p>
                                                                </div>
                                                            </div>
                                                        </motion.div>
                                                    );
                                                })}
                                            </div>
                                        </div>
                                    )}
                                </>
                            )}
                        </motion.div>
                    )}
                </div>
            </div>
        </div>
    );
}