import React, { useState, useEffect } from "react";
import { useParams, useNavigate, useSearchParams } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
    Brain,
    TrendingUp,
    BarChart3,
    Package,
    Truck,
    Users,
    DollarSign,
    PlayCircle,
    CheckCircle,
    Loader2,
    AlertCircle,
    ArrowLeft,
    Info,
    Target,
    Database,
    Sparkles,
    FileText,
    Download,
    Calendar,
    Activity,
    PieChart,
    BarChart2,
    TrendingDown
} from "lucide-react";
import {
    LineChart,
    Line,
    BarChart,
    Bar,
    PieChart as RechartsPie,
    Pie,
    Cell,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer
} from "recharts";

// Model metadata (same as before)
const MODEL_INFO = {
    "sales-forecasting": {
        id: "sales-forecasting",
        name: "Sales, Demand & Financial Forecasting Model",
        icon: TrendingUp,
        color: "blue",
        description: "Time-series forecasting for sales, demand, and financial metrics using advanced ML algorithms.",
        category: "Predictive Analytics"
    },
    "marketing-roi": {
        id: "marketing-roi",
        name: "Marketing ROI & Attribution Model",
        icon: BarChart3,
        color: "violet",
        description: "Marketing mix modeling and attribution to optimize marketing spend.",
        category: "Marketing Analytics"
    },
    "inventory-optimization": {
        id: "inventory-optimization",
        name: "Inventory & Replenishment Optimization Model",
        icon: Package,
        color: "emerald",
        description: "Optimize inventory levels, reorder points, and replenishment strategies.",
        category: "Operations Analytics"
    },
    "supplier-risk": {
        id: "supplier-risk",
        name: "Logistics & Supplier Risk Model",
        icon: Truck,
        color: "amber",
        description: "Assess supplier risk, optimize routing, and ensure supply chain resilience.",
        category: "Supply Chain Analytics"
    },
    "customer-segmentation": {
        id: "customer-segmentation",
        name: "Customer Segmentation & Modeling",
        icon: Users,
        color: "pink",
        description: "Segment customers based on behavior and characteristics, predict lifetime value.",
        category: "Customer Analytics"
    }
};

const COLORS = ['#4c6ef5', '#7c3aed', '#10b981', '#f59e0b', '#ec4899', '#06b6d4', '#6366f1', '#ef4444'];

export default function ModelDetailsPageIntegrated() {
    const { modelId } = useParams();
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();
    const preselectedFile = searchParams.get('file');

    const [selectedFile, setSelectedFile] = useState(preselectedFile || null);
    const [availableFiles, setAvailableFiles] = useState([]);
    const [modelResults, setModelResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [running, setRunning] = useState(false);

    const modelInfo = MODEL_INFO[modelId];

    useEffect(() => {
        // Fetch files that have been processed with this model
        fetchFilesForModel();
    }, [modelId]);

    useEffect(() => {
        // If file is selected, fetch its results
        if (selectedFile) {
            fetchModelResults(selectedFile);
        }
    }, [selectedFile, modelId]);

    const fetchFilesForModel = async () => {
        try {
            const response = await fetch(`http://localhost:5000/api/models/${modelId}/files`);
            const data = await response.json();
            if (data.success) {
                setAvailableFiles(data.files || []);
            }
        } catch (err) {
            console.error("Failed to fetch files:", err);
        }
    };

    const fetchModelResults = async (filename) => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch(`http://localhost:5000/api/models/${modelId}/results/${filename}`);
            const data = await response.json();

            if (data.success) {
                setModelResults(data.results);
            } else {
                setError(data.error || "Failed to load results");
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const runModel = async () => {
        if (!selectedFile) {
            setError("Please select a file first");
            return;
        }

        setRunning(true);
        setError(null);

        try {
            const response = await fetch("http://localhost:5000/api/models/run", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    filename: selectedFile,
                    model_type: modelInfo.name
                })
            });

            const data = await response.json();

            if (data.success) {
                // Refresh results
                await fetchModelResults(selectedFile);
                await fetchFilesForModel();
            } else {
                setError(data.error || "Model execution failed");
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setRunning(false);
        }
    };

    if (!modelInfo) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="text-center">
                    <AlertCircle size={48} className="text-red-400 mx-auto mb-4" />
                    <h2 className="text-2xl font-bold text-slate-200 mb-2">Model Not Found</h2>
                    <button
                        onClick={() => navigate("/models")}
                        className="px-6 py-3 rounded-full bg-gradient-to-r from-blue-500 to-violet-500 text-white font-medium"
                    >
                        Back to Models
                    </button>
                </div>
            </div>
        );
    }

    const Icon = modelInfo.icon;

    return (
        <div className="min-h-screen relative overflow-hidden">
            {/* Background effects */}
            <div className="absolute top-20 left-32 w-96 h-96 bg-blue-600/10 rounded-full blur-[120px]" />
            <div className="absolute bottom-10 right-40 w-[28rem] h-[28rem] bg-violet-600/10 rounded-full blur-[120px]" />

            <div className="relative z-10 p-8">
                <div className="max-w-7xl mx-auto">

                    {/* Back Button */}
                    <button
                        onClick={() => navigate("/models")}
                        className="flex items-center gap-2 text-slate-400 hover:text-slate-200 transition-colors mb-6"
                    >
                        <ArrowLeft size={20} />
                        <span>Back to Models</span>
                    </button>

                    {/* Header */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="mb-8"
                    >
                        <div className="flex items-start gap-6">
                            <div className={`w-20 h-20 rounded-2xl bg-${modelInfo.color}-500/20 border border-${modelInfo.color}-400/30 flex items-center justify-center flex-shrink-0`}>
                                <Icon size={40} className={`text-${modelInfo.color}-400`} />
                            </div>
                            <div className="flex-1">
                                <span className={`inline-block px-3 py-1 rounded-full text-xs font-medium bg-${modelInfo.color}-500/10 border border-${modelInfo.color}-400/30 text-${modelInfo.color}-400 mb-3`}>
                                    {modelInfo.category}
                                </span>
                                <h1 className="text-4xl font-bold text-slate-50 mb-3">{modelInfo.name}</h1>
                                <p className="text-lg text-slate-400 mb-4">{modelInfo.description}</p>
                            </div>
                        </div>
                    </motion.div>

                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                        {/* Main Content - 2 columns */}
                        <div className="lg:col-span-2 space-y-6">

                            {/* File Selection */}
                            <Card title="Select Data File" icon={FileText}>
                                <select
                                    value={selectedFile || ""}
                                    onChange={(e) => setSelectedFile(e.target.value)}
                                    className="w-full px-4 py-3 rounded-lg bg-white/5 border border-white/10 text-slate-200 focus:outline-none focus:border-blue-400/50 transition-all mb-4"
                                >
                                    <option value="">Choose a file...</option>
                                    {availableFiles.map((file) => (
                                        <option key={file.filename} value={file.filename}>
                                            {file.filename} (Last run: {new Date(file.last_run).toLocaleDateString()})
                                        </option>
                                    ))}
                                </select>

                                {selectedFile && (
                                    <button
                                        onClick={runModel}
                                        disabled={running}
                                        className={`w-full px-6 py-3 rounded-full font-medium transition-all ${running
                                                ? "bg-slate-700 text-slate-500 cursor-not-allowed"
                                                : `bg-gradient-to-r from-${modelInfo.color}-500 to-${modelInfo.color}-600 text-white hover:scale-105 shadow-lg`
                                            }`}
                                    >
                                        {running ? (
                                            <span className="flex items-center justify-center gap-2">
                                                <Loader2 size={20} className="animate-spin" />
                                                Running Model...
                                            </span>
                                        ) : (
                                            <span className="flex items-center justify-center gap-2">
                                                <PlayCircle size={20} />
                                                Run Model Again
                                            </span>
                                        )}
                                    </button>
                                )}
                            </Card>

                            {/* Error Display */}
                            {error && (
                                <Card title="Error" icon={AlertCircle}>
                                    <div className="text-red-400">{error}</div>
                                </Card>
                            )}

                            {/* Loading State */}
                            {loading && (
                                <Card title="Loading Results" icon={Loader2}>
                                    <div className="flex items-center justify-center py-8">
                                        <Loader2 size={48} className="text-blue-400 animate-spin" />
                                    </div>
                                </Card>
                            )}

                            {/* Results Display */}
                            {!loading && modelResults && selectedFile && (
                                <ResultsDisplay
                                    modelId={modelId}
                                    results={modelResults}
                                    modelInfo={modelInfo}
                                />
                            )}
                        </div>

                        {/* Sidebar */}
                        <div className="space-y-6">
                            <ResultsSidebar
                                modelResults={modelResults}
                                selectedFile={selectedFile}
                                loading={loading}
                            />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

// Results Display Component
function ResultsDisplay({ modelId, results, modelInfo }) {
    if (!results || !results.available) {
        return (
            <Card title="No Results" icon={AlertCircle}>
                <p className="text-slate-400">No results available for this file.</p>
            </Card>
        );
    }

    return (
        <>
            {/* Metadata Card */}
            <Card title="Model Execution Details" icon={Info}>
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <div className="text-xs text-slate-500 mb-1">Created</div>
                        <div className="text-sm text-slate-300">
                            {new Date(results.created_at).toLocaleString()}
                        </div>
                    </div>
                    {results.target_column && (
                        <div>
                            <div className="text-xs text-slate-500 mb-1">Target Column</div>
                            <div className="text-sm text-slate-300">{results.target_column}</div>
                        </div>
                    )}
                    {results.solver_used && (
                        <div>
                            <div className="text-xs text-slate-500 mb-1">Solver Used</div>
                            <div className="text-sm text-slate-300">{results.solver_used}</div>
                        </div>
                    )}
                    {results.best_model_name && (
                        <div>
                            <div className="text-xs text-slate-500 mb-1">Best Model</div>
                            <div className="text-sm text-slate-300">{results.best_model_name}</div>
                        </div>
                    )}
                </div>
            </Card>

            {/* Model-Specific Results */}
            {modelId === "sales-forecasting" && <SalesForecastingResults results={results} />}
            {modelId === "marketing-roi" && <MarketingROIResults results={results} />}
            {modelId === "inventory-optimization" && <InventoryResults results={results} />}
            {modelId === "supplier-risk" && <SupplierRiskResults results={results} />}
            {modelId === "customer-segmentation" && <CustomerSegmentationResults results={results} />}
        </>
    );
}

// Sales Forecasting Results
function SalesForecastingResults({ results }) {
    const forecast = results.best_model?.forecast || [];
    const metrics = results.best_model?.metrics || {};

    const forecastData = forecast.slice(0, 20).map((value, index) => ({
        period: `T+${index + 1}`,
        forecast: parseFloat(value.toFixed(2))
    }));

    return (
        <>
            <Card title="Performance Metrics" icon={Activity}>
                <div className="grid grid-cols-3 gap-4">
                    {Object.entries(metrics).map(([key, value]) => (
                        <div key={key} className="p-4 rounded-lg bg-white/5">
                            <div className="text-xs text-slate-500 uppercase mb-1">{key}</div>
                            <div className="text-2xl font-bold text-slate-200">
                                {typeof value === 'number' ? value.toFixed(2) : value}
                            </div>
                        </div>
                    ))}
                </div>
            </Card>

            <Card title="Forecast (Next 20 Periods)" icon={TrendingUp}>
                <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={forecastData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                        <XAxis dataKey="period" stroke="#94a3b8" />
                        <YAxis stroke="#94a3b8" />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#1e293b',
                                border: '1px solid #334155',
                                borderRadius: '8px'
                            }}
                        />
                        <Line
                            type="monotone"
                            dataKey="forecast"
                            stroke="#4c6ef5"
                            strokeWidth={2}
                            dot={{ fill: '#4c6ef5' }}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </Card>

            {results.exogenous_features && results.exogenous_features.length > 0 && (
                <Card title="Features Used" icon={Database}>
                    <div className="flex flex-wrap gap-2">
                        {results.exogenous_features.map((feature) => (
                            <span key={feature} className="px-3 py-1 rounded-lg bg-blue-500/10 border border-blue-400/30 text-blue-400 text-sm">
                                {feature}
                            </span>
                        ))}
                    </div>
                </Card>
            )}
        </>
    );
}

// Marketing ROI Results
function MarketingROIResults({ results }) {
    const attribution = results.attribution || [];

    const chartData = attribution.map((item) => ({
        channel: item.channel,
        roi: item.roi ? parseFloat(item.roi.toFixed(2)) : 0,
        spend: item.total_spend || 0
    }));

    return (
        <>
            <Card title="ROI by Channel" icon={BarChart3}>
                <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                        <XAxis dataKey="channel" stroke="#94a3b8" angle={-45} textAnchor="end" height={100} />
                        <YAxis stroke="#94a3b8" />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#1e293b',
                                border: '1px solid #334155',
                                borderRadius: '8px'
                            }}
                        />
                        <Bar dataKey="roi" fill="#7c3aed" radius={[4, 4, 0, 0]} />
                    </BarChart>
                </ResponsiveContainer>
            </Card>

            <Card title="Channel Performance" icon={DollarSign}>
                <div className="space-y-3">
                    {attribution.slice(0, 5).map((item) => (
                        <div key={item.channel} className="p-4 rounded-lg bg-white/5 border border-white/10">
                            <div className="flex items-center justify-between mb-2">
                                <span className="font-medium text-slate-200">{item.channel}</span>
                                <span className={`text-sm font-bold ${item.roi > 1 ? 'text-emerald-400' : 'text-red-400'}`}>
                                    ROI: {item.roi ? item.roi.toFixed(2) : 'N/A'}
                                </span>
                            </div>
                            <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                    <span className="text-slate-500">Spend: </span>
                                    <span className="text-slate-300">${item.total_spend?.toFixed(2) || 0}</span>
                                </div>
                                <div>
                                    <span className="text-slate-500">Contribution: </span>
                                    <span className="text-slate-300">${item.contribution?.toFixed(2) || 0}</span>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </Card>
        </>
    );
}

// Inventory Optimization Results
function InventoryResults({ results }) {
    const skuAnalysis = results.sku_analysis || [];
    const scheduleSummary = results.schedule_summary || {};

    return (
        <>
            <Card title="Optimization Summary" icon={Package}>
                <div className="grid grid-cols-3 gap-4">
                    <div className="p-4 rounded-lg bg-emerald-500/10 border border-emerald-400/30">
                        <div className="text-xs text-emerald-400 uppercase mb-1">SKUs with Orders</div>
                        <div className="text-2xl font-bold text-emerald-400">
                            {scheduleSummary.total_skus_with_orders || 0}
                        </div>
                    </div>
                    <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-400/30">
                        <div className="text-xs text-blue-400 uppercase mb-1">Total Orders</div>
                        <div className="text-2xl font-bold text-blue-400">
                            {scheduleSummary.total_orders || 0}
                        </div>
                    </div>
                    <div className="p-4 rounded-lg bg-violet-500/10 border border-violet-400/30">
                        <div className="text-xs text-violet-400 uppercase mb-1">Total Quantity</div>
                        <div className="text-2xl font-bold text-violet-400">
                            {scheduleSummary.total_quantity?.toFixed(0) || 0}
                        </div>
                    </div>
                </div>
            </Card>

            <Card title="SKU Analysis (Top 10)" icon={BarChart2}>
                <div className="space-y-2">
                    {skuAnalysis.slice(0, 10).map((sku) => (
                        <div key={sku.sku} className="p-3 rounded-lg bg-white/5 border border-white/10">
                            <div className="flex items-center justify-between mb-2">
                                <span className="font-medium text-slate-200">{sku.sku}</span>
                                <span className="text-sm text-emerald-400">EOQ: {sku.eoq?.toFixed(2) || 'N/A'}</span>
                            </div>
                            <div className="grid grid-cols-3 gap-2 text-xs text-slate-400">
                                <div>Demand: {sku.demand_rate?.toFixed(2) || 'N/A'}</div>
                                <div>Order Cost: ${sku.ordering_cost?.toFixed(2) || 'N/A'}</div>
                                <div>Hold Cost: ${sku.holding_cost?.toFixed(2) || 'N/A'}</div>
                            </div>
                        </div>
                    ))}
                </div>
            </Card>
        </>
    );
}

// Supplier Risk Results
function SupplierRiskResults({ results }) {
    const supplierAnalysis = results.supplier_analysis || [];
    const allocationSummary = results.allocation_summary || {};

    const riskData = supplierAnalysis.slice(0, 10).map((s) => ({
        name: s.supplier,
        risk: parseFloat((s.risk_score * 100).toFixed(1))
    }));

    return (
        <>
            <Card title="Supplier Risk Scores" icon={Truck}>
                <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={riskData} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                        <XAxis type="number" stroke="#94a3b8" />
                        <YAxis dataKey="name" type="category" stroke="#94a3b8" width={100} />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#1e293b',
                                border: '1px solid #334155',
                                borderRadius: '8px'
                            }}
                        />
                        <Bar dataKey="risk" fill="#f59e0b" radius={[0, 4, 4, 0]} />
                    </BarChart>
                </ResponsiveContainer>
            </Card>

            <Card title="Supplier Details" icon={Info}>
                <div className="space-y-3">
                    {supplierAnalysis.slice(0, 5).map((supplier) => (
                        <div key={supplier.supplier} className="p-4 rounded-lg bg-white/5 border border-white/10">
                            <div className="flex items-center justify-between mb-2">
                                <span className="font-medium text-slate-200">{supplier.supplier}</span>
                                <span className={`text-sm font-bold ${supplier.risk_score > 0.6 ? 'text-red-400' : supplier.risk_score > 0.4 ? 'text-amber-400' : 'text-emerald-400'
                                    }`}>
                                    Risk: {(supplier.risk_score * 100).toFixed(1)}%
                                </span>
                            </div>
                            <div className="grid grid-cols-2 gap-2 text-xs">
                                <div className="text-slate-400">
                                    Lead Time: <span className="text-slate-300">{supplier.avg_lead_time?.toFixed(1) || 'N/A'} days</span>
                                </div>
                                <div className="text-slate-400">
                                    On-Time: <span className="text-slate-300">{(supplier.on_time_rate * 100)?.toFixed(1) || 'N/A'}%</span>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </Card>

            <Card title="Allocation Summary" icon={Target}>
                <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 rounded-lg bg-white/5">
                        <div className="text-xs text-slate-500 mb-1">Suppliers Used</div>
                        <div className="text-2xl font-bold text-slate-200">
                            {allocationSummary.suppliers_used || 0}
                        </div>
                    </div>
                    <div className="p-4 rounded-lg bg-white/5">
                        <div className="text-xs text-slate-500 mb-1">Total Allocated</div>
                        <div className="text-2xl font-bold text-slate-200">
                            {allocationSummary.total_allocated?.toFixed(0) || 0}
                        </div>
                    </div>
                </div>
            </Card>
        </>
    );
}

// Customer Segmentation Results
function CustomerSegmentationResults({ results }) {
    const segmentDistribution = results.segment_distribution || {};
    const predictive = results.predictive || {};

    // Convert segment distribution to chart data
    const segmentData = Object.entries(segmentDistribution).flatMap(([method, segments]) =>
        Object.entries(segments).map(([segment, count]) => ({
            method,
            segment: `Segment ${segment}`,
            count
        }))
    );

    return (
        <>
            <Card title="Segmentation Overview" icon={Users}>
                <div className="grid grid-cols-2 gap-4 mb-6">
                    <div className="p-4 rounded-lg bg-pink-500/10 border border-pink-400/30">
                        <div className="text-xs text-pink-400 uppercase mb-1">Total Customers</div>
                        <div className="text-2xl font-bold text-pink-400">{results.customers_count || 0}</div>
                    </div>
                    <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-400/30">
                        <div className="text-xs text-blue-400 uppercase mb-1">Segmentation Methods</div>
                        <div className="text-2xl font-bold text-blue-400">{Object.keys(segmentDistribution).length}</div>
                    </div>
                </div>

                {segmentData.length > 0 && (
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={segmentData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                            <XAxis dataKey="segment" stroke="#94a3b8" />
                            <YAxis stroke="#94a3b8" />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: '#1e293b',
                                    border: '1px solid #334155',
                                    borderRadius: '8px'
                                }}
                            />
                            <Legend />
                            <Bar dataKey="count" fill="#ec4899" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                )}
            </Card>

            {predictive.trained && (
                <Card title="Predictive Model Results" icon={Activity}>
                    <div className="grid grid-cols-2 gap-4 mb-4">
                        <div className="p-4 rounded-lg bg-white/5">
                            <div className="text-xs text-slate-500 mb-1">Model AUC</div>
                            <div className="text-2xl font-bold text-slate-200">
                                {predictive.metrics?.auc?.toFixed(3) || 'N/A'}
                            </div>
                        </div>
                        <div className="p-4 rounded-lg bg-red-500/10 border border-red-400/30">
                            <div className="text-xs text-red-400 uppercase mb-1">High Risk Customers</div>
                            <div className="text-2xl font-bold text-red-400">
                                {predictive.high_risk_customers_count || 0}
                            </div>
                        </div>
                    </div>
                </Card>
            )}

            {results.customer_sample && results.customer_sample.length > 0 && (
                <Card title="Customer Sample" icon={FileText}>
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b border-white/10">
                                    <th className="text-left py-2 text-slate-400">Customer ID</th>
                                    <th className="text-left py-2 text-slate-400">RFM Score</th>
                                    <th className="text-left py-2 text-slate-400">LTV</th>
                                </tr>
                            </thead>
                            <tbody>
                                {results.customer_sample.slice(0, 5).map((customer) => (
                                    <tr key={customer.customer_id} className="border-b border-white/5">
                                        <td className="py-2 text-slate-300">{customer.customer_id}</td>
                                        <td className="py-2 text-slate-300">{customer.RFM_score?.toFixed(1) || 'N/A'}</td>
                                        <td className="py-2 text-slate-300">${customer.LTV?.toFixed(2) || 0}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </Card>
            )}
        </>
    );
}

// Sidebar Component
function ResultsSidebar({ modelResults, selectedFile, loading }) {
    if (!selectedFile) {
        return (
            <Card title="Select a File" icon={Info}>
                <p className="text-sm text-slate-400">
                    Choose a file from the dropdown to view results and run the model.
                </p>
            </Card>
        );
    }

    if (loading) {
        return (
            <Card title="Loading..." icon={Loader2}>
                <div className="flex items-center justify-center py-8">
                    <Loader2 size={32} className="text-blue-400 animate-spin" />
                </div>
            </Card>
        );
    }

    if (!modelResults) {
        return (
            <Card title="No Results" icon={AlertCircle}>
                <p className="text-sm text-slate-400">
                    No results available yet. Run the model to generate results.
                </p>
            </Card>
        );
    }

    return (
        <>
            <Card title="Result Status" icon={CheckCircle}>
                <div className="space-y-3">
                    <div className="flex items-center gap-2 text-emerald-400">
                        <CheckCircle size={20} />
                        <span className="text-sm font-medium">Results Available</span>
                    </div>
                    <div className="p-3 rounded-lg bg-white/5 text-xs">
                        <div className="text-slate-500 mb-1">Created At</div>
                        <div className="text-slate-300">
                            {new Date(modelResults.created_at).toLocaleString()}
                        </div>
                    </div>
                </div>
            </Card>

            <Card title="Quick Actions" icon={Sparkles}>
                <div className="space-y-2">
                    <button className="w-full px-4 py-2 rounded-lg bg-blue-500/10 border border-blue-400/30 text-blue-400 hover:bg-blue-500/20 transition-all text-sm">
                        Download Results
                    </button>
                    <button className="w-full px-4 py-2 rounded-lg bg-white/5 border border-white/10 text-slate-300 hover:bg-white/10 transition-all text-sm">
                        Export Report
                    </button>
                </div>
            </Card>
        </>
    );
}

// Reusable Card Component
function Card({ title, icon: Icon, children }) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-2xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 backdrop-blur-xl p-6"
        >
            <div className="flex items-center gap-3 mb-4">
                {Icon && <Icon size={20} className="text-blue-400" />}
                <h3 className="text-lg font-semibold text-slate-100">{title}</h3>
            </div>
            {children}
        </motion.div>
    );
}