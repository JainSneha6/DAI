import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
    FileText,
    TrendingUp,
    BarChart3,
    Package,
    Truck,
    Users,
    Trash2,
    Eye,
    Sparkles,
    ChevronRight,
    CheckCircle,
    Clock,
    PlayCircle
} from "lucide-react";

// Model icon mapping
const MODEL_ICONS = {
    "Sales, Demand & Financial Forecasting Model": TrendingUp,
    "Marketing ROI & Attribution Model": BarChart3,
    "Inventory & Replenishment Optimization Model": Package,
    "Logistics & Supplier Risk Model": Truck,
    "Customer Segmentation & Modeling": Users,
};

const MODEL_TO_URL = {
    "Sales, Demand & Financial Forecasting Model": "sales-forecasting",
    "Marketing ROI & Attribution Model": "marketing-roi",
    "Inventory & Replenishment Optimization Model": "inventory-optimization",
    "Logistics & Supplier Risk Model": "supplier-risk",
    "Customer Segmentation & Modeling": "customer-segmentation",
};

const MODEL_COLORS = {
    "Sales, Demand & Financial Forecasting Model": "blue",
    "Marketing ROI & Attribution Model": "violet",
    "Inventory & Replenishment Optimization Model": "emerald",
    "Logistics & Supplier Risk Model": "amber",
    "Customer Segmentation & Modeling": "pink",
};

// Category colors
const categoryColors = {
    sales: "emerald",
    customer: "blue",
    product: "violet",
    financial: "amber",
    inventory: "cyan",
    marketing: "pink",
    operations: "indigo",
    finance: "amber",
    "supply chain": "emerald"
};

export default function FilesListWithModels() {
    const [files, setFiles] = useState([]);
    const [loading, setLoading] = useState(true);
    const [expandedFile, setExpandedFile] = useState(null);
    const [fileModels, setFileModels] = useState({});
    const navigate = useNavigate();

    useEffect(() => {
        fetchFiles();
    }, []);

    const fetchFiles = async () => {
        try {
            const response = await fetch("http://localhost:5000/api/files");
            const data = await response.json();

            console.log("Raw API response:", data);

            if (data.success) {
                const filesData = data.files || [];
                console.log("Files data:", filesData);

                // Normalize the files data to handle category as object
                const normalizedFiles = filesData
                    .filter(file => file && file.filename) // Use 'filename' not 'name'
                    .map(file => ({
                        name: file.filename, // Map filename to name
                        size: formatFileSize(file.size),
                        uploaded_at: file.uploaded_at,
                        rows: file.row_count ? parseInt(file.row_count) : null,
                        columns: file.columns ? file.columns.length : null,
                        // Extract category string from object
                        category: typeof file.category === 'object' && file.category
                            ? file.category.data_domain
                            : (typeof file.category === 'string' ? file.category : null),
                        // Extract suggested models - need to derive from category
                        suggested_models: file.suggested_models || (
                            file.category && typeof file.category === 'object' && file.category.data_domain
                                ? getSuggestedModelsForDomain(file.category.data_domain)
                                : []
                        )
                    }));

                console.log("Normalized files:", normalizedFiles);
                setFiles(normalizedFiles);

                // Fetch model history for each file
                normalizedFiles.forEach(file => {
                    fetchFileModels(file.name);
                });
            }
        } catch (error) {
            console.error("Error fetching files:", error);
        } finally {
            setLoading(false);
        }
    };

    // Helper function to get suggested models based on domain
    const getSuggestedModelsForDomain = (domain) => {
        const domainToModels = {
            "Sales": [
                "Sales, Demand & Financial Forecasting Model",
                "Pricing & Revenue Optimization Model"
            ],
            "Inventory": [
                "Inventory & Replenishment Optimization Model",
                "Sales, Demand & Financial Forecasting Model"
            ],
            "Marketing": [
                "Marketing ROI & Attribution Model",
                "Pricing & Revenue Optimization Model"
            ],
            "Customer": [
                "Customer Segmentation & Modeling",
                "Customer Value & Retention Model"
            ],
            "Finance": [
                "Sales, Demand & Financial Forecasting Model",
                "Pricing & Revenue Optimization Model"
            ],
            "Operations": [
                "Inventory & Replenishment Optimization Model",
                "Logistics & Supplier Risk Model"
            ],
            "Logistics": [
                "Logistics & Supplier Risk Model"
            ],
            "Supply Chain": [
                "Sales, Demand & Financial Forecasting Model",
                "Inventory & Replenishment Optimization Model",
                "Logistics & Supplier Risk Model"
            ]
        };

        return domainToModels[domain] || [];
    };

    // Helper to format file size
    const formatFileSize = (size) => {
        if (!size) return 'Unknown';
        if (typeof size === 'string' && size.includes('KB')) return size;
        // If it's a number, assume bytes
        const bytes = parseFloat(size);
        if (isNaN(bytes)) return size;
        return `${(bytes / 1024).toFixed(1)} KB`;
    };

    const fetchFileModels = async (filename) => {
        try {
            const response = await fetch(`http://localhost:5000/api/files/${filename}/models`);
            const data = await response.json();

            if (data.success && data.models) {
                setFileModels(prev => ({
                    ...prev,
                    [filename]: data.models
                }));
            }
        } catch (error) {
            console.error(`Error fetching models for ${filename}:`, error);
        }
    };

    const deleteFile = async (filename) => {
        if (!window.confirm(`Delete ${filename}?`)) return;

        try {
            const response = await fetch(`http://localhost:5000/api/files/${filename}`, {
                method: "DELETE"
            });
            const data = await response.json();
            if (data.success) {
                fetchFiles();
            }
        } catch (error) {
            console.error("Error deleting file:", error);
        }
    };

    const getCategoryColor = (category) => {
        if (!category || typeof category !== 'string') return "slate";
        const cat = category.toLowerCase();
        for (const [key, color] of Object.entries(categoryColors)) {
            if (cat.includes(key)) return color;
        }
        return "slate";
    };

    if (loading) {
        return (
            <div className="rounded-2xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 backdrop-blur-xl p-8 text-center">
                <div className="animate-spin w-8 h-8 border-2 border-blue-400 border-t-transparent rounded-full mx-auto mb-4" />
                <p className="text-slate-400">Loading files...</p>
            </div>
        );
    }

    if (files.length === 0) {
        return (
            <div className="rounded-2xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 backdrop-blur-xl p-8 text-center">
                <FileText size={48} className="text-slate-500 mx-auto mb-4" />
                <p className="text-slate-400">No files uploaded yet</p>
            </div>
        );
    }

    return (
        <div className="rounded-2xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 backdrop-blur-xl overflow-hidden">
            <div className="p-6 border-b border-white/10">
                <h2 className="text-xl font-semibold text-slate-100">Your Data Files</h2>
                <p className="text-sm text-slate-400 mt-1">{files.length} file{files.length !== 1 ? 's' : ''} ready for analysis</p>
            </div>

            <div className="divide-y divide-white/5">
                {files.map((file) => {
                    const models = fileModels[file.name] || [];
                    const suggestedModels = file.suggested_models || [];
                    const runModels = models.map(m => m.model_type);
                    const notRunModels = suggestedModels.filter(m => !runModels.includes(m));

                    return (
                        <FileCard
                            key={`file-${file.name}`}
                            file={file}
                            index={files.indexOf(file)}
                            models={models}
                            suggestedModels={suggestedModels}
                            notRunModels={notRunModels}
                            isExpanded={expandedFile === file.name}
                            onToggleExpand={() => setExpandedFile(expandedFile === file.name ? null : file.name)}
                            onDelete={() => deleteFile(file.name)}
                            onAnalyze={() => navigate(`/analysis/${file.name}`)}
                            onViewModel={(modelUrl) => navigate(`/models/${modelUrl}?file=${file.name}`)}
                            getCategoryColor={getCategoryColor}
                        />
                    );
                })}
            </div>
        </div>
    );
}

function FileCard({
    file,
    index,
    models,
    suggestedModels,
    notRunModels,
    isExpanded,
    onToggleExpand,
    onDelete,
    onAnalyze,
    onViewModel,
    getCategoryColor
}) {
    // Extract category string - it might be an object from Gemini analysis
    let categoryString = '';
    if (file.category) {
        if (typeof file.category === 'string') {
            categoryString = file.category;
        } else if (typeof file.category === 'object' && file.category.data_domain) {
            // If it's the Gemini analysis object, extract data_domain
            categoryString = file.category.data_domain;
        }
    }

    const color = getCategoryColor(categoryString);

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            className="group hover:bg-white/5 transition-all"
        >
            <div className="p-6">
                <div className="flex items-start gap-4">
                    {/* File Icon */}
                    <div className={`w-12 h-12 rounded-xl bg-${color}-500/10 border border-${color}-400/30 flex items-center justify-center flex-shrink-0`}>
                        <FileText size={24} className={`text-${color}-400`} />
                    </div>

                    {/* File Info */}
                    <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between gap-4 mb-2">
                            <div className="flex-1">
                                <h3 className="text-lg font-semibold text-slate-100 truncate">{file.name}</h3>
                                {categoryString && (
                                    <span className={`inline-block px-2 py-1 rounded text-xs font-medium bg-${color}-500/10 border border-${color}-400/30 text-${color}-400 mt-1`}>
                                        {categoryString}
                                    </span>
                                )}
                            </div>

                            {/* Action Buttons */}
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={onAnalyze}
                                    className="p-2 rounded-lg bg-blue-500/10 border border-blue-400/30 text-blue-400 hover:bg-blue-500/20 transition-all"
                                    title="Analyze File"
                                >
                                    <Eye size={18} />
                                </button>
                                <button
                                    onClick={onDelete}
                                    className="p-2 rounded-lg bg-red-500/10 border border-red-400/30 text-red-400 hover:bg-red-500/20 transition-all"
                                    title="Delete File"
                                >
                                    <Trash2 size={18} />
                                </button>
                            </div>
                        </div>

                        {/* File Details */}
                        <div className="flex flex-wrap gap-4 text-sm text-slate-400 mb-3">
                            {file.size && <span>{file.size}</span>}
                            {file.rows && <span>{file.rows.toLocaleString()} rows</span>}
                            {file.columns && <span>{file.columns} columns</span>}
                            {file.uploaded_at && <span>{new Date(file.uploaded_at).toLocaleDateString()}</span>}
                        </div>

                        {/* Models Section */}
                        {(models.length > 0 || suggestedModels.length > 0) && (
                            <div>
                                <button
                                    onClick={onToggleExpand}
                                    className="flex items-center gap-2 text-sm font-medium text-slate-300 hover:text-slate-100 transition-colors"
                                >
                                    <Sparkles size={16} className="text-violet-400" />
                                    <span>
                                        {models.length > 0 && `${models.length} Model${models.length !== 1 ? 's' : ''} Run`}
                                        {models.length > 0 && notRunModels.length > 0 && " • "}
                                        {notRunModels.length > 0 && `${notRunModels.length} Available`}
                                    </span>
                                    <ChevronRight
                                        size={16}
                                        className={`transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                                    />
                                </button>

                                <AnimatePresence>
                                    {isExpanded && (
                                        <motion.div
                                            initial={{ height: 0, opacity: 0 }}
                                            animate={{ height: "auto", opacity: 1 }}
                                            exit={{ height: 0, opacity: 0 }}
                                            transition={{ duration: 0.2 }}
                                            className="overflow-hidden"
                                        >
                                            <div className="mt-4 space-y-4">

                                                {/* Models that have been run */}
                                                {models.length > 0 && (
                                                    <div>
                                                        <div className="text-xs font-semibold text-slate-400 mb-2 flex items-center gap-2">
                                                            <CheckCircle size={14} className="text-emerald-400" />
                                                            Completed Models
                                                        </div>
                                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                                            {models.map((modelInfo) => {
                                                                const Icon = MODEL_ICONS[modelInfo.model_type] || Sparkles;
                                                                const modelUrl = MODEL_TO_URL[modelInfo.model_type];
                                                                const colorClass = MODEL_COLORS[modelInfo.model_type] || "blue";

                                                                return (
                                                                    <motion.button
                                                                        key={`model-completed-${file.name}-${modelInfo.model_type}`}
                                                                        initial={{ opacity: 0, scale: 0.95 }}
                                                                        animate={{ opacity: 1, scale: 1 }}
                                                                        onClick={() => modelUrl && onViewModel(modelUrl)}
                                                                        className="flex items-center gap-3 p-3 rounded-lg bg-emerald-500/10 border border-emerald-400/30 hover:bg-emerald-500/20 transition-all text-left group"
                                                                    >
                                                                        <div className={`w-8 h-8 rounded-lg bg-${colorClass}-500/10 border border-${colorClass}-400/30 flex items-center justify-center flex-shrink-0`}>
                                                                            <Icon size={16} className={`text-${colorClass}-400`} />
                                                                        </div>
                                                                        <div className="flex-1 min-w-0">
                                                                            <div className="text-sm font-medium text-slate-200 truncate">
                                                                                {modelInfo.model_type}
                                                                            </div>
                                                                            <div className="text-xs text-emerald-400 flex items-center gap-1">
                                                                                <CheckCircle size={12} />
                                                                                Results available
                                                                            </div>
                                                                        </div>
                                                                        <ChevronRight size={16} className="text-slate-400 group-hover:text-emerald-400 transition-colors" />
                                                                    </motion.button>
                                                                );
                                                            })}
                                                        </div>
                                                    </div>
                                                )}

                                                {/* Models that haven't been run yet */}
                                                {notRunModels.length > 0 && (
                                                    <div>
                                                        <div className="text-xs font-semibold text-slate-400 mb-2 flex items-center gap-2">
                                                            <Clock size={14} className="text-blue-400" />
                                                            Available Models
                                                        </div>
                                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                                            {notRunModels.map((model) => {
                                                                const Icon = MODEL_ICONS[model] || Sparkles;
                                                                const modelUrl = MODEL_TO_URL[model];
                                                                const colorClass = MODEL_COLORS[model] || "blue";

                                                                return (
                                                                    <motion.button
                                                                        key={`model-available-${file.name}-${model}`}
                                                                        initial={{ opacity: 0, scale: 0.95 }}
                                                                        animate={{ opacity: 1, scale: 1 }}
                                                                        onClick={() => modelUrl && onViewModel(modelUrl)}
                                                                        className="flex items-center gap-3 p-3 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 hover:border-blue-400/30 transition-all text-left group"
                                                                    >
                                                                        <div className={`w-8 h-8 rounded-lg bg-${colorClass}-500/10 border border-${colorClass}-400/30 flex items-center justify-center flex-shrink-0`}>
                                                                            <Icon size={16} className={`text-${colorClass}-400`} />
                                                                        </div>
                                                                        <div className="flex-1 min-w-0">
                                                                            <div className="text-sm font-medium text-slate-200 truncate">
                                                                                {model}
                                                                            </div>
                                                                            <div className="text-xs text-blue-400 flex items-center gap-1">
                                                                                <PlayCircle size={12} />
                                                                                Click to run
                                                                            </div>
                                                                        </div>
                                                                        <ChevronRight size={16} className="text-slate-400 group-hover:text-blue-400 transition-colors" />
                                                                    </motion.button>
                                                                );
                                                            })}
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </motion.div>
    );
}