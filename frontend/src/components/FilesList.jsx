import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { FileText, Download, Loader2, AlertCircle, CheckCircle2, BarChart3 } from "lucide-react";

export default function FilesList() {
    const [files, setFiles] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const navigate = useNavigate();

    useEffect(() => {
        setLoading(true);
        setError(null);
        fetch("http://localhost:5000/api/files")
            .then((r) => r.json())
            .then((j) => {
                if (j && j.success && Array.isArray(j.files)) {
                    setFiles(j.files);
                } else {
                    setFiles([]);
                }
            })
            .catch((e) => {
                console.error("Failed to fetch files", e);
                setError("Failed to load files");
                setFiles([]);
            })
            .finally(() => setLoading(false));
    }, []);

    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center py-12">
                <Loader2 className="w-8 h-8 text-blue-400 animate-spin mb-4" />
                <p className="text-slate-400 text-sm">Loading files...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex flex-col items-center justify-center py-12">
                <div className="w-12 h-12 rounded-full bg-red-500/10 border border-red-400/30 flex items-center justify-center mb-4">
                    <AlertCircle className="w-6 h-6 text-red-400" />
                </div>
                <p className="text-red-400 text-sm">{error}</p>
            </div>
        );
    }

    if (files.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center py-12">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-slate-700/20 to-slate-800/20 border border-white/10 flex items-center justify-center mb-4">
                    <FileText className="w-8 h-8 text-slate-500" strokeWidth={1.5} />
                </div>
                <p className="text-slate-400 text-sm text-center">No files uploaded yet</p>
            </div>
        );
    }

    return (
        <div>
            <div className="mb-6">
                <h3 className="text-xl font-semibold text-slate-100 mb-2">Uploaded Files</h3>
                <p className="text-sm text-slate-400">
                    {files.length} file{files.length > 1 ? "s" : ""} processed and categorized
                </p>
            </div>

            <div className="space-y-4 max-h-[600px] overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-white/10 pr-2">
                {files.map((f, index) => {
                    const filename = f.filename || "unknown-file";

                    // Category extraction
                    let category = "Unknown";
                    if (typeof f.category === "string") {
                        category = f.category;
                    } else if (f.category && typeof f.category === "object") {
                        category = f.category.data_domain || f.category.category || "Unknown";
                    }

                    const rowCount = f.row_count || 0;
                    const columns = Array.isArray(f.columns) ? f.columns : [];
                    const classification =
                        f.classification && typeof f.classification === "object"
                            ? f.classification
                            : null;

                    // Category color coding
                    const getCategoryColor = (cat) => {
                        const lower = cat.toLowerCase();
                        if (lower.includes("sales") || lower.includes("revenue")) return "emerald";
                        if (lower.includes("customer") || lower.includes("user")) return "blue";
                        if (lower.includes("product") || lower.includes("inventory")) return "violet";
                        if (lower.includes("financial") || lower.includes("finance")) return "amber";
                        return "slate";
                    };

                    const categoryColor = getCategoryColor(category);

                    return (
                        <motion.div
                            key={filename}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.3, delay: index * 0.05 }}
                            className="group p-5 rounded-xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 hover:border-blue-400/30 transition-all duration-300 cursor-pointer"
                            onClick={() => navigate(`/analysis/${encodeURIComponent(filename)}`)}
                        >
                            <div className="flex items-start justify-between gap-4">

                                {/* File Info */}
                                <div className="flex-1 min-w-0">

                                    {/* Header */}
                                    <div className="flex items-start gap-3 mb-3">
                                        <div className={`w-10 h-10 rounded-lg bg-gradient-to-br from-${categoryColor}-500/10 to-${categoryColor}-600/10 border border-${categoryColor}-400/30 flex items-center justify-center flex-shrink-0`}>
                                            <FileText size={20} className={`text-${categoryColor}-400`} strokeWidth={1.5} />
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <h4 className="font-semibold text-slate-100 truncate mb-1 group-hover:text-blue-400 transition-colors">
                                                {filename}
                                            </h4>

                                            {/* Category Badge */}
                                            <div className="flex items-center gap-2 flex-wrap">
                                                <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-${categoryColor}-500/10 border border-${categoryColor}-400/30 text-${categoryColor}-400`}>
                                                    <CheckCircle2 size={12} />
                                                    {category}
                                                </span>
                                                {classification?.confidence != null && (
                                                    <span className="text-xs text-slate-500">
                                                        {Math.round(classification.confidence * 100)}% confidence
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                    </div>

                                    {/* Stats */}
                                    <div className="flex items-center gap-4 text-xs text-slate-400 mb-3">
                                        <div className="flex items-center gap-1.5">
                                            <div className="w-1 h-1 rounded-full bg-blue-400" />
                                            <span>{rowCount.toLocaleString()} rows</span>
                                        </div>
                                        <div className="flex items-center gap-1.5">
                                            <div className="w-1 h-1 rounded-full bg-violet-400" />
                                            <span>{columns.length} columns</span>
                                        </div>
                                    </div>

                                    {/* Columns Preview */}
                                    {columns.length > 0 && (
                                        <div className="p-3 rounded-lg bg-white/5 border border-white/10">
                                            <div className="text-xs text-slate-400 mb-2 font-medium">
                                                Columns:
                                            </div>
                                            <div className="flex flex-wrap gap-2">
                                                {columns.slice(0, 6).map((col, idx) => (
                                                    <span
                                                        key={idx}
                                                        className="px-2 py-1 rounded text-xs bg-slate-700/30 text-slate-300 border border-white/10"
                                                    >
                                                        {col}
                                                    </span>
                                                ))}
                                                {columns.length > 6 && (
                                                    <span className="px-2 py-1 rounded text-xs text-slate-500">
                                                        +{columns.length - 6} more
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {/* Action Buttons */}
                                <div className="flex flex-col gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            navigate(`/analysis/${encodeURIComponent(filename)}`);
                                        }}
                                        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-blue-500 to-violet-500 hover:from-blue-600 hover:to-violet-600 transition-all text-sm text-white font-medium whitespace-nowrap"
                                    >
                                        <BarChart3 size={16} />
                                        Analyze
                                    </button>

                                    <a
                                        onClick={(e) => e.stopPropagation()}
                                        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 hover:border-blue-400/30 transition-all text-sm text-slate-300 hover:text-blue-400 whitespace-nowrap"
                                        href={`/models/${encodeURIComponent(filename)}`}
                                        target="_blank"
                                        rel="noreferrer"
                                    >
                                        <Download size={16} />
                                        Download
                                    </a>
                                </div>
                            </div>
                        </motion.div>
                    );
                })}
            </div>
        </div>
    );
}