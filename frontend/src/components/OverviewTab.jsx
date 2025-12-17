import React from "react";
import { ChartCard } from "../pages/FileAnalysisPage";
import { Table, Grid3x3 } from "lucide-react";

// Overview Tab - Fixed to handle sample rows properly
export default function OverviewTab({ data }) {
    // Parse sample rows if they come as strings
    const getSampleRows = () => {
        if (!data?.sample_rows || data.sample_rows.length === 0) {
            return [];
        }

        // Check if sample_rows are strings that need parsing
        if (typeof data.sample_rows[0] === 'string') {
            try {
                // Try to parse as JSON-like strings
                return data.sample_rows.map(rowStr => {
                    // Replace single quotes with double quotes for valid JSON
                    const jsonStr = rowStr.replace(/'/g, '"');
                    return JSON.parse(jsonStr);
                });
            } catch (e) {
                console.error("Failed to parse sample rows:", e);
                return [];
            }
        }

        // Already proper objects
        return data.sample_rows;
    };

    const sampleRows = getSampleRows();
    const columns = data?.columns || [];

    return (
        <div className="space-y-6">

            {/* Sample Data */}
            <ChartCard title="Sample Data" icon={Table}>
                {sampleRows.length === 0 ? (
                    <div className="text-center py-8 text-slate-400">
                        No sample data available
                    </div>
                ) : (
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b border-white/10">
                                    {columns.slice(0, 10).map((col, idx) => (
                                        <th key={idx} className="text-left p-3 text-slate-300 font-medium whitespace-nowrap">
                                            {col}
                                        </th>
                                    ))}
                                    {columns.length > 10 && (
                                        <th className="text-left p-3 text-slate-300 font-medium">
                                            +{columns.length - 10} more...
                                        </th>
                                    )}
                                </tr>
                            </thead>
                            <tbody>
                                {sampleRows.map((row, idx) => (
                                    <tr key={idx} className="border-b border-white/5 hover:bg-white/5">
                                        {columns.slice(0, 10).map((col, vidx) => {
                                            const val = row[col];
                                            return (
                                                <td key={vidx} className="p-3 text-slate-400 whitespace-nowrap">
                                                    {val !== null && val !== undefined ? String(val) : "-"}
                                                </td>
                                            );
                                        })}
                                        {columns.length > 10 && (
                                            <td className="p-3 text-slate-500">...</td>
                                        )}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </ChartCard>

            {/* Column Types */}
            <ChartCard title="Column Data Types" icon={Grid3x3}>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                    {Object.entries(data?.dtypes || {}).map(([col, dtype], idx) => (
                        <div key={idx} className="p-3 rounded-lg bg-white/5 border border-white/10">
                            <div className="text-sm font-medium text-slate-200 mb-1 truncate" title={col}>
                                {col}
                            </div>
                            <div className="text-xs text-slate-400">{dtype}</div>
                        </div>
                    ))}
                </div>
            </ChartCard>

            {/* Data Quality Summary */}
            <ChartCard title="Data Quality" icon={Grid3x3}>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <QualityCard
                        label="Total Rows"
                        value={data?.basic_info?.row_count?.toLocaleString() || "0"}
                        color="blue"
                    />
                    <QualityCard
                        label="Total Columns"
                        value={data?.basic_info?.column_count || "0"}
                        color="violet"
                    />
                    <QualityCard
                        label="Memory Usage"
                        value={data?.basic_info?.memory_usage || "0 MB"}
                        color="emerald"
                    />
                    <QualityCard
                        label="Data Types"
                        value={Object.keys(data?.dtypes || {}).length}
                        color="cyan"
                    />
                </div>
            </ChartCard>

            {/* Column Statistics Overview */}
            <ChartCard title="Column Overview" icon={Table}>
                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b border-white/10">
                                <th className="text-left p-3 text-slate-300 font-medium">Column Name</th>
                                <th className="text-left p-3 text-slate-300 font-medium">Data Type</th>
                                <th className="text-left p-3 text-slate-300 font-medium">Category</th>
                                <th className="text-right p-3 text-slate-300 font-medium">Sample Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            {columns.map((col, idx) => {
                                const dtype = data?.dtypes?.[col] || "unknown";
                                const isNumeric = data?.numerical_summary?.[col];
                                const isCategorical = data?.categorical_summary?.[col];
                                const isTime = data?.time_series_data?.[col];

                                let category = "Other";
                                let categoryColor = "slate";

                                if (isNumeric) {
                                    category = "Numeric";
                                    categoryColor = "blue";
                                } else if (isCategorical) {
                                    category = "Categorical";
                                    categoryColor = "violet";
                                } else if (isTime) {
                                    category = "Time";
                                    categoryColor = "emerald";
                                }

                                // Get sample value
                                const sampleValue = sampleRows[0]?.[col];

                                return (
                                    <tr key={idx} className="border-b border-white/5 hover:bg-white/5">
                                        <td className="p-3 text-slate-200 font-medium">{col}</td>
                                        <td className="p-3 text-slate-400">
                                            <code className="px-2 py-1 rounded bg-slate-800/50 text-xs">{dtype}</code>
                                        </td>
                                        <td className="p-3">
                                            <span className={`inline-flex px-2 py-1 rounded-full text-xs font-medium bg-${categoryColor}-500/10 border border-${categoryColor}-400/30 text-${categoryColor}-400`}>
                                                {category}
                                            </span>
                                        </td>
                                        <td className="p-3 text-right text-slate-400 truncate max-w-xs" title={String(sampleValue)}>
                                            {sampleValue !== null && sampleValue !== undefined ? String(sampleValue) : "-"}
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </ChartCard>
        </div>
    );
}

function QualityCard({ label, value, color }) {
    const colorClasses = {
        blue: "from-blue-500/10 to-blue-600/10 border-blue-400/30 text-blue-400",
        violet: "from-violet-500/10 to-violet-600/10 border-violet-400/30 text-violet-400",
        emerald: "from-emerald-500/10 to-emerald-600/10 border-emerald-400/30 text-emerald-400",
        cyan: "from-cyan-500/10 to-cyan-600/10 border-cyan-400/30 text-cyan-400"
    };

    return (
        <div className={`rounded-xl bg-gradient-to-br ${colorClasses[color]} border backdrop-blur-xl p-4`}>
            <div className="text-xs text-slate-400 mb-1">{label}</div>
            <div className="text-2xl font-bold text-slate-100">{value}</div>
        </div>
    );
}