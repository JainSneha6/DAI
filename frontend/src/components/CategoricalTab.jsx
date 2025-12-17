import React from "react";
import { ChartCard } from "../pages/FileAnalysisPage";
import { PieChart as PieChartIcon, BarChart3 } from "lucide-react";
import {
    PieChart,
    Pie,
    Cell,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer
} from "recharts";

const CHART_COLORS = [
    "#4c6ef5", "#7c3aed", "#10b981", "#06b6d4", "#f59e0b",
    "#ef4444", "#ec4899", "#8b5cf6", "#14b8a6", "#f97316"
];

export default function CategoricalTab({ data }) {
    const categoricalSummary = data?.categorical_summary || {};
    const categoricalFrequency = data?.categorical_frequency || {};

    const columns = Object.keys(categoricalSummary);

    if (columns.length === 0) {
        return (
            <div className="rounded-2xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 backdrop-blur-xl p-12 text-center">
                <PieChartIcon size={48} className="text-slate-500 mx-auto mb-4" />
                <p className="text-slate-400">No categorical columns found in this dataset.</p>
            </div>
        );
    }

    // Helper to safely parse top_values if they come as strings
    const parseTopValues = (topValues) => {
        if (!topValues || !Array.isArray(topValues)) return [];

        // Check if first item is a string that needs parsing
        if (topValues.length > 0 && typeof topValues[0] === 'string') {
            try {
                return topValues.map(str => {
                    const jsonStr = str.replace(/'/g, '"');
                    return JSON.parse(jsonStr);
                });
            } catch (e) {
                console.error("Failed to parse top_values:", e);
                return [];
            }
        }

        // Already proper objects
        return topValues;
    };

    // Helper to safely get numeric values
    const safeNumber = (val, defaultVal = 0) => {
        if (val === null || val === undefined) return defaultVal;
        const parsed = typeof val === 'string' ? parseFloat(val) : val;
        return isNaN(parsed) ? defaultVal : parsed;
    };

    return (
        <div className="space-y-6">

            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {columns.map((col, idx) => {
                    const summary = categoricalSummary[col];
                    const topValues = parseTopValues(summary.top_values);
                    const uniqueCount = safeNumber(summary.unique_count, 0);
                    const missingCount = safeNumber(summary.missing_count, 0);

                    return (
                        <div
                            key={col}
                            className="rounded-xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 backdrop-blur-xl p-6"
                        >
                            <h4 className="text-lg font-semibold text-slate-100 mb-4 truncate" title={col}>
                                {col}
                            </h4>

                            <div className="space-y-3">
                                <div className="flex items-center justify-between p-3 rounded-lg bg-white/5">
                                    <span className="text-sm text-slate-400">Unique Values</span>
                                    <span className="text-lg font-bold text-blue-400">
                                        {uniqueCount.toLocaleString()}
                                    </span>
                                </div>

                                <div className="flex items-center justify-between p-3 rounded-lg bg-white/5">
                                    <span className="text-sm text-slate-400">Missing</span>
                                    <span className="text-lg font-bold text-red-400">
                                        {missingCount.toLocaleString()}
                                    </span>
                                </div>

                                {/* Top Values */}
                                <div className="pt-3 border-t border-white/10">
                                    <div className="text-xs font-semibold text-slate-400 mb-2">Top Values</div>
                                    <div className="space-y-2 max-h-40 overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-white/10">
                                        {topValues.slice(0, 5).map((item, i) => {
                                            const count = safeNumber(item.count, 0);
                                            const pct = safeNumber(item.pct, 0);
                                            const value = item.value !== null && item.value !== undefined ? String(item.value) : "(null)";

                                            return (
                                                <div key={i} className="flex items-center justify-between text-xs">
                                                    <span className="text-slate-300 truncate flex-1 mr-2" title={value}>
                                                        {value}
                                                    </span>
                                                    <span className="text-slate-500 whitespace-nowrap">
                                                        {count.toLocaleString()} ({(pct * 100).toFixed(1)}%)
                                                    </span>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Frequency Charts */}
            {columns.slice(0, 6).map((col, idx) => {
                const frequency = categoricalFrequency[col];
                if (!frequency || !frequency.labels || frequency.labels.length === 0) return null;

                // Prepare data for charts
                const chartData = frequency.labels.map((label, i) => ({
                    name: String(label).substring(0, 30), // Truncate long labels
                    value: safeNumber(frequency.values[i], 0),
                    fullName: String(label)
                })).filter(item => item.value > 0); // Filter out zero values

                const total = chartData.reduce((sum, item) => sum + item.value, 0);

                if (chartData.length === 0 || total === 0) return null;

                return (
                    <ChartCard key={col} title={`Frequency Distribution: ${col}`} icon={BarChart3}>
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

                            {/* Bar Chart - FIXED: Using vertical orientation */}
                            <div style={{ height: '400px' }}>
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 80 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                        <XAxis
                                            dataKey="name"
                                            stroke="#94a3b8"
                                            tick={{ fill: '#94a3b8', fontSize: 11 }}
                                            angle={-45}
                                            textAnchor="end"
                                            height={80}
                                            interval={0}
                                        />
                                        <YAxis
                                            stroke="#94a3b8"
                                            tick={{ fill: '#94a3b8' }}
                                            label={{ value: 'Count', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                                        />
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                                                border: '1px solid rgba(255, 255, 255, 0.1)',
                                                borderRadius: '8px',
                                                color: '#e2e8f0'
                                            }}
                                            formatter={(value, name, props) => [
                                                `${value.toLocaleString()} (${((value / total) * 100).toFixed(1)}%)`,
                                                props.payload.fullName
                                            ]}
                                        />
                                        <Bar
                                            dataKey="value"
                                            fill={CHART_COLORS[idx % CHART_COLORS.length]}
                                            radius={[4, 4, 0, 0]}
                                        />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Pie Chart */}
                            <div style={{ height: '400px' }}>
                                <ResponsiveContainer width="100%" height="100%">
                                    <PieChart>
                                        <Pie
                                            data={chartData}
                                            cx="50%"
                                            cy="50%"
                                            labelLine={false}
                                            label={renderCustomLabel}
                                            outerRadius={120}
                                            fill="#8884d8"
                                            dataKey="value"
                                        >
                                            {chartData.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                                            ))}
                                        </Pie>
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                                                border: '1px solid rgba(255, 255, 255, 0.1)',
                                                borderRadius: '8px',
                                                color: '#e2e8f0'
                                            }}
                                            formatter={(value, name, props) => [
                                                `${value.toLocaleString()} (${((value / total) * 100).toFixed(1)}%)`,
                                                props.payload.fullName
                                            ]}
                                        />
                                    </PieChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </ChartCard>
                );
            })}

            {/* Detailed Value Table */}
            <ChartCard title="Detailed Value Counts" icon={PieChartIcon}>
                <div className="space-y-6">
                    {columns.slice(0, 5).map((col) => {
                        const summary = categoricalSummary[col];
                        const topValues = parseTopValues(summary.top_values);

                        if (!topValues || topValues.length === 0) return null;

                        return (
                            <div key={col}>
                                <h5 className="text-sm font-semibold text-slate-300 mb-3">{col}</h5>
                                <div className="overflow-x-auto">
                                    <table className="w-full text-sm">
                                        <thead>
                                            <tr className="border-b border-white/10">
                                                <th className="text-left p-3 text-slate-400 font-medium">Value</th>
                                                <th className="text-right p-3 text-slate-400 font-medium">Count</th>
                                                <th className="text-right p-3 text-slate-400 font-medium">Percentage</th>
                                                <th className="text-left p-3 text-slate-400 font-medium">Visual</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {topValues.slice(0, 10).map((item, idx) => {
                                                const count = safeNumber(item.count, 0);
                                                const pct = safeNumber(item.pct, 0);
                                                const value = item.value !== null && item.value !== undefined ? String(item.value) : "(null)";

                                                return (
                                                    <tr key={idx} className="border-b border-white/5 hover:bg-white/5">
                                                        <td className="p-3 text-slate-200" title={value}>
                                                            {value}
                                                        </td>
                                                        <td className="p-3 text-right text-slate-400">
                                                            {count.toLocaleString()}
                                                        </td>
                                                        <td className="p-3 text-right text-slate-400">
                                                            {(pct * 100).toFixed(2)}%
                                                        </td>
                                                        <td className="p-3">
                                                            <div className="w-full bg-white/5 rounded-full h-2">
                                                                <div
                                                                    className="h-2 rounded-full transition-all duration-500"
                                                                    style={{
                                                                        width: `${pct * 100}%`,
                                                                        backgroundColor: CHART_COLORS[idx % CHART_COLORS.length]
                                                                    }}
                                                                />
                                                            </div>
                                                        </td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </ChartCard>
        </div>
    );
}

// Custom label for pie chart
const renderCustomLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }) => {
    if (percent < 0.05) return null; // Don't show labels for small slices

    const RADIAN = Math.PI / 180;
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    return (
        <text
            x={x}
            y={y}
            fill="white"
            textAnchor={x > cx ? 'start' : 'end'}
            dominantBaseline="central"
            fontSize="12"
            fontWeight="bold"
        >
            {`${(percent * 100).toFixed(0)}%`}
        </text>
    );
};