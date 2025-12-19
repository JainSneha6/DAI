import React from "react";
import { ChartCard } from "../pages/FileAnalysisPage";
import { BarChart3, TrendingUp, Activity } from "lucide-react";
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    AreaChart,
    Area,
    LineChart,
    Line
} from "recharts";

const CHART_COLORS = [
    "#4c6ef5", "#7c3aed", "#10b981", "#06b6d4", "#f59e0b",
    "#ef4444", "#ec4899", "#8b5cf6", "#14b8a6", "#f97316"
];

export default function NumericalTab({ data }) {
    const numericalSummary = data?.numerical_summary || {};
    const distributionData = data?.distribution_data || {};
    const boxPlotData = data?.box_plot_data || {};

    const columns = Object.keys(numericalSummary);

    if (columns.length === 0) {
        return (
            <div className="rounded-2xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 backdrop-blur-xl p-12 text-center">
                <Activity size={48} className="text-slate-500 mx-auto mb-4" />
                <p className="text-slate-400">No numerical columns found in this dataset.</p>
            </div>
        );
    }

    return (
        <div className="space-y-6">

            {/* Summary Statistics Table */}
            <ChartCard title="Summary Statistics" icon={BarChart3}>
                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b border-white/10">
                                <th className="text-left p-3 text-slate-300 font-medium">Column</th>
                                <th className="text-right p-3 text-slate-300 font-medium">Count</th>
                                <th className="text-right p-3 text-slate-300 font-medium">Mean</th>
                                <th className="text-right p-3 text-slate-300 font-medium">Std Dev</th>
                                <th className="text-right p-3 text-slate-300 font-medium">Min</th>
                                <th className="text-right p-3 text-slate-300 font-medium">Max</th>
                                <th className="text-right p-3 text-slate-300 font-medium">Sum</th>
                                <th className="text-right p-3 text-slate-300 font-medium">Missing</th>
                            </tr>
                        </thead>
                        <tbody>
                            {columns.map((col, idx) => {
                                const stats = numericalSummary[col];
                                return (
                                    <tr key={idx} className="border-b border-white/5 hover:bg-white/5">
                                        <td className="p-3 text-slate-200 font-medium">{col}</td>
                                        <td className="p-3 text-right text-slate-400">{stats.count?.toLocaleString()}</td>
                                        <td className="p-3 text-right text-slate-400">
                                            {stats.mean !== null ? Number(stats.mean).toFixed(2) : "-"}
                                        </td>
                                        <td className="p-3 text-right text-slate-400">
                                            {stats.std !== null ? Number(stats.std).toFixed(2) : "-"}
                                        </td>
                                        <td className="p-3 text-right text-slate-400">
                                            {stats.min !== null ? Number(stats.min).toFixed(2) : "-"}
                                        </td>
                                        <td className="p-3 text-right text-slate-400">
                                            {stats.max !== null ? Number(stats.max).toFixed(2) : "-"}
                                        </td>
                                        <td className="p-3 text-right text-slate-400">
                                            {stats.sum !== null ? Number(stats.sum).toFixed(2) : "-"}
                                        </td>
                                        <td className="p-3 text-right text-slate-400">{stats.missing_count || 0}</td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </ChartCard>

            {/* Distribution Charts */}
            {columns.slice(0, 6).map((col, idx) => {
                const dist = distributionData[col];
                if (!dist) return null;

                // Prepare histogram data
                const histogramData = dist.counts.map((count, i) => ({
                    bin: `${dist.bins[i].toFixed(1)}-${dist.bins[i + 1].toFixed(1)}`,
                    binStart: dist.bins[i],
                    count: count,
                    frequency: count
                }));

                return (
                    <ChartCard key={col} title={`Distribution: ${col}`} icon={TrendingUp} height="400px">
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">

                            {/* Statistics Panel */}
                            <div className="space-y-4">
                                <h4 className="text-sm font-semibold text-slate-300 mb-4">Statistics</h4>
                                <StatItem label="Mean" value={dist.stats.mean.toFixed(2)} />
                                <StatItem label="Median" value={dist.stats.median.toFixed(2)} />
                                <StatItem label="Std Dev" value={dist.stats.std.toFixed(2)} />
                                <StatItem label="Min" value={dist.stats.min.toFixed(2)} />
                                <StatItem label="Max" value={dist.stats.max.toFixed(2)} />
                                <div className="pt-4 border-t border-white/10">
                                    <StatItem
                                        label="Range"
                                        value={(dist.stats.max - dist.stats.min).toFixed(2)}
                                    />
                                </div>
                            </div>

                            {/* Histogram */}
                            <div className="lg:col-span-2">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={histogramData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                        <XAxis
                                            dataKey="bin"
                                            stroke="#94a3b8"
                                            tick={{ fill: '#94a3b8', fontSize: 11 }}
                                            angle={-45}
                                            textAnchor="end"
                                            height={80}
                                        />
                                        <YAxis
                                            stroke="#94a3b8"
                                            tick={{ fill: '#94a3b8' }}
                                        />
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                                                border: '1px solid rgba(255, 255, 255, 0.1)',
                                                borderRadius: '8px',
                                                color: '#e2e8f0'
                                            }}
                                        />
                                        <Bar dataKey="frequency" fill={CHART_COLORS[idx % CHART_COLORS.length]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </ChartCard>
                );
            })}

            {/* Box Plots */}
            <ChartCard title="Box Plot Comparison" icon={Activity} height="auto">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 p-4">
                    {columns.slice(0, 9).map((col, idx) => {
                        const boxData = boxPlotData[col];
                        if (!boxData) return null;

                        return (
                            <div key={col} className="p-4 rounded-lg bg-white/5 border border-white/10">
                                <h5 className="text-sm font-semibold text-slate-200 mb-4 truncate">{col}</h5>
                                <BoxPlot data={boxData} color={CHART_COLORS[idx % CHART_COLORS.length]} />
                            </div>
                        );
                    })}
                </div>
            </ChartCard>
        </div>
    );
}

// Helper Components
function StatItem({ label, value }) {
    return (
        <div className="flex items-center justify-between p-3 rounded-lg bg-white/5">
            <span className="text-xs text-slate-400">{label}</span>
            <span className="text-sm font-semibold text-slate-200">{value}</span>
        </div>
    );
}

function BoxPlot({ data, color }) {
    const { min, q1, median, q3, max } = data;
    const range = max - min;
    const scale = 100 / range;

    const positions = {
        min: 0,
        q1: (q1 - min) * scale,
        median: (median - min) * scale,
        q3: (q3 - min) * scale,
        max: 100
    };

    return (
        <div className="relative h-48 flex items-center">
            <div className="w-full relative" style={{ height: '60px' }}>

                {/* Whisker line */}
                <div
                    className="absolute top-1/2 h-0.5 bg-slate-600"
                    style={{
                        left: `${positions.min}%`,
                        right: `${100 - positions.max}%`,
                        transform: 'translateY(-50%)'
                    }}
                />

                {/* Box */}
                <div
                    className="absolute top-1/2 border-2 rounded"
                    style={{
                        left: `${positions.q1}%`,
                        width: `${positions.q3 - positions.q1}%`,
                        height: '40px',
                        transform: 'translateY(-50%)',
                        borderColor: color,
                        backgroundColor: `${color}20`
                    }}
                >
                    {/* Median line */}
                    <div
                        className="absolute top-0 bottom-0 w-0.5"
                        style={{
                            left: `${((median - q1) / (q3 - q1)) * 100}%`,
                            backgroundColor: color
                        }}
                    />
                </div>

                {/* Min whisker */}
                <div
                    className="absolute top-1/2 w-0.5 h-4 bg-slate-600"
                    style={{ left: `${positions.min}%`, transform: 'translateY(-50%)' }}
                />

                {/* Max whisker */}
                <div
                    className="absolute top-1/2 w-0.5 h-4 bg-slate-600"
                    style={{ left: `${positions.max}%`, transform: 'translateY(-50%)' }}
                />
            </div>

            {/* Labels */}
            <div className="absolute -bottom-2 w-full text-xs text-slate-500 flex justify-between">
                <span>{min.toFixed(1)}</span>
                <span>{max.toFixed(1)}</span>
            </div>
        </div>
    );
}