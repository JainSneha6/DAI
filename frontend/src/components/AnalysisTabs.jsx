import React from "react";
import { ChartCard } from "../pages/FileAnalysisPage";
import { Activity, TrendingUp, Layers } from "lucide-react";
import {
    LineChart,
    Line,
    AreaChart,
    Area,
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

// Correlations Tab
export function CorrelationsTab({ data }) {
    const correlationMatrix = data?.correlation_matrix;

    if (!correlationMatrix || !correlationMatrix.columns || correlationMatrix.columns.length === 0) {
        return (
            <div className="rounded-2xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 backdrop-blur-xl p-12 text-center">
                <Activity size={48} className="text-slate-500 mx-auto mb-4" />
                <p className="text-slate-400">No correlation data available. Need at least 2 numeric columns.</p>
            </div>
        );
    }

    const columns = correlationMatrix.columns;
    const values = correlationMatrix.values;

    // Find strong correlations
    const strongCorrelations = [];
    for (let i = 0; i < columns.length; i++) {
        for (let j = i + 1; j < columns.length; j++) {
            const corr = values[i][j];
            if (Math.abs(corr) > 0.5) {
                strongCorrelations.push({
                    col1: columns[i],
                    col2: columns[j],
                    correlation: corr
                });
            }
        }
    }

    strongCorrelations.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));

    return (
        <div className="space-y-6">

            {/* Strong Correlations */}
            {strongCorrelations.length > 0 && (
                <ChartCard title="Strong Correlations (|r| > 0.5)" icon={Activity}>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {strongCorrelations.slice(0, 10).map((item, idx) => (
                            <div
                                key={idx}
                                className="p-4 rounded-lg bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10"
                            >
                                <div className="flex items-center justify-between mb-3">
                                    <div className="flex-1">
                                        <div className="text-sm font-medium text-slate-200 mb-1">{item.col1}</div>
                                        <div className="text-xs text-slate-400">↔</div>
                                        <div className="text-sm font-medium text-slate-200 mt-1">{item.col2}</div>
                                    </div>
                                    <div className="text-right">
                                        <div className={`text-2xl font-bold ${item.correlation > 0 ? 'text-emerald-400' : 'text-red-400'
                                            }`}>
                                            {item.correlation.toFixed(3)}
                                        </div>
                                        <div className="text-xs text-slate-500">
                                            {Math.abs(item.correlation) > 0.8 ? 'Very Strong' :
                                                Math.abs(item.correlation) > 0.6 ? 'Strong' : 'Moderate'}
                                        </div>
                                    </div>
                                </div>
                                <div className="w-full bg-white/5 rounded-full h-2">
                                    <div
                                        className={`h-2 rounded-full ${item.correlation > 0 ? 'bg-emerald-500' : 'bg-red-500'
                                            }`}
                                        style={{ width: `${Math.abs(item.correlation) * 100}%` }}
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                </ChartCard>
            )}

            {/* Correlation Matrix Heatmap */}
            <ChartCard title="Correlation Matrix" icon={Activity}>
                <div className="overflow-x-auto">
                    <table className="text-xs border-separate" style={{ borderSpacing: 0 }}>
                        <thead>
                            <tr>
                                {/* Empty corner cell with extra height for rotated labels */}
                                <th className="p-2 text-left text-slate-400 font-medium sticky left-0 bg-gradient-to-r from-[#0d111a] to-transparent" style={{ height: '120px', verticalAlign: 'bottom' }}>
                                    <div className="h-full"></div>
                                </th>
                                {columns.map((col, idx) => (
                                    <th
                                        key={idx}
                                        className="p-2 text-slate-400 font-medium min-w-[80px] relative"
                                        style={{ height: '120px', verticalAlign: 'bottom' }}
                                    >
                                        <div className="absolute bottom-2 left-2 transform -rotate-45 origin-bottom-left whitespace-nowrap">
                                            <span className="text-xs">{col}</span>
                                        </div>
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {columns.map((rowCol, i) => (
                                <tr key={i} className="border-t border-white/5">
                                    <td className="p-2 text-slate-300 font-medium sticky left-0 bg-gradient-to-r from-[#0d111a] to-transparent whitespace-nowrap">
                                        {rowCol}
                                    </td>
                                    {columns.map((colCol, j) => {
                                        const corr = values[i][j];
                                        const absCorr = Math.abs(corr);
                                        const intensity = absCorr * 0.8;
                                        const bgColor = corr > 0
                                            ? `rgba(16, 185, 129, ${intensity})`
                                            : `rgba(239, 68, 68, ${intensity})`;

                                        return (
                                            <td
                                                key={j}
                                                className="p-2 text-center border-l border-white/5"
                                                style={{ backgroundColor: bgColor }}
                                            >
                                                <span className={absCorr > 0.5 ? 'font-bold text-white' : 'text-slate-300'}>
                                                    {corr.toFixed(2)}
                                                </span>
                                            </td>
                                        );
                                    })}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </ChartCard>
        </div>
    );
}

// Time Series Tab
export function TimeSeriesTab({ data }) {
    const timeSeriesData = data?.time_series_data || {};
    const timeColumns = Object.keys(timeSeriesData);

    if (timeColumns.length === 0) {
        return (
            <div className="rounded-2xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 backdrop-blur-xl p-12 text-center">
                <TrendingUp size={48} className="text-slate-500 mx-auto mb-4" />
                <p className="text-slate-400">No time series data available in this dataset.</p>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {timeColumns.map((timeCol) => {
                const tsData = timeSeriesData[timeCol];
                const seriesNames = Object.keys(tsData.series);

                if (seriesNames.length === 0) return null;

                // Prepare data for charts
                const chartData = tsData.dates.map((date, idx) => {
                    const point = { date: new Date(date).toLocaleDateString() };
                    seriesNames.forEach(series => {
                        point[series] = tsData.series[series][idx];
                    });
                    return point;
                });

                return (
                    <div key={timeCol}>
                        {/* Line Chart */}
                        <ChartCard title={`Time Series: ${timeCol}`} icon={TrendingUp} height="500px">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                    <XAxis
                                        dataKey="date"
                                        stroke="#94a3b8"
                                        tick={{ fill: '#94a3b8', fontSize: 11 }}
                                        angle={-45}
                                        textAnchor="end"
                                        height={80}
                                    />
                                    <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8' }} />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: 'rgba(15, 23, 42, 0.9)',
                                            border: '1px solid rgba(255, 255, 255, 0.1)',
                                            borderRadius: '8px',
                                            color: '#e2e8f0'
                                        }}
                                    />
                                    <Legend />
                                    {seriesNames.slice(0, 5).map((series, idx) => (
                                        <Line
                                            key={series}
                                            type="monotone"
                                            dataKey={series}
                                            stroke={CHART_COLORS[idx % CHART_COLORS.length]}
                                            strokeWidth={2}
                                            dot={false}
                                        />
                                    ))}
                                </LineChart>
                            </ResponsiveContainer>
                        </ChartCard>

                        {/* Area Chart */}
                        <ChartCard title={`Area Chart: ${timeCol}`} icon={TrendingUp} height="500px">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={chartData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                    <XAxis
                                        dataKey="date"
                                        stroke="#94a3b8"
                                        tick={{ fill: '#94a3b8', fontSize: 11 }}
                                        angle={-45}
                                        textAnchor="end"
                                        height={80}
                                    />
                                    <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8' }} />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: 'rgba(15, 23, 42, 0.9)',
                                            border: '1px solid rgba(255, 255, 255, 0.1)',
                                            borderRadius: '8px',
                                            color: '#e2e8f0'
                                        }}
                                    />
                                    <Legend />
                                    {seriesNames.slice(0, 5).map((series, idx) => (
                                        <Area
                                            key={series}
                                            type="monotone"
                                            dataKey={series}
                                            stackId="1"
                                            stroke={CHART_COLORS[idx % CHART_COLORS.length]}
                                            fill={CHART_COLORS[idx % CHART_COLORS.length]}
                                            fillOpacity={0.6}
                                        />
                                    ))}
                                </AreaChart>
                            </ResponsiveContainer>
                        </ChartCard>
                    </div>
                );
            })}
        </div>
    );
}

// Grouped Analysis Tab - FIXED
export function GroupedTab({ data }) {
    const groupedAnalysis = data?.grouped_analysis || {};
    const categoryColumns = Object.keys(groupedAnalysis);

    if (categoryColumns.length === 0) {
        return (
            <div className="rounded-2xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 backdrop-blur-xl p-12 text-center">
                <Layers size={48} className="text-slate-500 mx-auto mb-4" />
                <p className="text-slate-400">No grouped analysis available for this dataset.</p>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {categoryColumns.map((catCol) => {
                const numericAnalyses = groupedAnalysis[catCol];
                const numericColumns = Object.keys(numericAnalyses);

                return (
                    <div key={catCol} className="space-y-6">
                        <h3 className="text-2xl font-bold text-slate-100">Grouped by: {catCol}</h3>

                        {numericColumns.map((numCol, idx) => {
                            const groups = numericAnalyses[numCol];
                            if (!groups || groups.length === 0) return null;

                            // Prepare chart data
                            const chartData = groups.map(g => ({
                                category: g.category_value !== null ? String(g.category_value) : "(null)",
                                count: g.count,
                                mean: g.mean,
                                sum: g.sum,
                                min: g.min,
                                max: g.max
                            }));

                            return (
                                <ChartCard
                                    key={numCol}
                                    title={`${numCol} by ${catCol}`}
                                    icon={Layers}
                                >
                                    {/* Bar Charts Section */}
                                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">

                                        {/* Mean Bar Chart */}
                                        <div>
                                            <h5 className="text-sm font-semibold text-slate-300 mb-4">Average {numCol}</h5>
                                            <div style={{ height: '400px' }}>
                                                <ResponsiveContainer width="100%" height="100%">
                                                    <BarChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 80 }}>
                                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                                        <XAxis
                                                            dataKey="category"
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
                                                            label={{ value: 'Mean', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                                                        />
                                                        <Tooltip
                                                            contentStyle={{
                                                                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                                                                border: '1px solid rgba(255, 255, 255, 0.1)',
                                                                borderRadius: '8px',
                                                                color: '#e2e8f0'
                                                            }}
                                                        />
                                                        <Bar
                                                            dataKey="mean"
                                                            fill={CHART_COLORS[idx % CHART_COLORS.length]}
                                                            radius={[4, 4, 0, 0]}
                                                        />
                                                    </BarChart>
                                                </ResponsiveContainer>
                                            </div>
                                        </div>

                                        {/* Sum Bar Chart */}
                                        <div>
                                            <h5 className="text-sm font-semibold text-slate-300 mb-4">Total {numCol}</h5>
                                            <div style={{ height: '400px' }}>
                                                <ResponsiveContainer width="100%" height="100%">
                                                    <BarChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 80 }}>
                                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                                        <XAxis
                                                            dataKey="category"
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
                                                            label={{ value: 'Sum', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                                                        />
                                                        <Tooltip
                                                            contentStyle={{
                                                                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                                                                border: '1px solid rgba(255, 255, 255, 0.1)',
                                                                borderRadius: '8px',
                                                                color: '#e2e8f0'
                                                            }}
                                                        />
                                                        <Bar
                                                            dataKey="sum"
                                                            fill={CHART_COLORS[(idx + 1) % CHART_COLORS.length]}
                                                            radius={[4, 4, 0, 0]}
                                                        />
                                                    </BarChart>
                                                </ResponsiveContainer>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Detailed Table - Now properly visible */}
                                    <div className="border-t border-white/10 pt-6">
                                        <h5 className="text-sm font-semibold text-slate-300 mb-4">Detailed Statistics</h5>
                                        <div className="overflow-x-auto">
                                            <table className="w-full text-sm">
                                                <thead>
                                                    <tr className="border-b border-white/10">
                                                        <th className="text-left p-3 text-slate-300 font-medium bg-white/5">{catCol}</th>
                                                        <th className="text-right p-3 text-slate-300 font-medium bg-white/5">Count</th>
                                                        <th className="text-right p-3 text-slate-300 font-medium bg-white/5">Mean</th>
                                                        <th className="text-right p-3 text-slate-300 font-medium bg-white/5">Sum</th>
                                                        <th className="text-right p-3 text-slate-300 font-medium bg-white/5">Min</th>
                                                        <th className="text-right p-3 text-slate-300 font-medium bg-white/5">Max</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {groups.map((g, gidx) => (
                                                        <tr key={gidx} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                                                            <td className="p-3 text-slate-200 font-medium">
                                                                {g.category_value !== null ? String(g.category_value) : "(null)"}
                                                            </td>
                                                            <td className="p-3 text-right text-slate-400">{g.count?.toLocaleString()}</td>
                                                            <td className="p-3 text-right text-slate-400">
                                                                {g.mean !== null ? Number(g.mean).toFixed(2) : "-"}
                                                            </td>
                                                            <td className="p-3 text-right text-slate-400">
                                                                {g.sum !== null ? Number(g.sum).toFixed(2) : "-"}
                                                            </td>
                                                            <td className="p-3 text-right text-slate-400">
                                                                {g.min !== null ? Number(g.min).toFixed(2) : "-"}
                                                            </td>
                                                            <td className="p-3 text-right text-slate-400">
                                                                {g.max !== null ? Number(g.max).toFixed(2) : "-"}
                                                            </td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                </ChartCard>
                            );
                        })}
                    </div>
                );
            })}
        </div>
    );
}