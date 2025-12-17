import React, { useState, useRef, useEffect } from 'react';
import { Send, TrendingUp, DollarSign, BarChart3, Zap, Target, PieChart, Bot, User, Sparkles, Loader2, ArrowUpRight, ArrowDownRight } from 'lucide-react';

export default function MarketingChat() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [availableModels, setAvailableModels] = useState([]);
    const [campaignPlan, setCampaignPlan] = useState(null);
    const [isCampaignLoading, setIsCampaignLoading] = useState(true);
    const messagesEndRef = useRef(null);

    useEffect(() => {
        fetchAvailableModels();
        fetchCampaignPlan();
    }, []);

    const fetchAvailableModels = async () => {
        try {
            const response = await fetch('http://localhost:5000/api/marketing/chat/models');
            const data = await response.json();
            if (data.success) {
                setAvailableModels(data.models);
            }
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    };

    const fetchCampaignPlan = async () => {
        try {
            setIsCampaignLoading(true);
            const response = await fetch('http://localhost:5000/api/marketing/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: 'show campaign plan' })
            });
            const data = await response.json();
            if (data.success && data.campaign_plan) {
                setCampaignPlan(data.campaign_plan);
            }
        } catch (error) {
            console.error('Failed to load campaign plan:', error);
        } finally {
            setIsCampaignLoading(false);
        }
    };

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMessage = {
            role: 'user',
            content: input,
            timestamp: new Date()
        };

        setMessages((prev) => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await fetch('http://localhost:5000/api/marketing/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: input })
            });

            const data = await response.json();

            if (data.success) {
                const assistantMessage = {
                    role: 'assistant',
                    content: data.answer,
                    timestamp: new Date(),
                    intent: data.intent,
                    roiData: data.roi_data,
                    attributionData: data.attribution_data,
                    optimizationData: data.optimization_data,
                    simulationData: data.simulation_data,
                    modelInfo: data.model_info,
                    sources: data.sources
                };
                setMessages((prev) => [...prev, assistantMessage]);
            } else {
                setMessages((prev) => [
                    ...prev,
                    {
                        role: 'assistant',
                        content: `I encountered an error: ${data.error}`,
                        timestamp: new Date(),
                        isError: true
                    }
                ]);
            }
        } catch (error) {
            setMessages((prev) => [
                ...prev,
                {
                    role: 'assistant',
                    content: `Failed to get response: ${error.message}`,
                    timestamp: new Date(),
                    isError: true
                }
            ]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const quickActions = [
        { label: 'Show ROI by channel', query: 'Show me ROI analysis for all channels' },
        { label: 'Optimize my budget', query: 'How should I optimize my marketing budget?' },
        { label: 'Compare channels', query: 'Compare performance of my top channels' },
        { label: 'Attribution analysis', query: 'Show me marketing attribution breakdown' }
    ];

    // ROI Visualization Component
    const ROIVisualization = ({ data }) => {
        if (!data || !data.roi_by_channel) return null;

        const channels = Object.entries(data.roi_by_channel)
            .sort((a, b) => Math.abs(b[1].relative_impact) - Math.abs(a[1].relative_impact))
            .slice(0, 5);

        const maxImpact = Math.max(...channels.map(([_, d]) => Math.abs(d.relative_impact)));

        return (
            <div className="mt-4 p-4 bg-gradient-to-br from-white/5 to-white/[0.02] rounded-lg border border-white/10 backdrop-blur-xl">
                <h4 className="font-semibold text-slate-100 text-sm mb-3 flex items-center gap-2">
                    <DollarSign className="w-4 h-4 text-emerald-300" />
                    Channel ROI Analysis
                </h4>

                <div className="space-y-2">
                    {channels.map(([channel, channelData], idx) => {
                        const percentage = (Math.abs(channelData.relative_impact) / maxImpact) * 100;
                        const isPositive = channelData.coefficient > 0;
                        const contribution = channelData.contribution_pct || 0;

                        return (
                            <div key={idx}>
                                <div className="flex items-center justify-between mb-1">
                                    <span className="text-xs font-semibold text-slate-200">{channel}</span>
                                    <span className="text-xs text-slate-400">{contribution.toFixed(1)}%</span>
                                </div>
                                <div className="relative bg-slate-800/40 rounded-full h-6 overflow-hidden border border-slate-700">
                                    <div
                                        className={`h-full rounded-full transition-all duration-500 flex items-center justify-end pr-2 ${isPositive
                                            ? 'bg-gradient-to-r from-emerald-500 to-teal-500'
                                            : 'bg-gradient-to-r from-red-500 to-rose-500'
                                            }`}
                                        style={{ width: `${percentage}%` }}
                                    >
                                        <span className="text-xs font-bold text-white">{channelData.coefficient.toFixed(3)}</span>
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>
        );
    };

    // Attribution Chart Component
    const AttributionChart = ({ data }) => {
        if (!data || !data.roi_by_channel) return null;

        const channels = Object.entries(data.roi_by_channel)
            .sort((a, b) => (b[1].contribution_pct || 0) - (a[1].contribution_pct || 0))
            .slice(0, 5);

        return (
            <div className="mt-4 p-4 bg-gradient-to-br from-white/5 to-white/[0.02] rounded-lg border border-white/10 backdrop-blur-xl">
                <h4 className="font-semibold text-slate-100 text-sm mb-3 flex items-center gap-2">
                    <PieChart className="w-4 h-4 text-violet-300" />
                    Attribution Breakdown
                </h4>

                <div className="space-y-2">
                    {channels.map(([channel, channelData], idx) => {
                        const contribution = channelData.contribution_pct || 0;
                        const colors = ['bg-blue-500', 'bg-violet-500', 'bg-pink-500', 'bg-orange-500', 'bg-teal-500'];
                        return (
                            <div key={idx} className="flex items-center gap-2">
                                <div className={`w-2 h-2 rounded-full ${colors[idx % colors.length]}`} />
                                <span className="flex-1 text-xs font-medium text-slate-200">{channel}</span>
                                <span className="text-xs font-bold text-slate-100">{contribution.toFixed(1)}%</span>
                            </div>
                        );
                    })}
                </div>
            </div>
        );
    };

    // Optimization Suggestions Component
    const OptimizationSuggestions = ({ data }) => {
        if (!data || !data.ranked_channels) return null;

        const topChannels = data.ranked_channels.slice(0, 3);

        return (
            <div className="mt-4 p-4 bg-gradient-to-br from-white/5 to-white/[0.02] rounded-lg border border-white/10 backdrop-blur-xl">
                <h4 className="font-semibold text-slate-100 text-sm mb-3 flex items-center gap-2">
                    <Target className="w-4 h-4 text-blue-300" />
                    Top Performers
                </h4>

                <div className="space-y-2">
                    {topChannels.map(([channel, channelData], idx) => (
                        <div key={idx} className="flex items-center gap-2 p-2 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
                            <ArrowUpRight className="w-4 h-4 text-emerald-400" />
                            <span className="text-xs font-medium text-slate-200">{channel}</span>
                            <span className="ml-auto text-xs text-emerald-300 font-semibold">
                                {channelData.contribution_pct?.toFixed(1)}%
                            </span>
                        </div>
                    ))}
                </div>
            </div>
        );
    };

    // Simulation Result Component
    const SimulationResult = ({ data }) => {
        if (!data) return null;

        return (
            <div className="mt-4 p-4 bg-gradient-to-br from-white/5 to-white/[0.02] rounded-lg border border-white/10 backdrop-blur-xl">
                <h4 className="font-semibold text-slate-100 text-sm mb-3">Estimated Budget Impact</h4>
                <div className="space-y-2 text-xs text-slate-200">
                    <div className="flex justify-between">
                        <span className="text-slate-400">Channel:</span>
                        <span className="font-bold text-slate-100">{data.channel}</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-slate-400">Budget Change:</span>
                        <span className="font-bold text-blue-300">{data.budget_change_pct > 0 ? '+' : ''}{data.budget_change_pct}%</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-slate-400">Estimated Impact:</span>
                        <span className="font-bold text-emerald-300">{(data.estimated_impact * 100).toFixed(2)}%</span>
                    </div>
                </div>
            </div>
        );
    };

    // Campaign Plan Component
    const CampaignPlan = ({ data }) => {
        if (!data || !data.campaigns) return null;

        return (
            <div className="space-y-4">
                {/* Campaign Header */}
                <div className="p-5 bg-gradient-to-br from-orange-500/10 to-red-500/10 rounded-xl border border-orange-500/20 backdrop-blur-xl shadow-lg">
                    <div className="flex items-start gap-3">
                        <div className="p-3 bg-gradient-to-br from-orange-500 to-red-600 rounded-xl shadow-md">
                            <Sparkles className="w-7 h-7 text-white" />
                        </div>
                        <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                                <h3 className="text-xl font-bold text-slate-100">{data.campaign_name}</h3>
                                <span className={`px-2 py-0.5 rounded-full text-xs font-bold ${data.urgency === 'CRITICAL' ? 'bg-red-500/80 text-white' :
                                    data.urgency === 'HIGH' ? 'bg-orange-500/80 text-white' :
                                        'bg-amber-500/80 text-white'
                                    }`}>
                                    {data.urgency}
                                </span>
                            </div>
                            <p className="text-base font-semibold text-slate-200 mb-1">{data.theme}</p>
                            <div className="flex items-center gap-3 text-sm text-slate-400">
                                <span>📅 {data.period}</span>
                                <span>•</span>
                                <span>🎯 {data.target_audience}</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Channel Cards */}
                <div className="p-5 bg-gradient-to-br from-white/5 to-white/[0.02] rounded-xl border border-white/10 backdrop-blur-xl">
                    <h4 className="font-bold text-slate-100 text-base mb-4 flex items-center gap-2">
                        <BarChart3 className="w-5 h-5 text-violet-300" />
                        Estimated Campaign Channels & Budget
                    </h4>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
                        {data.campaigns && data.campaigns.map((campaign, idx) => {
                            const metrics = campaign.estimated_metrics;
                            const gradients = [
                                'from-blue-500/20 to-cyan-500/20',
                                'from-violet-500/20 to-pink-500/20',
                                'from-orange-500/20 to-red-500/20'
                            ];
                            const iconColors = [
                                'from-blue-500 to-cyan-600',
                                'from-violet-500 to-pink-600',
                                'from-orange-500 to-red-600'
                            ];

                            return (
                                <div key={idx} className={`p-4 bg-gradient-to-br ${gradients[idx % gradients.length]} rounded-lg border border-white/10 backdrop-blur-sm hover:border-white/20 transition-all`}>
                                    <div className="flex items-center gap-2 mb-3">
                                        <div className={`p-2 bg-gradient-to-br ${iconColors[idx % iconColors.length]} rounded-lg shadow-md`}>
                                            <span className="text-xl">{campaign.icon}</span>
                                        </div>
                                        <div className="flex-1">
                                            <h5 className="font-bold text-slate-100 text-sm">{campaign.channel}</h5>
                                            <p className="text-xs text-slate-400">{campaign.duration_days} days</p>
                                        </div>
                                    </div>

                                    <div className="p-3 bg-white/5 rounded-lg mb-3 border border-white/10">
                                        <div className="text-xs font-semibold text-slate-400 mb-0.5">Estimated Budget</div>
                                        <div className="text-xl font-black text-slate-100">
                                            Rs. {(campaign.budget / 1000).toFixed(0)}K
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-2">
                                        {metrics.impressions && (
                                            <div className="p-2 bg-white/5 rounded-lg border border-white/10">
                                                <div className="text-xs font-semibold text-slate-400">Est. Impressions</div>
                                                <div className="text-sm font-bold text-slate-100">
                                                    {(metrics.impressions / 1000).toFixed(0)}K
                                                </div>
                                            </div>
                                        )}
                                        {metrics.clicks && (
                                            <div className="p-2 bg-white/5 rounded-lg border border-white/10">
                                                <div className="text-xs font-semibold text-slate-400">Est. Clicks</div>
                                                <div className="text-sm font-bold text-slate-100">
                                                    {(metrics.clicks / 1000).toFixed(1)}K
                                                </div>
                                            </div>
                                        )}
                                        {metrics.conversions && (
                                            <div className="p-2 bg-white/5 rounded-lg border border-white/10">
                                                <div className="text-xs font-semibold text-slate-400">Est. Conversions</div>
                                                <div className="text-sm font-bold text-slate-100">
                                                    {metrics.conversions.toLocaleString()}
                                                </div>
                                            </div>
                                        )}
                                        {metrics.roas && (
                                            <div className="p-2 bg-white/5 rounded-lg border border-white/10">
                                                <div className="text-xs font-semibold text-slate-400">Est. ROAS</div>
                                                <div className="text-sm font-bold text-slate-100">
                                                    {metrics.roas.toFixed(1)}x
                                                </div>
                                            </div>
                                        )}
                                        {metrics.emails_sent && (
                                            <div className="p-2 bg-white/5 rounded-lg border border-white/10">
                                                <div className="text-xs font-semibold text-slate-400">Est. Emails</div>
                                                <div className="text-sm font-bold text-slate-100">
                                                    {(metrics.emails_sent / 1000).toFixed(0)}K
                                                </div>
                                            </div>
                                        )}
                                        {metrics.open_rate && (
                                            <div className="p-2 bg-white/5 rounded-lg border border-white/10">
                                                <div className="text-xs font-semibold text-slate-400">Est. Open Rate</div>
                                                <div className="text-sm font-bold text-slate-100">
                                                    {metrics.open_rate.toFixed(1)}%
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    {/* Campaign Summary */}
                    <div className="p-4 bg-gradient-to-br from-emerald-500/10 to-teal-500/10 rounded-xl border border-emerald-500/20 backdrop-blur-sm">
                        <h5 className="font-bold text-slate-100 text-sm mb-3 flex items-center gap-2">
                            <Target className="w-4 h-4 text-emerald-300" />
                            Estimated Campaign Performance Summary
                        </h5>

                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            <div className="p-3 bg-white/5 rounded-lg border border-white/10">
                                <div className="text-xs font-semibold text-slate-400 mb-0.5">Total Budget</div>
                                <div className="text-lg font-black text-emerald-300">
                                    Rs. {(data.totals.total_budget / 1000).toFixed(0)}K
                                </div>
                            </div>
                            <div className="p-3 bg-white/5 rounded-lg border border-white/10">
                                <div className="text-xs font-semibold text-slate-400 mb-0.5">Est. Conversions</div>
                                <div className="text-lg font-black text-violet-300">
                                    {data.totals.total_estimated_conversions.toLocaleString()}
                                </div>
                            </div>
                            <div className="p-3 bg-white/5 rounded-lg border border-white/10">
                                <div className="text-xs font-semibold text-slate-400 mb-0.5">Est. Avg ROAS</div>
                                <div className="text-lg font-black text-orange-300">
                                    {data.totals.weighted_average_roas}x
                                </div>
                            </div>
                            <div className="p-3 bg-white/5 rounded-lg border border-white/10">
                                <div className="text-xs font-semibold text-slate-400 mb-0.5">Est. Revenue</div>
                                <div className="text-lg font-black text-blue-300">
                                    Rs. {(data.totals.estimated_revenue / 1000).toFixed(0)}K
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Recommendations */}
                {data.recommendations && data.recommendations.length > 0 && (
                    <div className="p-5 bg-gradient-to-br from-amber-500/10 to-yellow-500/10 rounded-xl border border-amber-500/20 backdrop-blur-xl">
                        <h4 className="font-bold text-slate-100 text-base mb-3 flex items-center gap-2">
                            <Zap className="w-4 h-4 text-amber-300" />
                            Strategic Recommendations
                        </h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                            {data.recommendations.slice(0, 6).map((rec, idx) => (
                                <div key={idx} className="flex items-start gap-2 p-3 bg-white/5 rounded-lg border border-white/10">
                                    <div className="flex-shrink-0 w-5 h-5 bg-amber-500/80 text-white rounded-full flex items-center justify-center text-xs font-bold">
                                        {idx + 1}
                                    </div>
                                    <p className="text-xs text-slate-200 leading-relaxed">{rec}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-[#0f1724] via-[#111827] to-[#071029] relative overflow-hidden text-slate-100 flex flex-col">
            {/* Decorative backgrounds */}
            <div className="absolute top-16 left-28 w-96 h-96 bg-orange-600/10 rounded-full blur-[120px]" />
            <div className="absolute bottom-12 right-36 w-[28rem] h-[28rem] bg-violet-600/12 rounded-full blur-[120px]" />

            <div className="absolute inset-0 opacity-[0.02] pointer-events-none">
                <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <pattern id="dark-grid" width="60" height="60" patternUnits="userSpaceOnUse">
                            <path d="M 60 0 L 0 0 0 60" fill="none" stroke="white" strokeWidth="0.5" />
                        </pattern>
                    </defs>
                    <rect width="100%" height="100%" fill="url(#dark-grid)" />
                </svg>
            </div>

            {/* Header */}
            <div className="sticky top-0 z-50 p-4 border-b border-white/6 bg-[#0f1724]/95 backdrop-blur-md">
                <div className="max-w-7xl mx-auto flex items-center gap-3">
                    <div className="bg-gradient-to-br from-white/5 to-white/[0.02] p-3 rounded-xl backdrop-blur-sm border border-white/10">
                        <TrendingUp className="w-7 h-7 text-orange-300" />
                    </div>
                    <div className="flex-1">
                        <h1 className="text-xl font-bold text-slate-100">Marketing Mix Modeling & Campaign Assistant</h1>
                        <p className="text-slate-400 text-xs">AI-powered campaign planning & ROI optimization</p>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-slate-400">
                        <Sparkles className="w-4 h-4 text-amber-300" />
                        <span>{availableModels.length} model{availableModels.length !== 1 ? 's' : ''} available</span>
                    </div>
                </div>
            </div>

            {/* Main Container */}
            <div className="flex-1 flex flex-col overflow-hidden">
                {/* Top Section - Campaign Plan (50% height) */}
                <div className="h-1/2 overflow-y-auto border-b border-white/10">
                    <div className="max-w-7xl mx-auto p-4">
                        {isCampaignLoading ? (
                            <div className="flex items-center justify-center h-full min-h-[300px]">
                                <div className="text-center">
                                    <Loader2 className="w-10 h-10 animate-spin text-blue-400 mx-auto mb-3" />
                                    <p className="text-slate-300 font-medium text-sm">Loading campaign recommendations...</p>
                                </div>
                            </div>
                        ) : campaignPlan ? (
                            <CampaignPlan data={campaignPlan} />
                        ) : (
                            <div className="flex items-center justify-center h-full min-h-[300px]">
                                <div className="text-center p-6 bg-gradient-to-br from-white/5 to-white/[0.02] rounded-xl border border-white/10 backdrop-blur-xl">
                                    <Target className="w-12 h-12 text-slate-500 mx-auto mb-3" />
                                    <h3 className="text-lg font-bold text-slate-200 mb-2">No Campaign Data Available</h3>
                                    <p className="text-sm text-slate-400">Train a model to see campaign recommendations</p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Bottom Section - Chat Interface (50% height) */}
                <div className="h-1/2 flex flex-col">
                    <div className="flex-1 flex flex-col max-w-7xl mx-auto w-full">
                        {/* Chat Header */}
                        <div className="p-3 border-b border-white/6 bg-gradient-to-r from-blue-500/10 to-violet-500/10">
                            <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
                                <BarChart3 className="w-4 h-4 text-blue-300" />
                                Ask About Marketing Performance & Strategy
                            </h2>
                            <p className="text-xs text-slate-400 mt-0.5">
                                ROI analysis, budget optimization, channel comparison & more
                            </p>
                        </div>

                        {/* Messages area */}
                        <div className="flex-1 overflow-y-auto p-4 space-y-3">
                            {messages.length === 0 && (
                                <div className="text-center py-4">
                                    <p className="text-slate-400 text-sm mb-3">Quick Questions:</p>
                                    <div className="flex flex-wrap gap-2 justify-center max-w-3xl mx-auto">
                                        {quickActions.map((action, idx) => (
                                            <button
                                                key={idx}
                                                onClick={() => setInput(action.query)}
                                                className="px-3 py-2 bg-gradient-to-br from-white/5 to-white/[0.02] hover:from-white/7 hover:to-white/5 text-slate-200 text-xs rounded-lg border border-white/10 transition-all duration-200"
                                            >
                                                {action.label}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {messages.map((message, idx) => (
                                <div key={idx} className={`flex gap-2 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                    {message.role === 'assistant' && (
                                        <div className="flex-shrink-0 w-9 h-9 rounded-full bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center shadow-md">
                                            <Bot className="w-5 h-5 text-white" />
                                        </div>
                                    )}

                                    <div
                                        className={`max-w-2xl ${message.role === 'user'
                                            ? 'bg-gradient-to-br from-blue-600/80 to-violet-700/80 text-white'
                                            : message.isError
                                                ? 'bg-red-600/10 border border-red-400/20 text-red-200'
                                                : 'bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 text-slate-200'
                                            } rounded-2xl p-4 shadow-sm`}
                                    >
                                        <div className="whitespace-pre-wrap break-words text-sm">{message.content}</div>

                                        {message.roiData && <ROIVisualization data={message.roiData} />}
                                        {message.attributionData && <AttributionChart data={message.attributionData} />}
                                        {message.optimizationData && <OptimizationSuggestions data={message.optimizationData} />}
                                        {message.simulationData && <SimulationResult data={message.simulationData} />}

                                        {message.intent && (
                                            <div className="mt-2 pt-2 border-t border-white/6 text-xs text-slate-400">
                                                Intent: {message.intent}
                                            </div>
                                        )}

                                        <div className="mt-2 text-xs opacity-60 text-slate-400">{message.timestamp.toLocaleTimeString()}</div>
                                    </div>

                                    {message.role === 'user' && (
                                        <div className="flex-shrink-0 w-9 h-9 rounded-full bg-gradient-to-br from-slate-600 to-slate-800 flex items-center justify-center shadow-md">
                                            <User className="w-5 h-5 text-white" />
                                        </div>
                                    )}
                                </div>
                            ))}

                            {isLoading && (
                                <div className="flex gap-2 justify-start">
                                    <div className="flex-shrink-0 w-9 h-9 rounded-full bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center shadow-md">
                                        <Bot className="w-5 h-5 text-white" />
                                    </div>
                                    <div className="bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 rounded-2xl p-4 shadow-sm">
                                        <div className="flex items-center gap-2 text-slate-300">
                                            <Loader2 className="w-4 h-4 animate-spin" />
                                            <span className="text-sm">Analyzing...</span>
                                        </div>
                                    </div>
                                </div>
                            )}

                            <div ref={messagesEndRef} />
                        </div>

                        {/* Input area */}
                        <div className="p-4 border-t border-white/6">
                            <div className="flex gap-3 items-end">
                                <textarea
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    onKeyPress={handleKeyPress}
                                    placeholder="Ask about ROI, channels, budget optimization..."
                                    disabled={isLoading}
                                    rows={1}
                                    className="flex-1 px-4 py-3 border-2 border-transparent rounded-xl focus:outline-none focus:ring-2 focus:ring-orange-500 transition-all resize-none bg-white/5 text-slate-100 placeholder:text-slate-500"
                                    style={{ minHeight: '48px', maxHeight: '96px' }}
                                />
                                <button
                                    onClick={handleSend}
                                    disabled={!input.trim() || isLoading}
                                    className="px-6 py-3 bg-gradient-to-r from-orange-600 to-red-700 hover:from-orange-700 hover:to-red-800 text-white rounded-xl font-medium transition-all duration-200 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed shadow-md flex items-center gap-2"
                                >
                                    {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
                                    Send
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}