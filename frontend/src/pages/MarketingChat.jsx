import React, { useState, useRef, useEffect } from 'react';
import { Send, DollarSign, Bot, User, Sparkles, TrendingUp, PieChart, BarChart3, Loader2 } from 'lucide-react';

export default function MarketingMMMChat() {
    const [messages, setMessages] = useState([
        {
            role: 'assistant',
            content:
                'Hello! I\'m your Marketing Mix Modeling assistant.\n\nI can help you understand:\n• Which channels drive the most revenue\n• ROI and attribution per channel\n• Model performance and reliability\n• Budget optimization recommendations\n\nWhat would you like to know?',
            timestamp: new Date()
        }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [availableModels, setAvailableModels] = useState([]);
    const messagesEndRef = useRef(null);

    useEffect(() => {
        fetchAvailableModels();
    }, []);

    const fetchAvailableModels = async () => {
        try {
            const res = await fetch('http://localhost:5000/api/marketing_mmm/chat/models');
            const data = await res.json();
            if (data.success) setAvailableModels(data.models);
        } catch (err) {
            console.error(err);
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

        const userMsg = { role: 'user', content: input, timestamp: new Date() };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setIsLoading(true);

        try {
            const res = await fetch('http://localhost:5000/api/marketing_mmm/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: input })
            });
            const data = await res.json();

            if (data.success) {
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: data.answer,
                    timestamp: new Date(),
                    intent: data.intent,
                    attributionData: data.attribution_data,
                    modelInfo: data.model_info,
                    metrics: data.metrics
                }]);
            } else {
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: `Error: ${data.error}`,
                    timestamp: new Date(),
                    isError: true
                }]);
            }
        } catch (err) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `Connection error: ${err.message}`,
                timestamp: new Date(),
                isError: true
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const AttributionTable = ({ data }) => {
        if (!data) return null;

        const entries = Object.entries(data);
        const totalSpend = entries.reduce((sum, [_, v]) => sum + v.total_spend, 0);
        const totalContrib = entries.reduce((sum, [_, v]) => sum + v.approx_contribution, 0);

        return (
            <div className="mt-4 p-4 bg-gradient-to-br from-white/5 to-white/[0.02] rounded-lg border border-white/10 backdrop-blur-xl overflow-x-auto">
                <div className="flex items-center gap-2 mb-3">
                    <PieChart className="w-5 h-5 text-violet-400" />
                    <h4 className="font-semibold text-slate-100">Channel ROI & Attribution</h4>
                </div>
                <table className="w-full text-sm">
                    <thead>
                        <tr className="border-b border-white/10 text-slate-400">
                            <th className="text-left py-2">Channel</th>
                            <th className="text-right py-2">Spend</th>
                            <th className="text-right py-2">Contribution</th>
                            <th className="text-right py-2">ROI</th>
                            <th className="text-right py-2">% Impact</th>
                        </tr>
                    </thead>
                    <tbody>
                        {entries
                            .sort((a, b) => (b[1].approx_roi || 0) - (a[1].approx_roi || 0))
                            .map(([channel, info]) => {
                                const percent = totalContrib > 0 ? (info.approx_contribution / totalContrib * 100) : 0;
                                return (
                                    <tr key={channel} className="border-b border-white/5">
                                        <td className="py-2 text-slate-200">{channel}</td>
                                        <td className="py-2 text-right text-slate-300">${info.total_spend.toLocaleString()}</td>
                                        <td className="py-2 text-right text-violet-300">${info.approx_contribution.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                                        <td className="py-2 text-right font-medium text-violet-400">
                                            {info.approx_roi ? `${info.approx_roi.toFixed(2)}x` : 'N/A'}
                                        </td>
                                        <td className="py-2 text-right text-slate-400">{percent.toFixed(1)}%</td>
                                    </tr>
                                );
                            })}
                    </tbody>
                    <tfoot>
                        <tr className="font-semibold text-slate-100 border-t border-white/20">
                            <td className="py-3">Total</td>
                            <td className="py-3 text-right">${totalSpend.toLocaleString()}</td>
                            <td className="py-3 text-right">${totalContrib.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                            <td className="py-3 text-right"></td>
                            <td className="py-3 text-right">100%</td>
                        </tr>
                    </tfoot>
                </table>
            </div>
        );
    };

    const ModelInfoCard = ({ info }) => {
        if (!info) return null;
        return (
            <div className="mt-3 p-3 bg-gradient-to-br from-white/5 to-white/[0.02] rounded-lg border border-white/10">
                <div className="flex items-center gap-2 mb-2">
                    <BarChart3 className="w-4 h-4 text-violet-400" />
                    <h5 className="font-semibold text-sm text-slate-100">Model Summary</h5>
                </div>
                <div className="text-xs space-y-1 text-slate-300">
                    <div>Model: {info.model_name}</div>
                    <div>Target: {info.target_column}</div>
                    <div>Channels: {(info.unique_channels || info.spend_columns_detected || []).join(', ')}</div>
                    <div>Adstock: {info.adstock_applied ? 'Yes' : 'No'}</div>
                </div>
            </div>
        );
    };

    const quickActions = [
        { label: 'Show channel ROI', query: 'Which channels have the best ROI?' },
        { label: 'Model performance', query: 'How good is the model?' },
        { label: 'Budget recommendations', query: 'Where should I allocate budget?' },
        { label: 'Channel spend breakdown', query: 'How much was spent per channel?' }
    ];

    return (
        <div className="min-h-screen bg-gradient-to-br from-[#0f1724] via-[#111827] to-[#0f0e1a] relative overflow-hidden text-slate-100">
            {/* VIOLET theme backgrounds */}
            <div className="absolute top-20 left-32 w-96 h-96 bg-violet-600/10 rounded-full blur-[120px]" />
            <div className="absolute bottom-16 right-40 w-96 h-96 bg-violet-500/12 rounded-full blur-[120px]" />

            <div className="relative z-10 max-w-5xl mx-auto flex flex-col h-screen">
                <div className="p-6 border-b border-white/6">
                    <div className="flex items-center gap-3">
                        <div className="bg-gradient-to-br from-violet-500/20 to-violet-600/10 p-3 rounded-xl backdrop-blur-sm border border-violet-400/30">
                            <DollarSign className="w-8 h-8 text-violet-400" />
                        </div>
                        <div>
                            <h1 className="text-2xl font-bold">Marketing Mix Modeling Assistant</h1>
                            <p className="text-slate-400 text-sm">Understand ROI, attribution & optimize spend</p>
                        </div>
                        <div className="ml-auto flex items-center gap-3 text-sm text-slate-400">
                            <Sparkles className="w-4 h-4 text-violet-400" />
                            {availableModels.length > 0
                                ? `${availableModels.length} model${availableModels.length > 1 ? 's' : ''} loaded`
                                : 'No models yet'}
                        </div>
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto p-6 space-y-4">
                    {messages.map((msg, i) => (
                        <div key={i} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            {msg.role === 'assistant' && (
                                <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-violet-500 to-violet-600 flex items-center justify-center shadow-md">
                                    <Bot className="w-6 h-6 text-white" />
                                </div>
                            )}

                            <div className={`max-w-3xl ${msg.role === 'user'
                                ? 'bg-gradient-to-br from-violet-600/80 to-violet-700/80 text-white'
                                : msg.isError
                                    ? 'bg-red-600/10 border border-red-400/20 text-red-200'
                                    : 'bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 text-slate-200'
                                } rounded-2xl p-4 shadow-sm`}>
                                <div className="whitespace-pre-wrap break-words">{msg.content}</div>

                                {msg.attributionData && <AttributionTable data={msg.attributionData} />}
                                {msg.modelInfo && <ModelInfoCard info={msg.modelInfo} />}

                                {msg.intent && (
                                    <div className="mt-2 pt-2 border-t border-white/6 text-xs text-slate-400">
                                        Intent: {msg.intent.replace('_', ' ')}
                                    </div>
                                )}

                                <div className="mt-2 text-xs opacity-60 text-slate-400">
                                    {msg.timestamp.toLocaleTimeString()}
                                </div>
                            </div>

                            {msg.role === 'user' && (
                                <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-slate-600 to-slate-800 flex items-center justify-center shadow-md">
                                    <User className="w-6 h-6 text-white" />
                                </div>
                            )}
                        </div>
                    ))}

                    {isLoading && (
                        <div className="flex gap-3 justify-start">
                            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-violet-500 to-violet-600 flex items-center justify-center">
                                <Bot className="w-6 h-6 text-white" />
                            </div>
                            <div className="bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 rounded-2xl p-4">
                                <div className="flex items-center gap-2 text-slate-300">
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                    Analyzing marketing impact...
                                </div>
                            </div>
                        </div>
                    )}

                    <div ref={messagesEndRef} />
                </div>

                {messages.length <= 1 && (
                    <div className="px-6 py-4 border-t border-white/6">
                        <div className="text-sm font-medium text-slate-300 mb-2">Quick Actions:</div>
                        <div className="flex flex-wrap gap-2">
                            {quickActions.map((a, i) => (
                                <button
                                    key={i}
                                    onClick={() => setInput(a.query)}
                                    className="px-3 py-2 bg-gradient-to-br from-white/5 to-white/[0.02] hover:from-white/7 hover:to-white/5 text-slate-200 text-sm rounded-lg border border-white/10 transition-all"
                                >
                                    {a.label}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                <div className="p-6 border-t border-white/6">
                    <div className="flex gap-3 items-end">
                        <textarea
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSend())}
                            placeholder="Ask about ROI, channel performance, or budget optimization..."
                            disabled={isLoading}
                            rows={1}
                            className="flex-1 px-4 py-3 border-2 border-transparent rounded-xl focus:outline-none focus:ring-2 focus:ring-violet-500 bg-white/5 text-slate-100 resize-none"
                            style={{ minHeight: '52px', maxHeight: '120px' }}
                        />
                        <button
                            onClick={handleSend}
                            disabled={!input.trim() || isLoading}
                            className="px-6 py-3 my-1 bg-gradient-to-r from-violet-600 to-violet-700 hover:from-violet-700 hover:to-violet-800 text-white rounded-xl font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 shadow-md"
                        >
                            {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
                            Send
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}