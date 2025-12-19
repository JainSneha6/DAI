import React, { useState, useRef, useEffect } from 'react';
import { Send, Users, Bot, User, Sparkles, PieChart, DollarSign, Clock, AlertCircle, Loader2 } from 'lucide-react';

export default function CustomerSegmentationChat() {
    const [messages, setMessages] = useState([
        {
            role: 'assistant',
            content:
                'Hello! I can help you understand your customer segments.\n\nYou can ask me to:\n• Show customer segments\n• Explain RFM analysis\n• Identify high-value or at-risk customers\n• Describe the model\n\nWhat would you like to explore?',
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
            const res = await fetch('http://localhost:5000/api/customer_segmentation/chat/models');
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
            const res = await fetch('http://localhost:5000/api/customer_segmentation/chat', {
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
                    segmentInsights: data.segment_insights,
                    modelInfo: data.model_info
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

    const SegmentVisualization = ({ insights }) => {
        if (!insights || !insights.profiles) return null;

        const total = insights.total_customers;
        const profiles = Object.values(insights.profiles);

        return (
            <div className="mt-4 p-4 bg-gradient-to-br from-white/5 to-white/[0.02] rounded-lg border border-white/10 backdrop-blur-xl">
                <div className="flex items-center gap-2 mb-4">
                    <PieChart className="w-5 h-5 text-pink-400" />
                    <h4 className="font-semibold text-slate-100">Customer Segments ({insights.method})</h4>
                </div>

                <div className="space-y-3">
                    {profiles.sort((a, b) => b.size - a.size).map((seg, i) => {
                        const percent = ((seg.size / total) * 100).toFixed(1);
                        return (
                            <div key={i} className="bg-white/5 rounded-lg p-3 border border-white/10">
                                <div className="flex justify-between items-start mb-2">
                                    <div>
                                        <div className="font-medium text-slate-100">Segment {i + 1}</div>
                                        <div className="text-sm text-pink-300">{seg.description}</div>
                                    </div>
                                    <div className="text-right">
                                        <div className="text-lg font-bold text-slate-100">{seg.size.toLocaleString()}</div>
                                        <div className="text-xs text-slate-400">{percent}%</div>
                                    </div>
                                </div>
                                <div className="grid grid-cols-4 gap-2 text-xs">
                                    <div className="flex items-center gap-1"><Clock className="w-3 h-3" /> {seg.avg_recency_days.toFixed(0)}d</div>
                                    <div className="flex items-center gap-1"><Users className="w-3 h-3" /> {seg.avg_frequency.toFixed(1)}</div>
                                    <div className="flex items-center gap-1"><DollarSign className="w-3 h-3" /> ${seg.avg_monetary.toFixed(0)}</div>
                                    <div className="flex items-center gap-1 text-pink-400"><Sparkles className="w-3 h-3" /> ${seg.avg_ltv.toFixed(0)}</div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>
        );
    };

    const ModelInfoCard = ({ info }) => {
        if (!info) return null;
        return (
            <div className="mt-3 p-3 bg-gradient-to-br from-white/5 to-white/[0.02] rounded-lg border border-white/10">
                <div className="flex items-center gap-2 mb-2">
                    <Users className="w-4 h-4 text-pink-400" />
                    <h5 className="font-semibold text-sm text-slate-100">Model Details</h5>
                </div>
                <div className="text-xs space-y-1 text-slate-300">
                    <div>Methods: {info.segmentation_methods?.join(', ')}</div>
                    <div>Segments: {info.n_segments}</div>
                    <div>File: {info.input_file}</div>
                </div>
            </div>
        );
    };

    const quickActions = [
        { label: 'Show customer segments', query: 'Show me the customer segments' },
        { label: 'Explain RFM', query: 'What is the RFM analysis?' },
        { label: 'High-value customers', query: 'Who are the high-value customers?' },
        { label: 'Churn risk', query: 'Which customers are likely to churn?' }
    ];

    return (
        <div className="min-h-screen bg-gradient-to-br from-[#0f1724] via-[#111827] to-[#0a0e1a] relative overflow-hidden text-slate-100">
            {/* PINK theme backgrounds */}
            <div className="absolute top-20 left-32 w-96 h-96 bg-pink-600/10 rounded-full blur-[120px]" />
            <div className="absolute bottom-16 right-40 w-96 h-96 bg-pink-500/12 rounded-full blur-[120px]" />

            <div className="relative z-10 max-w-5xl mx-auto flex flex-col h-screen">
                <div className="p-6 border-b border-white/6">
                    <div className="flex items-center gap-3">
                        <div className="bg-gradient-to-br from-pink-500/20 to-pink-600/10 p-3 rounded-xl backdrop-blur-sm border border-pink-400/30">
                            <Users className="w-8 h-8 text-pink-400" />
                        </div>
                        <div>
                            <h1 className="text-2xl font-bold">Customer Segmentation Assistant</h1>
                            <p className="text-slate-400 text-sm">Understand your customers with RFM & clustering</p>
                        </div>
                        <div className="ml-auto flex items-center gap-3 text-sm text-slate-400">
                            <Sparkles className="w-4 h-4 text-pink-400" />
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
                                <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-pink-500 to-pink-600 flex items-center justify-center shadow-md">
                                    <Bot className="w-6 h-6 text-white" />
                                </div>
                            )}

                            <div className={`max-w-2xl ${msg.role === 'user'
                                ? 'bg-gradient-to-br from-pink-600/80 to-pink-700/80 text-white'
                                : msg.isError
                                    ? 'bg-red-600/10 border border-red-400/20 text-red-200'
                                    : 'bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 text-slate-200'
                                } rounded-2xl p-4 shadow-sm`}>
                                <div className="whitespace-pre-wrap break-words">{msg.content}</div>

                                {msg.segmentInsights && <SegmentVisualization insights={msg.segmentInsights} />}
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
                            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-pink-500 to-pink-600 flex items-center justify-center">
                                <Bot className="w-6 h-6 text-white" />
                            </div>
                            <div className="bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 rounded-2xl p-4">
                                <div className="flex items-center gap-2 text-slate-300">
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                    Analyzing customers...
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
                            placeholder="Ask about segments, RFM, churn risk..."
                            disabled={isLoading}
                            rows={1}
                            className="flex-1 px-4 py-3 border-2 border-transparent rounded-xl focus:outline-none focus:ring-2 focus:ring-pink-500 bg-white/5 text-slate-100 resize-none"
                            style={{ minHeight: '52px', maxHeight: '120px' }}
                        />
                        <button
                            onClick={handleSend}
                            disabled={!input.trim() || isLoading}
                            className="px-6 py-3 my-2 bg-gradient-to-r from-pink-600 to-pink-700 hover:from-pink-700 hover:to-pink-800 text-white rounded-xl font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 shadow-md"
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