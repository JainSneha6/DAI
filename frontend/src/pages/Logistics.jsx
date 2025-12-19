import React, { useState, useRef, useEffect } from 'react';
import {
  Send, Package, Bot, User, Sparkles, Database,
  Loader2, Download, MapPin, AlertTriangle, TrendingUp,
  Clock, DollarSign, Truck, CheckCircle2
} from 'lucide-react';

export default function LogisticsChat() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        'Hello! I\'m your AI logistics assistant. I can help you with:\n\n' +
        '  - Optimize supplier allocation\n' +
        '  - Assess supplier risk scores\n' +
        '  - Analyze routing plans\n' +
        '  - View supplier performance metrics\n' +
        '  - Check data quality\n' +
        '  - Compare cost efficiency\n\n' +
        'What would you like to know?',
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [availablePlans, setAvailablePlans] = useState([]);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    fetchAvailablePlans();
  }, []);

  const fetchAvailablePlans = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/logistics/chat/plans');
      const data = await response.json();
      if (data.success) {
        setAvailablePlans(data.plans);
      }
    } catch (error) {
      console.error('Failed to load plans:', error);
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
      const response = await fetch('http://localhost:5000/api/logistics/chat', {
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
          planData: data.plan_data,
          supplierDetails: data.supplier_details,
          riskData: data.risk_data,
          dataQuality: data.data_quality,
          routesData: data.routes_data,
          artifactInfo: data.artifact_info,
          sources: data.sources
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: `❌ Error: ${data.error}`,
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
          content: `❌ Failed to get response: ${error.message}`,
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

  const handleDownloadPlan = async (planName) => {
    try {
      const response = await fetch(
        `http://localhost:5000/api/logistics/chat/download_plan/${encodeURIComponent(planName)}`
      );

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `logistics_plan_${planName.replace(/\s+/g, '_')}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else {
        alert('Failed to download plan');
      }
    } catch (error) {
      console.error('Download error:', error);
      alert('Download failed');
    }
  };

  const quickActions = [
    {
      label: 'Show optimization plan',
      query: 'Show me the supplier allocation plan'
    },
    {
      label: 'Check risk scores',
      query: 'What are the risk scores?'
    },
    {
      label: 'View supplier details',
      query: 'Tell me about the suppliers'
    },
    {
      label: 'Check data quality',
      query: 'What is the data quality?'
    }
  ];

  // Enhanced visualization component
  const PlanVisualization = ({ data, artifactInfo }) => {
    if (!data) return null;

    const suppliers = data.suppliers || [];
    const topSuppliers = data.top_suppliers_by_risk || suppliers.slice(0, 5);
    const riskScores = data.risk_scores || {};
    const config = data.configuration || {};
    const dataSummary = data.data_summary || {};
    const mlMetrics = data.ml_metrics;

    const getRiskColor = (risk) => {
      if (typeof risk !== 'number') return 'text-slate-400';
      if (risk < 0.3) return 'text-green-400';
      if (risk < 0.7) return 'text-amber-400';
      return 'text-red-400';
    };

    const getRiskBadge = (risk) => {
      if (typeof risk !== 'number') return '';
      if (risk < 0.3) return 'LOW';
      if (risk < 0.7) return 'MED';
      return 'HIGH';
    };

    const getRiskBadgeColor = (risk) => {
      if (typeof risk !== 'number') return 'bg-slate-600/50 text-slate-300';
      if (risk < 0.3) return 'bg-green-600/20 text-green-300';
      if (risk < 0.7) return 'bg-amber-600/20 text-amber-300';
      return 'bg-red-600/20 text-red-300';
    };

    return (
      <div className="mt-4 space-y-3">
        {/* Header with download */}




        {/* ML Model Metrics */}
        {mlMetrics && (
          <div className="p-3 bg-gradient-to-br from-violet-600/10 to-purple-600/10 rounded-lg border border-violet-500/20">
            <div className="flex items-center gap-2 mb-2">
              <Sparkles className="w-4 h-4 text-violet-400" />
              <span className="text-sm font-semibold text-slate-100">ML Model Performance</span>
            </div>
            <div className="grid grid-cols-3 gap-3 text-xs">
              <div>
                <span className="text-slate-400">Model</span>
                <div className="font-semibold text-violet-300">{mlMetrics.model_type || 'N/A'}</div>
              </div>
              <div>
                <span className="text-slate-400">Accuracy</span>
                <div className="font-semibold text-green-400">
                  {mlMetrics.accuracy ? `${(mlMetrics.accuracy * 100).toFixed(1)}%` : 'N/A'}
                </div>
              </div>
              {mlMetrics.roc_auc && (
                <div>
                  <span className="text-slate-400">ROC-AUC</span>
                  <div className="font-semibold text-blue-400">{mlMetrics.roc_auc.toFixed(3)}</div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Top Suppliers by Risk */}
        <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
          <div className="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <TrendingUp className="w-4 h-4" />
            Top Suppliers (by risk - lowest first)
          </div>
          {topSuppliers.slice(0, 5).map((supplier, idx) => {
            const risk = riskScores[supplier];
            const stats = data.supplier_stats_preview?.[supplier] || {};
            const allocation = data.allocation_summary?.[supplier] || 0;
            const routes = data.routes_summary?.[supplier] || 0;

            return (
              <div key={supplier} className="bg-slate-800/40 p-3 rounded-lg border border-slate-700/50 hover:border-slate-600/50 transition-colors">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className={`text-xs font-bold px-2 py-0.5 rounded ${getRiskBadgeColor(risk)}`}>
                      {getRiskBadge(risk)}
                    </span>
                    <div>
                      <div className="font-semibold text-slate-100">{supplier}</div>
                      <div className={`text-xs font-medium ${getRiskColor(risk)}`}>
                        Risk: {typeof risk === 'number' ? risk.toFixed(3) : 'N/A'}
                      </div>
                    </div>
                  </div>
                  <div className="text-right text-xs">
                    <div className="text-slate-400">Rank</div>
                    <div className="text-lg font-bold text-amber-400">#{idx + 1}</div>
                  </div>
                </div>

                <div className="grid grid-cols-4 gap-2 text-xs">
                  <div>
                    <span className="text-slate-400 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" />
                      On-time
                    </span>
                    <div className="font-semibold text-green-300">
                      {stats.on_time_rate != null ? `${(stats.on_time_rate * 100).toFixed(0)}%` : 'N/A'}
                    </div>
                  </div>
                  <div>
                    <span className="text-slate-400 flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      Delay
                    </span>
                    <div className="font-semibold text-amber-300">
                      {stats.avg_delay_hours != null ? `${stats.avg_delay_hours.toFixed(1)}h` : 'N/A'}
                    </div>
                  </div>
                  <div>
                    <span className="text-slate-400 flex items-center gap-1">
                      <Package className="w-3 h-3" />
                      Assigned
                    </span>
                    <div className="font-semibold text-amber-300">{allocation}</div>
                  </div>
                  <div>
                    <span className="text-slate-400 flex items-center gap-1">
                      <Truck className="w-3 h-3" />
                      Routes
                    </span>
                    <div className="font-semibold text-purple-300">{routes}</div>
                  </div>
                </div>

                {/* Cost efficiency */}
                {stats.cost_per_kg != null && (
                  <div className="mt-2 pt-2 border-t border-slate-700/50 text-xs">
                    <span className="text-slate-400">Cost Efficiency: </span>
                    <span className="font-semibold text-slate-200">
                      ${stats.cost_per_kg.toFixed(2)}/kg
                    </span>
                    {stats.cost_per_km != null && (
                      <>
                        <span className="text-slate-500 mx-1">•</span>
                        <span className="font-semibold text-slate-200">
                          ${stats.cost_per_km.toFixed(2)}/km
                        </span>
                      </>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {suppliers.length > 5 && (
          <div className="text-center text-xs text-slate-400 py-2">
            Showing top 5 of {suppliers.length} suppliers. Download CSV for complete data.
          </div>
        )}
      </div>
    );
  };

  // Supplier details component
  const SupplierDetailsCard = ({ details }) => {
    if (!details) return null;

    const stats = details.statistics || {};
    const risk = details.risk_scores || {};
    const attrs = details.attributes || {};

    const getRiskColor = (score) => {
      if (score < 0.3) return 'text-green-400';
      if (score < 0.7) return 'text-amber-400';
      return 'text-red-400';
    };

    return (
      <div className="mt-4 p-4 bg-gradient-to-br from-white/5 to-white/[0.02] rounded-lg border border-white/10">
        <div className="flex items-center gap-2 mb-3">
          <Package className="w-5 h-5 text-amber-400" />
          <h4 className="font-semibold text-slate-100">{details.supplier_name}</h4>
          <span className={`ml-auto text-lg font-bold ${getRiskColor(risk.combined)}`}>
            {risk.combined?.toFixed(3)}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-3 text-sm">
          {stats.on_time_rate != null && (
            <div className="bg-slate-800/30 p-2 rounded">
              <div className="text-slate-400 text-xs">On-Time Rate</div>
              <div className="font-semibold text-green-400">{(stats.on_time_rate * 100).toFixed(1)}%</div>
            </div>
          )}
          {stats.delay_rate != null && (
            <div className="bg-slate-800/30 p-2 rounded">
              <div className="text-slate-400 text-xs">Delay Rate</div>
              <div className="font-semibold text-amber-400">{(stats.delay_rate * 100).toFixed(1)}%</div>
            </div>
          )}
          {stats.avg_cost_usd != null && (
            <div className="bg-slate-800/30 p-2 rounded">
              <div className="text-slate-400 text-xs">Avg Cost</div>
              <div className="font-semibold text-amber-400">${stats.avg_cost_usd.toFixed(2)}</div>
            </div>
          )}
          {stats.cost_per_kg != null && (
            <div className="bg-slate-800/30 p-2 rounded">
              <div className="text-slate-400 text-xs">Cost/kg</div>
              <div className="font-semibold text-purple-400">${stats.cost_per_kg.toFixed(2)}</div>
            </div>
          )}
        </div>
      </div>
    );
  };

  // Data quality card
  const DataQualityCard = ({ quality }) => {
    if (!quality) return null;

    const score = quality.overall_quality_score || 0;
    const getScoreColor = (s) => {
      if (s > 0.8) return 'text-green-400';
      if (s > 0.5) return 'text-amber-400';
      return 'text-red-400';
    };

    return (
      <div className="mt-4 p-4 bg-gradient-to-br from-white/5 to-white/[0.02] rounded-lg border border-white/10">
        <div className="flex items-center gap-2 mb-3">
          <Database className="w-5 h-5 text-violet-400" />
          <h4 className="font-semibold text-slate-100">Data Quality</h4>
          <span className={`ml-auto text-2xl font-bold ${getScoreColor(score)}`}>
            {(score * 100).toFixed(0)}%
          </span>
        </div>
        <div className="text-xs text-slate-400">
          {quality.total_rows} rows • {quality.total_columns} columns
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0f1724] via-[#111827] to-[#071029] relative overflow-hidden text-slate-100">

      {/* Decorative backgrounds */}
      <div className="absolute top-16 left-28 w-96 h-96 bg-yellow-600/10 rounded-full blur-[120px]" />
      <div className="absolute bottom-12 right-36 w-[28rem] h-[28rem] bg-amber-600/12 rounded-full blur-[120px]" />

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

      {/* Container */}
      <div className="relative z-10 max-w-6xl mx-auto flex flex-col h-screen">

        {/* Header */}
        <div className="p-6 border-b border-white/6">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-br from-yellow-600 to-amber-700 p-3 rounded-xl shadow-lg">
              <Package className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-slate-100">Logistics Assistant</h1>
              <p className="text-slate-400 text-sm">AI-powered supplier optimization</p>
            </div>
            <div className="ml-auto flex items-center gap-3">
              <div className="text-right text-sm">
                <div className="text-slate-400">Available Plans</div>
                <div className="font-semibold text-slate-200">
                  {availablePlans.length > 0 ? availablePlans.length : 'None'}
                </div>
              </div>
              {availablePlans.length > 0 && (
                <Sparkles className="w-5 h-5 text-amber-400" />
              )}
            </div>
          </div>
        </div>

        {/* Messages area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map((message, idx) => (
            <div key={idx} className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>

              {message.role === 'assistant' && (
                <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-yellow-500 to-amber-600 flex items-center justify-center shadow-md">
                  <Bot className="w-6 h-6 text-white" />
                </div>
              )}

              <div
                className={`max-w-3xl ${message.role === 'user'
                  ? 'bg-gradient-to-br from-yellow-600/80 to-amber-700/80 text-white shadow-lg'
                  : message.isError
                    ? 'bg-red-600/10 border border-red-400/20 text-red-200'
                    : 'bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 text-slate-200'
                  } rounded-2xl p-4 shadow-sm`}
              >
                <div className="whitespace-pre-wrap break-words font-sans text-sm leading-relaxed">
                  {message.content}
                </div>

                {message.planData && (
                  <PlanVisualization
                    data={message.planData}
                    artifactInfo={message.artifactInfo}
                  />
                )}

                {message.supplierDetails && (
                  <SupplierDetailsCard details={message.supplierDetails} />
                )}

                {message.dataQuality && (
                  <DataQualityCard quality={message.dataQuality} />
                )}

                {message.intent && (
                  <div className="mt-3 pt-3 border-t border-white/6 flex items-center gap-2 text-xs text-slate-400">
                    <Sparkles className="w-3 h-3" />
                    Intent: <span className="font-medium">{message.intent}</span>
                  </div>
                )}

                <div className="mt-2 text-xs opacity-60 text-slate-400">
                  {message.timestamp.toLocaleTimeString()}
                </div>
              </div>

              {message.role === 'user' && (
                <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-slate-600 to-slate-800 flex items-center justify-center shadow-md">
                  <User className="w-6 h-6 text-white" />
                </div>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="flex gap-3 justify-start">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-yellow-500 to-amber-600 flex items-center justify-center shadow-md">
                <Bot className="w-6 h-6 text-white" />
              </div>
              <div className="bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 rounded-2xl p-4 shadow-sm">
                <div className="flex items-center gap-2 text-slate-300">
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Analyzing...</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Quick Actions */}
        {messages.length <= 1 && (
          <div className="px-6 py-4 border-t border-white/6">
            <div className="text-sm text-slate-400 mb-2">Try asking:</div>
            <div className="flex gap-2">
              {quickActions.map((action, idx) => (
                <button
                  key={idx}
                  onClick={() => setInput(action.query)}
                  className="px-4 py-2 bg-white/5 hover:bg-white/10 text-slate-300 text-sm rounded-lg border border-white/10 transition-colors"
                >
                  {action.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Input area */}
        <div className="p-6 border-t border-white/6">
          <div className="flex gap-3 items-end">
            <div className="flex-1 relative">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about optimization, risk scores, supplier details, routes, costs..."
                disabled={isLoading}
                rows={1}
                className="w-full px-4 py-3 pr-12 border-2 border-transparent rounded-xl focus:outline-none focus:ring-2 focus:ring-yellow-500 transition-all resize-none bg-white/5 text-slate-100 placeholder-slate-500"
                style={{ minHeight: '52px', maxHeight: '120px' }}
              />
            </div>
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className="px-6 py-3 my-2 bg-gradient-to-r from-yellow-600 to-amber-700 hover:from-yellow-700 hover:to-amber-800 text-white rounded-xl font-medium transition-all duration-200 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed shadow-lg flex items-center gap-2"
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