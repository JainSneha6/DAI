import React, { useState, useRef, useEffect } from 'react';
import { Send, Package, Bot, User, Sparkles, BarChart3, Database, Loader2, Download, MapPin, AlertTriangle } from 'lucide-react';

export default function LogisticsChat() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        'Hello! I can help you with supplier risk assessment and routing optimization. You can ask me to:\n\n• Optimize allocations ("optimize supplier allocation for next 30 demands")\n• Show risk scores\n• Analyze routes\n• Get supplier details\n\nWhat would you like to optimize?',
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [availablePlans, setAvailablePlans] = useState([]);
  const [downloadUrl, setDownloadUrl] = useState(null);
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
          artifactInfo: data.artifact_info,
          sources: data.sources,
          planName: data.artifact_info?.model || 'latest'
        };
        setMessages((prev) => [...prev, assistantMessage]);
        
        // Set download URL if plan data available
        if (data.plan_data && data.artifact_info) {
          const planName = data.artifact_info.model || 'latest';
          setDownloadUrl(`/api/logistics/chat/download_plan/${encodeURIComponent(planName)}`);
        }
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

  const handleDownload = (e, planName) => {
    e.stopPropagation();
    if (downloadUrl) {
      const link = document.createElement('a');
      link.href = `http://localhost:5000${downloadUrl}`;
      link.download = `logistics_plan_${planName}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  const quickActions = [
    { label: 'Optimize supplier allocation', query: 'Optimize supplier allocation for next 10 demands' },
    { label: 'Show risk scores', query: 'What are the supplier risk scores?' },
    { label: 'Get routing plan', query: 'Show the routing plan' },
    { label: 'Explain logistics data', query: 'Tell me about the logistics data' }
  ];

  const PlanVisualization = ({ data, planName }) => {
    if (!data) return null;

    const suppliers = data.suppliers || [];
    const riskScores = data.combined_risk || {};
    const allocation = data.allocation || {};
    const routes = data.routes || {};

    // Compute summaries
    const allocationSummary = {};
    const routesSummary = {};
    let totalDemands = 0;

    suppliers.forEach(supplier => {
      const allocItems = allocation[supplier] || {};
      allocationSummary[supplier] = Object.keys(allocItems).length;
      totalDemands += allocationSummary[supplier];

      const supplierRoutes = routes[supplier] || [];
      routesSummary[supplier] = supplierRoutes.length;
    });

    return (
      <div className="mt-4 p-4 bg-gradient-to-br from-white/5 to-white/[0.02] rounded-lg border border-white/10 backdrop-blur-xl">
        <div className="flex items-center gap-2 mb-3">
          <MapPin className="w-5 h-5 text-blue-300" />
          <h4 className="font-semibold text-slate-100">Logistics Plan Preview</h4>
          {downloadUrl && (
            <button
              onClick={(e) => handleDownload(e, planName)}
              className="ml-auto px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded font-medium flex items-center gap-1"
            >
              <Download className="w-3 h-3" />
              Download Full CSV
            </button>
          )}
        </div>

        <div className="space-y-3 mb-4 max-h-60 overflow-y-auto">
          {suppliers.slice(0, 5).map((supplier) => {
            const risk = riskScores[supplier];
            const allocCount = allocationSummary[supplier] || 0;
            const routeCount = routesSummary[supplier] || 0;
            const riskDisplay = typeof risk === 'number' ? risk.toFixed(3) : (risk || 'N/A');
            const riskColor = typeof risk === 'number' 
              ? (risk > 0.7 ? 'text-red-300' : risk > 0.3 ? 'text-amber-300' : 'text-green-300')
              : 'text-slate-400';
            return (
              <div key={supplier} className="bg-slate-800/30 p-3 rounded border border-slate-700">
                <div className="font-medium text-slate-200 mb-2">{supplier}</div>
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div>
                    <span className="text-slate-400">Risk</span>
                    <div className={`font-semibold ${riskColor}`}>
                      {riskDisplay}
                    </div>
                  </div>
                  <div>
                    <span className="text-slate-400">Demands</span>
                    <div className="font-semibold text-blue-300">{allocCount}</div>
                  </div>
                  <div>
                    <span className="text-slate-400">Routes</span>
                    <div className="font-semibold text-purple-300">{routeCount}</div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {suppliers.length > 5 && (
          <div className="text-center text-xs text-slate-400 mb-3">
            Showing first 5 suppliers. Download full plan for all {suppliers.length} suppliers.
          </div>
        )}

        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="bg-gradient-to-br from-white/3 to-white/[0.02] p-2 rounded border border-white/6">
            <div className="text-slate-400 text-xs">Suppliers</div>
            <div className="font-bold text-slate-100">{suppliers.length}</div>
          </div>
          <div className="bg-gradient-to-br from-white/3 to-white/[0.02] p-2 rounded border border-white/6">
            <div className="text-slate-400 text-xs">Total Demands</div>
            <div className="font-bold text-slate-100">{totalDemands}</div>
          </div>
        </div>
      </div>
    );
  };

  const ArtifactInfoCard = ({ info }) => {
    if (!info) return null;

    return (
      <div className="mt-3 p-3 bg-gradient-to-br from-white/5 to-white/[0.02] rounded-lg border border-white/10 backdrop-blur-xl">
        <div className="flex items-center gap-2 mb-2">
          <Database className="w-4 h-4 text-violet-300" />
          <h5 className="font-semibold text-slate-100 text-sm">Plan Details</h5>
        </div>
        <div className="space-y-1 text-xs text-slate-200">
          <div>
            <span className="font-medium text-slate-300">Solver:</span> {info.solver_used}
          </div>
          <div>
            <span className="font-medium text-slate-300">Suppliers:</span> {info.suppliers_count || info.suppliers?.length || 0}
          </div>
          {info.risk_weight && (
            <div>
              <span className="font-medium text-slate-300">Risk Weight:</span> {info.risk_weight}
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0f1724] via-[#111827] to-[#071029] relative overflow-hidden text-slate-100">

      {/* Decorative backgrounds */}
      <div className="absolute top-16 left-28 w-96 h-96 bg-blue-600/10 rounded-full blur-[120px]" />
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

      {/* Container */}
      <div className="relative z-10 max-w-5xl mx-auto flex flex-col h-screen">

        {/* Header */}
        <div className="p-6 border-b border-white/6">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-br from-white/5 to-white/[0.02] p-3 rounded-xl backdrop-blur-sm border border-white/10">
              <Package className="w-8 h-8 text-blue-300" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-slate-100">Logistics & Supplier Risk Assistant</h1>
              <p className="text-slate-400 text-sm">AI-powered optimization & routing</p>
            </div>
            <div className="ml-auto text-sm text-slate-400 flex items-center gap-3">
              <Sparkles className="w-4 h-4 text-amber-300" />
              {availablePlans.length > 0 ? (
                <span>{availablePlans.length} plan{availablePlans.length !== 1 ? 's' : ''} available</span>
              ) : (
                <span>No plans loaded</span>
              )}
            </div>
          </div>
        </div>

        {/* Messages area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map((message, idx) => (
            <div key={idx} className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>

              {message.role === 'assistant' && (
                <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center shadow-md">
                  <Bot className="w-6 h-6 text-white" />
                </div>
              )}

              <div
                className={`max-w-2xl ${
                  message.role === 'user'
                    ? 'bg-gradient-to-br from-blue-600/80 to-violet-700/80 text-white'
                    : message.isError
                    ? 'bg-red-600/10 border border-red-400/20 text-red-200'
                    : 'bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 text-slate-200'
                } rounded-2xl p-4 shadow-sm`}
              >
                <div className="whitespace-pre-wrap break-words">{message.content}</div>

                {message.planData && <PlanVisualization data={message.planData} planName={message.planName} />}
                {message.artifactInfo && <ArtifactInfoCard info={message.artifactInfo} />}

                {message.intent && (
                  <div className="mt-2 pt-2 border-t border-white/6 text-xs text-slate-400">
                    Intent: {message.intent}
                  </div>
                )}

                <div className="mt-2 text-xs opacity-60 text-slate-400">{message.timestamp.toLocaleTimeString()}</div>
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
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center shadow-md">
                <Bot className="w-6 h-6 text-white" />
              </div>
              <div className="bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 rounded-2xl p-4 shadow-sm">
                <div className="flex items-center gap-2 text-slate-300">
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Optimizing...</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Quick Actions (shown when only initial message) */}
        {messages.length <= 1 && (
          <div className="px-6 py-4 border-t border-white/6">
            <div className="text-sm font-medium text-slate-300 mb-2">Quick Actions:</div>
            <div className="flex flex-wrap gap-2">
              {quickActions.map((action, idx) => (
                <button
                  key={idx}
                  onClick={() => setInput(action.query)}
                  className="px-3 py-2 bg-gradient-to-br from-white/5 to-white/[0.02] hover:from-white/7 hover:to-white/5 text-slate-200 text-sm rounded-lg border border-white/10 transition-all duration-200"
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
                placeholder="Ask about optimization, risk scores, routes, or your logistics data..."
                disabled={isLoading}
                rows={1}
                className="w-full px-4 py-3 pr-12 border-2 border-transparent rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all resize-none bg-white/5 text-slate-100"
                style={{ minHeight: '52px', maxHeight: '120px' }}
              />
            </div>
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className="px-6 py-3 my-2 bg-gradient-to-r from-blue-600 to-violet-700 hover:from-blue-700 hover:to-violet-800 text-white rounded-xl font-medium transition-all duration-200 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed shadow-md flex items-center gap-2"
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