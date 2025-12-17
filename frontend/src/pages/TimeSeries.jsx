import React, { useState, useRef, useEffect } from 'react';
import { Send, TrendingUp, Bot, User, Sparkles, BarChart3, Database, Loader2 } from 'lucide-react';

export default function TimeSeriesChat() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        'Hello! I can help you with time series forecasting. You can ask me to:\n\n• Generate forecasts ("forecast the next 30 days")\n• Explain model performance\n• Analyze your data\n• Compare models\n\nWhat would you like to know?',
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
      const response = await fetch('http://localhost:5000/api/timeseries/chat/models');
      const data = await response.json();
      if (data.success) {
        setAvailableModels(data.models);
      }
    } catch (error) {
      console.error('Failed to load models:', error);
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
      const response = await fetch('http://localhost:5000/api/timeseries/chat', {
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
          forecastData: data.forecast_data,
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
    { label: 'Forecast next 10 steps', query: 'Forecast the next 10 steps' },
    { label: 'Forecast next 30 days', query: 'Predict the next 30 days' },
    { label: 'Show model performance', query: 'What are the model metrics?' },
    { label: 'Explain my data', query: 'Tell me about the uploaded data' }
  ];

  const ForecastVisualization = ({ data }) => {
    if (!data || !data.forecast) return null;

    const max = Math.max(...data.forecast);
    const min = Math.min(...data.forecast);
    const range = max - min;

    return (
      <div className="mt-4 p-4 bg-gradient-to-br from-white/5 to-white/[0.02] rounded-lg border border-white/10 backdrop-blur-xl">
        <div className="flex items-center gap-2 mb-3">
          <TrendingUp className="w-5 h-5 text-blue-300" />
          <h4 className="font-semibold text-slate-100">Forecast Visualization</h4>
        </div>

        <div className="space-y-2 mb-4 text-slate-200">
          {data.forecast.slice(0, 10).map((value, idx) => {
            const percentage = range > 0 ? ((value - min) / range) * 100 : 50;
            return (
              <div key={idx} className="flex items-center gap-3">
                <span className="text-xs font-mono text-slate-400 w-20">
                  {data.forecast_dates ? data.forecast_dates[idx] : `Step ${idx + 1}`}
                </span>
                <div className="flex-1 bg-slate-800/40 rounded-full h-6 relative overflow-hidden border border-slate-700">
                  <div
                    className="bg-gradient-to-r from-blue-500 to-violet-500 h-full rounded-full transition-all duration-300 flex items-center justify-end pr-2"
                    style={{ width: `${percentage}%` }}
                  >
                    <span className="text-xs font-semibold text-white">{value.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <div className="grid grid-cols-3 gap-3 text-sm">
          <div className="bg-gradient-to-br from-white/3 to-white/[0.02] p-2 rounded border border-white/6">
            <div className="text-slate-400 text-xs">Mean</div>
            <div className="font-bold text-slate-100">
              {(data.forecast.reduce((a, b) => a + b, 0) / data.forecast.length).toFixed(2)}
            </div>
          </div>
          <div className="bg-gradient-to-br from-white/3 to-white/[0.02] p-2 rounded border border-white/6">
            <div className="text-slate-400 text-xs">Min</div>
            <div className="font-bold text-slate-100">{min.toFixed(2)}</div>
          </div>
          <div className="bg-gradient-to-br from-white/3 to-white/[0.02] p-2 rounded border border-white/6">
            <div className="text-slate-400 text-xs">Max</div>
            <div className="font-bold text-slate-100">{max.toFixed(2)}</div>
          </div>
        </div>
      </div>
    );
  };

  const ModelInfoCard = ({ info }) => {
    if (!info) return null;

    return (
      <div className="mt-3 p-3 bg-gradient-to-br from-white/5 to-white/[0.02] rounded-lg border border-white/10 backdrop-blur-xl">
        <div className="flex items-center gap-2 mb-2">
          <Database className="w-4 h-4 text-violet-300" />
          <h5 className="font-semibold text-slate-100 text-sm">Model Details</h5>
        </div>
        <div className="space-y-1 text-xs text-slate-200">
          <div>
            <span className="font-medium text-slate-300">Type:</span> {info.model_name}
          </div>
          <div>
            <span className="font-medium text-slate-300">Target:</span> {info.target_column}
          </div>
          {info.exogenous_features && info.exogenous_features.length > 0 && (
            <div>
              <span className="font-medium text-slate-300">Features:</span>{' '}
              {info.exogenous_features.join(', ')}
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
              <BarChart3 className="w-8 h-8 text-blue-300" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-slate-100">Sales, Demand & Financial Forecasting Assistant</h1>
              <p className="text-slate-400 text-sm">AI-powered forecasting & analysis</p>
            </div>
            <div className="ml-auto text-sm text-slate-400 flex items-center gap-3">
              <Sparkles className="w-4 h-4 text-amber-300" />
              {availableModels.length > 0 ? (
                <span>{availableModels.length} model{availableModels.length !== 1 ? 's' : ''} available</span>
              ) : (
                <span>No models loaded</span>
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

                {message.forecastData && <ForecastVisualization data={message.forecastData} />}
                {message.modelInfo && <ModelInfoCard info={message.modelInfo} />}

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
                  <span>Thinking...</span>
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
                placeholder="Ask about forecasts, models, or your data..."
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
