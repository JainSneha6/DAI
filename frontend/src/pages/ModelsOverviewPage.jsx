import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
    Brain,
    TrendingUp,
    Package,
    Users,
    DollarSign,
    Truck,
    BarChart3,
    Search,
    Filter,
    Sparkles,
    ArrowRight
} from "lucide-react";

const MODELS = [
    {
        id: "sales-forecasting",
        name: "Sales, Demand & Financial Forecasting",
        icon: TrendingUp,
        color: "blue",
        category: "Predictive Analytics",
        domains: ["Sales", "Finance", "Supply Chain"],
        description: "Forecast future sales, demand, and financial metrics using advanced time-series ML algorithms."
    },
    {
        id: "marketing-roi",
        name: "Marketing ROI & Attribution",
        icon: BarChart3,
        color: "violet",
        category: "Marketing Analytics",
        domains: ["Marketing"],
        description: "Optimize marketing spend and understand channel effectiveness through mix modeling and attribution."
    },
    {
        id: "inventory-optimization",
        name: "Inventory & Replenishment Optimization",
        icon: Package,
        color: "emerald",
        category: "Operations Analytics",
        domains: ["Inventory", "Operations", "Supply Chain"],
        description: "Optimize inventory levels, reorder points, and replenishment strategies to minimize costs."
    },
    {
        id: "logistics-optimization",
        name: "Logistics & Supplier Risk",
        icon: Truck,
        color: "amber",
        category: "Supply Chain Analytics",
        domains: ["Logistics", "Supply Chain", "Risk & Compliance"],
        description: "Assess supplier risk, optimize routing, and ensure supply chain resilience."
    },
    {
        id: "customer-segmentation",
        name: "Customer Segmentation & Modeling",
        icon: Users,
        color: "pink",
        category: "Customer Analytics",
        domains: ["Customer"],
        description: "Segment customers based on behavior, predict lifetime value, and identify churn risk."
    },
    {
        id: "pricing-optimization",
        name: "Pricing & Revenue Optimization",
        icon: DollarSign,
        color: "cyan",
        category: "Revenue Analytics",
        domains: ["Sales", "Marketing", "Finance"],
        description: "Optimize pricing strategies to maximize revenue and profitability."
    },
    {
        id: "sentiment-nlp",
        name: "Sentiment & Intent NLP",
        icon: Brain,
        color: "indigo",
        category: "Text Analytics",
        domains: ["Customer", "Support", "Product", "Marketing"],
        description: "Analyze text data to extract sentiment, intent, and key topics from customer feedback."
    },
    {
        id: "customer-retention",
        name: "Customer Value & Retention",
        icon: Users,
        color: "rose",
        category: "Customer Analytics",
        domains: ["Customer", "Support"],
        description: "Predict customer churn and lifetime value to optimize retention strategies."
    }
];

const colorClasses = {
    blue: "from-blue-500/10 to-blue-600/5 border-blue-400/30 hover:border-blue-400/50",
    violet: "from-violet-500/10 to-violet-600/5 border-violet-400/30 hover:border-violet-400/50",
    emerald: "from-emerald-500/10 to-emerald-600/5 border-emerald-400/30 hover:border-emerald-400/50",
    amber: "from-amber-500/10 to-amber-600/5 border-amber-400/30 hover:border-amber-400/50",
    pink: "from-pink-500/10 to-pink-600/5 border-pink-400/30 hover:border-pink-400/50",
    cyan: "from-cyan-500/10 to-cyan-600/5 border-cyan-400/30 hover:border-cyan-400/50",
    indigo: "from-indigo-500/10 to-indigo-600/5 border-indigo-400/30 hover:border-indigo-400/50",
    rose: "from-rose-500/10 to-rose-600/5 border-rose-400/30 hover:border-rose-400/50"
};

const iconColorClasses = {
    blue: "bg-blue-500/20 border-blue-400/30 text-blue-400",
    violet: "bg-violet-500/20 border-violet-400/30 text-violet-400",
    emerald: "bg-emerald-500/20 border-emerald-400/30 text-emerald-400",
    amber: "bg-amber-500/20 border-amber-400/30 text-amber-400",
    pink: "bg-pink-500/20 border-pink-400/30 text-pink-400",
    cyan: "bg-cyan-500/20 border-cyan-400/30 text-cyan-400",
    indigo: "bg-indigo-500/20 border-indigo-400/30 text-indigo-400",
    rose: "bg-rose-500/20 border-rose-400/30 text-rose-400"
};

const categories = ["All", "Predictive Analytics", "Marketing Analytics", "Operations Analytics", "Supply Chain Analytics", "Customer Analytics", "Revenue Analytics", "Text Analytics"];

export default function ModelsOverviewPage() {
    const navigate = useNavigate();
    const [searchQuery, setSearchQuery] = useState("");
    const [selectedCategory, setSelectedCategory] = useState("All");

    const filteredModels = MODELS.filter((model) => {
        const matchesSearch = model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            model.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
            model.domains.some(d => d.toLowerCase().includes(searchQuery.toLowerCase()));
        const matchesCategory = selectedCategory === "All" || model.category === selectedCategory;
        return matchesSearch && matchesCategory;
    });

    return (
        <div className="min-h-screen relative overflow-hidden">
            {/* Background effects */}
            <div className="absolute top-20 left-32 w-96 h-96 bg-blue-600/10 rounded-full blur-[120px]" />
            <div className="absolute bottom-10 right-40 w-[28rem] h-[28rem] bg-violet-600/10 rounded-full blur-[120px]" />

            <div className="relative z-10 p-8">
                <div className="max-w-7xl mx-auto">

                    {/* Header */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="mb-8"
                    >
                        <div className="flex items-center gap-3 mb-4">
                            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500/20 to-violet-500/20 border border-blue-400/30 flex items-center justify-center">
                                <Brain size={24} className="text-blue-400" />
                            </div>
                            <div>
                                <span className="inline-block px-3 py-1 rounded-full text-xs font-medium bg-blue-500/10 border border-blue-400/30 text-blue-400">
                                    AI Models
                                </span>
                            </div>
                        </div>
                        <h1 className="text-4xl font-bold text-slate-50 mb-3">Available Models</h1>
                        <p className="text-lg text-slate-400">
                            Choose from our collection of enterprise-grade AI models optimized for different business domains
                        </p>
                    </motion.div>

                    {/* Search and Filter */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 }}
                        className="mb-8 space-y-4"
                    >
                        {/* Search Bar */}
                        <div className="relative">
                            <Search size={20} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" />
                            <input
                                type="text"
                                placeholder="Search models by name, domain, or description..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="w-full pl-12 pr-4 py-4 rounded-xl bg-white/5 border border-white/10 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-blue-400/50 transition-all"
                            />
                        </div>

                        {/* Category Filter */}
                        <div className="flex items-center gap-3 overflow-x-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-white/10 pb-2">
                            <Filter size={20} className="text-slate-400 flex-shrink-0" />
                            {categories.map((category) => (
                                <button
                                    key={category}
                                    onClick={() => setSelectedCategory(category)}
                                    className={`px-4 py-2 rounded-lg whitespace-nowrap transition-all ${selectedCategory === category
                                            ? "bg-blue-500/20 border border-blue-400/30 text-blue-400"
                                            : "bg-white/5 border border-white/10 text-slate-400 hover:bg-white/10 hover:text-slate-200"
                                        }`}
                                >
                                    {category}
                                </button>
                            ))}
                        </div>
                    </motion.div>

                    {/* Models Grid */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.2 }}
                        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
                    >
                        {filteredModels.map((model, index) => {
                            const Icon = model.icon;
                            return (
                                <ModelCard
                                    key={model.id}
                                    model={model}
                                    Icon={Icon}
                                    index={index}
                                    onClick={() => navigate(`/${model.id}`)}
                                />
                            );
                        })}
                    </motion.div>

                    {/* No Results */}
                    {filteredModels.length === 0 && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="text-center py-16"
                        >
                            <Search size={48} className="text-slate-500 mx-auto mb-4" />
                            <h3 className="text-xl font-semibold text-slate-300 mb-2">No models found</h3>
                            <p className="text-slate-400">Try adjusting your search or filter criteria</p>
                        </motion.div>
                    )}

                    {/* Stats Footer */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                        className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6"
                    >
                        <div className="rounded-xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 backdrop-blur-xl p-6 text-center">
                            <div className="text-3xl font-bold text-blue-400 mb-2">{MODELS.length}</div>
                            <div className="text-sm text-slate-400">Total Models</div>
                        </div>
                        <div className="rounded-xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 backdrop-blur-xl p-6 text-center">
                            <div className="text-3xl font-bold text-violet-400 mb-2">{categories.length - 1}</div>
                            <div className="text-sm text-slate-400">Categories</div>
                        </div>
                        <div className="rounded-xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 backdrop-blur-xl p-6 text-center">
                            <div className="text-3xl font-bold text-emerald-400 mb-2">Enterprise</div>
                            <div className="text-sm text-slate-400">Grade Quality</div>
                        </div>
                    </motion.div>
                </div>
            </div>
        </div>
    );
}

function ModelCard({ model, Icon, index, onClick }) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            onClick={onClick}
            className={`group rounded-2xl bg-gradient-to-br ${colorClasses[model.color]} border backdrop-blur-xl p-6 cursor-pointer transition-all hover:scale-105 hover:shadow-xl`}
        >
            {/* Icon */}
            <div className={`w-14 h-14 rounded-xl ${iconColorClasses[model.color]} border flex items-center justify-center mb-4`}>
                <Icon size={28} />
            </div>

            {/* Category Badge */}
            <span className="inline-block px-2 py-1 rounded-md text-xs font-medium bg-white/10 text-slate-300 mb-3">
                {model.category}
            </span>

            {/* Title */}
            <h3 className="text-lg font-semibold text-slate-100 mb-2 group-hover:text-white transition-colors">
                {model.name}
            </h3>

            {/* Description */}
            <p className="text-sm text-slate-400 mb-4 line-clamp-3">
                {model.description}
            </p>

            {/* Domains */}
            <div className="flex flex-wrap gap-2 mb-4">
                {model.domains.slice(0, 2).map((domain) => (
                    <span key={domain} className="px-2 py-1 rounded text-xs bg-white/10 text-slate-300">
                        {domain}
                    </span>
                ))}
                {model.domains.length > 2 && (
                    <span className="px-2 py-1 rounded text-xs bg-white/10 text-slate-300">
                        +{model.domains.length - 2}
                    </span>
                )}
            </div>

            {/* CTA */}
            <div className="flex items-center gap-2 text-sm font-medium text-slate-300 group-hover:text-white transition-colors">
                <span>View Details</span>
                <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
            </div>

            {/* Sparkle effect on hover */}
            <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
                <Sparkles size={20} className={iconColorClasses[model.color].split(' ')[2]} />
            </div>
        </motion.div>
    );
}