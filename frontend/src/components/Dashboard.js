import React, { useState } from 'react';
import axios from 'axios';
import { Bar, Pie } from 'react-chartjs-2'; 
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
    ArcElement
} from 'chart.js';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement);

function Dashboard() {
    const [analysisResult, setAnalysisResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [selectedDataset, setSelectedDataset] = useState(null); // State to store selected dataset

    const handleAnalysis = async (dataType) => {
        setLoading(true);
        setError('');
        setAnalysisResult(null);

        try {
            const response = await axios.post('http://localhost:5000/api/grouped-analysis', { dataType });
            setAnalysisResult(response.data);
        } catch (error) {
            setError(`Failed to analyze the ${dataType}. Please try again.`);
        } finally {
            setLoading(false);
        }
    };

    // Dropdown to select the dataset to display
    const renderDatasetDropdown = (datasets, setSelectedDataset) => {
        return (
            <select
                className="mb-4 bg-gray-700 text-white p-2 rounded"
                onChange={(e) => setSelectedDataset(e.target.value)}
                defaultValue=""
            >
                <option value="" disabled>Select Field</option>
                {datasets.map((dataset, idx) => (
                    <option key={idx} value={dataset}>{dataset}</option>
                ))}
            </select>
        );
    };

    // Function to generate grouped bar chart for numerical data grouped by categorical data
    const renderGroupedBarChart = (groupedAnalysis) => {
        return Object.entries(groupedAnalysis).map(([category, data]) => {
            const labels = data.map((row) => row[category]); // Use category names as labels
            const availableDatasets = Object.keys(data[0]).filter(key => key !== category); // Available datasets (Sum, Mean, etc.)
            
            const selectedData = selectedDataset ? [selectedDataset] : availableDatasets; // Show the selected dataset or all by default

            const datasets = selectedData.map((key) => ({
                label: key,
                data: data.map((row) => row[key]),
                backgroundColor: (context) => {
                    const chart = context.chart;
                    const { ctx, chartArea } = chart;
                    
                    if (!chartArea) return null;

                    // Create a vertical gradient for bars
                    const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
                    gradient.addColorStop(0, `rgba(${Math.floor(Math.random() * 255)}, ${Math.floor(Math.random() * 255)}, ${Math.floor(Math.random() * 255)}, 0.9)`);
                    gradient.addColorStop(1, `rgba(${Math.floor(Math.random() * 255)}, ${Math.floor(Math.random() * 255)}, ${Math.floor(Math.random() * 255)}, 0.6)`);
                    return gradient;
                }
            }));

            const chartData = {
                labels: labels, // X-axis labels (e.g., city names)
                datasets: datasets, // Y-axis datasets (numerical values)
            };

            return (
                <div key={category} className="mb-6">
                    <h3 className="text-2xl font-bold mb-2 text-gray-300">{category} Analysis</h3>
                    
                    {/* Dataset selection dropdown */}
                    {renderDatasetDropdown(availableDatasets, setSelectedDataset)}

                    <Bar
                        data={chartData}
                        options={{
                            responsive: true,
                            plugins: {
                                legend: { position: 'top' },
                                title: { display: true, text: `${category} Grouped Analysis` },
                            },
                            scales: {
                                x: {
                                    title: {
                                        display: true,
                                        text: `${category}`, // X-axis label
                                        color: '#ffffff',
                                        font: {
                                            size: 16,
                                            weight: 'bold',
                                        },
                                    },
                                    ticks: {
                                        color: '#ffffff', // X-axis tick color
                                    },
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: 'Numerical Values (Sum, Mean, Max, Min)', // Y-axis title
                                        color: '#ffffff',
                                        font: {
                                            size: 16,
                                            weight: 'bold',
                                        },
                                    },
                                    ticks: {
                                        color: '#ffffff', // Y-axis tick color
                                    },
                                },
                            },
                        }}
                    />
                </div>
            );
        });
    };

    // Function to generate pie charts for categorical analysis
    const renderCategoricalAnalysisCharts = (categorical) => {
        return Object.keys(categorical).map((colName) => {
            const data = {
                labels: Object.keys(categorical[colName].frequency), // X-axis: category names
                datasets: [
                    {
                        label: `${colName} Frequency`,
                        data: Object.values(categorical[colName].frequency),
                        backgroundColor: (context) => {
                            const chart = context.chart;
                            const { ctx, chartArea } = chart;

                            if (!chartArea) return null;

                            // Generate gradients for each slice in the pie chart
                            return Object.keys(categorical[colName].frequency).map(() => {
                                const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
                                gradient.addColorStop(0, `rgba(${Math.floor(Math.random() * 255)}, ${Math.floor(Math.random() * 255)}, ${Math.floor(Math.random() * 255)}, 0.9)`);
                                gradient.addColorStop(1, `rgba(${Math.floor(Math.random() * 255)}, ${Math.floor(Math.random() * 255)}, ${Math.floor(Math.random() * 255)}, 0.6)`);
                                return gradient;
                            });
                        }
                    },
                ],
            };

            return (
                <div key={colName} className="mb-6">
                    <h3 className="text-2xl font-bold mb-2 text-gray-300">{colName} Analysis</h3>
                    <Pie
                        data={data}
                        options={{
                            responsive: true,
                            plugins: {
                                legend: { position: 'top' },
                                title: { display: true, text: `${colName} Data Analysis` },
                            },
                        }}
                    />
                </div>
            );
        });
    };

    return (
        <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 p-6 overflow-y-auto relative">
            <div className="container mx-auto">
                <h1 className="text-4xl font-bold mb-6 text-center text-white">Dashboard</h1>

                <div className="flex justify-center mb-6 space-x-4">
                    <button type="button" className="bg-gradient-to-r from-blue-500 to-blue-700 text-white font-bold py-2 px-4 rounded shadow hover:from-blue-600 hover:to-blue-800 transition duration-200" onClick={() => handleAnalysis('salesData')}>Sales Data</button>
                    <button type="button" className="bg-gradient-to-r from-green-500 to-green-700 text-white font-bold py-2 px-4 rounded shadow hover:from-green-600 hover:to-green-800 transition duration-200" onClick={() => handleAnalysis('customerData')}>Customer Data</button>
                    <button type="button" className="bg-gradient-to-r from-yellow-500 to-yellow-700 text-white font-bold py-2 px-4 rounded shadow hover:from-yellow-600 hover:to-yellow-800 transition duration-200" onClick={() => handleAnalysis('inventoryData')}>Inventory Data</button>
                    <button type="button" className="bg-gradient-to-r from-purple-500 to-purple-700 text-white font-bold py-2 px-4 rounded shadow hover:from-purple-600 hover:to-purple-800 transition duration-200" onClick={() => handleAnalysis('marketingCampaignsData')}>Marketing Data</button>
                </div>

                {loading && <p className="text-blue-300 text-center">Loading analysis data...</p>}
                {error && <p className="text-red-400 text-center">{error}</p>}

                {analysisResult && (
                    <div className="mt-8 p-6 bg-gray-900 rounded-lg shadow-lg">
                        {/* Render Categorical Data Analysis */}
                        {analysisResult.categorical && Object.keys(analysisResult.categorical).length > 0 && (
                            <>
                                {renderCategoricalAnalysisCharts(analysisResult.categorical)}
                            </>
                        )}

                        {/* Render Grouped Numerical Data Analysis */}
                        {analysisResult.grouped_analysis && Object.keys(analysisResult.grouped_analysis).length > 0 && (
                            <>
                                {renderGroupedBarChart(analysisResult.grouped_analysis)}
                            </>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

export default Dashboard;
