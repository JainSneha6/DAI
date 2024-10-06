import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Bar } from 'react-chartjs-2'; // Import Bar chart from Chart.js
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function Dashboard() {
    const [groupedAnalysis, setGroupedAnalysis] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleAnalysis = async (dataType) => {
        setLoading(true);
        setError('');
        setGroupedAnalysis(null);

        try {
            const response = await axios.post('http://localhost:5000/api/grouped-analysis', {
                dataType,
            });
            console.log(response.data.grouped_analysis)
            if (response.data && response.data.grouped_analysis) {
                setGroupedAnalysis(response.data.grouped_analysis);
                console.log(groupedAnalysis)
            } else {
                setError('No analysis data returned.');
            }
        } catch (error) {
            setError(`Failed to analyze the ${dataType}. Please try again.`);
            console.error('Error details:', error); // Log the error details
        } finally {
            setLoading(false);
        }
    };

    const renderChart = (data) => {
        const labels = Object.keys(data);
        const values = Object.values(data);

        const chartData = {
            labels: labels,
            datasets: [
                {
                    label: 'Analysis Data',
                    data: values,
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                },
            ],
        };

        return (
            <Bar
                data={chartData}
                options={{
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'top',
                        },
                        title: {
                            display: true,
                            text: 'Grouped Analysis Chart',
                        },
                    },
                }}
            />
        );
    };

    return (
        <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 p-6 overflow-y-auto relative">
            <div className="container mx-auto">
                <h1 className="text-4xl font-bold mb-6 text-center text-white">Grouped Analysis Dashboard</h1>

                {/* Buttons to trigger different data analyses */}
                <div className="flex justify-center mb-6 space-x-4">
                    <button type="button" className="bg-gradient-to-r from-blue-500 to-blue-700 text-white font-bold py-2 px-4 rounded shadow hover:from-blue-600 hover:to-blue-800 transition duration-200" onClick={() => handleAnalysis('salesData')}>Sales Data</button>
                    <button type="button" className="bg-gradient-to-r from-green-500 to-green-700 text-white font-bold py-2 px-4 rounded shadow hover:from-green-600 hover:to-green-800 transition duration-200" onClick={() => handleAnalysis('customerData')}>Customer Data</button>
                    <button type="button" className="bg-gradient-to-r from-yellow-500 to-yellow-700 text-white font-bold py-2 px-4 rounded shadow hover:from-yellow-600 hover:to-yellow-800 transition duration-200" onClick={() => handleAnalysis('inventoryData')}>Inventory Data</button>
                    <button type="button" className="bg-gradient-to-r from-purple-500 to-purple-700 text-white font-bold py-2 px-4 rounded shadow hover:from-purple-600 hover:to-purple-800 transition duration-200" onClick={() => handleAnalysis('marketingCampaignsData')}>Marketing Data</button>
                </div>

                {/* Loading state */}
                {loading && <p className="text-blue-300 text-center">Loading analysis data...</p>}

                {/* Render the chart based on grouped analysis data */}
                {groupedAnalysis && Object.entries(groupedAnalysis).map(([category, data]) => (
                    <div key={category} className="mb-6">
                        <h3 className="text-2xl font-bold mb-2 text-gray-300">{category} Analysis</h3>
                        {renderChart(data)}
                    </div>
                ))}

                {error && <p className="text-red-400 text-center">{error}</p>}
            </div>
        </div>
    );
}

export default Dashboard;
