import React, { useState } from 'react';
import axios from 'axios';
import QueryBot from './QueryBot'; // Import the QueryBot component

function InsightBot() {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showQueryBot, setShowQueryBot] = useState(false); // State to toggle QueryBot

  const handleAnalysis = async (dataType) => {
    setLoading(true);
    setError('');
    setAnalysisResult(null);

    try {
      const response = await axios.post('http://localhost:5000/api/analyze-data', {
        dataType,
      });

      setAnalysisResult(response.data);
    } catch (error) {
      setError(`Failed to analyze the ${dataType}. Please try again.`);
    } finally {
      setLoading(false);
    }
  };

  const renderFrequencyButtons = (frequency) => {
    return (
      <div className="flex flex-wrap gap-2">
        {Object.entries(frequency).map(([key, value]) => (
          <button
            key={key}
            className="bg-gray-700 text-gray-200 font-semibold py-1 px-3 rounded hover:bg-gray-600 transition duration-200"
            title={`Frequency: ${value}`} 
          >
            {key}
          </button>
        ))}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 p-6">
      <div className="container mx-auto">
        <h1 className="text-4xl font-bold mb-6 text-center text-white">InsightBot</h1>

        <div className="flex justify-center mb-6 space-x-4">
          <button
            type="button"
            className="bg-gradient-to-r from-blue-500 to-blue-700 text-white font-bold py-2 px-4 rounded shadow hover:from-blue-600 hover:to-blue-800 transition duration-200"
            onClick={() => handleAnalysis('salesData')}
          >
            Sales Data
          </button>
          <button
            type="button"
            className="bg-gradient-to-r from-green-500 to-green-700 text-white font-bold py-2 px-4 rounded shadow hover:from-green-600 hover:to-green-800 transition duration-200"
            onClick={() => handleAnalysis('customerData')}
          >
            Customer Data
          </button>
          <button
            type="button"
            className="bg-gradient-to-r from-yellow-500 to-yellow-700 text-white font-bold py-2 px-4 rounded shadow hover:from-yellow-600 hover:to-yellow-800 transition duration-200"
            onClick={() => handleAnalysis('inventoryData')}
          >
            Inventory Data
          </button>
          <button
            type="button"
            className="bg-gradient-to-r from-purple-500 to-purple-700 text-white font-bold py-2 px-4 rounded shadow hover:from-purple-600 hover:to-purple-800 transition duration-200"
            onClick={() => handleAnalysis('marketingCampaignsData')}
          >
            Marketing Data
          </button>
          <button
            type="button"
            className="bg-gradient-to-r from-teal-500 to-teal-700 text-white font-bold py-2 px-4 rounded shadow hover:from-teal-600 hover:to-teal-800 transition duration-200"
            onClick={() => setShowQueryBot((prev) => !prev)} // Toggle QueryBot
          >
            {showQueryBot ? 'Hide Bot' : 'Bot'}
          </button>
        </div>

        {loading && <p className="text-blue-300 text-center">Analyzing data, please wait...</p>}
        {error && <p className="text-red-400 text-center">{error}</p>}

        {analysisResult && (
          <div className="mt-8 p-6 bg-gray-900 rounded-lg shadow-lg">
            {analysisResult.categorical && Object.keys(analysisResult.categorical).length > 0 && (
              <div className="mb-6">
                <h3 className="text-2xl font-bold mb-2 text-gray-300">Categorical Data Analysis</h3>
                <table className="min-w-full bg-gray-800 border border-gray-700">
                  <thead>
                    <tr>
                      <th className="py-2 px-4 text-left text-gray-400">Column Name</th>
                      <th className="py-2 px-4 text-left text-gray-400">Unique Values</th>
                      <th className="py-2 px-4 text-left text-gray-400">Most Common Value</th>
                      <th className="py-2 px-4 text-left text-gray-400">Frequency</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.keys(analysisResult.categorical).map((colName, idx) => (
                      <tr key={idx} className="border-b border-gray-700">
                        <td className="py-2 px-4 text-gray-200">{colName}</td>
                        <td className="py-2 px-4 text-gray-200">{analysisResult.categorical[colName].unique_values}</td>
                        <td className="py-2 px-4 text-gray-200">{analysisResult.categorical[colName].most_common}</td>
                        <td className="py-2 px-4 text-gray-200">
                          {renderFrequencyButtons(analysisResult.categorical[colName].frequency)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {analysisResult.numerical && Object.keys(analysisResult.numerical).length > 0 && (
              <div className="mb-6">
                <h3 className="text-2xl font-bold mb-2 text-gray-300">Numerical Data Analysis</h3>
                <table className="min-w-full bg-gray-800 border border-gray-700">
                  <thead>
                    <tr>
                      <th className="py-2 px-4 text-left text-gray-400">Column Name</th>
                      <th className="py-2 px-4 text-left text-gray-400">Sum</th>
                      <th className="py-2 px-4 text-left text-gray-400">Mean</th>
                      <th className="py-2 px-4 text-left text-gray-400">Max</th>
                      <th className="py-2 px-4 text-left text-gray-400">Min</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.keys(analysisResult.numerical).map((colName, idx) => (
                      <tr key={idx} className="border-b border-gray-700">
                        <td className="py-2 px-4 text-gray-200">{colName}</td>
                        <td className="py-2 px-4 text-gray-200">{analysisResult.numerical[colName].sum}</td>
                        <td className="py-2 px-4 text-gray-200">{analysisResult.numerical[colName].mean}</td>
                        <td className="py-2 px-4 text-gray-200">{analysisResult.numerical[colName].max}</td>
                        <td className="py-2 px-4 text-gray-200">{analysisResult.numerical[colName].min}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {showQueryBot && <QueryBot />} 
      </div>
    </div>
  );
}

export default InsightBot;
