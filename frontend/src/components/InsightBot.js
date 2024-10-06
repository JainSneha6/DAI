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
      console.log(analysisResult);
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

  const [selectedGroupedDataset, setSelectedGroupedDataset] = useState('');

const renderGroupedDatasetDropdown = (datasets, setSelectedDataset) => {
    return (
        <select
            className="mb-4 bg-gray-700 text-white p-2 rounded"
            onChange={(e) => setSelectedDataset(e.target.value)}
            defaultValue=""
        >
            <option value="" disabled>Select Grouped Field</option>
            {datasets.map((dataset, idx) => (
                <option key={idx} value={dataset}>{dataset}</option>
            ))}
        </select>
    );
};

const renderGroupedAnalysis = (groupedAnalysis) => {
    return Object.entries(groupedAnalysis).map(([category, data]) => {
        const availableDatasets = Object.keys(data[0]).filter(key => key !== category); // Filter keys for available datasets

        return (
            <div key={category} className="mb-6">
                <h3 className="text-2xl font-bold mb-2 text-gray-300">{category} Analysis</h3>
                {renderGroupedDatasetDropdown(availableDatasets, setSelectedGroupedDataset)}
                <table className="min-w-full bg-gray-800 border border-gray-700">
                    <thead>
                        <tr>
                            <th className="py-2 px-4 text-left text-gray-400">Details</th>
                            {selectedGroupedDataset ? (
                                <th className="py-2 px-4 text-left text-gray-400">{selectedGroupedDataset}</th>
                            ) : (
                                availableDatasets.map((key) => (
                                    <th key={key} className="py-2 px-4 text-left text-gray-400">{key}</th>
                                ))
                            )}
                        </tr>
                    </thead>
                    <tbody>
                        {data.map((row, idx) => (
                            <tr key={idx} className="border-b border-gray-700">
                                <td className="py-2 px-4 text-gray-200">{row[category]}</td>
                                {selectedGroupedDataset ? (
                                    <td className="py-2 px-4 text-gray-200">{row[selectedGroupedDataset]}</td>
                                ) : (
                                    availableDatasets.map((key) => (
                                        key !== category && (
                                            <td key={key} className="py-2 px-4 text-gray-200">{row[key]}</td>
                                        )
                                    ))
                                )}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        );
    });
};



  const renderNumericalAnalysis = (numerical) => {
    return (
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
            {Object.entries(numerical).map(([colName, stats]) => (
              <tr key={colName} className="border-b border-gray-700">
                <td className="py-2 px-4 text-gray-200">{colName}</td>
                <td className="py-2 px-4 text-gray-200">{stats.sum}</td>
                <td className="py-2 px-4 text-gray-200">{stats.mean}</td>
                <td className="py-2 px-4 text-gray-200">{stats.max}</td>
                <td className="py-2 px-4 text-gray-200">{stats.min}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 p-6 overflow-y-auto">
      <div className="container mx-auto ">
        <h1 className="text-4xl font-bold mb-6 text-center text-white">InsightBot</h1>

        <div className="flex justify-center mb-6 space-x-4">
          <button type="button" className="bg-gradient-to-r from-blue-500 to-blue-700 text-white font-bold py-2 px-4 rounded shadow hover:from-blue-600 hover:to-blue-800 transition duration-200" onClick={() => handleAnalysis('salesData')}>Sales Data</button>
          <button type="button" className="bg-gradient-to-r from-green-500 to-green-700 text-white font-bold py-2 px-4 rounded shadow hover:from-green-600 hover:to-green-800 transition duration-200" onClick={() => handleAnalysis('customerData')}>Customer Data</button>
          <button type="button" className="bg-gradient-to-r from-yellow-500 to-yellow-700 text-white font-bold py-2 px-4 rounded shadow hover:from-yellow-600 hover:to-yellow-800 transition duration-200" onClick={() => handleAnalysis('inventoryData')}>Inventory Data</button>
          <button type="button" className="bg-gradient-to-r from-purple-500 to-purple-700 text-white font-bold py-2 px-4 rounded shadow hover:from-purple-600 hover:to-purple-800 transition duration-200" onClick={() => handleAnalysis('marketingCampaignsData')}>Marketing Data</button>
          <button type="button" className="bg-gradient-to-r from-teal-500 to-teal-700 text-white font-bold py-2 px-4 rounded shadow hover:from-teal-600 hover:to-teal-800 transition duration-200" onClick={() => setShowQueryBot((prev) => !prev)}>
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
                        <td className="py-2 px-4 text-gray-200">{renderFrequencyButtons(analysisResult.categorical[colName].frequency)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {renderNumericalAnalysis(analysisResult.numerical)}

            {analysisResult.grouped_analysis && Object.keys(analysisResult.grouped_analysis).length > 0 && (
              <>
                {renderGroupedAnalysis(analysisResult.grouped_analysis)}
              </>
            )}
          </div>
        )}

        {showQueryBot && <QueryBot />}
      </div>
    </div>
  );
}

export default InsightBot;
