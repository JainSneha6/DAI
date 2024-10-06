import React, { useState } from 'react';
import axios from 'axios';

function QueryBot() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleQuerySubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResponse('');

    try {
      const result = await axios.post('http://localhost:5000/query-analysis', { query });
      setResponse(result.data.answer);
    } catch (error) {
      setError('Failed to get a response. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-8 p-6 bg-gray-900 rounded-lg shadow-lg">
      <h3 className="text-2xl font-bold mb-4 text-gray-300">Bot</h3>
      <form onSubmit={handleQuerySubmit} className="flex">
        <input
          type="text"
          placeholder="Ask a question about the data..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="flex-grow p-2 border border-gray-700 rounded-l-lg bg-gray-800 text-gray-200"
        />
        <button
          type="submit"
          className="bg-blue-500 text-white font-bold py-2 px-4 rounded-r-lg hover:bg-blue-600 transition duration-200"
        >
          Ask
        </button>
      </form>
      {loading && <p className="text-blue-300 mt-4">Loading...</p>}
      {error && <p className="text-red-400 mt-4">{error}</p>}
      {response && (
        <div className="mt-4 p-4 bg-gray-800 rounded">
          <h4 className="text-gray-300">Response:</h4>
          <p className="text-gray-200">{response}</p>
        </div>
      )}
    </div>
  );
}

export default QueryBot;
