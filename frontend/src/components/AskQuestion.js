import React, { useState } from 'react';

function AskQuestion() {
  const [question, setQuestion] = useState('');

  const handleSubmit = (event) => {
    event.preventDefault();
    alert(`Your question has been submitted: "${question}"`);
    // Here you could add code to handle the question, e.g., send it to the server
    setQuestion('');
  };

  return (
    <div className="max-w-xl mx-auto p-6 bg-gradient-to-br from-blue-100 to-blue-200 rounded-lg shadow-lg border border-blue-300">
      <h2 className="text-3xl font-bold mb-4 text-blue-800 text-center">Ask Your Question</h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-gray-700 mb-1" htmlFor="question">Your Question</label>
          <textarea 
            id="question" 
            value={question} 
            onChange={(e) => setQuestion(e.target.value)} 
            className="border border-blue-300 p-2 rounded-lg w-full focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-300" 
            rows="3" 
            placeholder="Type your question here..."
            required
          />
        </div>
        <button 
          type="submit" 
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 rounded-lg transition duration-300"
        >
          Submit Question
        </button>
      </form>
    </div>
  );
}

export default AskQuestion;
