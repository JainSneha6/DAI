import React, { useState } from 'react';

function VirtualConsultant() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleQueryChange = (event) => {
    setQuery(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!query.trim()) return; // Prevent empty submissions

    const userMessage = { text: query, sender: 'user' };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);
    setQuery('');

    try {
      const response = await fetch('http://localhost:5000/consultant-query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (response.ok) {
        const data = await response.json();
        const botMessage = { text: data.answer, sender: 'bot' };
        setMessages((prev) => [...prev, botMessage]);
      } else {
        const errorMessage = { text: 'Error fetching answer. Please try again.', sender: 'bot' };
        setMessages((prev) => [...prev, errorMessage]);
      }
    } catch (error) {
      const errorMessage = { text: 'An error occurred. Please try again.', sender: 'bot' };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 p-6">
      <div className="container mx-auto">
        <header className="p-4 text-center font-bold text-4xl text-white mb-6">Virtual Consultant</header>
        <main className="flex-1 overflow-y-auto p-4">
          <div className="rounded-lg shadow-lg p-4 h-full overflow-auto">
            {messages.map((message, index) => (
              <div key={index} className={`mb-3 flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`flex items-center ${message.sender === 'user' ? 'flex-row-reverse' : ''}`}>
                  {message.sender === 'bot' && (
                    <img src="/bot-avatar.png" alt="Bot" className="w-8 h-8 rounded-full mr-2" />
                  )}
                  <div
                    className={`max-w-3xl p-3 rounded-lg text-sm shadow-md ${
                      message.sender === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-800'
                    }`}
                  >
                    {message.text}
                  </div>
                  {message.sender === 'user' && (
                    <img src="/user-avatar.png" alt="User" className="w-8 h-8 rounded-full ml-2" />
                  )}
                </div>
              </div>
            ))}
            {loading && <div className="text-center text-blue-300 animate-pulse">Loading...</div>}
          </div>
        </main>
        <footer className="p-4 shadow-md">
          <form onSubmit={handleSubmit} className="flex">
            <input
              type="text"
              value={query}
              onChange={handleQueryChange}
              className="border border-gray-300 text-black p-3 rounded-l-lg w-full focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-300"
              placeholder="Type your message..."
              required
            />
            <button
              type="submit"
              className="bg-gradient-to-r from-blue-500 to-blue-700 text-white font-bold py-3 px-4 rounded-r-lg transition duration-300 hover:from-blue-600 hover:to-blue-800"
            >
              Send
            </button>
          </form>
        </footer>
      </div>
    </div>
  );
}

export default VirtualConsultant;
