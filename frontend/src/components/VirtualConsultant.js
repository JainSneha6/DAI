import React, { useState, useEffect } from 'react';
import { FaStar } from 'react-icons/fa';
import { FaUser, FaRobot } from 'react-icons/fa'; // Importing icons for user and bot

function VirtualConsultant() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [feedback, setFeedback] = useState('');
  const [rating, setRating] = useState(0);
  const [feedbackVisible, setFeedbackVisible] = useState(false);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  useEffect(() => {
    const processAndStoreData = async () => {
      try {
        setInitialLoading(true);
        const response = await fetch('http://localhost:5000/process-and-store-data', {
          method: 'GET',
        });

        if (response.ok) {
          const data = await response.json();
          setMessages((prev) => [
            ...prev,
            { text: 'Data has been successfully processed and added to ChromaDB.', sender: 'bot' },
          ]);
        } else {
          setMessages((prev) => [
            ...prev,
            { text: 'Error processing data. Please try again.', sender: 'bot' },
          ]);
        }
      } catch (error) {
        console.error(error);
      } finally {
        setInitialLoading(false);
      }
    };

    processAndStoreData();
  }, []);

  const handleQueryChange = (event) => {
    setQuery(event.target.value);
  };

  const handleFeedbackChange = (event) => {
    setFeedback(event.target.value);
  };

  const handleRatingClick = (value) => {
    setRating(value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!query.trim()) return;

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

  const handleFeedbackSubmit = async (event) => {
    event.preventDefault();
    if (!feedback.trim() || !rating) return;

    try {
      const response = await fetch('http://localhost:5000/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ rating, comments: feedback }),
      });

      if (response.ok) {
        const successMessage = { text: 'Thank you for your feedback!', sender: 'bot' };
        setMessages((prev) => [...prev, successMessage]);
        setFeedbackSubmitted(true);
      } else {
        const errorMessage = { text: 'Error submitting feedback. Please try again.', sender: 'bot' };
        setMessages((prev) => [...prev, errorMessage]);
      }
    } catch (error) {
      const errorMessage = { text: 'An error occurred while submitting feedback.', sender: 'bot' };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setFeedback('');
      setRating(0);
      setFeedbackVisible(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 p-6">
      <div className="container mx-auto">
        <header className="p-4 text-center font-bold text-4xl text-white mb-6">Virtual Consultant</header>
        <main className="flex-1 overflow-y-auto p-4">
          <div className="rounded-lg shadow-lg p-4 h-full overflow-auto">
            {initialLoading ? (
              <div className="text-center text-blue-300 animate-pulse">Processing data...</div>
            ) : (
              <>
                {messages.map((message, index) => (
                  <div key={index} className={`mb-3 flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`flex items-center ${message.sender === 'user' ? 'flex-row-reverse' : ''}`}>
                      {message.sender === 'bot' ? (
                        <FaRobot className="w-8 h-8 text-gray-300 mr-2" />
                      ) : (
                        <FaUser className="w-8 h-8 text-blue-500 ml-2" />
                      )}
                      <div
                        className={`max-w-3xl p-3 rounded-lg text-sm shadow-md ${message.sender === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-800'
                          }`}
                      >
                        {message.text}
                      </div>
                    </div>
                  </div>
                ))}
              </>
            )}
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
      <button
        onClick={() => setFeedbackVisible(!feedbackVisible)}
        className="fixed bottom-4 right-4 bg-green-500 text-white font-bold py-3 px-5 rounded-full shadow-lg transition duration-300 hover:bg-green-600"
      >
        Feedback
      </button>

      {feedbackVisible && (
        <div className="fixed bottom-16 right-4 bg-white rounded-lg shadow-lg p-4 w-72 z-50">
          <h3 className="font-bold text-lg mb-2">Feedback</h3>
          <div className="flex mb-2">
            {[1, 2, 3, 4, 5].map((value) => (
              <FaStar
                key={value}
                className={`cursor-pointer ${rating >= value ? 'text-yellow-500' : 'text-gray-300'}`}
                onClick={() => handleRatingClick(value)}
                size={24}
              />
            ))}
          </div>
          <form onSubmit={handleFeedbackSubmit}>
            <input
              type="text"
              value={feedback}
              onChange={handleFeedbackChange}
              className="border border-gray-300 text-black p-2 mb-2 rounded w-full focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-300"
              placeholder="Your feedback..."
              required
            />
            <button
              type="submit"
              className="bg-gradient-to-r from-green-500 to-green-700 text-white font-bold py-2 rounded transition duration-300 hover:from-green-600 hover:to-green-800 w-full"
            >
              Submit
            </button>
          </form>
        </div>
      )}
    </div>
  );
}

export default VirtualConsultant;
