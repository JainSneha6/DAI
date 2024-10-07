import React from 'react';
import { Link } from 'react-router-dom';
import VoiceAssistant from './VoiceAssistant';

function NavBar() {
  return (
    <nav className="fixed w-full z-10 top-0">
      <div className="container mx-auto px-6 py-4 flex justify-between items-center">

        <a href="#" className="text-white text-3xl font-bold">DecisivAI</a>

        <div className="flex space-x-6">
          <Link to="/" className="text-white hover:text-blue-500 transition duration-300">Home</Link>
          <Link to="/upload" className="text-white hover:text-blue-500 transition duration-300">Your Business</Link>
          <Link to="/virtual-consultant" className="text-white hover:text-blue-500 transition duration-300">Virtual Consultant</Link>
          <Link to="/insight-bot" className="text-white hover:text-blue-500 transition duration-300">InsightBot</Link>
          <Link to="/grouped-analysis" className="text-white hover:text-blue-500 transition duration-300">Dashboard</Link>
          <VoiceAssistant /> 
        </div>
      </div>
    </nav>
  );
}

export default NavBar;
