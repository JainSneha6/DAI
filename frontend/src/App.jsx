import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './pages/Homepage';
import Input from './pages/Input';
import TimeSeries from './pages/TimeSeries';
import Dashboard from './pages/Dashboard';
import Chatbot from './pages/ChatWithData';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path='/input' element = {<Input />} />
        <Route path="/dashboard" element = {<Dashboard/>} />
        <Route path="/chat" element = {<Chatbot/>} />
        <Route path="/time-series" element = {<TimeSeries/>} />
      </Routes>
    </Router>
  );
}

export default App;