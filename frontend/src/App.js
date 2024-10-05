import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage'; // Adjust the path as necessary
import VirtualConsultant from './components/VirtualConsultant'; // Import the VirtualConsultant component
import NavBar from './components/NavBar';

function App() {
  return (
    <Router>
      <NavBar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/virtual-consultant" element={<VirtualConsultant />} />
      </Routes>
    </Router>
  );
}

export default App;
