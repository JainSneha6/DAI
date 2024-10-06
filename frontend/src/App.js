import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage'; 
import VirtualConsultant from './components/VirtualConsultant'; 
import NavBar from './components/NavBar';
import UploadForm from './components/UploadForm';
import InsightBot from './components/InsightBot';

function App() {
  return (
    <Router>
      <NavBar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/upload" element={<UploadForm/>}/>
        <Route path="/virtual-consultant" element={<VirtualConsultant />} />
        <Route path='/insight-bot' element={<InsightBot/>}/>
      </Routes>
    </Router>
  );
}

export default App;
