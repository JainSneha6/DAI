import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage'; 
import VirtualConsultant from './components/VirtualConsultant'; 
import NavBar from './components/NavBar';
import AskQuestion from './components/AskQuestion';
import UploadForm from './components/UploadForm';

function App() {
  return (
    <Router>
      <NavBar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/upload" element={<UploadForm/>}/>
        <Route path="/virtual-consultant" element={<VirtualConsultant />} />
        <Route path="/ask-question" element={<AskQuestion />} />
      </Routes>
    </Router>
  );
}

export default App;
