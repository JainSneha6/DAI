import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './pages/Homepage';
import TimeSeriesPage from './pages/TimeSeriesPage';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path='/time-series' element = {<TimeSeriesPage />} />
      </Routes>
    </Router>
  );
}

export default App;