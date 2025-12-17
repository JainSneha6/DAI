import React from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import Homepage from "./pages/Homepage";
import UploadPage from "./pages/Input";
import Dashboard from "./pages/Dashboard";
import ChatWithData from "./pages/ChatWithData";
import FileAnalysisPage from "./pages/FileAnalysisPage";
import ModelsOverviewPage from "./pages/ModelsOverviewPage";
import TimeSeriesChat from "./pages/TimeSeries";
import MarketingChat from "./pages/MarketingChat";

function App() {
  return (
    <Router>
      <Routes>
        {/* Homepage - No sidebar (landing page) */}
        <Route
          path="/"
          element={
            <Layout showSidebar={false}>
              <Homepage />
            </Layout>
          }
        />

        {/* Upload Page - With sidebar */}
        <Route
          path="/upload"
          element={
            <Layout>
              <UploadPage />
            </Layout>
          }
        />

        {/* Dashboard - With sidebar */}
        <Route
          path="/dashboard"
          element={
            <Layout>
              <Dashboard />
            </Layout>
          }
        />

        {/* Models Overview - With sidebar */}
        <Route
          path="/models"
          element={
            <Layout>
              <ModelsOverviewPage />
            </Layout>
          }
        />

        {/* Chat/Analysis - With sidebar */}
        <Route
          path="/chat"
          element={
            <Layout>
              <ChatWithData />
            </Layout>
          }
        />

        {/* File Analysis Page - With sidebar */}
        <Route
          path="/analysis/:filename"
          element={
            <Layout>
              <FileAnalysisPage />
            </Layout>
          }
        />

        <Route path="/sales-forecasting" element={
          <Layout>
            <TimeSeriesChat />
          </Layout>
        } />

        <Route path="/marketing-roi" element={
          <Layout>
            <MarketingChat />
          </Layout>
        } />

        {/* Redirect unknown routes to homepage */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}

export default App;