import React from 'react';
import NavBar from '../components/NavBar';

function HomePage() {
  return (
    <>
    <NavBar/>
    <div className="relative bg-cover bg-center min-h-screen text-white" style={{ backgroundImage: "url('/background.png')" }}>
      <div className="absolute inset-0 bg-black bg-opacity-50 backdrop-blur-sm"></div>
      <div className="relative flex flex-col items-center justify-center h-screen px-6">
        <h1 className="text-5xl font-bold mb-6 text-center">
          Welcome to DecisivAI
        </h1>
        <p className="text-2xl text-center mb-6 max-w-2xl">
          Empowering businesses with AI-driven insights for smarter, faster, and more informed decision-making. DecisivAI acts as your virtual consultant—simulating scenarios, visualizing decisions, and analyzing insights to keep your business ahead.
        </p>
      </div>
    </div>
    </>
  );
}

export default HomePage;
