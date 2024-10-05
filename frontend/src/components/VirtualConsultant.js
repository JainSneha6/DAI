import React from 'react';
import AskQuestion from './AskQuestion';

function VirtualConsultant() {
  return (
    <div className="relative bg-cover bg-center min-h-screen text-white" style={{ backgroundImage: "url('/background.png')" }}>
      <div className="absolute inset-0 bg-black bg-opacity-50 backdrop-blur-sm"></div>
      <div className="relative flex flex-col items-center justify-center h-screen px-6">
        <h1 className="text-5xl font-bold mb-6 text-center">Virtual Consultant</h1>
        <AskQuestion/>
      </div>
    </div>
  );
}

export default VirtualConsultant;
