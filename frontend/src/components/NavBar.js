import React from 'react';

function NavBar() {
  return (
    <nav className="fixed w-full z-10 top-0">
      <div className="container mx-auto px-6 py-4 flex justify-between items-center">

        <a href="#" className="text-white text-3xl font-bold">DecisivAI</a>
        
        <div className="flex space-x-6">
          <a href="#" className="text-white hover:text-blue-500 transition duration-300">Home</a>
          <a href="#" className="text-white hover:text-blue-500 transition duration-300">Your Business</a>
          <a href="#" className="text-white hover:text-blue-500 transition duration-300">Features</a>
        </div>
      </div>
    </nav>
  );
}

export default NavBar;
