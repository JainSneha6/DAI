import React, { useState } from 'react';

function UploadForm() {
  const [businessName, setBusinessName] = useState('');
  const [industry, setIndustry] = useState('');
  const [file, setFile] = useState(null);
  const [description, setDescription] = useState('');
  const [contactEmail, setContactEmail] = useState('');

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleBusinessNameChange = (event) => {
    setBusinessName(event.target.value);
  };

  const handleIndustryChange = (event) => {
    setIndustry(event.target.value);
  };

  const handleDescriptionChange = (event) => {
    setDescription(event.target.value);
  };

  const handleContactEmailChange = (event) => {
    setContactEmail(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    
    if (!file) {
      alert("Please select a file.");
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('businessName', businessName);
    formData.append('industry', industry);
    formData.append('description', description);
    formData.append('contactEmail', contactEmail);

    try {
      const response = await fetch('/api/upload-business-data', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Success:', data);
        alert('File uploaded successfully!');
        // Reset form fields
        setBusinessName('');
        setIndustry('');
        setFile(null);
        setDescription('');
        setContactEmail('');
      } else {
        console.error('Error:', response.statusText);
        alert('File upload failed.');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('An error occurred during file upload.');
    }
  };

  return (
    <div className="max-w-lg mx-auto p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4">Upload Your Business Data</h2>
      <form onSubmit={handleSubmit}>
        {/* Business Name */}
        <div className="mb-4">
          <label className="block text-gray-700 mb-2" htmlFor="businessName">Business Name</label>
          <input 
            type="text" 
            id="businessName" 
            value={businessName} 
            onChange={handleBusinessNameChange} 
            className="border border-gray-300 p-2 rounded w-full" 
            placeholder="Enter your business name"
            required
          />
        </div>

        {/* Industry Selection */}
        <div className="mb-4">
          <label className="block text-gray-700 mb-2" htmlFor="industry">Industry</label>
          <select 
            id="industry" 
            value={industry} 
            onChange={handleIndustryChange} 
            className="border border-gray-300 p-2 rounded w-full"
            required
          >
            <option value="">Select your industry</option>
            <option value="technology">Technology</option>
            <option value="healthcare">Healthcare</option>
            <option value="finance">Finance</option>
            <option value="education">Education</option>
            <option value="retail">Retail</option>
            <option value="manufacturing">Manufacturing</option>
            <option value="others">Others</option>
          </select>
        </div>

        {/* File Upload */}
        <div className="mb-4">
          <label className="block text-gray-700 mb-2" htmlFor="file">Select File</label>
          <input 
            type="file" 
            id="file" 
            onChange={handleFileChange} 
            className="border border-gray-300 p-2 rounded w-full"
            accept=".csv, .xlsx, .json" // Specify accepted file types
            required
          />
        </div>

        {/* Description */}
        <div className="mb-4">
          <label className="block text-gray-700 mb-2" htmlFor="description">Description</label>
          <textarea 
            id="description" 
            value={description} 
            onChange={handleDescriptionChange} 
            className="border border-gray-300 p-2 rounded w-full" 
            rows="4"
            placeholder="Enter a brief description of the data..."
            required
          />
        </div>

        {/* Contact Email */}
        <div className="mb-4">
          <label className="block text-gray-700 mb-2" htmlFor="contactEmail">Contact Email</label>
          <input 
            type="email" 
            id="contactEmail" 
            value={contactEmail} 
            onChange={handleContactEmailChange} 
            className="border border-gray-300 p-2 rounded w-full" 
            placeholder="Enter your contact email"
            required
          />
        </div>

        <button 
          type="submit" 
          className="bg-pink-500 hover:bg-pink-700 text-white font-bold py-2 px-4 rounded"
        >
          Upload
        </button>
      </form>
    </div>
  );
}

export default UploadForm;
