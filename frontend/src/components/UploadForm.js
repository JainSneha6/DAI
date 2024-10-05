import React, { useState } from 'react';

function UploadForm() {
  const [formData, setFormData] = useState({
    businessName: '',
    industry: '',
    file: null,
    description: '',
    contactEmail: '',
  });

  const handleFileChange = (event) => {
    setFormData({ ...formData, file: event.target.files[0] });
  };

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    
    if (!formData.file) {
      alert("Please select a file.");
      return;
    }

    const formDataToSend = new FormData();
    formDataToSend.append('file', formData.file);
    formDataToSend.append('businessName', formData.businessName);
    formDataToSend.append('industry', formData.industry);
    formDataToSend.append('description', formData.description);
    formDataToSend.append('contactEmail', formData.contactEmail);

    try {
      const response = await fetch('/api/upload-business-data', {
        method: 'POST',
        body: formDataToSend,
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Success:', data);
        alert('File uploaded successfully!');
        // Reset form fields
        setFormData({
          businessName: '',
          industry: '',
          file: null,
          description: '',
          contactEmail: '',
        });
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
    <div className="max-w-xl mx-auto p-6 bg-gradient-to-br from-blue-100 to-blue-200 rounded-lg shadow-lg border border-blue-300">
      <h2 className="text-3xl font-bold mb-4 text-blue-800 text-center">Upload Your Business Data</h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Business Name */}
        <div>
          <label className="block text-gray-700 mb-1" htmlFor="businessName">Business Name</label>
          <input 
            type="text" 
            id="businessName" 
            name="businessName" 
            value={formData.businessName} 
            onChange={handleChange} 
            className="border border-blue-300 p-2 rounded-lg w-full focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-300" 
            placeholder="Enter your business name"
            required
          />
        </div>

        {/* Industry Selection */}
        <div>
          <label className="block text-gray-700 mb-1" htmlFor="industry">Industry</label>
          <select 
            id="industry" 
            name="industry" 
            value={formData.industry} 
            onChange={handleChange} 
            className="border border-blue-300 p-2 text-gray-700 rounded-lg w-full focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-300"
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
        <div>
          <label className="block text-gray-700 mb-1 " htmlFor="file">Select File</label>
          <input 
            type="file" 
            id="file" 
            onChange={handleFileChange} 
            className="border text-black border-blue-300 p-2 rounded-lg w-full focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-300"
            accept=".csv, .xlsx, .json" 
            required
          />
        </div>

        {/* Description */}
        <div>
          <label className="block text-gray-700 mb-1" htmlFor="description">Description</label>
          <textarea 
            id="description" 
            name="description" 
            value={formData.description} 
            onChange={handleChange} 
            className="border border-blue-300 p-2 rounded-lg w-full focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-300" 
            rows="2" // Shorter height for the textarea
            placeholder="Enter a brief description of the data..."
            required
          />
        </div>

        {/* Contact Email */}
        <div>
          <label className="block text-gray-700 mb-1" htmlFor="contactEmail">Contact Email</label>
          <input 
            type="email" 
            id="contactEmail" 
            name="contactEmail" 
            value={formData.contactEmail} 
            onChange={handleChange} 
            className="border border-blue-300 p-2 rounded-lg w-full focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-300" 
            placeholder="Enter your contact email"
            required
          />
        </div>

        <button 
          type="submit" 
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 rounded-lg transition duration-300"
        >
          Upload
        </button>
      </form>
    </div>
  );
}

export default UploadForm;
