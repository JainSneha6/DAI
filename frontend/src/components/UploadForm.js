import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function UploadForm() {
  const [formData, setFormData] = useState({
    businessName: '',
    industry: '',
    salesData: null,
    customerData: null,
    inventoryData: null,
    marketingCampaignsData: null,
  });

  const navigate = useNavigate();

  const handleFileChange = (event) => {
    const { name, files } = event.target;
    setFormData({ ...formData, [name]: files[0] });
  };

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!formData.salesData || !formData.customerData || !formData.inventoryData || !formData.marketingCampaignsData) {
      alert("Please upload all required files.");
      return;
    }

    const formDataToSend = new FormData();
    formDataToSend.append('salesData', formData.salesData);
    formDataToSend.append('customerData', formData.customerData);
    formDataToSend.append('inventoryData', formData.inventoryData);
    formDataToSend.append('marketingCampaignsData', formData.marketingCampaignsData);
    formDataToSend.append('businessName', formData.businessName);
    formDataToSend.append('industry', formData.industry);

    try {
      const response = await fetch('http://localhost:5000/api/upload-business-data', {
        method: 'POST',
        body: formDataToSend,
      });

      if (response.ok) {
        const data = await response.json();
        console.log('Success:', data);
        alert('Files uploaded successfully! You can now ask a question.');
        navigate('/virtual-consultant');
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
    <div className="relative bg-cover bg-center min-h-screen text-white" style={{ backgroundImage: "url('/background.png')" }}>
      <div className="absolute inset-0 bg-black bg-opacity-50 backdrop-blur-sm">
        <div className="relative flex flex-col items-center justify-center h-screen px-6">
          <div className="max-w-xl mx-auto p-6 bg-gradient-to-br from-blue-100 to-blue-200 rounded-lg shadow-lg border border-blue-300">
            <h2 className="text-3xl font-bold mb-4 text-blue-800 text-center">Upload Your Business Data</h2>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-gray-700 mb-1" htmlFor="businessName">Business Name</label>
                <input
                  type="text"
                  id="businessName"
                  name="businessName"
                  value={formData.businessName}
                  onChange={handleChange}
                  className="border border-blue-300 text-black p-2 rounded-lg w-full focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-300"
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
                  className="border border-blue-300 p-2 text-black rounded-lg w-full focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-300"
                  required
                >
                  <option value="">Select your industry</option>
                  <option value="technology">Technology</option>
                  <option value="healthcare">Healthcare</option>
                  <option value="finance">Finance</option>
                  <option value="education">Education</option>
                  <option value="retail">Retail</option>
                  <option value="manufacturing">Manufacturing</option>
                  <option value="e-commerce">E-commerce</option>
                  <option value="others">Others</option>
                </select>
              </div>

              {/* File Upload Sections */}
              <div>
                <label className="block text-gray-700 mb-1" htmlFor="salesData">Sales Data</label>
                <input
                  type="file"
                  id="salesData"
                  name="salesData"
                  onChange={handleFileChange}
                  className="border text-black border-blue-300 p-2 rounded-lg w-full focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-300"
                  accept=".csv, .xlsx, .json"
                  required
                />
              </div>



              <div>
                <label className="block text-gray-700 mb-1" htmlFor="customerData">Customer Data</label>
                <input
                  type="file"
                  id="customerData"
                  name="customerData"
                  onChange={handleFileChange}
                  className="border text-black border-blue-300 p-2 rounded-lg w-full focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-300"
                  accept=".csv, .xlsx, .json"
                  required
                />
              </div>

              <div>
                <label className="block text-gray-700 mb-1" htmlFor="inventoryData">Inventory Data</label>
                <input
                  type="file"
                  id="inventoryData"
                  name="inventoryData"
                  onChange={handleFileChange}
                  className="border text-black border-blue-300 p-2 rounded-lg w-full focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-300"
                  accept=".csv, .xlsx, .json"
                  required
                />
              </div>

              <div>
                <label className="block text-gray-700 mb-1" htmlFor="marketingCampaignsData">Marketing Campaigns Data</label>
                <input
                  type="file"
                  id="marketingCampaignsData"
                  name="marketingCampaignsData"
                  onChange={handleFileChange}
                  className="border text-black border-blue-300 p-2 rounded-lg w-full focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-300"
                  accept=".csv, .xlsx, .json"
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
        </div>
      </div>
    </div>
  );
}

export default UploadForm;
