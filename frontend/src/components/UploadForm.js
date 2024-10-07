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
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 p-6 overflow-y-auto">
      <div className="container mx-auto ">
        <h1 className="text-4xl font-bold mb-6 text-center text-white">Upload Your Business Data</h1>

        <div className="max-w-xl mx-auto bg-gray-900 p-6 rounded-lg shadow-lg">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-gray-400 mb-2" htmlFor="businessName">Business Name</label>
              <input
                type="text"
                id="businessName"
                name="businessName"
                value={formData.businessName}
                onChange={handleChange}
                className="w-full p-2 bg-gray-800 text-white border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter your business name"
                required
              />
            </div>

            <div>
              <label className="block text-gray-400 mb-2" htmlFor="industry">Industry</label>
              <select
                id="industry"
                name="industry"
                value={formData.industry}
                onChange={handleChange}
                className="w-full p-2 bg-gray-800 text-white border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
              >
                <option value="" disabled>Select your industry</option>
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

            <div>
              <label className="block text-gray-400 mb-2" htmlFor="salesData">Sales Data</label>
              <input
                type="file"
                id="salesData"
                name="salesData"
                onChange={handleFileChange}
                className="w-full p-2 bg-gray-800 text-white border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                accept=".csv, .xlsx, .json"
                required
              />
            </div>

            <div>
              <label className="block text-gray-400 mb-2" htmlFor="customerData">Customer Data</label>
              <input
                type="file"
                id="customerData"
                name="customerData"
                onChange={handleFileChange}
                className="w-full p-2 bg-gray-800 text-white border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                accept=".csv, .xlsx, .json"
                required
              />
            </div>

            <div>
              <label className="block text-gray-400 mb-2" htmlFor="inventoryData">Inventory Data</label>
              <input
                type="file"
                id="inventoryData"
                name="inventoryData"
                onChange={handleFileChange}
                className="w-full p-2 bg-gray-800 text-white border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                accept=".csv, .xlsx, .json"
                required
              />
            </div>

            <div>
              <label className="block text-gray-400 mb-2" htmlFor="marketingCampaignsData">Marketing Campaigns Data</label>
              <input
                type="file"
                id="marketingCampaignsData"
                name="marketingCampaignsData"
                onChange={handleFileChange}
                className="w-full p-2 bg-gray-800 text-white border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                accept=".csv, .xlsx, .json"
                required
              />
            </div>

            <button
              type="submit"
              className="w-full bg-gray-700 text-white font-bold py-2 px-4 rounded shadow hover:bg-gray-800 transition duration-200"
            >
              Upload
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default UploadForm;
