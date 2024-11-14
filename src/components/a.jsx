import React, { useState } from 'react';
import axios from 'axios';

const UndertrialDashboard = () => {
  // State to hold form data
  const [formData, setFormData] = useState({
    case_id: '',
    name_of_undertrial: '',
    crimes: '',
    imprisonment_duration_served: 0,
    bail_bond: 0,
    surety_bond: 0,
    risk_of_escape: 0,
    risk_of_influence: 0,
    served_half_term: false,
    penalty: 'Fine',
  });

  // State to hold prediction result
  const [prediction, setPrediction] = useState(null);

  // Handle form input changes
  const handleInputChange = (e) => {
    const { name, value } = e.target;

    setFormData((prevFormData) => ({
      ...prevFormData,
      [name]: name === 'served_half_term' ? value === 'true' : value, // For boolean fields like served_half_term
    }));
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // Make a POST request to the Flask backend
      const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
        headers: {
          'Content-Type': 'application/json',
        },
      });
      // Update the state with the prediction result
      setPrediction(response.data.bail_eligibility);
    } catch (error) {
      console.error('Error during Axios request:', error);
      alert('An error occurred while submitting the form.');
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Undertrial Prisoner Dashboard</h1>
      <form onSubmit={handleSubmit} className="mb-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Case ID */}
          <div>
            <label className="block mb-2">Case ID</label>
            <input
              type="text"
              name="case_id"
              value={formData.case_id}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
              required
            />
          </div>

          {/* Name of Undertrial */}
          <div>
            <label className="block mb-2">Name of Undertrial</label>
            <input
              type="text"
              name="name_of_undertrial"
              value={formData.name_of_undertrial}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
              required
            />
          </div>

          {/* Crimes */}
          <div>
            <label className="block mb-2">Crimes</label>
            <input
              type="text"
              name="crimes"
              value={formData.crimes}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
              required
            />
          </div>

          {/* Imprisonment Duration Served */}
          <div>
            <label className="block mb-2">Imprisonment Duration Served (in years)</label>
            <input
              type="number"
              name="imprisonment_duration_served"
              value={formData.imprisonment_duration_served}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
              required
            />
          </div>

          {/* Bail Bond */}
          <div>
            <label className="block mb-2">Bail Bond</label>
            <input
              type="text"
              name="bail_bond"
              value={formData.bail_bond}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
              required
            />
          </div>

          {/* Surety Bond */}
          <div>
            <label className="block mb-2">Surety Bond</label>
            <input
              type="text"
              name="surety_bond"
              value={formData.surety_bond}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
              required
            />
          </div>

          {/* Risk of Escape */}
          <div>
            <label className="block mb-2">Risk of Escape (0-5)</label>
            <input
              type="number"
              name="risk_of_escape"
              value={formData.risk_of_escape}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
              min="0"
              max="5"
              required
            />
          </div>

          {/* Risk of Influence */}
          <div>
            <label className="block mb-2">Risk of Influence (0-5)</label>
            <input
              type="number"
              name="risk_of_influence"
              value={formData.risk_of_influence}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
              min="0"
              max="5"
              required
            />
          </div>

          {/* Served Half Term */}
          <div>
            <label className="block mb-2">Served Half Term</label>
            <select
              name="served_half_term"
              value={formData.served_half_term.toString()}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
              required
            >
              <option value="true">Yes</option>
              <option value="false">No</option>
            </select>
          </div>

          {/* Penalty */}
          <div>
            <label className="block mb-2">Penalty</label>
            <select
              name="penalty"
              value={formData.penalty}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
              required
            >
              <option value="Fine">Fine</option>
              <option value="Imprisonment">Imprisonment</option>
              <option value="Both">Both</option>
            </select>
          </div>
        </div>

        <button
          type="submit"
          className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Submit
        </button>
      </form>

      {/* Display prediction result */}
      {prediction !== null && (
        <div className="mt-4 p-4 bg-gray-100 rounded">
          <h2 className="text-xl font-semibold mb-2">Prediction Result</h2>
          <p>Bail Eligibility: {prediction ? 'Eligible' : 'Not Eligible'}</p>
        </div>
      )}
    </div>
  );
};

export default UndertrialDashboard;
