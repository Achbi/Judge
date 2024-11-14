import React, { useState } from 'react';
import axios from 'axios';
import jsPDF from 'jspdf';
import '../App.css';
import Icon_Balance from '../assets/Images/Icon_balance_scale.png';

function UndertrialDashboar() {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [selectedCrimes, setSelectedCrimes] = useState([]);
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
  const [prediction, setPrediction] = useState(null);

  const crimeOptions = [
    'Cyber Crimes',
    'Crimes Against SCs and STs',
    'Crimes Against Women',
    'Crimes Against Children',
    'Offences Against the State',
    'Economic Offence',
    'Crimes Against Foreigners',
  ];

  const toggleDropdown = (event) => {
    event.preventDefault();
    setDropdownOpen(!dropdownOpen);
  };

  const handleCheckboxChange = (event) => {
    const { value, checked } = event.target;
    if (checked) {
      setSelectedCrimes([...selectedCrimes, value]);
    } else {
      setSelectedCrimes(selectedCrimes.filter((crime) => crime !== value));
    }
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prevFormData) => ({
      ...prevFormData,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const generatePDF = () => {
    const doc = new jsPDF();
    
    doc.setFontSize(16);
    doc.text('Undertrial Case Details', 20, 20);
    
    doc.setFontSize(12);
    doc.text(`Case Number: ${formData.case_id}`, 20, 40);
    doc.text(`Name of Undertrial: ${formData.name_of_undertrial}`, 20, 50);
    doc.text(`Crimes: ${selectedCrimes.join(', ')}`, 20, 60);
    doc.text(`Imprisonment Served: ${formData.imprisonment_duration_served} years`, 20, 70);
    doc.text(`Bail Bond: ${formData.bail_bond}`, 20, 80);
    doc.text(`Surety Bond: ${formData.surety_bond}`, 20, 90);
    doc.text(`Risk of Escape: ${formData.risk_of_escape}`, 20, 100);
    doc.text(`Risk of Influence: ${formData.risk_of_influence}`, 20, 110);
    doc.text(`Served Half Term: ${formData.served_half_term ? 'Yes' : 'No'}`, 20, 120);
    doc.text(`Penalty: ${formData.penalty}`, 20, 130);
    
    if (prediction !== null) {
      doc.text(`Bail Eligibility: ${prediction ? 'Eligible' : 'Not Eligible'}`, 20, 150);
    }

    doc.save('undertrial_case_details.pdf');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', {
        ...formData,
        crimes: selectedCrimes.join(', '),
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
      });
      setPrediction(response.data.bail_eligibility);
      generatePDF();
    } catch (error) {
      console.error('Error during Axios request:', error);
      alert('An error occurred while submitting the form.');
    }
  };

  return (
    <div className="UndertrialDashboard">
      <header className="header">
        <div className='logo'>
          <img src={Icon_Balance} alt="Balance Icon" />
          <h1>Bail Reckoner</h1>
        </div>
        
        <div className="profile-menu">
          <span>Home</span>
          <span>Legal Aid Provider</span>
          <span>Judicial Authority</span>
        </div>
      </header>
      
      <main className="main-content">
        <h2>Undertrial Prisoner Dashboard</h2>
        <div className="dashboard-content">
          <div className="case-details">
            <h3>Case Details</h3>
            <form onSubmit={handleSubmit}>
              <input
                type="text"
                name="case_id"
                value={formData.case_id}
                onChange={handleInputChange}
                placeholder="Case ID"
                required
              />
              <input
                type="text"
                name="name_of_undertrial"
                value={formData.name_of_undertrial}
                onChange={handleInputChange}
                placeholder="Name of Undertrial"
                required
              />
              <input
                type="number"
                name="imprisonment_duration_served"
                value={formData.imprisonment_duration_served}
                onChange={handleInputChange}
                placeholder="Imprisonment Duration Served"
                required
              />
              <input
                type="number"
                name="bail_bond"
                value={formData.bail_bond}
                onChange={handleInputChange}
                placeholder="Bail Bond"
                required
              />
              <input
                type="number"
                name="surety_bond"
                value={formData.surety_bond}
                onChange={handleInputChange}
                placeholder="Surety Bond"
                required
              />
              <label>
                Served Half Term:
                <input
                  type="checkbox"
                  name="served_half_term"
                  checked={formData.served_half_term}
                  onChange={handleInputChange}
                />
              </label>
              <button className="submit-button" type="submit">Submit Details</button>
            </form>
          </div>

          <div className="nlp-results">
            <h3>NLP & ML Results</h3>
            {prediction !== null && (
              <div className="mt-4 p-4 bg-gray-100 rounded">
                <h2 className="text-xl font-semibold mb-2">Prediction Result</h2>
                <p>{prediction ? 'Eligible for Bail' : 'Not Eligible for Bail'}</p>
              </div>
            )}
          </div>
        </div>

        <div className="recommendations">
          <h3>AI Recommendations</h3>
          {/* Include recommendations content */}
        </div>

        <div className="steps">
          <h3>Steps to Apply for Bail</h3>
          {/* Include steps content */}
        </div>

        <button onClick={generatePDF} className="generate-pdf-button">Generate PDF</button>
      </main>

      <footer className="footer">
        <ul>
          <li>Contact Us</li>
          <li>Privacy Policy</li>
          <li>Terms of Service</li>
        </ul>
      </footer>
    </div>
  );
}

export default UndertrialDashboar;
