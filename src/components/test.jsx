import React, { useState } from 'react';
import axios from 'axios';
import '../App.css';
import Icon_Balance from '../assets/Images/Icon_balance_scale.png';
import jsPDF from 'jspdf';

function UndertrialDashboard() {
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
    const { name, value } = e.target;
    setFormData((prevFormData) => ({
      ...prevFormData,
      [name]: name === 'served_half_term' ? value === 'true' : value, // Handle boolean conversion
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
    doc.text(`bail eligibility: ${formData.penalty}`, 20, 130);
    
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
        crimes: selectedCrimes.join(', '), // Join selected crimes for backend
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
            <img src={Icon_Balance} alt="" />
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
              <label>
                Case Type
                <input
                  type="text"
                  name="case_id"
                  placeholder="Case Number"
                  value={formData.case_id}
                  onChange={handleInputChange}
                />
              </label>
              <label>
                Name of Undertrial
                <input
                  type="text"
                  name="name_of_undertrial"
                  placeholder="Name of Undertrial"
                  value={formData.name_of_undertrial}
                  onChange={handleInputChange}
                />
              </label>

              <div className="dropdown">
                <button onClick={toggleDropdown} className="dropdown-button">
                  Select Crimes
                </button>

                {dropdownOpen && (
                  <div className="dropdown-content">
                    {crimeOptions.map((crime, index) => (
                      <label key={index}>
                        <input
                          type="checkbox"
                          value={crime}
                          onChange={handleCheckboxChange}
                          checked={selectedCrimes.includes(crime)}
                        />
                        {crime}
                      </label>
                    ))}
                  </div>
                )}

                <p>Selected Crimes: {selectedCrimes.join(', ') || 'None'}</p>
              </div>

              <label>
                Imprisonment Served
                <input
                  type="text"
                  name="imprisonment_duration_served"
                  placeholder="Years of imprisonment"
                  value={formData.imprisonment_duration_served}
                  onChange={handleInputChange}
                />
              </label>
              <label>
                Bail Bond
                <input
                  type="text"
                  name="bail_bond"
                  placeholder="Bail Bond"
                  value={formData.bail_bond}
                  onChange={handleInputChange}
                />
              </label>
              <label>
                Surety Bond
                <input
                  type="text"
                  name="surety_bond"
                  placeholder="Surety Bond"
                  value={formData.surety_bond}
                  onChange={handleInputChange}
                />
              </label>
              <label>
                EVER ESCAPE
                <input
                  type="number"
                  name="risk_of_escape"
                  placeholder="0-5"
                  min="0"
                  max="5"
                  value={formData.risk_of_escape}
                  onChange={handleInputChange}
                />
              </label>
              <label>
                Risk of Case
                <input
                  type="number"
                  name="risk_of_influence"
                  placeholder="0-5"
                  min="0"
                  max="5"
                  value={formData.risk_of_influence}
                  onChange={handleInputChange}
                />
              </label>
              <label>
                Served Half Term
                <select
                  name="served_half_term"
                  value={formData.served_half_term.toString()}
                  onChange={handleInputChange}
                >
                  <option value="true">Yes</option>
                  <option value="false">No</option>
                </select>
              </label>
              <label>
                Penalty
                <select
                  name="penalty"
                  value={formData.penalty}
                  onChange={handleInputChange}
                >
                  <option value="Fine">Fine</option>
                  <option value="Imprisonment">Imprisonment</option>
                  <option value="Both">Both</option>
                </select>
              </label>
              <button className="submit-button" type="submit">Submit Details</button>
            </form>
          </div>

          <div className="nlp-results">
            <h3>NLP & ML Results</h3>
            <label>
            {prediction !== null && (
        <div className="mt-4 p-4 bg-gray-100 rounded">
          <h2 className="text-xl font-semibold mb-2">Prediction Result</h2>
          <p>{prediction ? 'Eligible for Bail' : 'Not Eligible for Bail'}</p>
        </div>
      )}
            
            </label>
          
          </div>
        </div>

        <div className="recommendations">
          <h3>AI Recommendations</h3>
          <div className="recommendation-content">
            <div className="recommendation-left">
              <p>Based on the charges and case details, it is recommended to highlight the following points in the bail application:</p>
              <ul>
                <li>No prior criminal record</li>
                <li>Strong family ties in the community</li>
                <li>Employment history showing stability</li>
                <li>Health concerns requiring regular treatment</li>
              </ul>
            </div>
            <div className="recommendation-right">
              <p>Additional considerations:</p>
              <ul>
                <li>Possibility of electronic monitoring</li>
                <li>Willingness to surrender passport</li>
                <li>Proposed guarantor details</li>
                <li>Community service options</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="steps">
          <h3>Steps to Apply for Bail</h3>
          <ul>
            <li>Gather all necessary documents (ID, case file, medical records)</li>
            <li>Fill out the bail application form accurately</li>
            <li>Submit the application to the court or relevant authority</li>
            <li>Attend the bail hearing, presenting the case details and recommendations</li>
            <li>Follow up with the court for the bail decision</li>
          </ul>
        </div>
      </main>

      <footer className="footer">
        <ul>
          <li>Contact Us</li>
          <li>Privacy Policy</li>
          <li>Terms of Service</li>
        </ul>
      </footer>

      {/* Display prediction result */}
    
    </div>
  );
}

export default UndertrialDashboard;




