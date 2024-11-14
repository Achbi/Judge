import React, { useState } from 'react';
import axios from 'axios';
import '../App.css';
import Icon_Balance from '../assets/Images/Icon_balance_scale.png';

const LegalAidDashboard = () => {
  const [formData, setFormData] = useState({
    clientName: '',
    lawyerName: '',
    clientId: '',
    caseDetails: ''
  });
  const [bailApplication, setBailApplication] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevState => ({
      ...prevState,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    setBailApplication('');

    try {
      const response = await axios.post('http://localhost:5000/api/generate_bail_application', {
        lawyer_name: formData.lawyerName,
        input_text: formData.caseDetails,
        n_words: 200
      });

      if (response.data.success) {
        setBailApplication(response.data.bail_application);
      } else {
        setError(response.data.error || 'Failed to generate bail application. Please try again.');
      }
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="legal-aid-dashboard">
      <header className="header">
        <div className="logo">
          <img src={Icon_Balance} alt="Balance Scale Icon" />
          <h1>Bail Reckoner</h1>
        </div>
      </header>

      <main className="main-content">
        <section className="content">
          <h2>Client Details</h2>
          <form onSubmit={handleSubmit}>
            <label htmlFor="clientName">
              Client Name:
              <input
                id="clientName"
                type="text"
                name="clientName"
                placeholder="Client Name"
                required
                value={formData.clientName}
                onChange={handleChange}
              />
            </label>
            <label htmlFor="lawyerName">
              Lawyer Name:
              <input
                id="lawyerName"
                type="text"
                name="lawyerName"
                placeholder="Lawyer Name"
                required
                value={formData.lawyerName}
                onChange={handleChange}
              />
            </label>
            <label htmlFor="clientId">
              Client Id:
              <input
                id="clientId"
                type="text"
                name="clientId"
                placeholder="Client Id"
                required
                value={formData.clientId}
                onChange={handleChange}
              />
            </label>
            <label htmlFor="caseDetails">
              Case Details:
              <textarea
                id="caseDetails"
                name="caseDetails"
                placeholder="Case Details"
                required
                value={formData.caseDetails}
                onChange={handleChange}
              />
            </label>
            <button className="submit-button" type="submit" disabled={isLoading}>
              {isLoading ? 'Generating...' : 'Generate Bail Application'}
            </button>
          </form>

          {error && <p className="error-message">{error}</p>}

          {bailApplication && (
            <div className="bail-application">
              <h3>Generated Bail Application</h3>
              <pre>{bailApplication}</pre>
            </div>
          )}

          <div className="summary-resources">
            <div className="case-summary">
              <h3>Case Summary</h3>
              <p>
                View and manage case details, track progress, and update
                information as needed.
              </p>
            </div>
            <div className="resources">
              <h3>Resources</h3>
              <p>
                Access legal documents, templates, and other resources to assist
                in case management.
              </p>
            </div>
          </div>
        </section>
      </main>

      <footer className="footer">
        <p>© 2023 Bail Reckoner. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default LegalAidDashboard;