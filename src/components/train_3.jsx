import React, { useState } from 'react';
import axios from 'axios';

function Train() {
    const [formData, setFormData] = useState({
        offense_category: '',
        risk_of_escape: '',
        risk_of_influence: '',
        charges: '',
        served_half_term: '',
        imprisonment_duration_served: ''  // <-- Add this
    });

    const [prediction, setPrediction] = useState(null);

    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const response = await axios.post('http://localhost:5000/predict', formData);
            setPrediction(response.data.bail_eligibility);
        } catch (error) {
            console.error('Error making prediction', error);
        }
    };

    return (
        <div className="App">
            <h1>Bail Eligibility Prediction</h1>
            <form onSubmit={handleSubmit}>
                <label>Offense Category</label>
                <input name="offense_category" value={formData.offense_category} onChange={handleChange} />

                <label>Risk of Escape</label>
                <input name="risk_of_escape" value={formData.risk_of_escape} onChange={handleChange} />

                <label>Risk of Influence</label>
                <input name="risk_of_influence" value={formData.risk_of_influence} onChange={handleChange} />

                <label>Charges</label>
                <input name="charges" value={formData.charges} onChange={handleChange} />

                <label>Served Half Term</label>
                <input name="served_half_term" value={formData.served_half_term} onChange={handleChange} />

                <label>Imprisonment Duration Served</label>  {/* <-- Add this */}
                <input name="imprisonment_duration_served" value={formData.imprisonment_duration_served} onChange={handleChange} />

                <button type="submit">Predict Bail Eligibility</button>
            </form>

            {prediction !== null && (
                <div>
                    <h2>Bail Eligibility: {prediction === 1 ? "Eligible" : "Not Eligible"}</h2>
                </div>
            )}
        </div>
    );
}

export default Train;
