import React, { useState } from 'react';
import axios from 'axios';

function Chatt() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleFileUpload = async () => {
    const formData = new FormData();
    formData.append('pdf', selectedFile);

    try {
      const response = await axios.post('http://127.0.0.1:5000/process-pdf', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      alert('PDF Processed Successfully');
    } catch (error) {
      console.error('Error uploading PDF:', error);
    }
  };

  const handleQuestionSubmit = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/ask-question', { question });
      setAnswer(response.data.answer);
    } catch (error) {
      console.error('Error submitting question:', error);
    }
  };

  return (
    <div className="App">
      <h1>PDF Chatbot</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleFileUpload}>Upload PDF</button>
      <br />
      <input
        type="text"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask a question"
      />
      <button onClick={handleQuestionSubmit}>Ask</button>
      <div>
        <h2>Answer:</h2>
        <p>{answer}</p>
      </div>
    </div>
  );
}

export default Chatt;
