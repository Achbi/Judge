// src/App.js
import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert('Please select a PDF file to upload.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setMessage(response.data.message);
    } catch (error) {
      console.error('Error uploading PDF:', error);
      setMessage('Error uploading PDF.');
    }
  };

  const handleAskQuestion = async () => {
    if (!question) {
      alert('Please enter a question.');
      return;
    }

    try {
      const response = await axios.post('http://127.0.0.1:5000/ask', { question });
      setAnswer(response.data.answer);
    } catch (error) {
      console.error('Error asking question:', error);
      setAnswer('Error getting answer.');
    }
  };

  return (
    <div className="App">
      <h1>PDF Upload and Question Answer</h1>
      <div>
        <input type="file" onChange={handleFileChange} />
        <button onClick={handleUpload}>Upload PDF</button>
        <p>{message}</p>
      </div>
      <div>
        <h2>Ask a Question</h2>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Enter your question"
        />
        <button onClick={handleAskQuestion}>Ask</button>
        <p>Answer: {answer}</p>
      </div>
    </div>
  );
}

export default App;
