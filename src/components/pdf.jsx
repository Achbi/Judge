import React, { useState } from "react";
import axios from "axios";

const Appp = () => {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleDownload = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/generate_pdf",
        { text },
        {
          responseType: "blob",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "generated.pdf");
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Error generating PDF:", error);
      setError("Failed to generate PDF. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>Text to PDF Converter</h1>
      <form onSubmit={handleDownload}>
        <textarea
          rows="10"
          cols="50"
          value={text}
          onChange={(e) => setText(e.target.value)}
          required
          placeholder="Enter your text here"
        ></textarea>
        <br />
        <button type="submit" disabled={loading}>
          {loading ? "Generating PDF..." : "Download as PDF"}
        </button>
      </form>
      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
};

export default Appp;