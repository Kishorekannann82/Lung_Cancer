import React, { useState } from "react";
import Header from "./components/Header";
import UploadSection from "./components/UploadSection";
import ResultDashboard from "./components/ResultDashboard";
import "./App.css";

function App() {
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const handleSubmit = async (formData) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res  = await fetch("/api/predict", {
        method: "POST",
        body:   formData,
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="bg-grid" />
      <div className="bg-glow" />
      <Header />
      <main className="main">
        {!result && !loading && (
          <UploadSection onSubmit={handleSubmit} error={error} />
        )}
        {loading && (
          <div className="analyzing">
            <div className="scanner">
              <div className="scanner-line" />
              <div className="scanner-ring r1" />
              <div className="scanner-ring r2" />
              <div className="scanner-ring r3" />
            </div>
            <p className="analyzing-text">Analyzing CT Scan</p>
            <p className="analyzing-sub">Running CNN · Grad-CAM · Risk Assessment</p>
          </div>
        )}
        {result && (
          <ResultDashboard result={result} onReset={() => setResult(null)} />
        )}
      </main>
    </div>
  );
}

export default App;