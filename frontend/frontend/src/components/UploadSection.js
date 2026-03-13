import React, { useState, useRef } from "react";

const SYMPTOMS = [
  { label: "Persistent Cough",     value: "persistent_cough" },
  { label: "Chest Pain",           value: "chest_pain" },
  { label: "Weight Loss",          value: "weight_loss" },
  { label: "Shortness of Breath",  value: "shortness_breath" },
  { label: "Hemoptysis",           value: "hemoptysis" },
  { label: "Fatigue",              value: "fatigue" },
];

function UploadSection({ onSubmit, error }) {
  const [file,     setFile]     = useState(null);
  const [preview,  setPreview]  = useState(null);
  const [dragging, setDragging] = useState(false);
  const [age,      setAge]      = useState(50);
  const [smoking,  setSmoking]  = useState("never");
  const [symptoms, setSymptoms] = useState([]);
  const inputRef = useRef();

  const handleFile = (f) => {
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const toggleSymptom = (val) => {
    setSymptoms(prev =>
      prev.includes(val) ? prev.filter(s => s !== val) : [...prev, val]
    );
  };

  const handleSubmit = () => {
    if (!file) return;
    const fd = new FormData();
    fd.append("image", file);
    fd.append("age", age);
    fd.append("smoking_status", smoking);
    symptoms.forEach(s => fd.append("symptoms", s));
    onSubmit(fd);
  };

  return (
    <div className="upload-section">

      {/* Hero */}
      <div className="upload-hero">
        <h1>
          Early Detection<br />
          <span className="line2">Saves Lives.</span>
        </h1>
        <p>Upload a CT scan and provide patient information for AI-powered pulmonary malignancy analysis.</p>
      </div>

      {/* Drop Zone */}
      <div
        className={`drop-zone ${dragging ? "active" : ""}`}
        onClick={() => inputRef.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault(); setDragging(false);
          const f = e.dataTransfer.files[0];
          if (f) handleFile(f);
        }}
      >
        <input
          ref={inputRef} type="file" accept="image/*"
          style={{ display: "none" }}
          onChange={(e) => e.target.files[0] && handleFile(e.target.files[0])}
        />
        {!preview ? (
          <>
            <div className="drop-icon">🔬</div>
            <div className="drop-title">Drop CT Scan Here</div>
            <div className="drop-sub">PNG, JPG, JPEG · Click or drag & drop</div>
          </>
        ) : (
          <>
            <img src={preview} alt="CT Preview" className="preview-img" />
            <div className="drop-sub" style={{ marginTop: "0.5rem" }}>
              ✅ {file.name} — click to change
            </div>
          </>
        )}
      </div>

      {/* Patient Form */}
      <div className="patient-form">
        <div className="form-title">Patient Information</div>
        <div className="form-grid">
          <div className="form-group">
            <label>Age</label>
            <input
              type="number" min="1" max="120"
              value={age}
              onChange={(e) => setAge(e.target.value)}
            />
          </div>
          <div className="form-group">
            <label>Smoking History</label>
            <select value={smoking} onChange={(e) => setSmoking(e.target.value)}>
              <option value="never">Never Smoked</option>
              <option value="former">Former Smoker</option>
              <option value="current">Current Smoker</option>
            </select>
          </div>
          <div className="form-group full">
            <label>Symptoms</label>
            <div className="checkbox-group">
              {SYMPTOMS.map(s => (
                <label
                  key={s.value}
                  className={`symptom-chip ${symptoms.includes(s.value) ? "checked" : ""}`}
                >
                  <input
                    type="checkbox"
                    checked={symptoms.includes(s.value)}
                    onChange={() => toggleSymptom(s.value)}
                  />
                  {s.label}
                </label>
              ))}
            </div>
          </div>
        </div>
      </div>

      {error && <div className="error-msg">⚠️ {error}</div>}

      <button className="submit-btn" onClick={handleSubmit} disabled={!file}>
        Analyze CT Scan →
      </button>

    </div>
  );
}

export default UploadSection;