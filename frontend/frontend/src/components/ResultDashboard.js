import React from "react";

function ResultDashboard({ result, onReset }) {
  const {
    classification, malignant_prob, benign_prob,
    risk, staging, gradcam_image, recommendations, disclaimer
  } = result;

  const isMalignant = classification === "Malignant";
  const tier        = risk.risk_tier.toLowerCase();

  // Staging helpers
  const stage         = staging?.cancer_stage;
  const tumor         = staging?.tumor_size;
  const survival      = staging?.survival_rate;
  const stageColor    = stage ? {
    "IA": "#00e676", "IB": "#69f0ae",
    "II": "#ffeb3b", "III": "#ff9800", "IV": "#f44336", "N/A": "#aaa"
  }[stage.stage_roman] || "#aaa" : "#aaa";
  const survivalColor = survival ? {
    "Excellent": "#00e676", "Good": "#69f0ae",
    "Moderate": "#ffeb3b", "Guarded": "#ff9800", "Poor": "#f44336"
  }[survival.survival_category] || "#aaa" : "#aaa";

  return (
    <div className="result-dashboard">

      {/* ── Top 3 Cards ── */}
      <div className="result-top">

        <div className="card">
          <div className="card-label">Classification</div>
          <div className={`card-value ${isMalignant ? "malignant" : "benign"}`}>
            {classification}
          </div>
          <div className="card-sub">
            {isMalignant ? "⚠️ Malignancy detected" : "✅ No malignancy detected"}
          </div>
          <div className="prob-bar">
            <div
              className={`prob-fill ${isMalignant ? "danger" : ""}`}
              style={{ width: `${(isMalignant ? malignant_prob : benign_prob) * 100}%` }}
            />
          </div>
          <div className="card-sub" style={{ marginTop: "0.4rem" }}>
            Confidence: {((isMalignant ? malignant_prob : benign_prob) * 100).toFixed(1)}%
          </div>
        </div>

        <div className="card">
          <div className="card-label">Risk Tier</div>
          <div className={`card-value ${tier}`}>{risk.risk_tier}</div>
          <div className="card-sub">Composite risk score</div>
          <div className="prob-bar">
            <div className="prob-fill danger" style={{ width: `${risk.risk_score * 100}%` }} />
          </div>
          <div className="card-sub" style={{ marginTop: "0.4rem" }}>
            Score: {risk.risk_score.toFixed(4)}
          </div>
        </div>

        <div className="card">
          <div className="card-label">Probabilities</div>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem", marginTop: "0.3rem" }}>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.72rem", marginBottom: "0.3rem" }}>
                <span style={{ color: "var(--danger)" }}>Malignant</span>
                <span>{(malignant_prob * 100).toFixed(1)}%</span>
              </div>
              <div className="prob-bar">
                <div className="prob-fill danger" style={{ width: `${malignant_prob * 100}%` }} />
              </div>
            </div>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.72rem", marginBottom: "0.3rem" }}>
                <span style={{ color: "var(--accent2)" }}>Benign</span>
                <span>{(benign_prob * 100).toFixed(1)}%</span>
              </div>
              <div className="prob-bar">
                <div className="prob-fill" style={{ width: `${benign_prob * 100}%` }} />
              </div>
            </div>
          </div>
        </div>

      </div>

      {/* ── Risk Factor Breakdown ── */}
      <div className="risk-factors">
        {[
          { label: "Malignancy Prob", value: (risk.malignancy_prob * 100).toFixed(1) + "%" },
          { label: "Age Factor",      value: (risk.age_factor * 100).toFixed(0) + "%" },
          { label: "Smoking Factor",  value: (risk.smoking_factor * 100).toFixed(0) + "%" },
          { label: "History Factor",  value: (risk.history_factor * 100).toFixed(0) + "%" },
        ].map(f => (
          <div className="factor-card" key={f.label}>
            <div className="factor-label">{f.label}</div>
            <div className="factor-value">{f.value}</div>
          </div>
        ))}
      </div>

      {/* ── Staging Row ── */}
      {staging && isMalignant && (
        <div className="staging-section">
          <div className="section-title">🔬 Advanced Analysis</div>
          <div className="staging-cards">

            {/* Cancer Stage */}
            <div className="staging-card">
              <div className="staging-icon">🎯</div>
              <div className="staging-label">Cancer Stage</div>
              <div className="staging-value" style={{ color: stageColor }}>
                Stage {stage.stage_roman}
              </div>
              <div className="staging-sub">{stage.stage_detail}</div>
              <div className="staging-desc">{stage.description}</div>
              <div className="staging-tnm">
                <span>T: {tumor?.t_classification}</span>
                <span>N: {stage.n_class}</span>
                <span>M: {stage.m_class}</span>
              </div>
            </div>

            {/* Tumor Size */}
            <div className="staging-card">
              <div className="staging-icon">📏</div>
              <div className="staging-label">Tumor Size Estimate</div>
              <div className="staging-value" style={{ color: "#64ffda" }}>
                {tumor?.size_range_cm}
              </div>
              <div className="staging-sub">{tumor?.size_category}</div>
              <div className="staging-desc">
                T-Class: {tumor?.t_classification}
              </div>
              <div className="staging-desc">
                Grad-CAM activation: {tumor?.activation_percent}%
              </div>
            </div>

            {/* Survival Rate */}
            <div className="staging-card">
              <div className="staging-icon">📊</div>
              <div className="staging-label">5-Year Survival Rate</div>
              <div className="staging-value" style={{ color: survivalColor }}>
                {survival?.five_year_survival}%
              </div>
              <div className="staging-sub">{survival?.survival_category} Prognosis</div>
              <div className="staging-desc">{survival?.note}</div>
            </div>

          </div>
        </div>
      )}

      {/* Benign staging message */}
      {staging && !isMalignant && (
        <div className="staging-section">
          <div className="staging-cards">
            <div className="staging-card benign-staging">
              <div className="staging-icon">✅</div>
              <div className="staging-label">Staging</div>
              <div className="staging-value" style={{ color: "#00e676" }}>N/A</div>
              <div className="staging-sub">No malignancy — staging not required</div>
              <div className="staging-desc">5-Year Survival: {survival?.five_year_survival}% — Excellent</div>
            </div>
          </div>
        </div>
      )}

      {/* ── Grad-CAM + CDSS ── */}
      <div className="result-mid">

        <div className="card gradcam-card">
          <div className="card-label">Grad-CAM Heatmap</div>
          <div className="card-sub">XAI — regions influencing prediction</div>
          <img
            src={`data:image/png;base64,${gradcam_image}`}
            alt="Grad-CAM Heatmap"
            className="gradcam-img"
          />
          <div className="legend">
            <div className="legend-item"><div className="legend-dot" style={{ background: "#ff0000" }} />High influence</div>
            <div className="legend-item"><div className="legend-dot" style={{ background: "#ffff00" }} />Medium</div>
            <div className="legend-item"><div className="legend-dot" style={{ background: "#0000ff" }} />Low influence</div>
          </div>
        </div>

        <div className="card cdss-card">
          <div className="card-label">Clinical Decision Support</div>
          <div className="card-sub">Powered by Groq · LLaMA 3.3 70B</div>
          <div className="cdss-body">{recommendations}</div>
          <div className="disclaimer">{disclaimer}</div>
        </div>

      </div>

      <button className="reset-btn" onClick={onReset}>
        ← Analyze Another Scan
      </button>

    </div>
  );
}

export default ResultDashboard;