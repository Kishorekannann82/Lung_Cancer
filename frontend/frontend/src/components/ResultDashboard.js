import React from "react";

function ResultDashboard({ result, onReset }) {
  const {
    classification, malignant_prob, benign_prob,
    risk, gradcam_image, recommendations, disclaimer
  } = result;

  const isMalignant = classification === "Malignant";
  const tier = risk.risk_tier.toLowerCase();

  return (
    <div className="result-dashboard">

      {/* ── Top 3 Cards ── */}
      <div className="result-top">

        {/* Classification */}
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

        {/* Risk Tier */}
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

        {/* Probabilities */}
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

      {/* ── Grad-CAM + CDSS ── */}
      <div className="result-mid">

        {/* Grad-CAM Heatmap */}
        <div className="card gradcam-card">
          <div className="card-label">Grad-CAM Heatmap</div>
          <div className="card-sub">XAI — regions influencing prediction</div>
          <img
            src={`data:image/png;base64,${gradcam_image}`}
            alt="Grad-CAM Heatmap"
            className="gradcam-img"
          />
          <div className="legend">
            <div className="legend-item">
              <div className="legend-dot" style={{ background: "#ff0000" }} />
              High influence
            </div>
            <div className="legend-item">
              <div className="legend-dot" style={{ background: "#ffff00" }} />
              Medium
            </div>
            <div className="legend-item">
              <div className="legend-dot" style={{ background: "#0000ff" }} />
              Low influence
            </div>
          </div>
        </div>

        {/* CDSS Recommendations */}
        <div className="card cdss-card">
          <div className="card-label">Clinical Decision Support</div>
          <div className="card-sub">Powered by Groq · LLaMA 3.3 70B</div>
          <div className="cdss-body">{recommendations}</div>
          <div className="disclaimer">{disclaimer}</div>
        </div>

      </div>

      {/* ── Reset ── */}
      <button className="reset-btn" onClick={onReset}>
        ← Analyze Another Scan
      </button>

    </div>
  );
}

export default ResultDashboard;