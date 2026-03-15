import React, { useState } from "react";

// ── Parse LLM text into sections ─────────────
function parseSections(text) {
  if (!text) return {};
  const sections = {};
  const sectionPatterns = [
    { key: "diagnosis",    pattern: /##\s*DIAGNOSIS SUMMARY([\s\S]*?)(?=##|$)/i },
    { key: "tests",        pattern: /##\s*RECOMMENDED DIAGNOSTIC TESTS([\s\S]*?)(?=##|$)/i },
    { key: "treatment",    pattern: /##\s*TREATMENT OPTIONS([\s\S]*?)(?=##|$)/i },
    { key: "medicines",    pattern: /##\s*PRESCRIBED MEDICINES([\s\S]*?)(?=##|$)/i },
    { key: "home",         pattern: /##\s*HOME REMEDIES[\s\S]*?([\s\S]*?)(?=##|$)/i },
    { key: "schedule",     pattern: /##\s*30-DAY PERSONAL TREATMENT SCHEDULE([\s\S]*?)(?=##\s*FOLLOW|$)/i },
    { key: "followup",     pattern: /##\s*FOLLOW-UP SCHEDULE([\s\S]*?)(?=##|$)/i },
    { key: "donor",        pattern: /##\s*ORGAN DONOR([\s\S]*?)(?=##|$)/i },
  ];
  sectionPatterns.forEach(({ key, pattern }) => {
    const match = text.match(pattern);
    if (match) sections[key] = match[1].trim();
  });
  return sections;
}

// ── Parse 30-day schedule into weeks ─────────
function parseScheduleWeeks(scheduleText) {
  if (!scheduleText) return [];
  const weekPattern = /WEEK\s*(\d+)[^:]*:\s*\[([^\]]*)\]([\s\S]*?)(?=WEEK\s*\d+|$)/gi;
  const weeks = [];
  let match;
  while ((match = weekPattern.exec(scheduleText)) !== null) {
    weeks.push({
      week:    parseInt(match[1]),
      focus:   match[2].trim(),
      content: match[3].trim()
    });
  }
  if (weeks.length === 0 && scheduleText.length > 10) {
    weeks.push({ week: 1, focus: "Full Schedule", content: scheduleText });
  }
  return weeks;
}

// ── Section Card Component ────────────────────
function SectionCard({ icon, title, content, accent }) {
  if (!content) return null;
  return (
    <div className="cdss-section-card" style={{ borderColor: accent || "rgba(100,255,218,0.15)" }}>
      <div className="cdss-section-header">
        <span className="cdss-section-icon">{icon}</span>
        <span className="cdss-section-title">{title}</span>
      </div>
      <div className="cdss-section-body">
        {content.split("\n").filter(l => l.trim()).map((line, i) => (
          <p key={i} className={line.startsWith("-") || line.match(/^\d+\./) ? "cdss-list-item" : "cdss-para"}>
            {line}
          </p>
        ))}
      </div>
    </div>
  );
}

// ── 30-Day Schedule Component ─────────────────
function SchedulePlan({ scheduleText }) {
  const [activeWeek, setActiveWeek] = useState(0);
  const weeks = parseScheduleWeeks(scheduleText);
  if (!weeks.length) return null;

  const weekColors = ["#64ffda", "#69f0ae", "#ffeb3b", "#ff9800"];

  return (
    <div className="schedule-section">
      <div className="schedule-header">
        <span className="schedule-icon">📅</span>
        <span className="schedule-title">30-Day Personal Treatment Schedule</span>
      </div>

      {/* Week tabs */}
      <div className="week-tabs">
        {weeks.map((w, i) => (
          <button
            key={i}
            className={`week-tab ${activeWeek === i ? "active" : ""}`}
            style={{ "--tab-color": weekColors[i] }}
            onClick={() => setActiveWeek(i)}
          >
            <span className="week-num">Week {w.week}</span>
            <span className="week-focus">{w.focus}</span>
          </button>
        ))}
      </div>

      {/* Week content */}
      <div className="week-content">
        {weeks[activeWeek]?.content.split("\n").filter(l => l.trim()).map((line, i) => {
          const isTime = line.match(/^\d{1,2}:\d{2}|^(Morning|Breakfast|Lunch|Dinner|Evening|Night|Mid-)/i);
          const isBullet = line.startsWith("-");
          return (
            <div key={i} className={`schedule-line ${isTime ? "time-entry" : isBullet ? "bullet-entry" : "text-entry"}`}>
              {line}
            </div>
          );
        })}
      </div>
    </div>
  );
}


// ── Main ResultDashboard ──────────────────────
function ResultDashboard({ result, onReset }) {
  const [cdssTab, setCdssTab] = useState("report");

  const {
    classification, malignant_prob, benign_prob,
    risk, staging, gradcam_image, recommendations, disclaimer
  } = result;

  const isMalignant = classification === "Malignant";
  const tier        = risk.risk_tier.toLowerCase();
  const sections    = parseSections(recommendations);

  const stage      = staging?.cancer_stage;
  const tumor      = staging?.tumor_size;
  const survival   = staging?.survival_rate;
  const stageColor = stage ? ({
    "IA": "#00e676", "IB": "#69f0ae",
    "II": "#ffeb3b", "III": "#ff9800", "IV": "#f44336", "N/A": "#aaa"
  }[stage.stage_roman] || "#aaa") : "#aaa";
  const survivalColor = survival ? ({
    "Excellent": "#00e676", "Good": "#69f0ae",
    "Moderate": "#ffeb3b", "Guarded": "#ff9800", "Poor": "#f44336"
  }[survival.survival_category] || "#aaa") : "#aaa";

  return (
    <div className="result-dashboard">

      {/* ── Top 3 Cards ── */}
      <div className="result-top">
        <div className="card">
          <div className="card-label">Classification</div>
          <div className={`card-value ${isMalignant ? "malignant" : "benign"}`}>{classification}</div>
          <div className="card-sub">{isMalignant ? "⚠️ Malignancy detected" : "✅ No malignancy detected"}</div>
          <div className="prob-bar">
            <div className={`prob-fill ${isMalignant ? "danger" : ""}`}
              style={{ width: `${(isMalignant ? malignant_prob : benign_prob) * 100}%` }} />
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
          <div className="card-sub" style={{ marginTop: "0.4rem" }}>Score: {risk.risk_score.toFixed(4)}</div>
        </div>

        <div className="card">
          <div className="card-label">Probabilities</div>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem", marginTop: "0.3rem" }}>
            {[
              { label: "Malignant", prob: malignant_prob, color: "var(--danger)", cls: "danger" },
              { label: "Benign",    prob: benign_prob,    color: "var(--accent2)", cls: "" }
            ].map(p => (
              <div key={p.label}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.72rem", marginBottom: "0.3rem" }}>
                  <span style={{ color: p.color }}>{p.label}</span>
                  <span>{(p.prob * 100).toFixed(1)}%</span>
                </div>
                <div className="prob-bar">
                  <div className={`prob-fill ${p.cls}`} style={{ width: `${p.prob * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Risk Factors ── */}
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

      {/* ── Staging ── */}
      {staging && isMalignant && (
        <div className="staging-section">
          <div className="section-title">🔬 Advanced Analysis</div>
          <div className="staging-cards">
            <div className="staging-card">
              <div className="staging-icon">🎯</div>
              <div className="staging-label">Cancer Stage</div>
              <div className="staging-value" style={{ color: stageColor }}>Stage {stage.stage_roman}</div>
              <div className="staging-sub">{stage.stage_detail}</div>
              <div className="staging-desc">{stage.description}</div>
              <div className="staging-tnm">
                <span>T: {tumor?.t_classification}</span>
                <span>N: {stage.n_class}</span>
                <span>M: {stage.m_class}</span>
              </div>
            </div>
            <div className="staging-card">
              <div className="staging-icon">📏</div>
              <div className="staging-label">Tumor Size Estimate</div>
              <div className="staging-value" style={{ color: "#64ffda" }}>{tumor?.size_range_cm}</div>
              <div className="staging-sub">{tumor?.size_category}</div>
              <div className="staging-desc">T-Class: {tumor?.t_classification}</div>
              <div className="staging-desc">Grad-CAM activation: {tumor?.activation_percent}%</div>
            </div>
            <div className="staging-card">
              <div className="staging-icon">📊</div>
              <div className="staging-label">5-Year Survival Rate</div>
              <div className="staging-value" style={{ color: survivalColor }}>{survival?.five_year_survival}%</div>
              <div className="staging-sub">{survival?.survival_category} Prognosis</div>
              <div className="staging-desc">{survival?.note}</div>
            </div>
          </div>
        </div>
      )}

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

      {/* ── Grad-CAM ── */}
      <div className="result-mid">
        <div className="card gradcam-card">
          <div className="card-label">Grad-CAM Heatmap</div>
          <div className="card-sub">XAI — regions influencing prediction</div>
          <img src={`data:image/png;base64,${gradcam_image}`} alt="Grad-CAM" className="gradcam-img" />
          <div className="legend">
            <div className="legend-item"><div className="legend-dot" style={{ background: "#ff0000" }} />High</div>
            <div className="legend-item"><div className="legend-dot" style={{ background: "#ffff00" }} />Medium</div>
            <div className="legend-item"><div className="legend-dot" style={{ background: "#0000ff" }} />Low</div>
          </div>
        </div>

        {/* ── CDSS Tabbed Panel ── */}
        <div className="card cdss-card">
          <div className="card-label">Clinical Decision Support</div>
          <div className="card-sub">Powered by Groq · LLaMA 3.3 70B</div>

          <div className="cdss-tabs">
            {[
              { key: "report",    label: "📋 Report" },
              { key: "medicines", label: "💊 Medicines" },
              { key: "home",      label: "🏠 Home Care" },
            ].map(t => (
              <button
                key={t.key}
                className={`cdss-tab-btn ${cdssTab === t.key ? "active" : ""}`}
                onClick={() => setCdssTab(t.key)}
              >
                {t.label}
              </button>
            ))}
          </div>

          <div className="cdss-tab-content">
            {cdssTab === "report" && (
              <div>
                <SectionCard icon="🩺" title="Diagnosis Summary"        content={sections.diagnosis} accent="rgba(100,255,218,0.2)" />
                <SectionCard icon="🔬" title="Diagnostic Tests"         content={sections.tests}     accent="rgba(255,235,59,0.2)" />
                <SectionCard icon="💉" title="Treatment Options"        content={sections.treatment} accent="rgba(255,152,0,0.2)" />
                <SectionCard icon="📆" title="Follow-up Schedule"       content={sections.followup}  accent="rgba(100,255,218,0.15)" />
                <SectionCard icon="🫀" title="Organ Donor Screening"    content={sections.donor}     accent="rgba(244,67,54,0.15)" />
              </div>
            )}
            {cdssTab === "medicines" && (
              <SectionCard icon="💊" title="Prescribed Medicines & Supplements" content={sections.medicines} accent="rgba(105,240,174,0.2)" />
            )}
            {cdssTab === "home" && (
              <SectionCard icon="🏠" title="Home Remedies & Natural Support" content={sections.home} accent="rgba(255,235,59,0.15)" />
            )}
          </div>

          <div className="disclaimer">{disclaimer}</div>
        </div>
      </div>

      {/* ── 30-Day Schedule ── */}
      {sections.schedule && <SchedulePlan scheduleText={sections.schedule} />}

      <button className="reset-btn" onClick={onReset}>← Analyze Another Scan</button>

    </div>
  );
}

export default ResultDashboard;