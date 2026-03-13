# 🫁 LungAI — Clinical Intelligence System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-red?style=for-the-badge)
![Railway](https://img.shields.io/badge/Deployed-Railway-purple?style=for-the-badge&logo=railway&logoColor=white)

**An AI-powered clinical decision support system for early-stage pulmonary malignancy detection.**

*Upload a CT scan → Get instant diagnosis, risk stratification, Grad-CAM heatmap & clinical recommendations.*

</div>

---

## 📸 Demo

| Malignant Detection | Benign Detection |
|---------------------|-----------------|
| 🔴 99.8% Confidence · Critical Risk | 🟢 89.3% Confidence · Low Risk |
| Grad-CAM highlights tumor region | Clean lung tissue confirmed |

---

## 🧠 How It Works

```
CT Scan Input
     ↓
Preprocessing  →  Denoise → Resize (224×224) → Normalize [0,1]
     ↓
CNN Feature Extraction  →  Conv × 3 + BatchNorm + MaxPool + GAP
     ↓
Binary Classification  →  Malignant / Benign (Softmax)
     ↓
Risk Stratification  →  R = α1·P(M|x) + α2·C_age + α3·C_smoke + α4·C_hist
     ↓
Grad-CAM XAI  →  Heatmap showing influential CT regions
     ↓
Groq CDSS  →  LLaMA 3.3 70B clinical recommendations
     ↓
Clinician Report Output
```

---

## ✨ Features

- 🔬 **CNN-based CT Analysis** — 3-block ConvNet with BatchNorm achieving 100% test accuracy
- 📊 **4-Tier Risk Stratification** — Low / Medium / High / Critical composite scoring
- 🗺️ **Grad-CAM Explainability** — Visual heatmaps showing *why* the model made its decision
- 🤖 **Groq CDSS** — LLaMA 3.3 70B generates treatment plans, diagnostic tests & follow-up schedules
- 💉 **Organ Donor Screening** — Lung health assessment for donor eligibility
- ⚡ **Real-time Inference** — Full pipeline in under 5 seconds

---

## 🏗️ Architecture

```
lung-cancer-ai/
├── backend/
│   ├── app.py                    # Flask REST API
│   ├── config.py                 # Central configuration
│   ├── preprocessing/
│   │   ├── preprocess.py         # Denoise → Resize → Normalize
│   │   └── dataset_split.py      # 80/20 train/test split
│   ├── model/
│   │   ├── cnn_model.py          # CNN architecture
│   │   ├── train.py              # Training pipeline
│   │   ├── gradcam.py            # Grad-CAM XAI (Eq. 14 & 15)
│   │   ├── risk_score.py         # Composite risk (Eq. 12 & 13)
│   │   └── predict.py            # Single image inference
│   └── cdss/
│       └── recommendations.py    # Groq API integration
├── frontend/
│   └── frontend/src/
│       ├── App.js
│       └── components/
│           ├── Header.js
│           ├── UploadSection.js
│           └── ResultDashboard.js
├── model/
│   └── cnn_model.h5              # Trained model (Git LFS)
├── Dockerfile
└── requirements.txt
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 100% |
| **Test Loss** | 0.0173 |
| **Val Accuracy** | 100% |
| **Dataset** | IQ-OTH/NCCD (1,097 CT scans) |
| **Train/Test Split** | 80% / 20% |
| **Malignant samples** | 561 |
| **Benign samples** | 536 |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Groq API key → [console.groq.com](https://console.groq.com)

### 1. Clone the repo
```bash
git clone https://github.com/Kishorekannann82/Lung_Cancer.git
cd Lung_Cancer
```

### 2. Setup Python environment
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 3. Configure environment
```bash
echo GROQ_API_KEY=your_key_here > backend/.env
```

### 4. Run preprocessing
```bash
cd backend
python preprocessing/preprocess.py
python preprocessing/dataset_split.py
```

### 5. Train the model
```bash
python model/train.py
```

### 6. Start backend
```bash
python app.py
```

### 7. Start frontend
```bash
cd ../frontend/frontend
npm install && npm start
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/test` | Test all modules |
| `POST` | `/api/predict` | Full prediction pipeline |

### POST `/api/predict` Response
```json
{
  "classification": "Malignant",
  "malignant_prob": 0.9978,
  "benign_prob": 0.0022,
  "risk": {
    "risk_tier": "Critical",
    "risk_score": 0.8689
  },
  "gradcam_image": "<base64_heatmap>",
  "recommendations": "## DIAGNOSIS SUMMARY..."
}
```

---

## 🧬 Risk Score Formula

```
R = α₁·P(M|x) + α₂·C_age + α₃·C_smoke + α₄·C_hist
α₁=0.5  α₂=0.2  α₃=0.2  α₄=0.1
```

| Tier | Score |
|------|-------|
| 🟢 Low | R < 0.35 |
| 🟡 Medium | 0.35 ≤ R < 0.55 |
| 🟠 High | 0.55 ≤ R < 0.75 |
| 🔴 Critical | R ≥ 0.75 |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow 2.15, Keras |
| Image Processing | OpenCV, Pillow |
| XAI | Grad-CAM (custom) |
| Backend | Flask 3.0 |
| CDSS | Groq API — LLaMA 3.3 70B |
| Frontend | React 18 (CRA) |
| Deployment | Docker, Railway |

---

## 📄 Reference Paper

> *AI-Driven Clinical Intelligence Framework for Early-Stage Pulmonary Malignancy Detection and Therapeutic Guidance*
> Mithuna R, Janani R, Sowmiya P K, Yuvasree M — Dr. Mahalingam College of Engineering & Technology

---

## ⚠️ Disclaimer

For **research and educational purposes only**. All AI-generated recommendations must be reviewed by a licensed physician.

---



<div align="center"><sub>Built for saving lives through AI</sub></div>