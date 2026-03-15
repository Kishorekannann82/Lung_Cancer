# 🫁 LungAI — Complete Setup Guide
### For running the project locally on a new laptop

---

## ✅ Prerequisites

Install these before starting:

| Tool | Download Link | Version |
|------|--------------|---------|
| **Anaconda** | https://www.anaconda.com/download | Latest |
| **Node.js** | https://nodejs.org | 18+ LTS |
| **Git** | https://git-scm.com/downloads | Latest |
| **VS Code** | https://code.visualstudio.com | Optional |

---

## 📥 Step 1 — Clone the Repository

Open **Anaconda Prompt** (search in Start Menu) and run:

```bash
git clone https://github.com/Kishorekannann82/Lung_Cancer.git
cd Lung_Cancer
```

---

## 🐍 Step 2 — Create Conda Environment

```bash
conda create -n lung_env python=3.11 -y
conda activate lung_env
```

You should see `(lung_env)` at the start of your terminal line.

---

## 📦 Step 3 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

⏳ This will take **5–10 minutes** — it installs TensorFlow, OpenCV, Flask etc.
Just wait until it finishes completely.

---

## 🔑 Step 4 — Set Up Groq API Key

**4a.** Get a free API key:
- Go to → https://console.groq.com
- Sign up / Login
- Click **API Keys → Create API Key**
- Copy the key (starts with `gsk_...`)

**4b.** Create the `.env` file:

```bash
# Make sure you are inside the Lung_Cancer folder
cd backend
```

Create a file called `.env` inside the `backend` folder with this content:

```
GROQ_API_KEY=gsk_your_key_here
```

> ⚠️ Replace `gsk_your_key_here` with your actual key!

**How to create the file:**
- Open the `backend` folder in File Explorer
- Right-click → New → Text Document
- Name it `.env` (make sure to remove the `.txt` extension)
- Open it with Notepad and paste: `GROQ_API_KEY=gsk_your_key_here`
- Save and close

---

## 🧠 Step 5 — Verify the Model Exists

The trained model is already included in the repo. Verify it:

```bash
# Go back to root folder
cd ..

# Check model exists
dir model\cnn_model.h5
```

You should see `cnn_model.h5` listed. If not, contact Kishore to share the model file.

---

## 🚀 Step 6 — Start the Backend (Flask)

Open **Anaconda Prompt** and run:

```bash
# Make sure you're in the Lung_Cancer folder
cd Lung_Cancer
conda activate lung_env
cd backend
python app.py
```

✅ You should see:
```
[INFO] Loading CNN model...
[INFO] Model loaded ✅
[INFO] Starting server on port 5000
 * Running on http://0.0.0.0:5000
```

> 🔴 Keep this terminal window open! Do NOT close it.

---

## 🌐 Step 7 — Start the Frontend (React)

Open a **NEW** Anaconda Prompt window and run:

```bash
cd Lung_Cancer\frontend\frontend
npm install
npm start
```

⏳ `npm install` takes 2–3 minutes the first time.

✅ You should see:
```
Compiled successfully!
Local: http://localhost:3000
```

Your browser will automatically open at **http://localhost:3000** 🎉

---

## 🧪 Step 8 — Test the Application

### Test with Malignant Case:
1. Go to http://localhost:3000
2. Upload this image:
   ```
   Lung_Cancer\The IQ-OTHNCCD lung cancer dataset\Malignant cases\Malignant case (3).jpg
   ```
3. Fill in the form:
   - Age: `65`
   - Smoking: `Current Smoker`
   - Symptoms: `Persistent Cough`, `Weight Loss`
4. Click **Analyze**

Expected result: 🔴 **Malignant ~99.8%** · Critical Risk

---

### Test with Benign Case:
1. Upload this image:
   ```
   Lung_Cancer\The IQ-OTHNCCD lung cancer dataset\Bengin cases\Bengin case (4).jpg
   ```
2. Fill in:
   - Age: `35`
   - Smoking: `Never`
   - Symptoms: None
3. Click **Analyze**

Expected result: 🟢 **Benign ~85-90%** · Low Risk

---

## 🗂️ Project Structure (What Each File Does)

```
Lung_Cancer/
│
├── backend/                    ← Flask API (Python)
│   ├── app.py                  ← Main server — RUN THIS
│   ├── config.py               ← All settings
│   ├── .env                    ← Your API key (create this!)
│   │
│   ├── preprocessing/
│   │   └── preprocess.py       ← Image cleaning pipeline
│   │
│   ├── model/
│   │   ├── cnn_model.py        ← CNN architecture definition
│   │   ├── train.py            ← Training script (already done!)
│   │   ├── gradcam.py          ← Heatmap generation
│   │   ├── risk_score.py       ← Risk calculation
│   │   ├── staging.py          ← Cancer stage detection
│   │   └── predict.py          ← Single image test script
│   │
│   └── cdss/
│       └── recommendations.py  ← Groq AI doctor advice
│
├── frontend/frontend/          ← React UI
│   └── src/
│       ├── App.js              ← Main React app
│       └── components/
│           ├── Header.js
│           ├── UploadSection.js
│           └── ResultDashboard.js
│
├── model/
│   └── cnn_model.h5            ← Trained model (1.5MB)
│
├── The IQ-OTHNCCD lung cancer dataset/
│   ├── Malignant cases/        ← 561 CT scans
│   ├── Bengin cases/           ← 120 CT scans
│   └── Normal cases/           ← 416 CT scans
│
└── requirements.txt            ← Python packages list
```

---

## ❗ Common Errors & Fixes

### Error: `ModuleNotFoundError: No module named 'flask'`
```bash
conda activate lung_env
pip install -r requirements.txt
```

### Error: `Could not generate recommendations: Connection error`
- Check your `backend/.env` file has the correct Groq API key
- Make sure the key starts with `gsk_`
- Try generating a new key at console.groq.com

### Error: `Port 5000 already in use`
```bash
# Windows — kill the process using port 5000
netstat -ano | findstr :5000
taskkill /PID <PID_NUMBER> /F
```

### Error: `npm: command not found`
- Download and install Node.js from https://nodejs.org
- Restart Anaconda Prompt after installing

### Error: `OSError: Unable to open file (file signature not found)`
- The `model/cnn_model.h5` file is missing or corrupted
- Ask Kishore to re-share the file
- Replace it in the `model/` folder

### Frontend shows blank page
- Make sure backend is running on port 5000 first
- Check browser console for errors (F12)

---

## 🔄 Every Time You Use the App

You need to start both servers each time:

**Terminal 1 (Backend):**
```bash
cd Lung_Cancer
conda activate lung_env
cd backend
python app.py
```

**Terminal 2 (Frontend):**
```bash
cd Lung_Cancer\frontend\frontend
npm start
```

Then open → http://localhost:3000

---

## 📞 Contact

If you face any issues, contact **Kishore Kannan**:
- GitHub: [@Kishorekannann82](https://github.com/Kishorekannann82)

---

<div align="center">
  <sub>🫁 LungAI — AI-Driven Clinical Intelligence for Pulmonary Malignancy Detection</sub>
</div>