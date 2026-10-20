# PhishShift: Intelligent Phishing & Threat Defense Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask 3.x](https://img.shields.io/badge/Framework-Flask%203.x-00f2fe.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-ffb800.svg)](https://scikit-learn.org/)
[![Test Suite](https://img.shields.io/badge/Tests-18%2F18%20Passing-00ffaa.svg)](tests/test_app.py)
[![Security Theme](https://img.shields.io/badge/UI-Dark%20Navy%20%26%20Cyan-0a1124.svg)](static/css/style.css)

**PhishShift** is an enterprise-grade AI phishing and spam detection platform. It combines real-world Kaggle threat intelligence, TF-IDF lexical n-gram vectorization, calibrated Logistic Regression, and heuristic cybersecurity rule validation into a responsive, tri-state threat verdict engine (**Phishing**, **Legitimate**, or **Uncertain**).

---

## Folder Structure

```text
PhishShift/
│
├── app.py                     # Flask web server, REST APIs, & SQLite persistence
├── train_model.py             # Data cleaning, column validation, TF-IDF + Logistic Regression training
├── evaluate_model.py          # 3-approach model benchmark (Rules vs LR vs Naive Bayes)
├── config.py                  # System paths, thresholds, and dynamic column detector
├── requirements.txt           # Python dependency specifications
├── README.md                  # Complete documentation & Windows PowerShell guide
├── REPORT.md                  # Comprehensive empirical research report & benchmarks
├── .gitignore                 # Environment and cache ignore patterns
│
├── data/
│   ├── phishing_email.csv     # Real Kaggle phishing email dataset (18,650 raw records)
│   └── cleaned_messages.csv   # Sanitized & deduplicated dataset (17,450 records)
│
├── models/
│   ├── phishing_model.pkl     # Serialized Logistic Regression model
│   ├── vectorizer.pkl         # Serialized 10,000-feature TF-IDF vectorizer
│   ├── metrics.json           # Real, un-fabricated test performance metrics
│   └── model_comparison.json  # 3-approach empirical comparison metrics
│
├── instance/
│   └── phishshift.db          # SQLite database storing real-time scan audit trail
│
├── templates/
│   ├── base.html              # Cyber-themed navigation, grid glow, and layout
│   ├── index.html             # Threat scanner, sample presets, & telemetry cards
│   ├── dashboard.html         # Real-time SQLite stats & model comparison charts
│   ├── results.html           # Detailed scan analysis, risk gauge, & rule matches
│   ├── history.html           # Searchable audit log with filter pills & CSV export
│   └── about.html             # Architecture, pipeline stages, & defense protocols
│
├── static/
│   ├── css/
│   │   └── style.css          # Modern dark navy & neon cyan cybersecurity theme
│   ├── js/
│   │   └── script.js          # Dynamic counters, scanner beam, & keybindings
│   └── images/
│       └── shield_logo.svg    # Cyber defense SVG vector insignia
│
└── tests/
    └── test_app.py            # Comprehensive 18-case test suite (unittest)
```

---

## Key Features

1. **Real Kaggle Phishing Dataset Integration**:
   - Integrates the authentic Kaggle Phishing Email dataset (`data/phishing_email.csv`).
   - Dynamic column resolution handles varying dataset column naming schemas (`Email Text`, `Email Type`, `label`, `text`, etc.) without hardcoded assumptions.
2. **Automated Cleaning Pipeline**:
   - Drops null and missing message values.
   - Identifies and eliminates 1,181 duplicate email records to avoid test set data leakage.
   - Normalizes URLs (`http_url`), IP addresses (`ip_addr`), and monetary amounts (`money_val`).
3. **Calibrated Machine Learning Engine**:
   - 10,000 sublinear TF-IDF n-gram features (unigrams + bigrams).
   - High-performance Logistic Regression with L2 regularization achieving **97.65% Test Accuracy**, **98.02% Precision**, and **0.9974 ROC-AUC**.
4. **Three-Approach Model Comparison**:
   - Compares **Rules-Based Baseline** (58.94% Acc), **Logistic Regression** (97.65% Acc), and **Multinomial Naive Bayes** (96.65% Acc) on the exact same held-out test split.
5. **Tri-State Verdict Architecture**:
   - **Phishing** ($P \ge 65\%$): Automated quarantine alert.
   - **Legitimate** ($P \le 35\%$): Clean benign correspondence.
   - **Uncertain** ($35\% < P < 65\%$ or rule-model conflict): Flagged for manual SOC review.
6. **SQLite Persistent Audit Trail**:
   - Every scan is logged into `instance/phishshift.db` with timestamps, prediction, probability breakdown, latency, rule triggers, and extracted keywords.
7. **SOC Operations Dashboard**:
   - Real-time interactive charts (Donut chart for live SQLite scan traffic; Bar charts comparing 3 model approaches; Confusion Matrix grid).
8. **Modern Cybersecurity Aesthetics**:
   - Dark navy abyss backgrounds (`#050811`, `#0e1626`), neon cyan accents (`#00f2fe`), glassmorphism cards, animated scanner beams, and responsive mobile navigation.

---

## Windows PowerShell Setup & Commands

Run these commands in Windows PowerShell from the `PhishShift` directory:

### 1. Create and Activate Virtual Environment (Optional but Recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Clean Dataset & Train the Model
```powershell
python train_model.py
```
*Outputs `data/cleaned_messages.csv`, `models/phishing_model.pkl`, `models/vectorizer.pkl`, and `models/metrics.json`.*

### 4. Run 3-Approach Benchmark Comparison
```powershell
python evaluate_model.py
```
*Outputs comparative accuracy, precision, recall, F1, and latency across Rules-Based, Logistic Regression, and Naive Bayes into `models/model_comparison.json`.*

### 5. Execute Test Suite
```powershell
python -m unittest tests/test_app.py -v
```
*Runs all 18 unit and integration tests covering preprocessing, rules, models, routes, and database operations.*

### 6. Launch PhishShift Web Application
```powershell
python app.py
```
*Launches the Flask server at **http://127.0.0.1:5000**.*

---

## Verified Model Benchmarks (Real Test Data: N=3,490)

| Model Approach | Accuracy | Precision | Recall | F1-Score | Latency (per sample) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Rules-Based Baseline** | 58.94% | 43.81% | 37.09% | 40.17% | 0.007 ms |
| **2. Logistic Regression (Primary)** | **97.65%** | **98.02%** | **95.61%** | **96.80%** | **0.001 ms** |
| **3. Multinomial Naive Bayes** | 96.65% | 96.97% | 93.91% | 95.42% | 0.001 ms |

---

## REST API Reference

### 1. Analyze Message
- **Endpoint**: `POST /api/scan`
- **Request Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "message": "URGENT: Your account has been suspended. Click here to verify credentials."
  }
  ```
- **Response (200 OK)**:
  ```json
  {
    "scan_id": 1,
    "timestamp": "2026-10-20 16:35:10",
    "prediction": "Phishing",
    "risk_level": "CRITICAL",
    "phishing_prob": 0.9642,
    "legit_prob": 0.0358,
    "confidence": 96.42,
    "rule_triggers": [
      "urgency_cues:urgent",
      "urgency_cues:account suspended",
      "credential_harvesting:click here"
    ],
    "duration_ms": 3.82,
    "model_name": "Logistic Regression + TF-IDF"
  }
  ```

### 2. Live Telemetry Stats
- **Endpoint**: `GET /api/stats`
- **Response (200 OK)**:
  ```json
  {
    "total_scans": 15,
    "phishing_count": 8,
    "legitimate_count": 5,
    "uncertain_count": 2
  }
  ```

---

## Hosting on Render

PhishShift includes full out-of-the-box support for hosting on [Render](https://render.com) using the included `Procfile` and `render.yaml`:

### Option A: One-Click / Blueprint Deploy
1. Sign in to your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** > **Blueprint**.
3. Connect your repository: `https://github.com/sugap2629-del/phishshift-message-risk-scanner.git`.
4. Render will read `render.yaml` and configure the Web Service automatically.
5. Click **Apply**.

### Option B: Manual Web Service Setup
1. On Render, click **New +** > **Web Service**.
2. Connect `https://github.com/sugap2629-del/phishshift-message-risk-scanner.git`.
3. Set the following configuration:
   - **Name**: `phishshift` (or your choice)
   - **Runtime**: `Python 3`
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: `Free`
4. Add Environment Variables (under Advanced):
   - `PYTHON_VERSION`: `3.11.9`
   - `SECRET_KEY`: `<your-random-secret-key>`
   - `FLASK_DEBUG`: `False`
5. Click **Deploy Web Service**.

Once deployed, your threat scanner is live worldwide on your custom `.onrender.com` URL!

---

## License & Attribution
- Dataset source: Real Kaggle Phishing Email Corpus.
- Framework: Flask 3.x, Scikit-Learn, SQLite3.
- Developed with enterprise security standards.
