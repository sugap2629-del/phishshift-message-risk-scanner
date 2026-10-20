"""
PhishShift - Enterprise AI Phishing & Spam Detection Platform
Flask Web Application with Tri-State ML Engine (Phishing, Legitimate, Uncertain)
Backed by SQLite persistent scan history and real Kaggle model evaluation metrics.
"""

import os
import sys
import json
import time
import sqlite3
import joblib
import numpy as np
from datetime import datetime
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    abort
)

import config
from train_model import preprocess_text
from evaluate_model import RuleBasedPhishingClassifier

# Initialize Flask App
app = Flask(
    __name__,
    template_folder=str(config.TEMPLATES_DIR),
    static_folder=str(config.STATIC_DIR)
)
app.config["SECRET_KEY"] = config.SECRET_KEY

# Global Model & Vectorizer Cache
_model = None
_vectorizer = None
_rules_detector = RuleBasedPhishingClassifier(threshold=2)


def get_db_connection():
    """Returns an active SQLite database connection with row factory."""
    conn = sqlite3.connect(str(config.DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initializes the SQLite schema in instance/phishshift.db (Requirement 8)."""
    conn = get_db_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                raw_message TEXT NOT NULL,
                clean_message TEXT,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                phishing_prob REAL NOT NULL,
                legit_prob REAL NOT NULL,
                risk_level TEXT NOT NULL,
                rule_triggers TEXT,
                keywords_found TEXT,
                scan_duration_ms REAL NOT NULL,
                model_name TEXT NOT NULL
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_scans_timestamp ON scans(timestamp);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_scans_prediction ON scans(prediction);
        """)
    conn.close()


def load_ml_artifacts():
    """Loads pre-trained model and TF-IDF vectorizer."""
    global _model, _vectorizer
    if _model is not None and _vectorizer is not None:
        return _model, _vectorizer

    if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.VECTORIZER_PATH):
        # Auto-train if artifacts are missing
        print("[!] Model artifacts missing. Auto-invoking training pipeline...")
        from train_model import train
        train()

    _model = joblib.load(config.MODEL_PATH)
    _vectorizer = joblib.load(config.VECTORIZER_PATH)
    return _model, _vectorizer


def analyze_text(raw_text: str):
    """
    Performs complete threat evaluation:
    1. Preprocesses text
    2. Computes TF-IDF vector & Logistic Regression probabilities
    3. Runs heuristic rule signatures
    4. Applies Tri-State Uncertainty thresholding (Requirement 6)
    5. Extracts token explainability attribution
    """
    t0 = time.time()
    model, vectorizer = load_ml_artifacts()

    clean_text = preprocess_text(raw_text)
    if not clean_text:
        return {
            "prediction": "Uncertain",
            "risk_level": "LOW",
            "phishing_prob": 0.0,
            "legit_prob": 1.0,
            "confidence": 50.0,
            "rule_triggers": [],
            "keywords_found": [],
            "duration_ms": 0.0,
            "clean_text": "",
            "model_name": "Logistic Regression + TF-IDF"
        }

    # Vectorize
    vec = vectorizer.transform([clean_text])
    probabilities = model.predict_proba(vec)[0]  # [P(0), P(1)]
    legit_prob = float(probabilities[0])
    phishing_prob = float(probabilities[1])

    # Rule heuristics
    rule_results = _rules_detector.analyze_message(clean_text)
    rule_triggers = rule_results["triggers"]
    critical_rules_count = len(rule_triggers)

    # Tri-State Decision Logic (Requirement 6)
    # Phishing threshold = 0.65, Legitimate threshold = 0.35
    if phishing_prob >= config.PHISHING_THRESHOLD:
        prediction = "Phishing"
        risk_level = "CRITICAL" if phishing_prob >= 0.85 else "HIGH"
        confidence = round(phishing_prob * 100, 2)
    elif phishing_prob <= config.LEGITIMATE_THRESHOLD:
        # If low model probability but multiple aggressive rules fired, mark Uncertain for caution
        if critical_rules_count >= 2:
            prediction = "Uncertain"
            risk_level = "MEDIUM"
            confidence = round((1 - abs(phishing_prob - 0.5) * 2) * 100, 2)
        else:
            prediction = "Legitimate"
            risk_level = "LOW"
            confidence = round(legit_prob * 100, 2)
    else:
        # Ambiguous boundary (35% to 65%)
        prediction = "Uncertain"
        risk_level = "MEDIUM"
        confidence = round((1 - abs(phishing_prob - 0.5) * 2) * 100, 2)

    # Explainability: Extract top influential words present in input
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # Get non-zero indices in input vector
    nonzero_indices = vec.indices
    tokens_attribution = []
    for idx in nonzero_indices:
        token = feature_names[idx]
        weight = float(coefficients[idx] * vec[0, idx])
        tokens_attribution.append({"feature": token, "weight": round(weight, 3)})

    # Sort tokens by absolute attribution weight
    tokens_attribution.sort(key=lambda x: abs(x["weight"]), reverse=True)
    top_keywords = tokens_attribution[:15]

    duration_ms = round((time.time() - t0) * 1000, 2)

    return {
        "prediction": prediction,
        "risk_level": risk_level,
        "phishing_prob": round(phishing_prob, 4),
        "legit_prob": round(legit_prob, 4),
        "confidence": confidence,
        "rule_triggers": rule_triggers,
        "keywords_found": top_keywords,
        "duration_ms": duration_ms,
        "clean_text": clean_text,
        "model_name": "Logistic Regression + TF-IDF"
    }


def record_scan_to_db(raw_message: str, result: dict) -> int:
    """Stores scan record in SQLite (Requirement 8)."""
    conn = get_db_connection()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with conn:
        cursor = conn.execute("""
            INSERT INTO scans (
                timestamp, raw_message, clean_message, prediction, confidence,
                phishing_prob, legit_prob, risk_level, rule_triggers,
                keywords_found, scan_duration_ms, model_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            timestamp,
            raw_message,
            result["clean_text"],
            result["prediction"],
            result["confidence"],
            result["phishing_prob"],
            result["legit_prob"],
            result["risk_level"],
            json.dumps(result["rule_triggers"]),
            json.dumps(result["keywords_found"]),
            result["duration_ms"],
            result["model_name"]
        ))
        scan_id = cursor.lastrowid
    conn.close()
    return scan_id


# Initialize DB and preload ML artifacts on import
init_database()
load_ml_artifacts()


# --------------------------------------------------------------------------
# Web Routes
# --------------------------------------------------------------------------

@app.route("/")
def index():
    """Scanner Home Page with interactive test presets."""
    conn = get_db_connection()
    total_scans = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    conn.close()
    return render_template("index.html", total_scans_logged=total_scans)


@app.route("/scan", methods=["POST"])
def scan_message():
    """Handles scan form submission from index.html."""
    raw_text = request.form.get("message_text", "").strip()
    if not raw_text:
        return redirect(url_for("index"))

    result = analyze_text(raw_text)
    scan_id = record_scan_to_db(raw_text, result)
    return redirect(url_for("view_result", scan_id=scan_id))


@app.route("/results/<int:scan_id>")
def view_result(scan_id: int):
    """Detailed Results Page showing risk breakdown, rules, and keywords."""
    conn = get_db_connection()
    scan = conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
    conn.close()

    if not scan:
        abort(404, description="Scan record not found in audit trail.")

    rule_triggers = json.loads(scan["rule_triggers"] or "[]")
    keywords_found = json.loads(scan["keywords_found"] or "[]")

    return render_template(
        "results.html",
        scan=scan,
        rule_triggers=rule_triggers,
        keywords_found=keywords_found
    )


@app.route("/dashboard")
def dashboard():
    """SOC Operations Dashboard with real SQLite telemetry and model metrics (Requirement 7)."""
    conn = get_db_connection()
    total_scans = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    phish_count = conn.execute("SELECT COUNT(*) FROM scans WHERE prediction = 'Phishing'").fetchone()[0]
    safe_count = conn.execute("SELECT COUNT(*) FROM scans WHERE prediction = 'Legitimate'").fetchone()[0]
    uncertain_count = conn.execute("SELECT COUNT(*) FROM scans WHERE prediction = 'Uncertain'").fetchone()[0]
    conn.close()

    stats = {
        "total_scans": total_scans,
        "phish_count": phish_count,
        "safe_count": safe_count,
        "uncertain_count": uncertain_count
    }

    # Load real, un-fabricated model metrics & comparisons
    metrics = {}
    if os.path.exists(config.METRICS_PATH):
        with open(config.METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    comparison = {}
    if os.path.exists(config.COMPARISON_PATH):
        with open(config.COMPARISON_PATH, "r", encoding="utf-8") as f:
            comparison = json.load(f)

    return render_template(
        "dashboard.html",
        stats=stats,
        metrics=metrics,
        comparison=comparison
    )


@app.route("/history")
def history():
    """Audit Trail & Scan History Page."""
    conn = get_db_connection()
    scans = conn.execute("SELECT * FROM scans ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("history.html", scans=scans)


@app.route("/history/delete/<int:scan_id>", methods=["POST"])
def delete_scan(scan_id: int):
    """Deletes an individual scan from SQLite."""
    conn = get_db_connection()
    with conn:
        conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
    conn.close()
    return redirect(url_for("history"))


@app.route("/history/clear", methods=["POST"])
def clear_history():
    """Clears all scan history from SQLite."""
    conn = get_db_connection()
    with conn:
        conn.execute("DELETE FROM scans;")
    conn.close()
    return jsonify({"status": "success", "message": "Scan history cleared."})


@app.route("/about")
def about():
    """System Architecture, Dataset Info, and Best Practices."""
    return render_template("about.html")


# --------------------------------------------------------------------------
# JSON REST API Endpoints
# --------------------------------------------------------------------------

@app.route("/api/scan", methods=["POST"])
def api_scan():
    """Programmatic JSON threat scanning endpoint."""
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Missing 'message' field in JSON payload."}), 400

    result = analyze_text(message)
    scan_id = record_scan_to_db(message, result)
    result["scan_id"] = scan_id
    result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify(result), 200


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Returns live telemetry statistics in JSON format."""
    conn = get_db_connection()
    total = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    phish = conn.execute("SELECT COUNT(*) FROM scans WHERE prediction = 'Phishing'").fetchone()[0]
    safe = conn.execute("SELECT COUNT(*) FROM scans WHERE prediction = 'Legitimate'").fetchone()[0]
    uncertain = conn.execute("SELECT COUNT(*) FROM scans WHERE prediction = 'Uncertain'").fetchone()[0]
    conn.close()

    return jsonify({
        "total_scans": total,
        "phishing_count": phish,
        "legitimate_count": safe,
        "uncertain_count": uncertain
    }), 200


# Error Handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template("base.html", error_message="404 - Resource Not Found"), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template("base.html", error_message="500 - Internal Server Error"), 500


if __name__ == "__main__":
    init_database()
    load_ml_artifacts()
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[*] PhishShift Security Server starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=config.DEBUG)
