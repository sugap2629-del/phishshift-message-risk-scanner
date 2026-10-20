"""
PhishShift - Model Evaluation & Comparison Engine
Compares three approaches:
  1. Rules-Based Cybersecurity Baseline
  2. TF-IDF + Logistic Regression Classifier
  3. TF-IDF + Multinomial Naive Bayes Classifier
Never fabricates metrics; computes real evaluation on identical held-out test data.
"""

import os
import re
import json
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

import config


# Rule-based heuristics dictionary
HEURISTIC_RULES = {
    "urgency_cues": [
        r"\b(urgent|immediately|immediate action|act now|critical update|account suspended|suspended|restricted|compromised)\b",
        r"\b(action required|security alert|warning|unauthorized|unusual activity|terminate|freeze|locked)\b",
        r"\b(within \d+ hours|24 hours|deadline|expires soon|final notice)\b"
    ],
    "credential_harvesting": [
        r"\b(verify your account|confirm your password|login details|credentials|enter password|reset link)\b",
        r"\b(confirm identity|security question|ssn|social security|passcode|pin code|two-factor code)\b",
        r"\b(click here to login|sign in to|update payment method|billing information|wallet)\b"
    ],
    "financial_bait": [
        r"\b(wire transfer|bank account|lottery|winner|inherited|tax refund|crypto|bitcoin|funds transferred)\b",
        r"\b(invoice attached|overdue payment|remittance|money_val|claim your prize|free gift card)\b"
    ],
    "suspicious_patterns": [
        r"\b(http_url|ip_addr)\b",
        r"\b(click here|click below|download attachment|open invoice|see details)\b"
    ]
}


class RuleBasedPhishingClassifier:
    """
    Expert rules-based baseline inspecting heuristic phishing signatures.
    """
    def __init__(self, threshold: int = 2):
        self.threshold = threshold

    def analyze_message(self, text: str):
        if not isinstance(text, str):
            return {"score": 0, "triggers": [], "prediction": 0}

        triggers = []
        score = 0

        for category, patterns in HEURISTIC_RULES.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    score += len(matches)
                    triggers.append(f"{category}:{matches[0] if isinstance(matches[0], str) else matches[0][0]}")

        # Predict 1 (Phishing) if triggers meet threshold, else 0 (Legitimate)
        pred = 1 if score >= self.threshold else 0
        return {
            "score": score,
            "triggers": triggers,
            "prediction": pred
        }

    def predict(self, texts):
        return np.array([self.analyze_message(t)["prediction"] for t in texts])


def evaluate_models():
    """Executes full evaluation on held-out test dataset."""
    start_time = time.time()
    if not os.path.exists(config.CLEANED_DATA_PATH):
        raise FileNotFoundError(
            f"Cleaned dataset not found at {config.CLEANED_DATA_PATH}. Run train_model.py first."
        )

    print(f"[*] Loading cleaned dataset: {config.CLEANED_DATA_PATH}")
    df = pd.read_csv(config.CLEANED_DATA_PATH)
    print(f"[+] Total samples available: {len(df):,}")

    X = df['clean_text'].astype(str)
    y = df['label'].astype(int)

    # Replicate exact stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Evaluating on {len(X_test):,} held-out test messages...")

    # Load or fit TF-IDF vectorizer
    if os.path.exists(config.VECTORIZER_PATH):
        print(f"[*] Loading vectorizer from: {config.VECTORIZER_PATH}")
        vectorizer = joblib.load(config.VECTORIZER_PATH)
        X_test_vec = vectorizer.transform(X_test)
        X_train_vec = vectorizer.transform(X_train)
    else:
        print("[*] Vectorizer not found on disk, fitting new vectorizer...")
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

    # --- Approach 1: Rules-Based Baseline ---
    print("\n[*] Evaluating Approach 1: Rules-Based Baseline...")
    rule_model = RuleBasedPhishingClassifier(threshold=2)
    t0 = time.time()
    y_pred_rules = rule_model.predict(X_test)
    time_rules = round((time.time() - t0) * 1000 / len(X_test), 3)  # ms per sample

    acc_rules = float(accuracy_score(y_test, y_pred_rules))
    prec_rules = float(precision_score(y_test, y_pred_rules, zero_division=0))
    rec_rules = float(recall_score(y_test, y_pred_rules, zero_division=0))
    f1_rules = float(f1_score(y_test, y_pred_rules, zero_division=0))
    cm_rules = confusion_matrix(y_test, y_pred_rules).tolist()

    # --- Approach 2: Logistic Regression ---
    print("[*] Evaluating Approach 2: TF-IDF + Logistic Regression...")
    if os.path.exists(config.MODEL_PATH):
        lr_model = joblib.load(config.MODEL_PATH)
    else:
        lr_model = LogisticRegression(C=2.0, max_iter=1000, solver='lbfgs', random_state=42)
        lr_model.fit(X_train_vec, y_train)

    t0 = time.time()
    y_pred_lr = lr_model.predict(X_test_vec)
    y_proba_lr = lr_model.predict_proba(X_test_vec)[:, 1]
    time_lr = round((time.time() - t0) * 1000 / len(X_test), 3)

    acc_lr = float(accuracy_score(y_test, y_pred_lr))
    prec_lr = float(precision_score(y_test, y_pred_lr, zero_division=0))
    rec_lr = float(recall_score(y_test, y_pred_lr, zero_division=0))
    f1_lr = float(f1_score(y_test, y_pred_lr, zero_division=0))
    roc_lr = float(roc_auc_score(y_test, y_proba_lr))
    cm_lr = confusion_matrix(y_test, y_pred_lr).tolist()

    # --- Approach 3: Naive Bayes ---
    print("[*] Evaluating Approach 3: TF-IDF + Multinomial Naive Bayes...")
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train_vec, y_train)

    t0 = time.time()
    y_pred_nb = nb_model.predict(X_test_vec)
    y_proba_nb = nb_model.predict_proba(X_test_vec)[:, 1]
    time_nb = round((time.time() - t0) * 1000 / len(X_test), 3)

    acc_nb = float(accuracy_score(y_test, y_pred_nb))
    prec_nb = float(precision_score(y_test, y_pred_nb, zero_division=0))
    rec_nb = float(recall_score(y_test, y_pred_nb, zero_division=0))
    f1_nb = float(f1_score(y_test, y_pred_nb, zero_division=0))
    roc_nb = float(roc_auc_score(y_test, y_proba_nb))
    cm_nb = confusion_matrix(y_test, y_pred_nb).tolist()

    # Summary table
    comparison_data = {
        "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_sample_size": len(y_test),
        "class_breakdown": {
            "legitimate": int((y_test == 0).sum()),
            "phishing": int((y_test == 1).sum())
        },
        "models": {
            "rules_based": {
                "name": "Rules-Based Baseline",
                "accuracy": acc_rules,
                "precision": prec_rules,
                "recall": rec_rules,
                "f1_score": f1_rules,
                "roc_auc": None,
                "latency_ms_per_sample": time_rules,
                "confusion_matrix": {
                    "tn": cm_rules[0][0], "fp": cm_rules[0][1],
                    "fn": cm_rules[1][0], "tp": cm_rules[1][1]
                },
                "strengths": "Fast, fully explainable, transparent heuristic rules.",
                "weaknesses": "Lacks nuance for zero-day phishing phrasing, lower recall."
            },
            "logistic_regression": {
                "name": "Logistic Regression + TF-IDF",
                "accuracy": acc_lr,
                "precision": prec_lr,
                "recall": rec_lr,
                "f1_score": f1_lr,
                "roc_auc": roc_lr,
                "latency_ms_per_sample": time_lr,
                "confusion_matrix": {
                    "tn": cm_lr[0][0], "fp": cm_lr[0][1],
                    "fn": cm_lr[1][0], "tp": cm_lr[1][1]
                },
                "strengths": "High precision and balanced F1-score, well-calibrated probabilities for uncertainty estimation.",
                "weaknesses": "Requires feature extraction pipeline; linear decision boundary."
            },
            "naive_bayes": {
                "name": "Multinomial Naive Bayes + TF-IDF",
                "accuracy": acc_nb,
                "precision": prec_nb,
                "recall": rec_nb,
                "f1_score": f1_nb,
                "roc_auc": roc_nb,
                "latency_ms_per_sample": time_nb,
                "confusion_matrix": {
                    "tn": cm_nb[0][0], "fp": cm_nb[0][1],
                    "fn": cm_nb[1][0], "tp": cm_nb[1][1]
                },
                "strengths": "Ultra-fast inference, probabilistic prior modeling.",
                "weaknesses": "Conditional independence assumption may underestimate combined term risks."
            }
        }
    }

    # Print comparative table
    print("\n" + "=" * 80)
    print(f"{'Model Approach':<30} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 80)
    print(f"{'1. Rules-Based Baseline':<30} | {acc_rules*100:>8.2f}% | {prec_rules*100:>8.2f}% | {rec_rules*100:>8.2f}% | {f1_rules*100:>8.2f}%")
    print(f"{'2. Logistic Regression':<30} | {acc_lr*100:>8.2f}% | {prec_lr*100:>8.2f}% | {rec_lr*100:>8.2f}% | {f1_lr*100:>8.2f}%")
    print(f"{'3. Naive Bayes':<30} | {acc_nb*100:>8.2f}% | {prec_nb*100:>8.2f}% | {rec_nb*100:>8.2f}% | {f1_nb*100:>8.2f}%")
    print("=" * 80)

    with open(config.COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\n[+] Comprehensive comparison saved to: {config.COMPARISON_PATH}")

    total_eval_time = round(time.time() - start_time, 2)
    print(f"[+] Evaluation finished in {total_eval_time}s.")
    return comparison_data


if __name__ == "__main__":
    evaluate_models()
