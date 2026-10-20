"""
PhishShift - Model Training Pipeline
Integrates real Kaggle phishing email dataset, cleans data, trains TF-IDF + Logistic Regression,
evaluates performance without fabricated metrics, and saves serializable artifacts.
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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import config


def preprocess_text(text: str) -> str:
    """Cleans raw email/message text while preserving cybersecurity-relevant tokens."""
    if not isinstance(text, str):
        return ""
    # Normalize unicode spaces and control characters
    text = re.sub(r'[\r\n\t]+', ' ', text)
    # Standardize links to retain semantic cue
    text = re.sub(r'https?://\S+|www\.\S+', ' http_url ', text)
    # Standardize email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' email_addr ', text)
    # Standardize IP addresses
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', ' ip_addr ', text)
    # Standardize monetary figures ($1,000, etc.)
    text = re.sub(r'[\$£€¥]\s*\d+([,\.]\d+)?', ' money_val ', text)
    # Normalize multiple whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def map_label(val) -> int:
    """Maps diverse label formats to 1 (Phishing) or 0 (Legitimate)."""
    s = str(val).strip().lower()
    if s in config.PHISHING_LABEL_VALUES:
        return 1
    if s in config.SAFE_LABEL_VALUES:
        return 0
    # Fallback heuristics for labels containing keywords
    if "phish" in s or "spam" in s or "malicious" in s:
        return 1
    if "safe" in s or "ham" in s or "legit" in s:
        return 0
    raise ValueError(f"Unrecognized class label value: '{val}'")


def load_and_clean_dataset():
    """
    Validates actual dataset columns instead of assuming their names (Requirement 11).
    Handles missing values and duplicates (Requirement 2).
    """
    if not os.path.exists(config.RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {config.RAW_DATA_PATH}. Please ensure phishing_email.csv is present in data/."
        )

    print(f"[*] Loading raw dataset from: {config.RAW_DATA_PATH}")
    df_raw = pd.read_csv(config.RAW_DATA_PATH)
    initial_rows = len(df_raw)
    print(f"[+] Initial row count: {initial_rows:,}")
    print(f"[+] Columns detected: {list(df_raw.columns)}")

    # Dynamically resolve text and label column names
    text_col, label_col = config.detect_columns(df_raw)
    print(f"[+] Validated columns -> Text column: '{text_col}', Label column: '{label_col}'")

    # Step 1: Remove missing values
    missing_before = df_raw[[text_col, label_col]].isnull().sum().to_dict()
    print(f"[*] Missing values before cleaning: {missing_before}")
    df_clean = df_raw.dropna(subset=[text_col, label_col]).copy()
    dropped_nulls = initial_rows - len(df_clean)
    print(f"[+] Dropped {dropped_nulls} row(s) with missing text or labels.")

    # Step 2: Map labels to 0 and 1
    df_clean['label'] = df_clean[label_col].apply(map_label)
    df_clean['label_text'] = df_clean['label'].map({1: 'Phishing', 0: 'Legitimate'})

    # Step 3: Text preprocessing
    print("[*] Preprocessing message text...")
    df_clean['clean_text'] = df_clean[text_col].astype(str).apply(preprocess_text)

    # Filter out empty strings after preprocessing
    df_clean = df_clean[df_clean['clean_text'].str.strip().str.len() > 3]

    # Step 4: Remove duplicates
    duplicates_count = df_clean.duplicated(subset=['clean_text']).sum()
    df_clean = df_clean.drop_duplicates(subset=['clean_text']).reset_index(drop=True)
    print(f"[+] Removed {duplicates_count:,} duplicate text records.")
    final_rows = len(df_clean)
    print(f"[+] Cleaned dataset size: {final_rows:,} rows")

    class_distribution = df_clean['label_text'].value_counts().to_dict()
    print(f"[+] Class distribution: {class_distribution}")

    # Step 5: Save cleaned dataset
    clean_cols = ['clean_text', 'label', 'label_text']
    df_clean[clean_cols].to_csv(config.CLEANED_DATA_PATH, index=False)
    print(f"[+] Cleaned dataset saved to: {config.CLEANED_DATA_PATH}")

    metadata = {
        "initial_rows": initial_rows,
        "dropped_nulls": dropped_nulls,
        "duplicates_removed": int(duplicates_count),
        "final_rows": final_rows,
        "text_column_detected": text_col,
        "label_column_detected": label_col,
        "class_distribution": class_distribution,
    }
    return df_clean, metadata


def train():
    """Trains TF-IDF vectorizer + Logistic Regression model and saves verified artifacts."""
    start_time = time.time()
    df, clean_metadata = load_and_clean_dataset()

    X = df['clean_text']
    y = df['label']

    print("\n[*] Splitting dataset (80% train / 20% test, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Train set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")

    print("\n[*] Fitting TF-IDF Vectorizer (max_features=10000, ngrams=(1,2))...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words='english',
        min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"[+] Vocabulary size: {len(vectorizer.vocabulary_):,} features")

    print("\n[*] Training Logistic Regression Classifier (C=2.0, solver='lbfgs')...")
    model = LogisticRegression(
        C=2.0,
        max_iter=1000,
        solver='lbfgs',
        random_state=42
    )
    model.fit(X_train_vec, y_train)
    training_duration = round(time.time() - start_time, 2)
    print(f"[+] Training completed in {training_duration} seconds.")

    print("\n[*] Evaluating model on test set...")
    y_train_pred = model.predict(X_train_vec)
    y_test_pred = model.predict(X_test_vec)
    y_test_proba = model.predict_proba(X_test_vec)[:, 1]

    # Real, un-fabricated metrics computation
    train_acc = float(accuracy_score(y_train, y_train_pred))
    test_acc = float(accuracy_score(y_test, y_test_pred))
    test_prec = float(precision_score(y_test, y_test_pred, zero_division=0))
    test_rec = float(recall_score(y_test, y_test_pred, zero_division=0))
    test_f1 = float(f1_score(y_test, y_test_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_test_proba))
    cm = confusion_matrix(y_test, y_test_pred).tolist()

    print(f"[+] Train Accuracy : {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"[+] Test Accuracy  : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"[+] Test Precision : {test_prec:.4f} ({test_prec*100:.2f}%)")
    print(f"[+] Test Recall    : {test_rec:.4f} ({test_rec*100:.2f}%)")
    print(f"[+] Test F1-Score  : {test_f1:.4f} ({test_f1*100:.2f}%)")
    print(f"[+] ROC-AUC Score  : {roc_auc:.4f}")
    print(f"[+] Confusion Matrix:\n    TN={cm[0][0]}, FP={cm[0][1]}\n    FN={cm[1][0]}, TP={cm[1][1]}")

    # Extract top influential keywords for explainability
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    top_phishing_indices = np.argsort(coefficients)[-20:][::-1]
    top_legit_indices = np.argsort(coefficients)[:20]

    top_phishing_features = [
        {"feature": str(feature_names[i]), "weight": float(round(coefficients[i], 4))}
        for i in top_phishing_indices
    ]
    top_legit_features = [
        {"feature": str(feature_names[i]), "weight": float(round(coefficients[i], 4))}
        for i in top_legit_indices
    ]

    # Save artifacts
    print(f"\n[*] Saving trained model to: {config.MODEL_PATH}")
    joblib.dump(model, config.MODEL_PATH)
    print(f"[*] Saving vectorizer to: {config.VECTORIZER_PATH}")
    joblib.dump(vectorizer, config.VECTORIZER_PATH)

    metrics_payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "Logistic Regression + TF-IDF",
        "training_duration_seconds": training_duration,
        "dataset_statistics": clean_metadata,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "performance": {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "precision": test_prec,
            "recall": test_rec,
            "f1_score": test_f1,
            "roc_auc": roc_auc,
            "confusion_matrix": {
                "true_negative": cm[0][0],
                "false_positive": cm[0][1],
                "false_negative": cm[1][0],
                "true_positive": cm[1][1]
            }
        },
        "top_phishing_indicators": top_phishing_features,
        "top_legitimate_indicators": top_legit_features,
        "thresholds": {
            "phishing": config.PHISHING_THRESHOLD,
            "legitimate": config.LEGITIMATE_THRESHOLD
        }
    }

    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[+] Real evaluation metrics saved to: {config.METRICS_PATH}")
    print("\n[SUCCESS] Model training and artifact generation completed successfully!")
    return metrics_payload


if __name__ == "__main__":
    train()
