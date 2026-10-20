import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
INSTANCE_DIR = BASE_DIR / "instance"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Ensure directories exist
for folder in [DATA_DIR, MODELS_DIR, INSTANCE_DIR, STATIC_DIR / "images"]:
    folder.mkdir(parents=True, exist_ok=True)

# Data files
RAW_DATA_PATH = DATA_DIR / "phishing_email.csv"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_messages.csv"

# Model artifacts
MODEL_PATH = MODELS_DIR / "phishing_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
COMPARISON_PATH = MODELS_DIR / "model_comparison.json"

# SQLite Database
DATABASE_PATH = INSTANCE_DIR / "phishshift.db"

# Flask Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "phishshift-sec-key-7f9a2b8e3c1d4a5b6e8f")
DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1")

# Classification Thresholds (Requirement 6: Phishing, Legitimate, Uncertain)
# Probabilities >= PHISHING_THRESHOLD -> Phishing
# Probabilities <= LEGITIMATE_THRESHOLD -> Legitimate
# In between -> Uncertain / Review Recommended
PHISHING_THRESHOLD = 0.65
LEGITIMATE_THRESHOLD = 0.35

# Text & Column Detection Candidates
POSSIBLE_TEXT_COLS = [
    "Email Text", "email_text", "text", "body", "message", "content", "Email", "v2"
]
POSSIBLE_LABEL_COLS = [
    "Email Type", "email_type", "label", "type", "class", "target", "Category", "v1"
]

PHISHING_LABEL_VALUES = {
    "phishing email", "phishing", "spam", "1", 1, "malicious", "fraud", "true"
}
SAFE_LABEL_VALUES = {
    "safe email", "safe", "ham", "0", 0, "legitimate", "normal", "false"
}


def detect_columns(df):
    """
    Validates actual dataset columns instead of assuming their names (Requirement 11).
    Returns (text_col, label_col) tuple.
    """
    text_col = None
    label_col = None

    # Check candidates
    for col in df.columns:
        clean_col = str(col).strip()
        for cand in POSSIBLE_TEXT_COLS:
            if clean_col.lower() == cand.lower():
                text_col = col
                break
        if text_col:
            break

    for col in df.columns:
        clean_col = str(col).strip()
        if col == text_col:
            continue
        for cand in POSSIBLE_LABEL_COLS:
            if clean_col.lower() == cand.lower():
                label_col = col
                break
        if label_col:
            break

    # Fallback to heuristics based on dtypes and content if column names are unfamiliar
    if not text_col:
        for col in df.columns:
            if col != label_col and df[col].dtype == object and df[col].astype(str).str.len().mean() > 25:
                text_col = col
                break

    if not label_col:
        for col in df.columns:
            if col != text_col and df[col].nunique() <= 10:
                label_col = col
                break

    if not text_col or not label_col:
        raise ValueError(
            f"Could not automatically detect text and label columns from dataset columns: {list(df.columns)}"
        )

    return text_col, label_col
