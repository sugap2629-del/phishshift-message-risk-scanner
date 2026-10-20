"""
PhishShift - Test Suite
Covers unit tests for preprocessing, heuristic rule signatures, ML tri-state classification,
Flask web routes, SQLite database persistence, and REST APIs.
Built using Python's standard `unittest` library for maximum compatibility.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
import app as flask_module
from train_model import preprocess_text
from evaluate_model import RuleBasedPhishingClassifier


class TestPreprocessing(unittest.TestCase):
    """Validates message text normalization and token preservation."""

    def test_url_normalization(self):
        text = "Visit https://secure-bank-login.com/auth now"
        cleaned = preprocess_text(text)
        self.assertIn("http_url", cleaned)
        self.assertNotIn("https://", cleaned)

    def test_ip_address_normalization(self):
        text = "Unauthorized login from 192.168.1.1"
        cleaned = preprocess_text(text)
        self.assertIn("ip_addr", cleaned)

    def test_money_normalization(self):
        text = "You won $5,000 in the cash prize pool"
        cleaned = preprocess_text(text)
        self.assertIn("money_val", cleaned)

    def test_empty_and_non_string_input(self):
        self.assertEqual(preprocess_text(""), "")
        self.assertEqual(preprocess_text(None), "")
        self.assertEqual(preprocess_text(12345), "")


class TestHeuristicRules(unittest.TestCase):
    """Validates expert heuristic signature analyzer."""

    def setUp(self):
        self.rules = RuleBasedPhishingClassifier(threshold=2)

    def test_phishing_rule_triggers(self):
        text = "Urgent: Your account is suspended. Verify your account password immediately!"
        res = self.rules.analyze_message(text)
        self.assertGreaterEqual(res["score"], 2)
        self.assertEqual(res["prediction"], 1)
        self.assertGreater(len(res["triggers"]), 0)

    def test_benign_rule_non_triggers(self):
        text = "Let us schedule our weekly team sync for Thursday at 2pm."
        res = self.rules.analyze_message(text)
        self.assertEqual(res["score"], 0)
        self.assertEqual(res["prediction"], 0)
        self.assertEqual(len(res["triggers"]), 0)


class TestModelAnalysis(unittest.TestCase):
    """Validates ML threat analysis and tri-state verdict logic."""

    def test_analyze_phishing_message(self):
        text = (
            "URGENT: Your bank account is locked due to suspicious activity. "
            "Please click http://phish-secure-bank.com/verify to enter your credentials immediately!"
        )
        result = flask_module.analyze_text(text)
        self.assertEqual(result["prediction"], "Phishing")
        self.assertIn(result["risk_level"], ("HIGH", "CRITICAL"))
        self.assertGreaterEqual(result["phishing_prob"], 0.50)
        self.assertIn("rule_triggers", result)
        self.assertIn("keywords_found", result)

    def test_analyze_legitimate_message(self):
        text = (
            "Hello colleagues, please find the quarterly report attached for your review. "
            "Let me know if any updates are needed before tomorrow's executive presentation."
        )
        result = flask_module.analyze_text(text)
        self.assertEqual(result["prediction"], "Legitimate")
        self.assertEqual(result["risk_level"], "LOW")
        self.assertGreaterEqual(result["legit_prob"], 0.50)

    def test_tri_state_uncertain_behavior(self):
        text = "Your invoice statement is ready. You can review your account receipt online."
        result = flask_module.analyze_text(text)
        self.assertIn(result["prediction"], ("Legitimate", "Uncertain", "Phishing"))
        self.assertTrue(0.0 <= result["phishing_prob"] <= 1.0)
        self.assertTrue(0.0 <= result["legit_prob"] <= 1.0)


class TestWebRoutesAndDatabase(unittest.TestCase):
    """Integration tests for web views, SQLite DB, and HTTP status codes."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = Path(self.temp_dir) / "test_phishshift.db"
        self.orig_db = config.DATABASE_PATH
        config.DATABASE_PATH = self.test_db

        flask_module.app.config["TESTING"] = True
        flask_module.init_database()
        self.client = flask_module.app.test_client()

    def tearDown(self):
        config.DATABASE_PATH = self.orig_db
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_index_page(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"PhishShift", res.data)
        self.assertIn(b"ADVANCED THREAT ANALYSIS ENGINE", res.data)

    def test_dashboard_page(self):
        res = self.client.get("/dashboard")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Threat Operations", res.data)
        self.assertIn(b"Benchmark Comparison", res.data)

    def test_history_page(self):
        res = self.client.get("/history")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Scan History", res.data)

    def test_about_page(self):
        res = self.client.get("/about")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"System Architecture", res.data)

    def test_scan_submission_and_results_view(self):
        post_res = self.client.post("/scan", data={"message_text": "Security warning: verify password now!"})
        self.assertEqual(post_res.status_code, 302)
        redirect_url = post_res.headers["Location"]
        self.assertIn("/results/", redirect_url)

        res = self.client.get(redirect_url)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Scan Analysis", res.data)
        self.assertIn(b"Probability Spectrum", res.data)

    def test_delete_scan(self):
        flask_module.record_scan_to_db("Test message", {
            "clean_text": "test message",
            "prediction": "Legitimate",
            "confidence": 95.0,
            "phishing_prob": 0.05,
            "legit_prob": 0.95,
            "risk_level": "LOW",
            "rule_triggers": [],
            "keywords_found": [],
            "duration_ms": 2.5,
            "model_name": "Logistic Regression"
        })
        res = self.client.post("/history/delete/1")
        self.assertEqual(res.status_code, 302)


class TestRestAPI(unittest.TestCase):
    """Validates JSON programmatic endpoints."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = Path(self.temp_dir) / "test_phishshift.db"
        self.orig_db = config.DATABASE_PATH
        config.DATABASE_PATH = self.test_db

        flask_module.app.config["TESTING"] = True
        flask_module.init_database()
        self.client = flask_module.app.test_client()

    def tearDown(self):
        config.DATABASE_PATH = self.orig_db
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_api_scan_success(self):
        payload = {"message": "Urgent alert: update your bank account information."}
        res = self.client.post("/api/scan", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("prediction", data)
        self.assertIn("phishing_prob", data)
        self.assertIn("scan_id", data)
        self.assertGreater(data["scan_id"], 0)

    def test_api_scan_empty_payload(self):
        res = self.client.post("/api/scan", json={"message": ""})
        self.assertEqual(res.status_code, 400)

    def test_api_stats(self):
        res = self.client.get("/api/stats")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("total_scans", data)
        self.assertIn("phishing_count", data)
        self.assertIn("legitimate_count", data)


if __name__ == "__main__":
    unittest.main()
