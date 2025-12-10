# services/gemini_analyzer.py

import google.generativeai as genai
import os
from typing import Optional, List, Dict
import csv
import json
import re
import logging

logger = logging.getLogger(__name__)

# IMPORTANT: use environment variable ONLY
GEMINI_API_KEY = "AIzaSyCeQmIwBMnF6AVQe_Uzy8lC7TE2HvhefNs"
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        pass


MODEL_RECOMMENDATIONS = {
    "Marketing ROI & Attribution Model": {
        "models": ["Markov Chain Attribution", "Shapley Value Attribution",
                   "Multi-Touch Attribution Models", "Uplift Modeling"],
        "description": "Measure marketing effectiveness and attribute ROI across channels"
    },
    "Customer Segmentation & Modeling": {
        "models": ["K-Means", "DBSCAN", "Gaussian Mixture Models", "Hierarchical Clustering"],
        "description": "Segment customers based on behavior and characteristics"
    },
    "Sales, Demand & Financial Forecasting Model": {
        "models": ["Prophet", "ARIMA", "Exponential Smoothing"],
        "description": "Time series forecasting for sales and demand prediction"
    },
    "Customer Value & Retention Model": {
        "models": ["Survival Analysis", "Random Forest", "XGBoost",
                   "Logistic Regression", "LTV Prediction Models"],
        "description": "Predict customer churn probability and lifetime value"
    },
    "Sentiment & Intent NLP Model": {
        "models": ["BERT", "RoBERTa", "DistilBERT", "LSTM Networks", "TextCNN"],
        "description": "Analyze customer sentiment and intent from text data"
    },
    "Inventory & Replenishment Optimization Model": {
        "models": ["Economic Order Quantity (EOQ)", "Multi-Echelon Inventory Optimization",
                   "Reinforcement Learning", "Stochastic Optimization"],
        "description": "Optimize inventory levels and stock management"
    },
    "Logistics & Supplier Risk Model": {
        "models": ["Supply Chain Risk Scoring Models", "Bayesian Networks",
                   "Monte Carlo Simulation", "Decision Trees"],
        "description": "Evaluate logistics performance and supplier risk"
    },
}


def extract_columns_from_csv_file(file_path: str) -> List[str]:
    """Read CSV and return header column names."""
    try:
        with open(file_path, newline="", encoding="utf-8") as csvfile:
            sample = csvfile.read(2048)
            csvfile.seek(0)

            has_header = csv.Sniffer().has_header(sample)
            try:
                dialect = csv.Sniffer().sniff(sample) if sample else csv.excel
            except Exception:
                dialect = csv.excel

            reader = csv.reader(csvfile, dialect)
            header = next(reader, [])
            return [col.strip() for col in header if col.strip()]
    except Exception:
        return []


def analyze_columns_with_gemini(columns: List[str],
                                model_type: Optional[str] = None) -> Dict:
    """
    Strict analysis using Gemini. NO fallbacks.
    Gemini MUST return:
        model_type, target_column, key_features (list), explanation

    Requirements enforced:
      • target_column MUST match EXACTLY one of the CSV headers
      • model_type MUST exist in MODEL_RECOMMENDATIONS
    """

    if not GEMINI_API_KEY:
        return {
            "success": False,
            "error": "Gemini API key is missing. Set GEMINI_API_KEY.",
            "analysis": {}
        }

    if not columns:
        return {
            "success": False,
            "error": "No CSV columns found.",
            "analysis": {}
        }

    prompt = f"""
Analyze ONLY the following CSV columns:

{columns}

Your response MUST be valid JSON with the structure:
{{
  "model_type": "...",
  "target_column": "...",
  "key_features": ["...", "..."],
  "explanation": "..."
}}

STRICT RULES:
• model_type MUST be one of these keys: {list(MODEL_RECOMMENDATIONS.keys())}
• target_column MUST be EXACTLY one of the provided CSV column names
• key_features MUST contain ONLY items from the CSV columns
• DO NOT invent or guess new column names
• DO NOT hallucinate fields like "Total Sales"
"""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        raw = getattr(resp, "text", str(resp))

        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            return {
                "success": False,
                "error": "Gemini returned non-JSON response.",
                "raw_response": raw,
                "analysis": {}
            }

        parsed = json.loads(json_match.group())

        # strict validation
        if parsed.get("model_type") not in MODEL_RECOMMENDATIONS:
            raise ValueError(f"Invalid model_type: {parsed.get('model_type')}")

        if parsed.get("target_column") not in columns:
            raise ValueError(f"Gemini selected non-existent target column: "
                             f"{parsed.get('target_column')}")

        key_feats = parsed.get("key_features", [])
        if not isinstance(key_feats, list) or any(k not in columns for k in key_feats):
            raise ValueError("key_features contains invalid or unknown columns.")

        return {
            "success": True,
            "analysis": parsed,
            "raw_response": raw
        }

    except Exception as e:
        logger.exception("Gemini analysis failed: %s", e)
        return {
            "success": False,
            "error": str(e),
            "analysis": {},
            "raw_response": raw if "raw" in locals() else None
        }


def analyze_file_with_gemini(file_path: str,
                             model_type: Optional[str] = None) -> Dict:
    """Strict Gemini-only analysis. No fallback."""
    columns = extract_columns_from_csv_file(file_path)
    if not columns:
        return {
            "success": False,
            "error": "CSV header could not be read.",
            "analysis": {}
        }

    return analyze_columns_with_gemini(columns, model_type)


def get_available_models() -> dict:
    return MODEL_RECOMMENDATIONS
