# services/gemini_analyzer.py
import os
import csv
import json
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# STRICT: environment variable only
GEMINI_API_KEY = "AIzaSyBw92Kb3L1GTN5lh_UaH4gqdYVyiUyMiYU"

if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        logger.exception("Failed to configure Gemini client")
else:
    logger.warning("GEMINI_API_KEY not set; Gemini calls will fail")


# Allowed enterprise data domains
ENTERPRISE_DATA_DOMAINS = [
    "Sales",
    "Inventory",
    "Marketing",
    "Finance",
    "Customer",
    "Operations",
    "Logistics",
    "Supply Chain",
    "Human Resources",
    "Product",
    "Support",
    "Risk & Compliance",
    "Other"
]


def extract_columns_from_csv_file(file_path: str) -> List[str]:
    """Read CSV and return header column names."""
    try:
        with open(file_path, newline="", encoding="utf-8") as csvfile:
            sample = csvfile.read(2048)
            csvfile.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample) if sample else csv.excel
            except Exception:
                dialect = csv.excel

            reader = csv.reader(csvfile, dialect)
            header = next(reader, [])
            return [c.strip() for c in header if c.strip()]
    except Exception:
        logger.exception("Failed to extract CSV columns: %s", file_path)
        return []


def analyze_columns_with_gemini(columns: List[str]) -> Dict[str, Any]:
    """
    STRICT Gemini-only enterprise data classification.

    Output JSON:
    {
      "data_domain": "<ENTERPRISE_DATA_DOMAINS>",
      "confidence": 0.0-1.0,
      "key_columns": [...],
      "explanation": "..."
    }
    """

    if not GEMINI_API_KEY:
        return {
            "success": False,
            "error": "GEMINI_API_KEY missing",
            "analysis": {}
        }

    if not columns:
        return {
            "success": False,
            "error": "No CSV columns found",
            "analysis": {}
        }

    prompt = f"""
You are an enterprise data architect.

Your task is to classify the following CSV header into ONE enterprise data domain.

Return ONLY a single JSON object with this exact shape:

{{
  "data_domain": "...",            // MUST be exactly one of: {ENTERPRISE_DATA_DOMAINS}
  "confidence": 0.0,               // number between 0.0 and 1.0
  "key_columns": ["col1", "col2"], // subset of provided columns ONLY
  "explanation": "..."             // 1–2 sentence explanation
}}

STRICT RULES:
- Use ONLY the column names provided
- DO NOT invent columns
- DO NOT suggest models, analytics, or ML
- This is NOT modeling, ONLY data classification
- If unsure, use "Other" with low confidence
- Output MUST be valid JSON and NOTHING ELSE

CSV header columns:
{columns}
"""

    try:
        import google.generativeai as genai
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        raw = getattr(resp, "text", str(resp))

        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return {
                "success": False,
                "error": "Gemini returned non-JSON",
                "raw_response": raw,
                "analysis": {}
            }

        parsed = json.loads(match.group())

        # Validate domain
        domain = parsed.get("data_domain")
        if domain not in ENTERPRISE_DATA_DOMAINS:
            raise ValueError(f"Invalid data_domain: {domain}")

        # Validate confidence
        confidence = float(parsed.get("confidence"))
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("confidence must be between 0 and 1")

        # Validate key columns
        key_columns = parsed.get("key_columns", [])
        if not isinstance(key_columns, list):
            raise ValueError("key_columns must be a list")
        for col in key_columns:
            if col not in columns:
                raise ValueError(f"Unknown column in key_columns: {col}")

        explanation = parsed.get("explanation", "")
        if not isinstance(explanation, str):
            raise ValueError("explanation must be string")

        analysis = {
            "data_domain": domain,
            "confidence": round(confidence, 3),
            "key_columns": key_columns,
            "explanation": explanation.strip()
        }

        logger.info("Gemini classified CSV as %s (%.2f)", domain, confidence)

        return {
            "success": True,
            "analysis": analysis,
            "raw_response": raw
        }

    except Exception as e:
        logger.exception("Gemini classification failed")
        return {
            "success": False,
            "error": str(e),
            "analysis": {},
            "raw_response": raw if "raw" in locals() else None
        }


def analyze_file_with_gemini(file_path: str) -> Dict[str, Any]:
    columns = extract_columns_from_csv_file(file_path)
    if not columns:
        return {
            "success": False,
            "error": "CSV header could not be read",
            "analysis": {}
        }
    return analyze_columns_with_gemini(columns)
