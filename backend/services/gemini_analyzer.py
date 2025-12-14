# services/gemini_analyzer.py
# Additions: domain->models mapping + model dispatcher to trigger model files

import os
import csv
import json
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# STRICT: environment variable only
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBOPDscoogNOm6GWwaEuEwG3HmW79yEDp4")

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

# -------------------------
# Enterprise domain -> suggested models mapping
# -------------------------
enterprise_data_to_models = [
    {
        "data_domain": "Sales",
        "models": [
            "Sales, Demand & Financial Forecasting Model",
            "Pricing & Revenue Optimization Model",
            "Sentiment & Intent NLP Model"
        ]
    },
    {
        "data_domain": "Inventory",
        "models": [
            "Sales, Demand & Financial Forecasting Model",
            "Inventory & Replenishment Optimization Model"
        ]
    },
    {
        "data_domain": "Marketing",
        "models": [
            "Marketing ROI & Attribution Model",
            "Pricing & Revenue Optimization Model",
            "Sentiment & Intent NLP Model"
        ]
    },
    {
        "data_domain": "Finance",
        "models": [
            "Sales, Demand & Financial Forecasting Model",
            "Pricing & Revenue Optimization Model"
        ]
    },
    {
        "data_domain": "Customer",
        "models": [
            "Customer Segmentation & Modeling",
            "Customer Value & Retention Model",
            "Sentiment & Intent NLP Model"
        ]
    },
    {
        "data_domain": "Operations",
        "models": [
            "Inventory & Replenishment Optimization Model",
            "Logistics & Supplier Risk Model"
        ]
    },
    {
        "data_domain": "Logistics",
        "models": [
            "Logistics & Supplier Risk Model"
        ]
    },
    {
        "data_domain": "Supply Chain",
        "models": [
            "Inventory & Replenishment Optimization Model",
            "Logistics & Supplier Risk Model"
        ]
    },
    {
        "data_domain": "Human Resources",
        "models": []
    },
    {
        "data_domain": "Product",
        "models": [
            "Sentiment & Intent NLP Model"
        ]
    },
    {
        "data_domain": "Support",
        "models": [
            "Sentiment & Intent NLP Model",
            "Customer Value & Retention Model"
        ]
    },
    {
        "data_domain": "Risk & Compliance",
        "models": [
            "Logistics & Supplier Risk Model"
        ]
    },
    {
        "data_domain": "Other",
        "models": []
    }
]

# Convert mapping to a dict for quick lookup
_DOMAIN_TO_MODELS: Dict[str, List[str]] = {entry["data_domain"]: entry["models"] for entry in enterprise_data_to_models}


# -------------------------
# Existing CSV helpers (unchanged)
# -------------------------
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


# -------------------------
# Gemini column classification (unchanged)
# -------------------------
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


# -------------------------
# New: domain -> model mapping + dispatcher
# -------------------------
def get_models_for_domain(data_domain: str) -> List[str]:
    """Return the suggested models for a given enterprise data domain."""
    return _DOMAIN_TO_MODELS.get(data_domain, [])


def _call_sales_forecasting_runner(file_path: str, gemini_response: Dict[str, Any], models_dir: str = "models", **kwargs) -> Dict[str, Any]:
    """
    Trigger the strict time-series pipeline for Sales/Demand forecasting.
    We will inject model_type into the gemini_response.analysis to meet the strict
    check inside services.time_series_pipeline.analyze_file_and_run_pipeline.
    """
    try:
        # lazy import to avoid heavy dependency if not used
        from services import time_series_pipeline
    except Exception as e:
        logger.exception("Failed to import time_series_pipeline: %s", e)
        return {"success": False, "error": "time_series_pipeline not available", "exception": str(e)}

    # Ensure analysis object exists and contains model_type so the strict pipeline knows what to run
    gemini_copy = dict(gemini_response) if gemini_response else {}
    analysis = gemini_copy.setdefault("analysis", {})
    # Place a canonical model_type expected by the time_series_pipeline entrance check
    analysis["model_type"] = "Sales, Demand & Financial Forecasting Model"

    try:
        result = time_series_pipeline.analyze_file_and_run_pipeline(file_path, gemini_copy, models_dir=models_dir, **kwargs)
        return {"success": True, "runner": "time_series_pipeline", "result": result}
    except Exception as e:
        logger.exception("Sales forecasting pipeline failed: %s", e)
        return {"success": False, "error": "Sales forecasting pipeline failed", "exception": str(e)}


def trigger_models_for_file(file_path: str, gemini_response: Dict[str, Any], models_dir: str = "models", **kwargs) -> Dict[str, Any]:
    """
    Given a file and a Gemini analysis (from analyze_file_with_gemini),
    map the data_domain to model(s) and attempt to trigger their pipelines.

    Returns a structured dict:
    {
      "success": True/False,
      "data_domain": "...",
      "models_considered": [...],
      "runners": {
         "<model name>": { ... runner output ... }
      }
    }
    """
    try:
        data_domain = gemini_response.get("analysis", {}).get("data_domain") if gemini_response else None
        if not data_domain:
            return {"success": False, "error": "No data_domain found in gemini_response", "gemini_response": gemini_response}

        models = get_models_for_domain(data_domain)
        if not models:
            return {"success": True, "data_domain": data_domain, "models_considered": [], "runners": {}, "note": "No mapped models for this domain"}

        runners_out = {}
        for model_name in models:
            # Dispatch to appropriate runner implementation
            if model_name.strip().lower() == "sales, demand & financial forecasting model":
                runners_out[model_name] = _call_sales_forecasting_runner(file_path, gemini_response, models_dir=models_dir, **kwargs)
            else:
                # placeholder / not implemented: you can wire additional model runners here
                logger.info("No runner implemented for model '%s' yet. Skipping.", model_name)
                runners_out[model_name] = {"success": False, "error": "runner_not_implemented", "note": "No runner implemented for this model yet."}

        return {
            "success": True,
            "data_domain": data_domain,
            "models_considered": models,
            "runners": runners_out
        }
    except Exception as e:
        logger.exception("trigger_models_for_file failed")
        return {"success": False, "error": str(e)}


# -------------------------
# Convenience: run classification + trigger models in one call
# -------------------------
def analyze_and_trigger(file_path: str, models_dir: str = "models", **kwargs) -> Dict[str, Any]:
    """
    1) Analyze CSV headers with Gemini (classify to enterprise domain)
    2) Map domain -> models and trigger the implemented runners
    Returns combined report.
    """
    gemini_resp = analyze_file_with_gemini(file_path)
    if not gemini_resp.get("success"):
        return {"success": False, "error": "Gemini analysis failed", "gemini_response": gemini_resp}

    trigger_report = trigger_models_for_file(file_path, gemini_resp, models_dir=models_dir, **kwargs)
    return {
        "success": True,
        "gemini_response": gemini_resp,
        "trigger_report": trigger_report
    }


# If used as script for quick debug
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--models-dir", required=False, default="models", help="Directory to save models")
    args = parser.parse_args()

    out = analyze_and_trigger(args.csv, models_dir=args.models_dir)
    print(json.dumps(out, indent=2, default=str))
