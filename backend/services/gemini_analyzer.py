import os
import csv
import json
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# STRICT: environment variable only
GEMINI_API_KEY = "AIzaSyB_cMKuBZPux9FttkqZSFEsDJjcUlyukqY"

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
            "Inventory & Replenishment Optimization Model",
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
    STRICT Gemini-only enterprise data classification + per-model target suggestion.

    Output JSON must be a single JSON object with this exact (top-level) shape:

    {
      "data_domain": "<one of ENTERPRISE_DATA_DOMAINS>",
      "confidence": 0.0-1.0,
      "key_columns": [...],            // subset of provided columns ONLY
      "explanation": "...",            // 1–2 sentence explanation
      "model_targets": {               // for each model mapped to the chosen domain
          "<model name>": "<target_column name or null>", 
          ...
      }
    }

    STRICT RULES (must be obeyed by the model):
    - Use ONLY the column names provided below (do NOT invent columns)
    - The top-level "data_domain" MUST be exactly one of ENTERPRISE_DATA_DOMAINS
    - "confidence" must be a number between 0.0 and 1.0
    - "key_columns" must be a list and only contain names from the provided columns
    - "model_targets" must be an object. For each model that maps to the returned domain,
      return either a single column name (must be one of the provided columns) or null
      if no suitable target exists.
    - Do NOT suggest models, analytics, or ML process descriptions beyond the required fields
    - If unsure about the domain or targets, use "Other" and null targets with low confidence
    - Output MUST be valid JSON and NOTHING ELSE
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

    # Build a compact mapping string for the model to reference which models are mapped to each domain.
    domain_to_models_text = json.dumps(_DOMAIN_TO_MODELS, indent=2)

    prompt = f"""
You are an enterprise data architect.

Classify the following CSV header into EXACTLY ONE enterprise data domain and for each model that is mapped to that domain, propose a single primary TARGET COLUMN (from the provided CSV header) that would be appropriate as the target variable for that model.

You are given:
- The allowed enterprise data domains (use exactly these strings): {json.dumps(ENTERPRISE_DATA_DOMAINS)}
- A domain -> models mapping (do NOT change model names): {domain_to_models_text}

Task:
1) Choose exactly one "data_domain" from the allowed list that best fits the provided CSV header columns.
2) Return the model_targets map containing an entry for every model that is mapped to the chosen domain (as shown in the domain->models mapping above).
   - For each model, set the value to either a single column name from the provided CSV header (the primary target variable), OR null if no suitable target column is present.
   - Do NOT invent or modify column names; use only names from the CSV header provided below.
3) Provide a "confidence" score (0.0-1.0) for the domain classification.
4) Provide "key_columns" as a subset of the provided columns that support your classification.
5) Provide "explanation" (1-2 sentences).

Return ONLY a single JSON object with this EXACT shape (no extra fields, no commentary, valid JSON only):

{{
  "data_domain": "...",            // exactly one of the allowed domains
  "confidence": 0.0,               // number between 0.0 and 1.0
  "key_columns": ["col1", "col2"], // subset of provided columns ONLY
  "explanation": "...",            // 1–2 sentence explanation
  "model_targets": {{
      "<model name>": "<column name or null>",
      "...": "..."
  }}
}}

CSV header columns:
{json.dumps(columns)}

STRICT RULES:
- Use ONLY the column names provided in the CSV header above.
- DO NOT invent columns.
- Each model in model_targets MUST be one of the models listed for the chosen domain.
- Each target value must be either a string equal to a column name above, or null.
- If unsure, pick "Other" domain, set low confidence, and set all model targets to null.
- Output MUST be valid JSON and NOTHING ELSE.
"""

    try:
        import google.generativeai as genai
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        raw = getattr(resp, "text", str(resp))

        # extract first JSON object found
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

        # Validate model_targets
        model_targets_raw = parsed.get("model_targets", {})
        if not isinstance(model_targets_raw, dict):
            raise ValueError("model_targets must be a dict")

        # Allowed models for the detected domain
        allowed_models_for_domain = _DOMAIN_TO_MODELS.get(domain, [])

        model_targets: Dict[str, Optional[str]] = {}
        for model_name, target_value in model_targets_raw.items():
            # model name must be one of allowed for chosen domain
            if model_name not in allowed_models_for_domain:
                raise ValueError(f"Model '{model_name}' not mapped to domain '{domain}'")

            # allow either a string (column name) or null
            if target_value is None:
                model_targets[model_name] = None
                continue

            if isinstance(target_value, str):
                t = target_value.strip()
                if t == "" or t.lower() == "null":
                    model_targets[model_name] = None
                else:
                    if t not in columns:
                        raise ValueError(f"Unknown column in model_targets for model '{model_name}': {t}")
                    model_targets[model_name] = t
            else:
                # in case the model returned a small object (e.g. {"target_column": "X", ...})
                if isinstance(target_value, dict):
                    # try to pick a field named target_column or target
                    t = target_value.get("target_column") or target_value.get("target")
                    if t is None:
                        model_targets[model_name] = None
                    elif not isinstance(t, str):
                        raise ValueError(f"Invalid target value type for model '{model_name}'")
                    else:
                        if t not in columns:
                            raise ValueError(f"Unknown column in model_targets for model '{model_name}': {t}")
                        model_targets[model_name] = t
                else:
                    raise ValueError(f"Invalid target value for model '{model_name}': {target_value}")

        analysis = {
            "data_domain": domain,
            "confidence": round(confidence, 3),
            "key_columns": key_columns,
            "explanation": explanation.strip(),
            "model_targets": model_targets
        }

        print(raw)

        logger.info("Gemini classified CSV as %s (%.2f) with model targets: %s", domain, confidence, model_targets)

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

def _call_marketing_mmm_runner(file_path: str, gemini_response: Dict[str, Any], models_dir: str = "models", **kwargs) -> Dict[str, Any]:
    """
    Trigger the Marketing ROI & Attribution pipeline.
    Injects the canonical model_type so the marketing_mmm_pipeline entry point will accept it.
    """
    try:
        # lazy import so we only pay the cost when needed
        from services import marketing_mmm_pipeline
    except Exception as e:
        logger.exception("Failed to import marketing_mmm_pipeline: %s", e)
        return {"success": False, "error": "marketing_mmm_pipeline not available", "exception": str(e)}

    gemini_copy = dict(gemini_response) if gemini_response else {}
    analysis = gemini_copy.setdefault("analysis", {})
    analysis["model_type"] = "Marketing ROI & Attribution Model"

    try:
        result = marketing_mmm_pipeline.analyze_file_and_run_pipeline(file_path, gemini_copy, models_dir=models_dir, **kwargs)
        return {"success": True, "runner": "marketing_mmm_pipeline", "result": result}
    except Exception as e:
        logger.exception("Marketing MMM pipeline failed: %s", e)
        return {"success": False, "error": "Marketing MMM pipeline failed", "exception": str(e)}


def _call_inventory_runner(
    file_path: str,
    gemini_response: Dict[str, Any],
    models_dir: str = "models",
    **kwargs
) -> Dict[str, Any]:
    """
    Trigger the Inventory & Replenishment Optimization pipeline.
    Filters incompatible kwargs coming from generic dispatch.
    """
    try:
        from services import inventory_optimization_pipeline
    except Exception as e:
        logger.exception("Failed to import inventory_optimization_pipeline: %s", e)
        return {
            "success": False,
            "error": "inventory_optimization_pipeline not available",
            "exception": str(e),
        }

    # Defensive copy
    gemini_copy = dict(gemini_response) if gemini_response else {}
    analysis = gemini_copy.setdefault("analysis", {})
    analysis["model_type"] = "Inventory & Replenishment Optimization Model"

    # ✅ Only allow kwargs supported by inventory pipeline
    allowed_kwargs = {
        "date_col_hint",
        "holding_cost_rate",
        "ordering_cost",
        "stockout_cost",
        "service_level",
        "lead_time_days",
        "review_period_days",
        "models_dir",
    }

    filtered_kwargs = {
        k: v for k, v in kwargs.items() if k in allowed_kwargs
    }

    try:
        result = inventory_optimization_pipeline.analyze_file_and_run_pipeline(
            file_path,
            gemini_copy,
            models_dir=models_dir,
            **filtered_kwargs
        )
        return {
            "success": True,
            "runner": "inventory_optimization_pipeline",
            "result": result,
        }
    except Exception as e:
        logger.exception("Inventory optimization pipeline failed: %s", e)
        return {
            "success": False,
            "error": "Inventory optimization pipeline failed",
            "exception": str(e),
        }



def _call_supplier_runner(
    file_path: str,
    gemini_response: Dict[str, Any],
    models_dir: str = "models",
    **kwargs
) -> Dict[str, Any]:
    """
    Trigger the Supplier Risk & Routing Optimization pipeline.
    Filters incompatible kwargs coming from generic dispatch.
    """
    try:
        from services import supplier_risk_and_routing_pipeline
    except Exception as e:
        logger.exception("Failed to import supplier_risk_and_routing_pipeline: %s", e)
        return {
            "success": False,
            "error": "supplier_risk_and_routing_pipeline not available",
            "exception": str(e),
        }

    # Defensive copy and force model_type so the pipeline runs in strict mode
    gemini_copy = dict(gemini_response) if gemini_response else {}
    analysis = gemini_copy.setdefault("analysis", {})
    analysis["model_type"] = "Logistics & Supplier Risk Model"

    # Allowed kwargs for supplier pipeline
    allowed_kwargs = {
        "supplier_col_hint",
        "demand_col_hint",
        "date_col_hint",
        "default_unit_cost",
        "default_capacity",
        "default_lead_time",
        "use_milp",
        "risk_weight",
        "vehicle_capacity",
        "time_horizon_limit",
        "milp_timeout_seconds",
        "models_dir",
    }

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_kwargs}

    try:
        result = supplier_risk_and_routing_pipeline.analyze_file_and_run_pipeline(
            file_path,
            gemini_copy,
            models_dir=models_dir,
            **filtered_kwargs
        )
        return {
            "success": True,
            "runner": "supplier_risk_and_routing_pipeline",
            "result": result,
        }
    except Exception as e:
        logger.exception("Supplier risk & routing pipeline failed: %s", e)
        return {
            "success": False,
            "error": "supplier_risk_and_routing_pipeline failed",
            "exception": str(e),
        }


def _call_customer_segmentation_runner(
    file_path: str,
    gemini_response: Dict[str, Any],
    models_dir: str = "models",
    **kwargs
) -> Dict[str, Any]:
    """
    Trigger the Customer Segmentation & Modeling pipeline.
    Filters incompatible kwargs coming from generic dispatch.
    """
    try:
        from services import customer_segmentation_pipeline
    except Exception as e:
        logger.exception("Failed to import customer_segmentation_pipeline: %s", e)
        return {
            "success": False,
            "error": "customer_segmentation_pipeline not available",
            "exception": str(e),
        }

    # Defensive copy and force model_type so the pipeline runs in strict mode
    gemini_copy = dict(gemini_response) if gemini_response else {}
    analysis = gemini_copy.setdefault("analysis", {})
    # Must match the strict model_type expected by the pipeline
    analysis["model_type"] = "Customer Segmentation & Modeling"

    # Allowed kwargs for customer segmentation pipeline (keep in sync with pipeline signature)
    allowed_kwargs = {
        "customer_col_hint",
        "date_col_hint",
        "monetary_col_hint",
        "segmentation_methods",
        "n_segments",
        "run_predictive",
        "predictive_method",
        "churn_label_col_hint",
        "models_dir",
        "random_state",
    }

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_kwargs}

    try:
        result = customer_segmentation_pipeline.analyze_file_and_run_pipeline(
            file_path,
            gemini_copy,
            models_dir=models_dir,
            **filtered_kwargs
        )
        return {
            "success": True,
            "runner": "customer_segmentation_pipeline",
            "result": result,
        }
    except Exception as e:
        logger.exception("Customer segmentation pipeline failed: %s", e)
        return {
            "success": False,
            "error": "customer_segmentation_pipeline failed",
            "exception": str(e),
        }


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
            mkey = model_name.strip().lower()
            if mkey == "sales, demand & financial forecasting model":
                runners_out[model_name] = _call_sales_forecasting_runner(file_path, gemini_response, models_dir=models_dir, **kwargs)
            elif mkey == "marketing roi & attribution model":
                runners_out[model_name] = _call_marketing_mmm_runner(file_path, gemini_response, models_dir=models_dir, **kwargs)
            elif mkey == "inventory & replenishment optimization model":
                runners_out[model_name] = _call_inventory_runner(file_path, gemini_response, models_dir=models_dir, **kwargs)
            elif mkey in ("supplier risk & routing optimization model", "logistics & supplier risk model"):
                # Dispatch to the supplier risk & routing pipeline (supporting both canonical and legacy names)
                runners_out[model_name] = _call_supplier_runner(file_path, gemini_response, models_dir=models_dir, **kwargs)
            elif mkey in ("customer segmentation & modeling", "customer segmentation pipeline", "customer segmentation"):
                # Dispatch to the customer segmentation pipeline (supports canonical and shorthand names)
                runners_out[model_name] = _call_customer_segmentation_runner(file_path, gemini_response, models_dir=models_dir, **kwargs)
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


def analyze_and_trigger(file_path: str, models_dir: str = "models", **kwargs) -> Dict[str, Any]:
    """
    1) Analyze CSV headers with Gemini (classify to enterprise domain and suggest per-model targets)
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--models-dir", required=False, default="models", help="Directory to save models")
    args = parser.parse_args()

    out = analyze_and_trigger(args.csv, models_dir=args.models_dir)
    print(json.dumps(out, indent=2, default=str))
