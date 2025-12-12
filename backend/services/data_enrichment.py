# services/data_enrichment.py
import os
import json
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# NOTE: pandas is optional but recommended
try:
    import pandas as pd
except Exception:
    pd = None
    logger.warning("pandas not available: summarization will be limited")

# Basic category keys mapped to your MODEL_RECOMMENDATIONS labels
CATEGORY_KEYWORDS = {
    "Sales, Demand & Financial Forecasting Model": [
        "sales", "revenue", "profit", "price", "quantity", "order", "amount", "total"
    ],
    "Pricing & Revenue Optimization Model": [
        "price", "discount", "margin", "revenue", "rate", "charge", "pricing"
    ],
    "Marketing ROI & Attribution Model": [
        "campaign", "channel", "click", "impression", "ctr", "cost", "cpa", "cac", "utm", "source"
    ],
    "Customer Segmentation & Modeling": [
        "customer", "segment", "age", "gender", "region", "cluster", "persona"
    ],
    "Customer Value & Retention Model": [
        "churn", "retention", "lifetime", "ltv", "renewal", "cancel", "subscriber"
    ],
    "Sentiment & Intent NLP Model": [
        "review", "comment", "feedback", "text", "sentiment", "message", "tweet"
    ],
    "Inventory & Replenishment Optimization Model": [
        "inventory", "stock", "sku", "warehouse", "on_hand", "reorder", "lead_time"
    ],
    "Logistics & Supplier Risk Model": [
        "supplier", "shipment", "carrier", "delivery", "eta", "transit", "lead_time", "delay"
    ],
}

def fallback_classify(columns: List[str]) -> str:
    """
    Choose the best category by matching header keywords.
    Returns the best-matching category or 'Auto-detect (Best-fit)' if unsure.
    """
    if not columns:
        return "Auto-detect (Best-fit)"

    cols_lower = [c.lower() for c in columns]
    scores = {cat: 0 for cat in CATEGORY_KEYWORDS}
    for idx, c in enumerate(cols_lower):
        for cat, kws in CATEGORY_KEYWORDS.items():
            for kw in kws:
                if kw in c or c in kw:
                    scores[cat] += 1
    # Choose highest score
    best = max(scores.items(), key=lambda kv: kv[1])
    if best[1] == 0:
        return "Auto-detect (Best-fit)"
    return best[0]

def combine_classification(gemini_analysis: Optional[Dict], columns: List[str]) -> str:
    """
    Determine category: prefer validated Gemini output; if missing/fails, fallback to keyword classifier.
    """
    try:
        if isinstance(gemini_analysis, dict) and gemini_analysis.get("success") and gemini_analysis.get("analysis"):
            m = gemini_analysis.get("analysis")
            model_type = m.get("model_type")
            if model_type:
                # keep as-is if in known categories
                if model_type in CATEGORY_KEYWORDS or model_type == "Auto-detect (Best-fit)":
                    return model_type
    except Exception:
        logger.debug("Gemini analysis unusable, using fallback")

    # fallback
    return fallback_classify(columns)


def summarize_csv(file_path: str, sample_rows: int = 3) -> Dict:
    """
    Produce a summary dict: columns, dtypes, row_count (if pandas), sample rows,
    candidate time columns, numeric columns, simple aggregations.
    """
    summary = {
        "columns": [],
        "dtypes": {},
        "row_count": None,
        "sample_rows": [],
        "time_columns": [],
        "numeric_columns": [],
        "aggregations": {},  # per numeric column: mean/min/max/sum
    }

    # Try to use pandas for robust parsing
    if pd is None:
        # fallback to reading first line as header
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                header = fh.readline().strip().split(",")
                cols = [c.strip() for c in header if c.strip()]
                summary["columns"] = cols
        except Exception as e:
            logger.exception("Could not read CSV header: %s", e)
        return summary

    try:
        df = pd.read_csv(file_path, nrows=100000)  # try to load - memory permitting
        summary["row_count"] = int(len(df))
        summary["columns"] = list(df.columns.astype(str))
        # dtypes
        for c in df.columns:
            summary["dtypes"][str(c)] = str(df[c].dtype)
            if pd.api.types.is_numeric_dtype(df[c].dtype):
                summary["numeric_columns"].append(str(c))
            # naive time detection
            if "date" in str(c).lower() or "time" in str(c).lower() or pd.api.types.is_datetime64_any_dtype(df[c].dtype):
                summary["time_columns"].append(str(c))
        # sample rows
        sample_df = df.head(sample_rows)
        summary["sample_rows"] = sample_df.to_dict(orient="records")
        # aggregations
        for c in summary["numeric_columns"]:
            try:
                col = pd.to_numeric(df[c], errors="coerce")
                summary["aggregations"][c] = {
                    "count": int(col.count()),
                    "mean": float(col.mean()) if col.count() else None,
                    "min": float(col.min()) if col.count() else None,
                    "max": float(col.max()) if col.count() else None,
                    "sum": float(col.sum()) if col.count() else None,
                }
            except Exception:
                continue

        # if time + numeric present, compute simple trend for the top numeric column (daily/weekly)
        if summary["time_columns"] and summary["numeric_columns"]:
            tcol = summary["time_columns"][0]
            ncol = summary["numeric_columns"][0]
            try:
                df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
                df[ncol] = pd.to_numeric(df[ncol], errors="coerce")
                ts = df.dropna(subset=[tcol, ncol])
                if not ts.empty:
                    # aggregate by date
                    ts_agg = ts.set_index(tcol).resample("D")[ncol].sum().fillna(0)
                    # keep last 30 days
                    last = ts_agg.tail(30)
                    summary["trend_last_30_days"] = last.reset_index().rename(columns={tcol: "date", ncol: "value"}).to_dict(orient="records")
            except Exception:
                logger.debug("Failed to compute trend", exc_info=True)
    except Exception as e:
        logger.exception("Failed to summarize CSV %s: %s", file_path, e)

    return summary


def write_upload_metadata(upload_folder: str, filename: str, metadata: Dict):
    """
    Persist metadata for an uploaded file next to the uploads folder.
    """
    try:
        meta_dir = os.path.join(upload_folder, "metadata")
        os.makedirs(meta_dir, exist_ok=True)
        meta_path = os.path.join(meta_dir, f"{filename}.meta.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, default=str, indent=2)
        return meta_path
    except Exception as e:
        logger.exception("Failed to write metadata: %s", e)
        return None
