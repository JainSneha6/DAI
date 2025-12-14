# services/data_enrichment.py
import os
import json
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# NOTE: pandas is optional but recommended
try:
    import pandas as pd
except Exception:
    pd = None
    logger.warning("pandas not available: summarization will be limited")


def combine_classification(
    gemini_analysis: Optional[Dict],
    columns: List[str]
) -> Optional[Dict]:
    """
    Use Gemini analyzer as the single source of truth for enterprise data classification.

    Returns:
    {
        "data_domain": str,
        "confidence": float,
        "key_columns": List[str],
        "explanation": str
    }
    or None if unavailable
    """

    if not isinstance(gemini_analysis, dict):
        return None

    if not gemini_analysis.get("success"):
        return None

    analysis = gemini_analysis.get("analysis")
    if not isinstance(analysis, dict):
        return None

    data_domain = analysis.get("data_domain") or analysis.get("category")
    if not data_domain:
        return None

    return {
        "data_domain": data_domain,
        "confidence": analysis.get("confidence"),
        "key_columns": analysis.get("key_columns", []),
        "explanation": analysis.get("explanation", "")
    }


def summarize_csv(file_path: str, sample_rows: int = 3) -> Dict:
    """
    Produce a structural summary of a CSV.
    """
    summary = {
        "columns": [],
        "dtypes": {},
        "row_count": None,
        "sample_rows": [],
        "time_columns": [],
        "numeric_columns": [],
        "aggregations": {},
    }

    if pd is None:
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                header = fh.readline().strip().split(",")
                summary["columns"] = [c.strip() for c in header if c.strip()]
        except Exception:
            logger.exception("Failed to read CSV header")
        return summary

    try:
        df = pd.read_csv(file_path, nrows=100000)
        summary["row_count"] = int(len(df))
        summary["columns"] = list(df.columns.astype(str))

        for c in df.columns:
            cname = str(c)
            summary["dtypes"][cname] = str(df[c].dtype)

            if pd.api.types.is_numeric_dtype(df[c]):
                summary["numeric_columns"].append(cname)

            if (
                "date" in cname.lower()
                or "time" in cname.lower()
                or pd.api.types.is_datetime64_any_dtype(df[c])
            ):
                summary["time_columns"].append(cname)

        summary["sample_rows"] = df.head(sample_rows).to_dict(orient="records")

        for c in summary["numeric_columns"]:
            col = pd.to_numeric(df[c], errors="coerce")
            if col.count():
                summary["aggregations"][c] = {
                    "count": int(col.count()),
                    "mean": float(col.mean()),
                    "min": float(col.min()),
                    "max": float(col.max()),
                    "sum": float(col.sum()),
                }

    except Exception:
        logger.exception("Failed to summarize CSV")

    return summary


def write_upload_metadata(upload_folder: str, filename: str, metadata: Dict):
    try:
        meta_dir = os.path.join(upload_folder, "metadata")
        os.makedirs(meta_dir, exist_ok=True)
        meta_path = os.path.join(meta_dir, f"{filename}.meta.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, ensure_ascii=False, indent=2)
        return meta_path
    except Exception:
        logger.exception("Failed to write metadata")
        return None
