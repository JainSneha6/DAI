# services/data_enrichment.py
import os
import json
import math
from typing import List, Dict, Optional, Any
import logging
from collections import Counter

logger = logging.getLogger(__name__)

# NOTE: pandas is optional but strongly recommended
try:
    import pandas as pd
    import numpy as np
except Exception:
    pd = None
    np = None
    logger.warning("pandas/numpy not available: summarization will be limited")


def _to_serializable(value: Any):
    """
    Convert pandas / numpy types (and others) to JSON-serializable Python types.
    """
    try:
        # pandas/numpy scalars
        if pd is not None and isinstance(value, (pd.Timestamp, pd.Timedelta)):
            try:
                return value.isoformat()
            except Exception:
                return str(value)
        if np is not None and isinstance(value, (np.integer, np.int_)):
            return int(value)
        if np is not None and isinstance(value, (np.floating, np.float_)):
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if np is not None and isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, (int, float, str, bool)):
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    return None
            return value
        if value is None:
            return None
        # for numpy arrays or pandas arrays -> list
        if np is not None and isinstance(value, (np.ndarray,)):
            return [_to_serializable(v) for v in value.tolist()]
        # fallback
        return str(value)
    except Exception:
        return str(value)


def _sanitize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize all values in a dict for JSON encoding."""
    out = {}
    for k, v in (rec or {}).items():
        if isinstance(v, dict):
            out[k] = _sanitize_record(v)
        elif isinstance(v, list):
            out[k] = [_to_serializable(x) for x in v]
        else:
            out[k] = _to_serializable(v)
    return out


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


def summarize_csv(file_path: str, sample_rows: int = 3, max_rows: Optional[int] = 100000) -> Dict:
    """
    Produce a comprehensive summary metadata for a CSV file.

    Output keys include:
    - columns, dtypes, row_count, sample_rows, time_columns, numeric_columns
    - categorical_columns
    - categorical_summary: per categorical column -> { unique_count, top_values: [{value, count, pct}], missing_count }
    - numerical_summary: per numeric column -> { count, mean, min, max, sum, std, missing_count }
    - grouped_analysis: for each categorical column (if cardinality is reasonable) -> for each numeric column a list of group-aggregations
      Each group-aggregation = { category_value, count, mean, sum, min, max }
    - aggregations: same as numerical_summary for convenience
    """
    summary = {
        "columns": [],
        "dtypes": {},
        "row_count": None,
        "sample_rows": [],
        "time_columns": [],
        "numeric_columns": [],
        "categorical_columns": [],
        "categorical_summary": {},  # per cat col: { unique_count, missing_count, top_values: [{value,count,pct}] }
        "numerical_summary": {},    # per num col: { count, mean, min, max, sum, std, missing_count }
        "grouped_analysis": {},     # per cat col: { numeric_col: [ {category_value, count, mean, sum, min, max}, ... ] }
        "aggregations": {},
    }

    # If pandas not available -- fallback: minimal header and sample
    if pd is None:
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                header = fh.readline().strip().split(",")
                cols = [c.strip() for c in header if c.strip()]
                summary["columns"] = cols
                # sample rows
                rows = []
                for i, line in enumerate(fh):
                    if i >= sample_rows:
                        break
                    rows.append([v.strip() for v in line.strip().split(",")])
                summary["sample_rows"] = rows
        except Exception:
            logger.exception("Could not read CSV header (no pandas)")
        return summary

    # With pandas available: robust profiling
    try:
        # load up to max_rows rows (None = full)
        if max_rows:
            df = pd.read_csv(file_path, nrows=max_rows)
        else:
            df = pd.read_csv(file_path)

        # core info
        summary["row_count"] = int(len(df))
        summary["columns"] = list(df.columns.astype(str))

        # sample rows (safe serialization)
        try:
            sample_df = df.head(sample_rows)
            summary["sample_rows"] = sample_df.replace({np.nan: None}).to_dict(orient="records")
        except Exception:
            summary["sample_rows"] = []

        # dtypes and classification of numeric / time / categorical
        for c in df.columns:
            cname = str(c)
            dtype = str(df[c].dtype)
            summary["dtypes"][cname] = dtype

            # numeric
            if pd.api.types.is_numeric_dtype(df[c].dtype):
                summary["numeric_columns"].append(cname)

            # datetime-like
            if ("date" in cname.lower()) or ("time" in cname.lower()) or pd.api.types.is_datetime64_any_dtype(df[c].dtype):
                summary["time_columns"].append(cname)

        # identify categorical columns: object, bool, or numeric with low cardinality
        for c in df.columns:
            cname = str(c)
            is_numeric = pd.api.types.is_numeric_dtype(df[c].dtype)
            is_object = pd.api.types.is_object_dtype(df[c].dtype) or pd.api.types.is_bool_dtype(df[c].dtype) or pd.api.types.is_categorical_dtype(df[c].dtype)

            # treat numeric with low unique values as categorical (e.g., codes)
            try:
                nunique = int(df[c].nunique(dropna=True))
            except Exception:
                nunique = None

            if is_object or (not is_numeric) or (is_numeric and nunique is not None and nunique <= 50):
                summary["categorical_columns"].append(cname)

        # Numerical summary
        for c in summary["numeric_columns"]:
            try:
                col = pd.to_numeric(df[c], errors="coerce")
                count = int(col.count())
                missing = int(len(col) - count)
                if count:
                    mean = float(col.mean())
                    _min = float(col.min())
                    _max = float(col.max())
                    _sum = float(col.sum())
                    std = float(col.std()) if count > 1 else 0.0
                else:
                    mean = None
                    _min = None
                    _max = None
                    _sum = None
                    std = None

                summary["numerical_summary"][c] = {
                    "count": count,
                    "missing_count": missing,
                    "mean": _to_serializable(mean),
                    "min": _to_serializable(_min),
                    "max": _to_serializable(_max),
                    "sum": _to_serializable(_sum),
                    "std": _to_serializable(std),
                }
                # also aggregate key fields for backwards compatibility
                summary["aggregations"][c] = summary["numerical_summary"][c]
            except Exception:
                logger.exception("Failed numerical summary for column %s", c)

        # Categorical summary (unique counts and top values)
        for c in summary["categorical_columns"]:
            try:
                # treat values as strings for counting but preserve NaNs
                ser = df[c].fillna("__NULL__").astype(str)
                vc = ser.value_counts(dropna=False)
                total = int(vc.sum())
                # top values (limit to top 10)
                top_n = 10
                top = []
                for val, cnt in vc.head(top_n).items():
                    if val == "__NULL__":
                        display_val = None
                    else:
                        display_val = val
                    top.append({
                        "value": _to_serializable(display_val),
                        "count": int(cnt),
                        "pct": round(int(cnt) / total, 4) if total else 0.0
                    })
                unique_count = int(vc.shape[0])
                missing_count = int((df[c].isna()).sum())
                summary["categorical_summary"][c] = {
                    "unique_count": unique_count,
                    "missing_count": missing_count,
                    "top_values": top
                }
            except Exception:
                logger.exception("Failed categorical summary for column %s", c)

        # Grouped analysis: for each categorical column (only if cardinality reasonable),
        # compute aggregated numeric stats per category for each numeric column.
        grouped = {}
        # threshold config
        MAX_CARDINALITY_FOR_GROUPING = 500   # skip grouping for columns with more than this many unique values
        MAX_GROUPS_RETURN = 50              # return only top N groups by count to avoid huge payloads

        for cat_col in summary["categorical_columns"]:
            try:
                nunique = int(df[cat_col].nunique(dropna=True))
            except Exception:
                nunique = None

            # skip grouping when cardinality too high (or when no numeric columns)
            if not summary["numeric_columns"] or (nunique is not None and nunique > MAX_CARDINALITY_FOR_GROUPING):
                continue

            # build grouped aggregations for each numeric column
            cat_group = {}
            for num_col in summary["numeric_columns"]:
                try:
                    # compute groupby aggregations; convert to records; limit to most frequent groups
                    gb = (
                        df.groupby(cat_col, dropna=False)[num_col]
                        .agg(["count", "mean", "sum", "min", "max"])
                        .reset_index()
                    )
                    # replace NaN keys with None for readability
                    # sort groups by count desc
                    if "count" in gb.columns:
                        gb = gb.sort_values(by="count", ascending=False)
                    # limit groups
                    gb = gb.head(MAX_GROUPS_RETURN)
                    groups_list = []
                    for _, row in gb.iterrows():
                        key = row[cat_col]
                        # normalize key (NaN -> None)
                        if pd.isna(key):
                            key_val = None
                        else:
                            key_val = _to_serializable(key)
                        groups_list.append({
                            "category_value": key_val,
                            "count": int(row["count"]) if not pd.isna(row["count"]) else 0,
                            "mean": _to_serializable(row["mean"]),
                            "sum": _to_serializable(row["sum"]),
                            "min": _to_serializable(row["min"]),
                            "max": _to_serializable(row["max"]),
                        })
                    cat_group[num_col] = groups_list
                except Exception:
                    logger.exception("Failed grouped analysis for cat %s, num %s", cat_col, num_col)
            if cat_group:
                grouped[cat_col] = cat_group

        summary["grouped_analysis"] = grouped

    except Exception as e:
        logger.exception("Failed to summarize CSV %s: %s", file_path, e)

    # Final sanitization - ensure all values JSON serializable
    summary = _sanitize_record(summary)
    return summary


def write_upload_metadata(upload_folder: str, filename: str, metadata: Dict):
    """
    Persist file metadata as JSON (ensures folder exists).
    """
    try:
        meta_dir = os.path.join(upload_folder, "metadata")
        os.makedirs(meta_dir, exist_ok=True)
        meta_path = os.path.join(meta_dir, f"{filename}.meta.json")
        # sanitize before writing
        safe = _sanitize_record(metadata)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(safe, fh, ensure_ascii=False, indent=2)
        return meta_path
    except Exception:
        logger.exception("Failed to write metadata")
        return None
