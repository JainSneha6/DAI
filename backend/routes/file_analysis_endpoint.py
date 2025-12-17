# Complete fix for file_analysis_endpoint.py
# This version properly serializes ALL data types

from flask import Blueprint, jsonify, request, current_app
import os
import json
import logging
from services.data_enrichment import summarize_csv
from services.gemini_analyzer import analyze_file_with_gemini
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

analysis_bp = Blueprint("analysis", __name__)


def _safe_serialize(obj):
    """Convert pandas/numpy types to JSON-serializable Python types."""
    if isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_serialize(v) for v in obj]
    if pd.isna(obj):
        return None
    return obj


def _extract_categorical_summary(df):
    """
    Extract categorical summary with proper serialization.
    Returns a dict with proper data types, not strings.
    """
    categorical_summary = {}
    
    # Identify categorical columns
    categorical_cols = []
    for col in df.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df[col].dtype)
        is_object = pd.api.types.is_object_dtype(df[col].dtype) or pd.api.types.is_bool_dtype(df[col].dtype)
        
        try:
            nunique = int(df[col].nunique(dropna=True))
        except Exception:
            nunique = None
        
        if is_object or (not is_numeric) or (is_numeric and nunique is not None and nunique <= 50):
            categorical_cols.append(col)
    
    # Process each categorical column
    for col in categorical_cols:
        try:
            # Get value counts
            ser = df[col].fillna("__NULL__").astype(str)
            vc = ser.value_counts(dropna=False)
            total = int(vc.sum())
            
            # Top values (limit to top 10)
            top_values = []
            for val, cnt in vc.head(10).items():
                if val == "__NULL__":
                    display_val = None
                else:
                    display_val = val
                top_values.append({
                    "value": display_val,
                    "count": int(cnt),
                    "pct": round(float(cnt) / total, 4) if total else 0.0
                })
            
            unique_count = int(vc.shape[0])
            missing_count = int((df[col].isna()).sum())
            
            categorical_summary[col] = {
                "unique_count": unique_count,
                "missing_count": missing_count,
                "top_values": top_values
            }
        except Exception as e:
            logger.exception(f"Failed categorical summary for column {col}: {e}")
    
    return categorical_summary


def _extract_numerical_summary(df):
    """
    Extract numerical summary with proper serialization.
    """
    numerical_summary = {}
    
    # Identify numeric columns
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            numeric_cols.append(col)
    
    # Process each numeric column
    for col in numeric_cols:
        try:
            col_data = pd.to_numeric(df[col], errors="coerce")
            count = int(col_data.count())
            missing = int(len(col_data) - count)
            
            if count > 0:
                mean = float(col_data.mean())
                _min = float(col_data.min())
                _max = float(col_data.max())
                _sum = float(col_data.sum())
                std = float(col_data.std()) if count > 1 else 0.0
            else:
                mean = None
                _min = None
                _max = None
                _sum = None
                std = None
            
            numerical_summary[col] = {
                "count": count,
                "missing_count": missing,
                "mean": mean,
                "min": _min,
                "max": _max,
                "sum": _sum,
                "std": std
            }
        except Exception as e:
            logger.exception(f"Failed numerical summary for column {col}: {e}")
    
    return numerical_summary


@analysis_bp.route("/api/files/<filename>/analysis", methods=["GET"])
def get_file_analysis(filename):
    """
    Get comprehensive analysis for a specific file.
    ALL data is properly serialized as native JSON types.
    """
    try:
        upload_folder = current_app.config.get("UPLOAD_FOLDER", os.path.join(os.getcwd(), "uploads"))
        file_path = os.path.join(upload_folder, filename)

        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": "File not found"}), 404

        # Load dataframe
        df = pd.read_csv(file_path)

        # Get basic info
        columns = [str(col) for col in df.columns]
        dtypes = {str(col): str(df[col].dtype) for col in df.columns}

        # Get sample rows (properly formatted)
        sample_rows = []
        try:
            sample_df = df.head(5)
            for _, row in sample_df.iterrows():
                row_dict = {}
                for col in sample_df.columns:
                    val = row[col]
                    if pd.isna(val):
                        row_dict[str(col)] = None
                    elif isinstance(val, (np.integer, np.int_)):
                        row_dict[str(col)] = int(val)
                    elif isinstance(val, (np.floating, np.float64)):
                        if np.isnan(val) or np.isinf(val):
                            row_dict[str(col)] = None
                        else:
                            row_dict[str(col)] = float(val)
                    elif isinstance(val, (pd.Timestamp, pd.Timedelta)):
                        row_dict[str(col)] = str(val)
                    else:
                        row_dict[str(col)] = str(val)
                sample_rows.append(row_dict)
        except Exception as e:
            logger.exception("Failed to get sample rows")
            sample_rows = []

        # Get numerical and categorical summaries (properly formatted)
        numerical_summary = _extract_numerical_summary(df)
        categorical_summary = _extract_categorical_summary(df)
        
        # Identify column types
        numeric_columns = list(numerical_summary.keys())
        categorical_columns = list(categorical_summary.keys())
        
        # Identify time columns
        time_columns = []
        for col in df.columns:
            col_str = str(col)
            if ("date" in col_str.lower()) or ("time" in col_str.lower()):
                time_columns.append(col_str)

        # Generate correlation matrix for numeric columns
        correlation_data = None
        if len(numeric_columns) > 1:
            try:
                numeric_df = df[numeric_columns].select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    corr_matrix = numeric_df.corr()
                    correlation_data = {
                        "columns": [str(c) for c in corr_matrix.columns],
                        "values": [[float(v) if not (np.isnan(v) or np.isinf(v)) else None 
                                   for v in row] for row in corr_matrix.values]
                    }
            except Exception as e:
                logger.exception("Failed to compute correlation matrix")

        # Generate distribution data for charts (histograms)
        distribution_data = {}
        for col in numeric_columns[:10]:
            try:
                col_data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(col_data) > 0:
                    hist, bin_edges = np.histogram(col_data, bins=20)
                    distribution_data[col] = {
                        "counts": [int(x) for x in hist],
                        "bins": [float(x) for x in bin_edges],
                        "stats": {
                            "mean": float(col_data.mean()),
                            "median": float(col_data.median()),
                            "std": float(col_data.std()),
                            "min": float(col_data.min()),
                            "max": float(col_data.max())
                        }
                    }
            except Exception as e:
                logger.exception(f"Failed to generate distribution for {col}")

        # Generate time series data if time columns exist
        time_series_data = {}
        if time_columns and numeric_columns:
            for time_col in time_columns[:2]:
                try:
                    df_copy = df.copy()
                    df_copy[time_col] = pd.to_datetime(df_copy[time_col], errors='coerce')
                    df_copy = df_copy.dropna(subset=[time_col])
                    df_copy = df_copy.sort_values(time_col)

                    if len(df_copy) > 200:
                        df_copy = df_copy.iloc[::len(df_copy)//200]

                    time_series_data[time_col] = {
                        "dates": [str(d) for d in df_copy[time_col].tolist()],
                        "series": {}
                    }

                    for num_col in numeric_columns[:5]:
                        if num_col in df_copy.columns:
                            values = []
                            for v in df_copy[num_col].tolist():
                                if pd.isna(v):
                                    values.append(None)
                                elif isinstance(v, (np.floating, float)):
                                    values.append(None if (np.isnan(v) or np.isinf(v)) else float(v))
                                elif isinstance(v, (np.integer, int)):
                                    values.append(int(v))
                                else:
                                    values.append(float(v))
                            time_series_data[time_col]["series"][num_col] = values
                except Exception as e:
                    logger.exception(f"Failed to generate time series for {time_col}")

        # Generate categorical frequency data for charts
        categorical_frequency = {}
        for col in categorical_columns[:10]:
            try:
                value_counts = df[col].value_counts().head(15)
                categorical_frequency[col] = {
                    "labels": [str(x) if not pd.isna(x) else None for x in value_counts.index.tolist()],
                    "values": [int(x) for x in value_counts.values.tolist()]
                }
            except Exception as e:
                logger.exception(f"Failed to generate frequency for {col}")

        # Box plot data for numeric columns
        box_plot_data = {}
        for col in numeric_columns[:10]:
            try:
                col_data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(col_data) > 0:
                    q1 = float(col_data.quantile(0.25))
                    q2 = float(col_data.quantile(0.5))
                    q3 = float(col_data.quantile(0.75))
                    iqr = q3 - q1
                    whisker_low = float(col_data[col_data >= q1 - 1.5 * iqr].min())
                    whisker_high = float(col_data[col_data <= q3 + 1.5 * iqr].max())

                    box_plot_data[col] = {
                        "min": whisker_low,
                        "q1": q1,
                        "median": q2,
                        "q3": q3,
                        "max": whisker_high,
                        "outliers": [float(x) for x in col_data[(col_data < whisker_low) | (col_data > whisker_high)].tolist()[:50]]
                    }
            except Exception as e:
                logger.exception(f"Failed to generate box plot for {col}")

        # Grouped analysis
        grouped_analysis = {}
        MAX_CARDINALITY = 500
        MAX_GROUPS = 50
        
        for cat_col in categorical_columns:
            try:
                nunique = int(df[cat_col].nunique(dropna=True))
            except Exception:
                nunique = None
            
            if not numeric_columns or (nunique is not None and nunique > MAX_CARDINALITY):
                continue
            
            cat_group = {}
            for num_col in numeric_columns:
                try:
                    gb = (
                        df.groupby(cat_col, dropna=False)[num_col]
                        .agg(["count", "mean", "sum", "min", "max"])
                        .reset_index()
                    )
                    
                    if "count" in gb.columns:
                        gb = gb.sort_values(by="count", ascending=False)
                    
                    gb = gb.head(MAX_GROUPS)
                    
                    groups_list = []
                    for _, row in gb.iterrows():
                        key = row[cat_col]
                        key_val = None if pd.isna(key) else str(key)
                        
                        groups_list.append({
                            "category_value": key_val,
                            "count": int(row["count"]) if not pd.isna(row["count"]) else 0,
                            "mean": float(row["mean"]) if not pd.isna(row["mean"]) else None,
                            "sum": float(row["sum"]) if not pd.isna(row["sum"]) else None,
                            "min": float(row["min"]) if not pd.isna(row["min"]) else None,
                            "max": float(row["max"]) if not pd.isna(row["max"]) else None,
                        })
                    
                    cat_group[num_col] = groups_list
                except Exception as e:
                    logger.exception(f"Failed grouped analysis for cat {cat_col}, num {num_col}")
            
            if cat_group:
                grouped_analysis[cat_col] = cat_group

        # Compile response
        response = {
            "success": True,
            "filename": filename,
            "basic_info": {
                "row_count": int(len(df)),
                "column_count": len(columns),
                "numeric_columns": len(numeric_columns),
                "categorical_columns": len(categorical_columns),
                "time_columns": len(time_columns),
                "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
            },
            "columns": columns,
            "dtypes": dtypes,
            "sample_rows": sample_rows,
            "numerical_summary": numerical_summary,
            "categorical_summary": categorical_summary,
            "grouped_analysis": grouped_analysis,
            "correlation_matrix": correlation_data,
            "distribution_data": distribution_data,
            "time_series_data": time_series_data,
            "categorical_frequency": categorical_frequency,
            "box_plot_data": box_plot_data
        }

        return jsonify(response)

    except Exception as e:
        logger.exception(f"Failed to analyze file {filename}")
        return jsonify({"success": False, "error": str(e)}), 500


@analysis_bp.route("/api/files/<filename>/column/<column_name>", methods=["GET"])
def get_column_details(filename, column_name):
    """Get detailed analysis for a specific column."""
    try:
        upload_folder = current_app.config.get("UPLOAD_FOLDER", os.path.join(os.getcwd(), "uploads"))
        file_path = os.path.join(upload_folder, filename)

        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": "File not found"}), 404

        df = pd.read_csv(file_path)

        if column_name not in df.columns:
            return jsonify({"success": False, "error": "Column not found"}), 404

        col_data = df[column_name]
        is_numeric = pd.api.types.is_numeric_dtype(col_data)

        response = {
            "success": True,
            "column_name": column_name,
            "dtype": str(col_data.dtype),
            "is_numeric": is_numeric,
            "total_count": int(len(col_data)),
            "missing_count": int(col_data.isna().sum()),
            "missing_percentage": float(col_data.isna().sum() / len(col_data) * 100)
        }

        if is_numeric:
            numeric_col = pd.to_numeric(col_data, errors='coerce')
            response["statistics"] = {
                "count": int(numeric_col.count()),
                "mean": float(numeric_col.mean()) if not pd.isna(numeric_col.mean()) else None,
                "std": float(numeric_col.std()) if not pd.isna(numeric_col.std()) else None,
                "min": float(numeric_col.min()) if not pd.isna(numeric_col.min()) else None,
                "25%": float(numeric_col.quantile(0.25)) if not pd.isna(numeric_col.quantile(0.25)) else None,
                "50%": float(numeric_col.quantile(0.50)) if not pd.isna(numeric_col.quantile(0.50)) else None,
                "75%": float(numeric_col.quantile(0.75)) if not pd.isna(numeric_col.quantile(0.75)) else None,
                "max": float(numeric_col.max()) if not pd.isna(numeric_col.max()) else None,
                "sum": float(numeric_col.sum()) if not pd.isna(numeric_col.sum()) else None,
                "variance": float(numeric_col.var()) if not pd.isna(numeric_col.var()) else None
            }

            hist, bins = np.histogram(numeric_col.dropna(), bins=30)
            response["histogram"] = {
                "counts": [int(x) for x in hist],
                "bins": [float(x) for x in bins]
            }
        else:
            value_counts = col_data.value_counts()
            response["unique_count"] = int(col_data.nunique())
            response["top_values"] = {
                "labels": [str(x) if not pd.isna(x) else None for x in value_counts.head(20).index.tolist()],
                "counts": [int(x) for x in value_counts.head(20).values.tolist()]
            }

        return jsonify(response)

    except Exception as e:
        logger.exception(f"Failed to get column details for {column_name} in {filename}")
        return jsonify({"success": False, "error": str(e)}), 500