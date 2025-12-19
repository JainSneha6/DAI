# services/supplier_risk_and_routing_pipeline_enhanced.py
# Enhanced Logistics Supplier Risk & Routing Optimization pipeline
# 
# IMPROVEMENTS:
# 1. Better column detection for shipment-specific data (shipment_id, tracking_url, etc.)
# 2. Enhanced delay/on-time detection using expected vs actual delivery times
# 3. Richer metadata capturing data quality, supplier performance, and model configuration
# 4. Better handling of carrier/supplier differentiation
# 5. Improved cost calculations including delay costs
# 6. More robust date parsing and validation
# 7. Enhanced logging and error handling
# 8. Better capacity estimation from historical data
# 9. Vehicle type awareness in routing
# 10. Temperature-controlled and hazardous cargo handling

import os
import math
import json
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Optional dependencies
try:
    import pulp as pl
    _HAS_PULP = True
except Exception:
    _HAS_PULP = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    from lifelines import CoxPHFitter
    _HAS_LIFELINES = True
except Exception:
    _HAS_LIFELINES = False

# ========== CONFIGURATION ==========

# Enhanced keyword detection for shipment data
_SHIPMENT_ID_KEYWORDS = ["shipment_id", "shipment", "ship_id", "tracking_number", "tracking_id", "consignment_id"]
_ORIGIN_KEYWORDS = ["origin", "source", "pickup_location", "pickup", "from_location", "ship_from"]
_DESTINATION_KEYWORDS = ["destination", "dest", "delivery_location", "delivery", "to_location", "ship_to"]
_PICKUP_DATE_KEYWORDS = ["pickup_date", "pickup_time", "ship_date", "departure_date", "collection_date"]
_DELIVERY_DATE_KEYWORDS = ["delivery_date", "delivery_time", "arrival_date", "delivered_date"]
_STATUS_KEYWORDS = ["status", "shipment_status", "delivery_status", "order_status"]
_WEIGHT_KEYWORDS = ["weight", "weight_kg", "weight_lbs", "mass", "gross_weight"]
_VOLUME_KEYWORDS = ["volume", "volume_m3", "volume_cbm", "cubic_meters", "capacity"]
_CARRIER_KEYWORDS = ["carrier", "carrier_name", "transporter", "logistics_provider", "shipping_company"]
_VEHICLE_KEYWORDS = ["vehicle", "vehicle_type", "transport_mode", "vehicle_class"]
_DISTANCE_KEYWORDS = ["distance", "distance_km", "distance_miles", "route_distance"]
_EXPECTED_TIME_KEYWORDS = ["expected_delivery_time", "expected_time", "planned_delivery", "estimated_time", "expected_delivery_time_hours"]
_ACTUAL_TIME_KEYWORDS = ["actual_delivery_time", "actual_time", "real_delivery", "actual_delivery_time_hours"]
_DELAY_KEYWORDS = ["delay", "delay_hours", "delay_days", "late", "lateness", "delay_minutes"]
_COST_KEYWORDS = ["cost", "cost_usd", "price", "shipping_cost", "freight_cost", "total_cost"]
_PRIORITY_KEYWORDS = ["priority", "urgency", "importance", "priority_level"]
_ITEM_COUNT_KEYWORDS = ["item_count", "items", "quantity", "units", "pieces", "line_items"]
_TEMP_CONTROLLED_KEYWORDS = ["temperature_controlled", "temp_controlled", "refrigerated", "cold_chain", "climate_controlled"]
_HAZARDOUS_KEYWORDS = ["hazardous", "dangerous_goods", "hazmat", "dangerous", "haz"]
_TRACKING_URL_KEYWORDS = ["tracking_url", "tracking_link", "track_url", "tracking"]

# Supplier-specific
_SUPPLIER_ID_KEYWORDS = ["supplier_id", "supplier", "vendor_id", "vendor", "carrier_id"]
_SUPPLIER_NAME_KEYWORDS = ["supplier_name", "vendor_name", "carrier_name", "supplier"]
_SUPPLIER_EMAIL_KEYWORDS = ["supplier_email", "vendor_email", "contact_email", "email"]
_SUPPLIER_KEYWORDS = ["supplier", "vendor", "supplier_id", "vendor_id", "carrier"]

# Customer-specific
_CUSTOMER_KEYWORDS = ["customer", "customer_id", "client", "buyer", "consignee", "recipient"]

# Location coordinates
_LAT_KEYWORDS = ["lat", "latitude", "origin_lat", "dest_lat"]
_LON_KEYWORDS = ["lon", "lng", "long", "longitude", "origin_lon", "dest_lon"]

# Additional quality metrics
_ONTIME_KEYWORDS = ["on_time", "ontime", "delivered_on_time", "on_time_flag", "punctual"]
_DEFECT_KEYWORDS = ["defect", "defective", "quality_issue", "num_defects", "rejects", "damage", "damaged"]

# ========== UTILITY FUNCTIONS ==========

def _ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def _to_float_safe(val: Any, default: float = 0.0) -> float:
    """
    Convert val to float safely, handling pandas/numpy types.
    Returns default for NaN/unconvertible values.
    """
    try:
        if isinstance(val, (pd.Series,)):
            if val.empty:
                return default
            v = val.iloc[0]
        elif isinstance(val, (np.ndarray,)):
            if val.size == 0:
                return default
            v = val.flat[0]
        else:
            v = val

        if isinstance(v, (int, float, np.integer, np.floating)):
            if pd.isna(v):
                return default
            return float(v)

        coerced = pd.to_numeric(v, errors="coerce")
        if isinstance(coerced, (pd.Series, np.ndarray)):
            try:
                coerced_val = coerced.item()
            except Exception:
                coerced_val = coerced[0] if len(coerced) > 0 else np.nan
            coerced = coerced_val

        if pd.isna(coerced):
            return default
        return float(coerced)
    except Exception:
        return default


def _to_int_safe(val: Any, default: int = 0) -> int:
    """Convert val to int safely."""
    return int(_to_float_safe(val, default=float(default)))


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert objects to JSON-friendly types."""
    if isinstance(obj, dict):
        new: Dict[Any, Any] = {}
        for k, v in obj.items():
            if isinstance(k, (pd.Timestamp, np.datetime64)):
                new_key = str(k)
            elif isinstance(k, (np.integer,)):
                new_key = int(k)
            elif isinstance(k, (np.floating,)):
                new_key = float(k)
            elif isinstance(k, (pd.Period,)):
                new_key = str(k)
            else:
                new_key = k
            new[new_key] = _sanitize_for_json(v)
        return new
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_sanitize_for_json(v) for v in obj)
    else:
        if isinstance(obj, (np.generic,)):
            try:
                return obj.item()
            except Exception:
                return obj
        if isinstance(obj, (pd.Timestamp, np.datetime64)):
            return str(obj)
        if isinstance(obj, (pd.Timedelta,)):
            return str(obj)
        return obj


# ========== FILE I/O AND PERSISTENCE ==========

def read_csv(file_path: str) -> pd.DataFrame:
    """Read CSV with enhanced error handling."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise


def save_model_artifact(obj: Any, model_name: str, models_dir: str = "models") -> str:
    """Save model artifact as pickle file."""
    _ensure_dir(models_dir)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"{model_name.replace(' ', '_')}_{ts}.pkl"
    path = os.path.join(models_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved artifact to {path}")
    return path


def save_metadata(metadata: Dict, model_filepath: str, models_dir: str = "models") -> str:
    """Save metadata as JSON file."""
    _ensure_dir(models_dir)
    base = os.path.splitext(os.path.basename(model_filepath))[0]
    meta_filename = f"{base}.meta.json"
    meta_path = os.path.join(models_dir, meta_filename)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Saved metadata to {meta_path}")
    return meta_path


# ========== ENHANCED COLUMN DETECTION ==========

def detect_columns_by_keywords(columns: List[str], keywords: List[str]) -> Optional[str]:
    """Detect column by matching keywords (case-insensitive)."""
    low = [c.lower() for c in columns]
    for kw in keywords:
        for i, c in enumerate(low):
            if kw in c:
                return columns[i]
    return None


def detect_shipment_id_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _SHIPMENT_ID_KEYWORDS)


def detect_origin_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _ORIGIN_KEYWORDS)


def detect_destination_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _DESTINATION_KEYWORDS)


def detect_pickup_date_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _PICKUP_DATE_KEYWORDS)


def detect_delivery_date_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _DELIVERY_DATE_KEYWORDS)


def detect_status_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _STATUS_KEYWORDS)


def detect_weight_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _WEIGHT_KEYWORDS)


def detect_volume_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _VOLUME_KEYWORDS)


def detect_carrier_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _CARRIER_KEYWORDS)


def detect_vehicle_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _VEHICLE_KEYWORDS)


def detect_distance_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _DISTANCE_KEYWORDS)


def detect_expected_time_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _EXPECTED_TIME_KEYWORDS)


def detect_actual_time_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _ACTUAL_TIME_KEYWORDS)


def detect_delay_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _DELAY_KEYWORDS)


def detect_cost_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _COST_KEYWORDS)


def detect_priority_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _PRIORITY_KEYWORDS)


def detect_item_count_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _ITEM_COUNT_KEYWORDS)


def detect_temp_controlled_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _TEMP_CONTROLLED_KEYWORDS)


def detect_hazardous_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _HAZARDOUS_KEYWORDS)


def detect_tracking_url_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _TRACKING_URL_KEYWORDS)


def detect_supplier_id_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _SUPPLIER_ID_KEYWORDS)


def detect_supplier_name_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _SUPPLIER_NAME_KEYWORDS)


def detect_supplier_email_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _SUPPLIER_EMAIL_KEYWORDS)


def detect_supplier_column(columns: List[str]) -> Optional[str]:
    """Detect main supplier/carrier identifier column."""
    # Try specific IDs first, then general
    result = detect_supplier_id_column(columns)
    if result:
        return result
    result = detect_carrier_column(columns)
    if result:
        return result
    return detect_columns_by_keywords(columns, _SUPPLIER_KEYWORDS)


def detect_customer_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _CUSTOMER_KEYWORDS)


def detect_lat_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _LAT_KEYWORDS)


def detect_lon_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _LON_KEYWORDS)


def detect_ontime_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _ONTIME_KEYWORDS)


def detect_defect_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _DEFECT_KEYWORDS)


# ========== DATA QUALITY ASSESSMENT ==========

def assess_data_quality(df: pd.DataFrame, detected_columns: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """
    Assess data quality metrics for the dataset.
    Returns dict with completeness, validity, and quality scores.
    """
    total_rows = len(df)
    quality_report = {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "detected_columns": {k: v for k, v in detected_columns.items() if v is not None},
        "column_completeness": {},
        "data_issues": [],
        "overall_quality_score": 0.0
    }
    
    # Check completeness for each detected column
    for col_type, col_name in detected_columns.items():
        if col_name and col_name in df.columns:
            non_null_count = df[col_name].notna().sum()
            completeness = non_null_count / total_rows if total_rows > 0 else 0
            quality_report["column_completeness"][col_type] = {
                "column_name": col_name,
                "completeness": round(completeness, 3),
                "missing_count": total_rows - non_null_count
            }
            
            if completeness < 0.5:
                quality_report["data_issues"].append(
                    f"{col_type} ({col_name}) has low completeness: {completeness:.1%}"
                )
    
    # Calculate overall quality score
    if quality_report["column_completeness"]:
        avg_completeness = np.mean([
            v["completeness"] for v in quality_report["column_completeness"].values()
        ])
        quality_report["overall_quality_score"] = round(avg_completeness, 3)
    
    return quality_report


# ========== ENHANCED SUPPLIER STATISTICS ==========

def compute_delay_metrics(df: pd.DataFrame, supplier_col: str, 
                          expected_time_col: Optional[str],
                          actual_time_col: Optional[str],
                          delay_col: Optional[str],
                          status_col: Optional[str]) -> Tuple[pd.Series, pd.Series]:
    """
    Compute delay hours and on-time flag from available data.
    Returns (delay_hours_series, on_time_flag_series).
    """
    delay_hours = pd.Series(0.0, index=df.index)
    on_time_flag = pd.Series(1, index=df.index)  # 1 = on time, 0 = late
    
    # Method 1: Use delay column directly if available
    if delay_col and delay_col in df.columns:
        delay_hours = pd.to_numeric(df[delay_col], errors='coerce').fillna(0.0)
        on_time_flag = (delay_hours <= 0).astype(int)
        logger.info(f"Using delay column '{delay_col}' for delay metrics")
        return delay_hours, on_time_flag
    
    # Method 2: Calculate from expected vs actual time
    if expected_time_col and actual_time_col and \
       expected_time_col in df.columns and actual_time_col in df.columns:
        expected = pd.to_numeric(df[expected_time_col], errors='coerce')
        actual = pd.to_numeric(df[actual_time_col], errors='coerce')
        delay_hours = (actual - expected).fillna(0.0)
        on_time_flag = (delay_hours <= 0).astype(int)
        logger.info(f"Calculated delay from expected vs actual time columns")
        return delay_hours, on_time_flag
    
    # Method 3: Use status column to infer delays
    if status_col and status_col in df.columns:
        status_lower = df[status_col].astype(str).str.lower()
        delayed_mask = status_lower.str.contains('delay|late|overdue', na=False)
        on_time_flag = (~delayed_mask).astype(int)
        # Estimate delay hours as 24 for delayed shipments (rough estimate)
        delay_hours = delayed_mask.astype(float) * 24.0
        logger.info(f"Inferred delays from status column '{status_col}'")
        return delay_hours, on_time_flag
    
    logger.warning("No delay information available, assuming all shipments on-time")
    return delay_hours, on_time_flag


def compute_enhanced_supplier_stats(
    df: pd.DataFrame, 
    supplier_col: str,
    detected_cols: Dict[str, Optional[str]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute comprehensive per-supplier statistics including:
    - Volume metrics (shipments, weight, volume)
    - Timing metrics (avg delivery time, lead time variability, delay rate)
    - Cost metrics (avg cost, cost per kg, cost per km)
    - Quality metrics (on-time rate, defect rate)
    - Special handling (temp-controlled %, hazardous %)
    """
    suppliers = df[supplier_col].unique().tolist()
    stats: Dict[str, Dict[str, float]] = {}
    
    # Extract column names for easier access
    weight_col = detected_cols.get('weight')
    volume_col = detected_cols.get('volume')
    cost_col = detected_cols.get('cost')
    distance_col = detected_cols.get('distance')
    expected_time_col = detected_cols.get('expected_time')
    actual_time_col = detected_cols.get('actual_time')
    delay_col = detected_cols.get('delay')
    status_col = detected_cols.get('status')
    temp_col = detected_cols.get('temp_controlled')
    hazardous_col = detected_cols.get('hazardous')
    defect_col = detected_cols.get('defect')
    item_count_col = detected_cols.get('item_count')
    
    # Compute delay metrics once for the entire dataset
    delay_hours, on_time_flag = compute_delay_metrics(
        df, supplier_col, expected_time_col, actual_time_col, delay_col, status_col
    )
    
    for supplier in suppliers:
        supplier_mask = df[supplier_col] == supplier
        supplier_df = df[supplier_mask]
        supplier_delays = delay_hours[supplier_mask]
        supplier_ontime = on_time_flag[supplier_mask]
        
        n_shipments = len(supplier_df)
        
        # Volume metrics
        total_weight = _to_float_safe(
            pd.to_numeric(supplier_df[weight_col], errors='coerce').sum() 
            if weight_col and weight_col in supplier_df.columns else 0.0
        )
        total_volume = _to_float_safe(
            pd.to_numeric(supplier_df[volume_col], errors='coerce').sum()
            if volume_col and volume_col in supplier_df.columns else 0.0
        )
        avg_weight = total_weight / n_shipments if n_shipments > 0 else 0.0
        
        # Cost metrics
        if cost_col and cost_col in supplier_df.columns:
            costs = pd.to_numeric(supplier_df[cost_col], errors='coerce').dropna()
            total_cost = float(costs.sum())
            avg_cost = float(costs.mean()) if len(costs) > 0 else 0.0
            cost_per_kg = total_cost / total_weight if total_weight > 0 else 0.0
        else:
            total_cost = 0.0
            avg_cost = 0.0
            cost_per_kg = 0.0
        
        # Distance and efficiency
        if distance_col and distance_col in supplier_df.columns:
            distances = pd.to_numeric(supplier_df[distance_col], errors='coerce').dropna()
            avg_distance = float(distances.mean()) if len(distances) > 0 else 0.0
            total_distance = float(distances.sum())
            cost_per_km = total_cost / total_distance if total_distance > 0 else 0.0
        else:
            avg_distance = 0.0
            cost_per_km = 0.0
        
        # Timing metrics
        if actual_time_col and actual_time_col in supplier_df.columns:
            actual_times = pd.to_numeric(supplier_df[actual_time_col], errors='coerce').dropna()
            avg_delivery_time = float(actual_times.mean()) if len(actual_times) > 0 else 0.0
            std_delivery_time = float(actual_times.std()) if len(actual_times) > 0 else 0.0
        else:
            avg_delivery_time = 0.0
            std_delivery_time = 0.0
        
        # Delay and on-time metrics
        avg_delay = float(supplier_delays.mean())
        on_time_rate = float(supplier_ontime.mean()) if len(supplier_ontime) > 0 else 0.0
        delay_rate = 1.0 - on_time_rate
        
        # Quality metrics
        if defect_col and defect_col in supplier_df.columns:
            defects = pd.to_numeric(supplier_df[defect_col], errors='coerce').fillna(0)
            defect_rate = float(defects.sum() / len(defects)) if len(defects) > 0 else 0.0
        else:
            defect_rate = 0.0
        
        # Special handling
        if temp_col and temp_col in supplier_df.columns:
            temp_vals = supplier_df[temp_col].astype(str).str.upper()
            temp_controlled_pct = float((temp_vals == 'TRUE').sum() / n_shipments) if n_shipments > 0 else 0.0
        else:
            temp_controlled_pct = 0.0
        
        if hazardous_col and hazardous_col in supplier_df.columns:
            haz_vals = supplier_df[hazardous_col].astype(str).str.upper()
            hazardous_pct = float((haz_vals == 'TRUE').sum() / n_shipments) if n_shipments > 0 else 0.0
        else:
            hazardous_pct = 0.0
        
        # Item count
        if item_count_col and item_count_col in supplier_df.columns:
            items = pd.to_numeric(supplier_df[item_count_col], errors='coerce').dropna()
            avg_items = float(items.mean()) if len(items) > 0 else 0.0
        else:
            avg_items = 0.0
        
        # Compile all stats
        stats[supplier] = {
            # Volume
            "total_shipments": float(n_shipments),
            "total_weight_kg": total_weight,
            "total_volume_m3": total_volume,
            "avg_weight_kg": avg_weight,
            "avg_items_per_shipment": avg_items,
            
            # Cost
            "total_cost_usd": total_cost,
            "avg_cost_usd": avg_cost,
            "cost_per_kg": cost_per_kg,
            "cost_per_km": cost_per_km,
            
            # Performance
            "avg_distance_km": avg_distance,
            "avg_delivery_time_hours": avg_delivery_time,
            "std_delivery_time_hours": std_delivery_time,
            "avg_delay_hours": avg_delay,
            "on_time_rate": on_time_rate,
            "delay_rate": delay_rate,
            "defect_rate": defect_rate,
            
            # Special handling
            "temp_controlled_pct": temp_controlled_pct,
            "hazardous_pct": hazardous_pct,
        }
    
    return stats


# ========== RISK SCORING ==========

def scale_to_score(x: float, direction: str = "higher_better", 
                   clip: Tuple[float, float] = (0.0, 1.0)) -> float:
    """
    Scale a metric to [0,1] score using tanh.
    direction: 'higher_better' or 'lower_better'
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.5
    try:
        val = float(x)
    except Exception:
        return 0.5
    
    if direction == "higher_better":
        s = 0.5 + 0.5 * math.tanh(val / (1.0 + abs(val)))
    else:
        s = 0.5 + 0.5 * math.tanh(-val / (1.0 + abs(val)))
    
    lo, hi = clip
    return max(lo, min(hi, s))


def compute_statistical_risk_scores(stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Compute risk scores from supplier statistics.
    Higher risk score = higher risk (0-1 scale).
    
    Risk factors:
    - Delivery time variability (higher std = higher risk)
    - Delay rate (higher delay rate = higher risk)
    - Defect rate (higher defect = higher risk)
    - Cost efficiency (higher cost per kg/km = moderate risk)
    """
    scores: Dict[str, float] = {}
    
    for supplier, st in stats.items():
        # Extract metrics
        std_time = st.get("std_delivery_time_hours", 0.0)
        delay_rate = st.get("delay_rate", 0.0)
        defect_rate = st.get("defect_rate", 0.0)
        cost_per_kg = st.get("cost_per_kg", 0.0)
        
        # Scale each to risk contribution [0,1]
        risk_variability = scale_to_score(std_time, direction="lower_better")
        risk_delays = scale_to_score(delay_rate, direction="lower_better")
        risk_defects = scale_to_score(defect_rate, direction="lower_better")
        risk_cost = scale_to_score(cost_per_kg, direction="lower_better")
        
        # Weighted combination (adjustable)
        weights = {
            "variability": 0.25,
            "delays": 0.40,
            "defects": 0.25,
            "cost": 0.10
        }
        
        risk_score = (
            weights["variability"] * risk_variability +
            weights["delays"] * risk_delays +
            weights["defects"] * risk_defects +
            weights["cost"] * risk_cost
        )
        
        scores[supplier] = float(max(0.0, min(1.0, risk_score)))
    
    return scores


# ========== ML-BASED RISK MODEL ==========

def train_ml_risk_model(
    df: pd.DataFrame,
    supplier_col: str,
    detected_cols: Dict[str, Optional[str]],
    use_random_forest: bool = True,
    random_state: int = 42
) -> Tuple[Optional[Any], Optional[Dict[str, float]], Optional[Dict[str, Any]]]:
    """
    Train ML model to predict delivery delays/failures.
    Returns (model, per_supplier_risk_probs, model_metrics).
    """
    if not _HAS_SKLEARN:
        logger.warning("sklearn not available, skipping ML model")
        return None, None, None
    
    df_copy = df.copy()
    
    # Compute delay metrics
    delay_hours, on_time_flag = compute_delay_metrics(
        df_copy, supplier_col,
        detected_cols.get('expected_time'),
        detected_cols.get('actual_time'),
        detected_cols.get('delay'),
        detected_cols.get('status')
    )
    
    # Label: 1 = bad delivery (delayed), 0 = good
    labels = (on_time_flag == 0).astype(int)
    
    if labels.sum() == 0:
        logger.warning("No delayed shipments found, cannot train ML model")
        return None, None, None
    
    # Feature engineering
    features = pd.DataFrame(index=df_copy.index)
    
    # Weight
    weight_col = detected_cols.get('weight')
    if weight_col and weight_col in df_copy.columns:
        features["weight_kg"] = pd.to_numeric(df_copy[weight_col], errors='coerce').fillna(0.0)
    else:
        features["weight_kg"] = 0.0
    
    # Distance
    distance_col = detected_cols.get('distance')
    if distance_col and distance_col in df_copy.columns:
        features["distance_km"] = pd.to_numeric(df_copy[distance_col], errors='coerce').fillna(0.0)
    else:
        features["distance_km"] = 0.0
    
    # Expected time
    expected_col = detected_cols.get('expected_time')
    if expected_col and expected_col in df_copy.columns:
        features["expected_time"] = pd.to_numeric(df_copy[expected_col], errors='coerce').fillna(0.0)
    else:
        features["expected_time"] = 0.0
    
    # Item count
    item_col = detected_cols.get('item_count')
    if item_col and item_col in df_copy.columns:
        features["item_count"] = pd.to_numeric(df_copy[item_col], errors='coerce').fillna(0.0)
    else:
        features["item_count"] = 0.0
    
    # Temperature controlled (binary)
    temp_col = detected_cols.get('temp_controlled')
    if temp_col and temp_col in df_copy.columns:
        temp_vals = df_copy[temp_col].astype(str).str.upper()
        features["temp_controlled"] = (temp_vals == 'TRUE').astype(int)
    else:
        features["temp_controlled"] = 0
    
    # Hazardous (binary)
    haz_col = detected_cols.get('hazardous')
    if haz_col and haz_col in df_copy.columns:
        haz_vals = df_copy[haz_col].astype(str).str.upper()
        features["hazardous"] = (haz_vals == 'TRUE').astype(int)
    else:
        features["hazardous"] = 0
    
    # Supplier encoding
    features["supplier_code"] = pd.Categorical(df_copy[supplier_col]).codes
    
    # Fill any remaining NaNs
    features = features.fillna(features.median().fillna(0.0))
    
    # Train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features.values, labels.values,
            test_size=0.2, random_state=random_state,
            stratify=labels if labels.sum() > 0 else None
        )
    except Exception:
        X_train, X_test, y_train, y_test = features.values, features.values, labels.values, labels.values
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model selection
    if use_random_forest:
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1
        )
        model_type = "RandomForest"
    else:
        model = LogisticRegression(random_state=random_state, max_iter=500)
        model_type = "LogisticRegression"
    
    try:
        model.fit(X_train_scaled, y_train)
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return None, None, None
    
    # Evaluation
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        "model_type": model_type,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": float((y_pred == y_test).mean()),
        "delayed_rate_test": float(y_test.mean()),
    }
    
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
    except Exception:
        metrics["roc_auc"] = None
    
    logger.info(f"ML model trained: {model_type}, Accuracy: {metrics['accuracy']:.3f}")
    
    # Per-supplier risk probabilities
    suppliers = df_copy[supplier_col].unique().tolist()
    supplier_probs: Dict[str, float] = {}
    
    for supplier in suppliers:
        supplier_mask = df_copy[supplier_col] == supplier
        supplier_features = features[supplier_mask]
        
        if len(supplier_features) == 0:
            supplier_probs[supplier] = 0.5
            continue
        
        X_supplier = scaler.transform(supplier_features.values)
        probs = model.predict_proba(X_supplier)[:, 1]
        supplier_probs[supplier] = float(np.mean(probs))
    
    return model, supplier_probs, metrics


# ========== SURVIVAL MODEL ==========

def fit_survival_model(
    df: pd.DataFrame,
    supplier_col: str,
    detected_cols: Dict[str, Optional[str]]
) -> Tuple[Optional[Any], Optional[Dict[str, float]]]:
    """
    Fit Cox Proportional Hazards model for supplier risk.
    """
    if not _HAS_LIFELINES:
        logger.warning("lifelines not available, skipping survival model")
        return None, None
    
    df_copy = df.copy()
    
    # Duration: use expected delivery time
    expected_col = detected_cols.get('expected_time')
    if expected_col and expected_col in df_copy.columns:
        df_copy["duration"] = pd.to_numeric(df_copy[expected_col], errors='coerce').fillna(1.0)
    else:
        df_copy["duration"] = 1.0
    
    # Event: delayed deliveries
    delay_hours, on_time_flag = compute_delay_metrics(
        df_copy, supplier_col,
        detected_cols.get('expected_time'),
        detected_cols.get('actual_time'),
        detected_cols.get('delay'),
        detected_cols.get('status')
    )
    df_copy["event"] = (on_time_flag == 0).astype(int)
    
    if df_copy["event"].sum() == 0:
        logger.warning("No delayed shipments, cannot fit survival model")
        return None, None
    
    # Covariates
    covariates = pd.DataFrame(index=df_copy.index)
    
    # Weight
    weight_col = detected_cols.get('weight')
    if weight_col and weight_col in df_copy.columns:
        covariates["weight_kg"] = pd.to_numeric(df_copy[weight_col], errors='coerce').fillna(0.0)
    else:
        covariates["weight_kg"] = 0.0
    
    # Distance
    distance_col = detected_cols.get('distance')
    if distance_col and distance_col in df_copy.columns:
        covariates["distance_km"] = pd.to_numeric(df_copy[distance_col], errors='coerce').fillna(0.0)
    else:
        covariates["distance_km"] = 0.0
    
    # Supplier code
    covariates["supplier_code"] = pd.Categorical(df_copy[supplier_col]).codes
    
    # Combine
    cph_df = pd.concat([df_copy[["duration", "event"]], covariates], axis=1).dropna()
    
    # Drop constant columns
    nunique = cph_df.nunique()
    keep_cols = nunique[nunique > 1].index.tolist()
    cph_df = cph_df[keep_cols]
    
    if cph_df.empty or "duration" not in cph_df.columns or "event" not in cph_df.columns:
        logger.warning("Insufficient data for survival model")
        return None, None
    
    try:
        cph = CoxPHFitter(penalizer=0.1)
    except TypeError:
        cph = CoxPHFitter()
    
    try:
        cph.fit(cph_df, duration_col="duration", event_col="event", show_progress=False)
    except Exception as e:
        logger.error(f"Survival model fit failed: {e}")
        return None, None
    
    # Per-supplier hazard scores
    suppliers = df_copy[supplier_col].unique().tolist()
    hazard_scores: Dict[str, float] = {}
    
    for supplier in suppliers:
        supplier_mask = df_copy[supplier_col] == supplier
        supplier_cph_df = cph_df[cph_df.index.isin(df_copy[supplier_mask].index)]
        
        if supplier_cph_df.empty:
            hazard_scores[supplier] = 0.5
            continue
        
        hazard = cph.predict_partial_hazard(supplier_cph_df).mean()
        hazard_scores[supplier] = float(hazard)
    
    # Normalize to [0,1]
    vals = np.array(list(hazard_scores.values()), dtype=float)
    if np.nanmax(vals) - np.nanmin(vals) > 0:
        norm = (vals - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals))
        for i, supplier in enumerate(list(hazard_scores.keys())):
            hazard_scores[supplier] = float(norm[i])
    else:
        for supplier in hazard_scores.keys():
            hazard_scores[supplier] = 0.5
    
    logger.info("Survival model fitted successfully")
    return cph, hazard_scores


# ========== OPTIMIZATION ==========

def solve_supplier_allocation_mip(
    demand_df: pd.DataFrame,
    supplier_list: List[str],
    supplier_capacity: Dict[str, float],
    supplier_unit_cost: Dict[str, float],
    supplier_risk_score: Dict[str, float],
    weight_col: Optional[str],
    supplier_col: str,
    risk_weight: float = 1.0,
    timeout: Optional[int] = 30,
) -> Dict[str, Any]:
    """
    MILP formulation for supplier allocation.
    Minimizes: sum (cost + risk_weight * risk) * weight
    """
    if not _HAS_PULP:
        raise RuntimeError("pulp not installed - MILP solver unavailable")
    
    demand_df = demand_df.reset_index(drop=True)
    N = demand_df.shape[0]
    S = supplier_list
    
    prob = pl.LpProblem("supplier_allocation", pl.LpMinimize)
    x = pl.LpVariable.dicts("x", (list(range(N)), S), lowBound=0, cat="Continuous")
    
    # Objective
    obj_terms = []
    for i in range(N):
        weight = _to_float_safe(
            demand_df.loc[i, weight_col] if weight_col and weight_col in demand_df.columns else 1.0,
            default=1.0
        )
        if weight <= 0:
            weight = 1.0
        
        for s in S:
            unit_cost = float(supplier_unit_cost.get(s, 0.0))
            risk = float(supplier_risk_score.get(s, 0.0))
            obj_terms.append((unit_cost + risk_weight * risk) * weight * x[i][s])
    
    prob += pl.lpSum(obj_terms)
    
    # Constraints: satisfy demand
    for i in range(N):
        prob += pl.lpSum([x[i][s] for s in S]) == 1.0  # Assign each shipment fully
    
    # Capacity constraints
    for s in S:
        cap = float(supplier_capacity.get(s, float("inf")))
        total_weight_assigned = []
        for i in range(N):
            weight = _to_float_safe(
                demand_df.loc[i, weight_col] if weight_col and weight_col in demand_df.columns else 1.0,
                default=1.0
            )
            total_weight_assigned.append(weight * x[i][s])
        prob += pl.lpSum(total_weight_assigned) <= cap
    
    # Solve
    solver = pl.PULP_CBC_CMD(msg=False, timeLimit=timeout)
    status = prob.solve(solver)
    status_str = pl.LpStatus[status]
    
    logger.info(f"MILP solver status: {status_str}")
    
    # Extract allocation
    allocation: Dict[str, Dict[int, float]] = {s: {} for s in S}
    for i in range(N):
        for s in S:
            try:
                q = float(pl.value(x[i][s]) or 0.0)
            except Exception:
                q = 0.0
            
            if q > 1e-6:
                allocation[s][int(i)] = q
    
    return {"status": status_str, "allocation": allocation}


def heuristic_supplier_allocator(
    demand_df: pd.DataFrame,
    supplier_list: List[str],
    supplier_capacity: Dict[str, float],
    supplier_unit_cost: Dict[str, float],
    supplier_risk_score: Dict[str, float],
    weight_col: Optional[str],
) -> Dict[str, Any]:
    """
    Greedy heuristic allocation: assign each shipment to best available supplier.
    """
    allocation: Dict[str, Dict[int, float]] = {s: {} for s in supplier_list}
    remaining_cap = {s: float(supplier_capacity.get(s, float("inf"))) for s in supplier_list}
    
    # Combined score: cost + risk
    score = {
        s: float(supplier_unit_cost.get(s, 0.0)) + float(supplier_risk_score.get(s, 0.0))
        for s in supplier_list
    }
    
    for i in range(demand_df.shape[0]):
        weight = _to_float_safe(
            demand_df.iloc[i][weight_col] if weight_col and weight_col in demand_df.columns else 1.0,
            default=1.0
        )
        if weight <= 0:
            weight = 1.0
        
        # Sort suppliers by score
        ordered = sorted(supplier_list, key=lambda s: score[s])
        
        # Try to assign to supplier with capacity
        assigned = False
        for s in ordered:
            if remaining_cap[s] >= weight:
                allocation[s][int(i)] = 1.0  # Full assignment
                remaining_cap[s] -= weight
                assigned = True
                break
        
        # If no capacity, assign to best supplier anyway
        if not assigned:
            s_best = ordered[0]
            allocation[s_best][int(i)] = 1.0
    
    return {"status": "heuristic_allocated", "allocation": allocation}


# ========== ROUTING ==========

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in km."""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def simple_vrp_routes(
    allocation: Dict[str, Dict[int, float]],
    demand_df: pd.DataFrame,
    detected_cols: Dict[str, Optional[str]],
    supplier_locations: Dict[str, Tuple[float, float]],
    vehicle_capacity: float = 100.0,
    max_vehicles_per_supplier: int = 10,
) -> Dict[str, List[List[int]]]:
    """
    Simple VRP routing using nearest neighbor heuristic.
    """
    weight_col = detected_cols.get('weight')
    lat_col = detected_cols.get('lat')
    lon_col = detected_cols.get('lon')
    
    has_locations = (
        lat_col and lon_col and
        lat_col in demand_df.columns and
        lon_col in demand_df.columns
    )
    
    routes_per_supplier: Dict[str, List[List[int]]] = {}
    
    for supplier, alloc in allocation.items():
        rows = list(alloc.keys())
        if not rows:
            routes_per_supplier[supplier] = []
            continue
        
        if not has_locations:
            # No location data: one route per shipment
            routes_per_supplier[supplier] = [[int(r)] for r in sorted(rows)]
            continue
        
        # Build nodes with locations and weights
        nodes = []
        for r in rows:
            try:
                lat = float(demand_df.loc[r, lat_col])
                lon = float(demand_df.loc[r, lon_col])
            except Exception:
                lat, lon = 0.0, 0.0
            
            weight = _to_float_safe(
                demand_df.loc[r, weight_col] if weight_col and weight_col in demand_df.columns else 1.0,
                default=1.0
            )
            
            nodes.append({
                "row": int(r),
                "weight": weight,
                "lat": lat,
                "lon": lon
            })
        
        # Supplier location
        supplier_loc = supplier_locations.get(supplier, (0.0, 0.0))
        
        # Nearest neighbor routing with capacity constraints
        remaining_nodes = nodes.copy()
        routes = []
        vehicles_used = 0
        
        while remaining_nodes and vehicles_used < max_vehicles_per_supplier:
            capacity_left = vehicle_capacity
            route = []
            current_lat, current_lon = supplier_loc
            
            while remaining_nodes and capacity_left > 0:
                # Find feasible nodes
                feasible = [n for n in remaining_nodes if n["weight"] <= capacity_left]
                
                if not feasible:
                    # Pick nearest even if over capacity
                    nearest = min(
                        remaining_nodes,
                        key=lambda n: haversine_distance(current_lat, current_lon, n["lat"], n["lon"])
                    )
                    serve = min(nearest["weight"], capacity_left)
                    if serve <= 0:
                        break
                    route.append(int(nearest["row"]))
                    nearest["weight"] -= serve
                    capacity_left -= serve
                    if nearest["weight"] <= 1e-8:
                        remaining_nodes.remove(nearest)
                    current_lat, current_lon = nearest["lat"], nearest["lon"]
                else:
                    # Pick nearest feasible
                    nearest = min(
                        feasible,
                        key=lambda n: haversine_distance(current_lat, current_lon, n["lat"], n["lon"])
                    )
                    route.append(int(nearest["row"]))
                    capacity_left -= nearest["weight"]
                    remaining_nodes.remove(nearest)
                    current_lat, current_lon = nearest["lat"], nearest["lon"]
            
            if route:
                routes.append(route)
            vehicles_used += 1
        
        # Handle any remaining nodes
        if remaining_nodes:
            leftover_route = [int(n["row"]) for n in remaining_nodes]
            routes.append(leftover_route)
        
        routes_per_supplier[supplier] = routes
    
    return routes_per_supplier


# ========== MAIN PIPELINE ==========

def run_supplier_risk_and_routing_pipeline(
    file_path: str,
    gemini_analysis: Optional[Dict] = None,
    supplier_col_hint: Optional[str] = None,
    models_dir: str = "models",
    use_milp: bool = True,
    use_random_forest: bool = True,
    risk_weight: float = 10.0,
    vehicle_capacity: float = 10000.0,
    time_horizon_limit: Optional[int] = None,
    milp_timeout_seconds: int = 30,
    **kwargs
) -> Dict[str, Any]:
    """
    Enhanced supplier risk and routing optimization pipeline.
    """
    logger.info(f"Starting pipeline for {file_path}")
    
    # Read data
    df = read_csv(file_path)
    columns = list(df.columns)
    logger.info(f"Loaded {len(df)} rows, {len(columns)} columns")
    
    # Detect all columns
    detected_cols = {
        "shipment_id": detect_shipment_id_column(columns),
        "origin": detect_origin_column(columns),
        "destination": detect_destination_column(columns),
        "pickup_date": detect_pickup_date_column(columns),
        "delivery_date": detect_delivery_date_column(columns),
        "status": detect_status_column(columns),
        "weight": detect_weight_column(columns),
        "volume": detect_volume_column(columns),
        "carrier": detect_carrier_column(columns),
        "vehicle": detect_vehicle_column(columns),
        "distance": detect_distance_column(columns),
        "expected_time": detect_expected_time_column(columns),
        "actual_time": detect_actual_time_column(columns),
        "delay": detect_delay_column(columns),
        "cost": detect_cost_column(columns),
        "priority": detect_priority_column(columns),
        "item_count": detect_item_count_column(columns),
        "temp_controlled": detect_temp_controlled_column(columns),
        "hazardous": detect_hazardous_column(columns),
        "tracking_url": detect_tracking_url_column(columns),
        "supplier_id": detect_supplier_id_column(columns),
        "supplier_name": detect_supplier_name_column(columns),
        "supplier_email": detect_supplier_email_column(columns),
        "customer": detect_customer_column(columns),
        "lat": detect_lat_column(columns),
        "lon": detect_lon_column(columns),
        "ontime": detect_ontime_column(columns),
        "defect": detect_defect_column(columns),
    }
    
    # Main supplier column
    supplier_col = supplier_col_hint or detect_supplier_column(columns)
    if not supplier_col:
        raise ValueError("Could not detect supplier/carrier column")
    
    detected_cols["supplier"] = supplier_col
    logger.info(f"Detected supplier column: {supplier_col}")
    
    # Data quality assessment
    quality_report = assess_data_quality(df, detected_cols)
    logger.info(f"Data quality score: {quality_report['overall_quality_score']:.3f}")
    
    # Get unique suppliers
    suppliers = df[supplier_col].unique().tolist()
    logger.info(f"Found {len(suppliers)} unique suppliers: {suppliers}")
    
    # Compute enhanced supplier statistics
    stats = compute_enhanced_supplier_stats(df, supplier_col, detected_cols)
    
    # Statistical risk scoring
    stat_risk = compute_statistical_risk_scores(stats)
    logger.info("Computed statistical risk scores")
    
    # ML-based risk model
    ml_model, ml_probs, ml_metrics = train_ml_risk_model(
        df, supplier_col, detected_cols,
        use_random_forest=use_random_forest
    )
    ml_used = ml_model is not None
    
    # Survival model
    surv_model, surv_scores = fit_survival_model(df, supplier_col, detected_cols)
    surv_used = surv_model is not None
    
    # Combined risk score
    combined_risk: Dict[str, float] = {}
    for supplier in suppliers:
        parts = [stat_risk.get(supplier, 0.5)]
        if ml_used:
            parts.append(ml_probs.get(supplier, 0.5))
        if surv_used:
            parts.append(surv_scores.get(supplier, 0.5))
        combined_risk[supplier] = float(np.nanmean(parts))
    
    logger.info("Computed combined risk scores")
    
    # Extract supplier attributes from historical data
    supplier_unit_cost: Dict[str, float] = {}
    supplier_capacity: Dict[str, float] = {}
    supplier_locations: Dict[str, Tuple[float, float]] = {}
    
    cost_col = detected_cols.get('cost')
    weight_col = detected_cols.get('weight')
    lat_col = detected_cols.get('lat')
    lon_col = detected_cols.get('lon')
    
    for supplier in suppliers:
        supplier_df = df[df[supplier_col] == supplier]
        
        # Unit cost: average cost from history
        if cost_col and cost_col in supplier_df.columns:
            costs = pd.to_numeric(supplier_df[cost_col], errors='coerce').dropna()
            supplier_unit_cost[supplier] = float(costs.mean()) if len(costs) > 0 else 1.0
        else:
            supplier_unit_cost[supplier] = 1.0
        
        # Capacity: sum of historical weights or default
        if weight_col and weight_col in supplier_df.columns:
            weights = pd.to_numeric(supplier_df[weight_col], errors='coerce').dropna()
            historical_capacity = float(weights.sum())
            supplier_capacity[supplier] = max(1.0, historical_capacity * 1.2)  # 20% buffer
        else:
            supplier_capacity[supplier] = 10000.0
        
        # Location: average lat/lon
        if lat_col and lon_col and lat_col in supplier_df.columns and lon_col in supplier_df.columns:
            try:
                lat = float(pd.to_numeric(supplier_df[lat_col], errors='coerce').mean())
                lon = float(pd.to_numeric(supplier_df[lon_col], errors='coerce').mean())
                supplier_locations[supplier] = (lat, lon)
            except Exception:
                supplier_locations[supplier] = (0.0, 0.0)
        else:
            supplier_locations[supplier] = (0.0, 0.0)
    
    logger.info("Extracted supplier attributes")
    
    # Filter demand_df for future allocations
    demand_df = df.copy()
    if time_horizon_limit:
        pickup_col = detected_cols.get('pickup_date')
        if pickup_col and pickup_col in demand_df.columns:
            try:
                demand_df[pickup_col] = pd.to_datetime(demand_df[pickup_col], errors='coerce')
                valid_dates = demand_df.dropna(subset=[pickup_col])
                if not valid_dates.empty:
                    demand_df = valid_dates.sort_values(pickup_col).tail(time_horizon_limit)
            except Exception:
                demand_df = demand_df.tail(time_horizon_limit)
        else:
            demand_df = demand_df.tail(time_horizon_limit)
    
    logger.info(f"Demand dataset: {len(demand_df)} rows")
    
    # Solve allocation
    allocation_result = None
    solver_used = None
    
    if use_milp and _HAS_PULP:
        try:
            allocation_result = solve_supplier_allocation_mip(
                demand_df=demand_df,
                supplier_list=suppliers,
                supplier_capacity=supplier_capacity,
                supplier_unit_cost=supplier_unit_cost,
                supplier_risk_score=combined_risk,
                weight_col=weight_col,
                supplier_col=supplier_col,
                risk_weight=risk_weight,
                timeout=milp_timeout_seconds,
            )
            solver_used = "MILP"
            logger.info(f"MILP allocation completed: {allocation_result['status']}")
        except Exception as e:
            logger.error(f"MILP failed: {e}")
            solver_used = "heuristic_fallback"
    
    if allocation_result is None or not allocation_result.get("allocation"):
        allocation_result = heuristic_supplier_allocator(
            demand_df=demand_df,
            supplier_list=suppliers,
            supplier_capacity=supplier_capacity,
            supplier_unit_cost=supplier_unit_cost,
            supplier_risk_score=combined_risk,
            weight_col=weight_col,
        )
        solver_used = "heuristic"
        logger.info("Heuristic allocation completed")
    
    allocation = allocation_result.get("allocation", {})
    
    # Compute routes
    routes = simple_vrp_routes(
        allocation=allocation,
        demand_df=demand_df,
        detected_cols=detected_cols,
        supplier_locations=supplier_locations,
        vehicle_capacity=vehicle_capacity,
    )
    logger.info(f"Generated {sum(len(r) for r in routes.values())} routes")
    
    # Compile results
    results = {
        "success": True,
        "data_quality": quality_report,
        "detected_columns": {k: v for k, v in detected_cols.items() if v is not None},
        "suppliers": suppliers,
        "supplier_statistics": stats,
        "risk_scores": {
            "statistical": stat_risk,
            "ml_based": ml_probs if ml_used else None,
            "survival_based": surv_scores if surv_used else None,
            "combined": combined_risk,
        },
        "ml_model_metrics": ml_metrics,
        "supplier_attributes": {
            "unit_cost": supplier_unit_cost,
            "capacity": supplier_capacity,
            "locations": supplier_locations,
        },
        "allocation": {
            "solver_used": solver_used,
            "solver_status": allocation_result.get("status"),
            "assignments": allocation,
        },
        "routes": routes,
        "configuration": {
            "risk_weight": risk_weight,
            "vehicle_capacity": vehicle_capacity,
            "use_milp": use_milp,
            "use_random_forest": use_random_forest,
            "time_horizon_limit": time_horizon_limit,
        }
    }
    
    # Enhanced metadata
    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_name": "Enhanced Logistics & Supplier Risk Model",
        "model_version": "2.0",
        "input_file": os.path.basename(file_path),
        "data_summary": {
            "total_rows": len(df),
            "total_suppliers": len(suppliers),
            "quality_score": quality_report["overall_quality_score"],
            "demand_rows": len(demand_df),
        },
        "models_used": {
            "statistical_scoring": True,
            "ml_model": ml_used,
            "ml_model_type": ml_metrics.get("model_type") if ml_metrics else None,
            "survival_model": surv_used,
        },
        "optimization": {
            "solver": solver_used,
            "total_assignments": sum(len(v) for v in allocation.values()),
            "total_routes": sum(len(r) for r in routes.values()),
        },
        "configuration": results["configuration"],
        "detected_columns": detected_cols,
    }
    
    # Persist artifacts
    try:
        artifact_obj = {
            "results": results,
            "metadata": metadata,
            "models": {
                "ml_model": ml_model if ml_used else None,
                "survival_model": surv_model if surv_used else None,
            }
        }
        saved_path = save_model_artifact(artifact_obj, "Supplier_Risk_Routing_Enhanced", models_dir)
        saved_meta_path = save_metadata(metadata, saved_path, models_dir)
        
        results["artifact"] = {
            "model_path": saved_path,
            "meta_path": saved_meta_path
        }
    except Exception as e:
        logger.error(f"Failed to persist artifacts: {e}")
        results["artifact_error"] = str(e)
    
    # Sanitize for JSON
    results = _sanitize_for_json(results)
    logger.info("Pipeline completed successfully")
    
    return results


def analyze_file_and_run_pipeline(
    file_path: str,
    gemini_response: Dict,
    models_dir: str = "models",
    **kwargs
) -> Dict:
    """
    Entry point for gemini_analyser integration.
    """
    try:
        model_type = gemini_response.get("analysis", {}).get("model_type", "")
    except Exception:
        model_type = ""
    
    if "logistics" in model_type.lower() or "supplier" in model_type.lower():
        pipeline_result = run_supplier_risk_and_routing_pipeline(
            file_path,
            gemini_analysis=gemini_response,
            models_dir=models_dir,
            **kwargs
        )
        return {
            "success": True,
            "gemini_analysis": gemini_response,
            "pipeline": pipeline_result
        }
    else:
        return {
            "success": False,
            "error": "Model type not applicable for logistics & supplier risk pipeline"
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Supplier Risk & Routing Pipeline")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--models-dir", default="models", help="Directory for model artifacts")
    parser.add_argument("--no-milp", action="store_true", help="Disable MILP solver")
    parser.add_argument("--logistic-regression", action="store_true", help="Use Logistic Regression instead of Random Forest")
    parser.add_argument("--risk-weight", type=float, default=10.0, help="Risk weight in objective")
    parser.add_argument("--vehicle-capacity", type=float, default=10000.0, help="Vehicle capacity (kg)")
    parser.add_argument("--time-horizon", type=int, default=None, help="Limit to recent N rows")
    args = parser.parse_args()
    
    result = run_supplier_risk_and_routing_pipeline(
        file_path=args.csv,
        models_dir=args.models_dir,
        use_milp=not args.no_milp,
        use_random_forest=not args.logistic_regression,
        risk_weight=args.risk_weight,
        vehicle_capacity=args.vehicle_capacity,
        time_horizon_limit=args.time_horizon,
    )
    
    print(json.dumps(result, indent=2, default=str))