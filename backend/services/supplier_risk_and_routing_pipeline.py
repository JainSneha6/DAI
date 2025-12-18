# services/supplier_risk_and_routing_pipeline.py
# Logistics Supplier Risk & Routing Optimization pipeline (full file).
# - Supplier risk scoring using statistical heuristics, optional ML (sklearn), and optional survival modeling (lifelines).
# - Supplier allocation optimization via MILP (pulp) when available; otherwise deterministic heuristic.
# - Simple VRP routing heuristic (capacity-constrained nearest neighbor).
# - Aggregates demand, uses supplier capacities, costs, lead times, locations.
# - Persists chosen plan/artifact and metadata, returns ONLY JSON-serializable results.
#
# Improvements:
# - If demand_col detected but all values zero/NaN, fallback to unit demands (d_i=1.0).
# - Default capacity per supplier at least 1.0 to allow full unit assignments.
# - Enhanced logging for debugging.
# - Fixed date filtering to handle invalid dates better.
#
# Usage:
#   from services.supplier_risk_and_routing_pipeline import analyze_file_and_run_pipeline
#   analyze_file_and_run_pipeline(csv_path, gemini_response, models_dir="models")

import os
import math
import json
import pickle
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

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
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    from lifelines import CoxPHFitter
    _HAS_LIFELINES = True
except Exception:
    _HAS_LIFELINES = False

# ---- persistence helpers ----


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_model_artifact(obj: Any, model_name: str, models_dir: str = "models") -> str:
    _ensure_dir(models_dir)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"{model_name.replace(' ', '_')}_{ts}.pkl"
    path = os.path.join(models_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def save_metadata(metadata: Dict, model_filepath: str, models_dir: str = "models") -> str:
    _ensure_dir(models_dir)
    base = os.path.splitext(os.path.basename(model_filepath))[0]
    meta_filename = f"{base}.meta.json"
    meta_path = os.path.join(models_dir, meta_filename)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return meta_path


# ---- detection heuristics ----

COMMON_DATE_KEYWORDS = ["date", "datetime", "ds", "timestamp", "time"]
_SUPPLIER_KEYWORDS = ["supplier", "vendor", "vendor_id", "supplier_id", "vendor_code", "vendor_name",
                      "carrier", "carrier_id", "carrier_code", "carrier_name", "hauler"]
_CUSTOMER_KEYWORDS = ["customer", "cust", "order_id", "order", "shipment", "consignment"]
_DEMAND_KEYWORDS = ["demand", "quantity", "qty", "units", "order_qty", "demand_qty", "quantity_ordered",
                    "shipment_qty", "volume", "weight"]
_LEADTIME_KEYWORDS = ["lead_time", "leadtime", "lt", "lead-time"]
_CAPACITY_KEYWORDS = ["capacity", "max_capacity", "cap"]
_COST_KEYWORDS = ["cost", "unit_cost", "price", "shipping_cost", "order_cost", "fixed_order_cost"]
_LAT_KEYWORDS = ["lat", "latitude"]
_LON_KEYWORDS = ["lon", "lng", "long", "longitude"]
_ONTIME_KEYWORDS = ["on_time", "ontime", "delivered_on_time", "on_time_flag"]
_DEFECT_KEYWORDS = ["defect", "defective", "quality_issue", "num_defects", "rejects"]


def read_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def detect_columns_by_keywords(columns: List[str], keywords: List[str]) -> Optional[str]:
    low = [c.lower() for c in columns]
    for kw in keywords:
        for i, c in enumerate(low):
            if kw in c:
                return columns[i]
    return None


def detect_supplier_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _SUPPLIER_KEYWORDS)


def detect_customer_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _CUSTOMER_KEYWORDS)


def detect_demand_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _DEMAND_KEYWORDS)


def detect_date_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, COMMON_DATE_KEYWORDS)


def detect_leadtime_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _LEADTIME_KEYWORDS)


def detect_capacity_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _CAPACITY_KEYWORDS)


def detect_cost_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _COST_KEYWORDS)


def detect_lat_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _LAT_KEYWORDS)


def detect_lon_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _LON_KEYWORDS)


def detect_ontime_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _ONTIME_KEYWORDS)


def detect_defect_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _DEFECT_KEYWORDS)


# ---- utility: safe numeric conversion ----


def _to_float_safe(val: Any, default: float = 0.0) -> float:
    """
    Convert val (scalar, numpy scalar, pandas Series/ndarray or string) into a float.
    Returns default (0.0) for NaN/unconvertible values.
    """
    try:
        # If it's a pandas Series (e.g. loc[[i], col]) take first element
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

        # Numeric check first
        if isinstance(v, (int, float, np.integer, np.floating)):
            if pd.isna(v):
                return default
            return float(v)

        # Attempt to coerce to numeric
        coerced = pd.to_numeric(v, errors="coerce")
        if isinstance(coerced, (pd.Series, np.ndarray)):
            # pick first element
            try:
                coerced_val = coerced.item()  # works for length-1
            except Exception:
                coerced_val = coerced[0] if len(coerced) > 0 else np.nan
            coerced = coerced_val

        if pd.isna(coerced):
            return default
        return float(coerced)
    except Exception:
        return default


# ---- scoring helpers (statistical heuristics) ----


def compute_basic_supplier_stats(df: pd.DataFrame, supplier_col: str, demand_col: Optional[str], lead_time_col: Optional[str], ontime_col: Optional[str], defect_col: Optional[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute per-supplier basic stats:
     - total_supply (sum of demand)
     - avg_lead_time, std_lead_time
     - on_time_rate
     - defect_rate
    """
    suppliers = df[supplier_col].unique().tolist()
    stats: Dict[str, Dict[str, float]] = {}
    for s in suppliers:
        sub = df[df[supplier_col] == s]
        total = float(pd.to_numeric(sub[demand_col], errors="coerce").sum()) if demand_col and demand_col in sub.columns else float(len(sub))
        # lead time metrics
        if lead_time_col and lead_time_col in sub.columns:
            lt_vals = pd.to_numeric(sub[lead_time_col], errors="coerce").dropna()
            avg_lt = float(lt_vals.mean()) if not lt_vals.empty else float(np.nan)
            std_lt = float(lt_vals.std()) if not lt_vals.empty else float(np.nan)
        else:
            avg_lt = float(np.nan)
            std_lt = float(np.nan)
        # on-time
        if ontime_col and ontime_col in sub.columns:
            ontime_vals = pd.to_numeric(sub[ontime_col], errors="coerce").dropna()
            ontime_rate = float((ontime_vals == 1).sum() / len(ontime_vals)) if len(ontime_vals) > 0 else float(np.nan)
        else:
            ontime_rate = float(np.nan)
        # defect
        if defect_col and defect_col in sub.columns:
            defect_vals = pd.to_numeric(sub[defect_col], errors="coerce").dropna()
            defect_rate = float(defect_vals.sum() / len(defect_vals)) if len(defect_vals) > 0 else float(np.nan)
        else:
            defect_rate = float(np.nan)
        stats[s] = {
            "total_supply": total,
            "avg_lead_time": avg_lt,
            "std_lead_time": std_lt,
            "on_time_rate": ontime_rate,
            "defect_rate": defect_rate,
        }
    return stats


def scale_to_score(x: float, direction: str = "higher_better", clip: Tuple[float, float] = (0.0, 1.0)) -> float:
    """
    Simple scaling helper:
     - if x is nan -> 0.5 neutral
     - direction: 'higher_better' or 'lower_better'
     - maps to [clip_min, clip_max]
    """
    if x is None or (isinstance(x, float) and (np.isnan(x))):
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
    Combine basic stats into a risk score in [0,1], where 1 = highest risk.
    Heuristic: high lead_time std, low on_time_rate, high defect_rate -> higher risk.
    """
    scores: Dict[str, float] = {}
    for s, st in stats.items():
        std_lt = st.get("std_lead_time", float(np.nan))
        on_time = st.get("on_time_rate", float(np.nan))
        defect = st.get("defect_rate", float(np.nan))

        c_std = scale_to_score(std_lt, direction="lower_better")
        c_ont = scale_to_score(on_time, direction="higher_better")
        c_def = scale_to_score(defect, direction="lower_better")

        inv_ont = 1.0 - c_ont

        w_std, w_ont, w_def = 0.35, 0.35, 0.30
        risk_raw = w_std * c_std + w_ont * inv_ont + w_def * c_def
        scores[s] = float(max(0.0, min(1.0, risk_raw)))
    return scores


# ---- optional ML-based scorer (logistic regression) ----


def train_ml_risk_model(df: pd.DataFrame, supplier_col: str, demand_col: Optional[str], lead_time_col: Optional[str], ontime_col: Optional[str], defect_col: Optional[str], random_state: int = 42) -> Tuple[Optional[Any], Optional[Dict[str, float]]]:
    """
    Train a simple logistic regression to predict 'bad' deliveries.
    - labels derived from ontime/defect (if available). If neither available -> returns None.
    - returns (model, per-supplier_prob_of_bad) where higher prob -> higher risk
    """
    if not _HAS_SKLEARN:
        return None, None

    df_copy = df.copy()
    labels = None
    if ontime_col and ontime_col in df_copy.columns:
        on_time = pd.to_numeric(df_copy[ontime_col], errors="coerce")
        labels = (on_time == 0).astype(int)
    if (labels is None or labels.sum() == 0) and defect_col and defect_col in df_copy.columns:
        defect_vals = pd.to_numeric(df_copy[defect_col], errors="coerce").fillna(0)
        labels = (defect_vals > 0).astype(int)
    if labels is None:
        return None, None

    feats = pd.DataFrame(index=df_copy.index)
    if demand_col and demand_col in df_copy.columns:
        feats["demand"] = pd.to_numeric(df_copy[demand_col], errors="coerce").fillna(0.0)
    else:
        feats["demand"] = 1.0

    if lead_time_col and lead_time_col in df_copy.columns:
        feats["lead_time"] = pd.to_numeric(df_copy[lead_time_col], errors="coerce").fillna(feats["demand"].median() if "demand" in feats else 0.0)
    else:
        feats["lead_time"] = feats["demand"] * 0.0 + np.nan

    feats["supplier_code"] = pd.Categorical(df_copy[supplier_col]).codes

    feats = feats.fillna(feats.median().fillna(0.0))

    try:
        X_train, X_test, y_train, y_test = train_test_split(feats.values, labels.values, test_size=0.2, random_state=random_state, stratify=labels if labels.sum() > 0 else None)
    except Exception:
        X_train, X_test, y_train, y_test = feats.values, feats.values, labels.values, labels.values
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(random_state=random_state, max_iter=500)
    try:
        model.fit(X_train_s, y_train)
    except Exception:
        return None, None

    supplier_probs: Dict[str, float] = {}
    suppliers = df_copy[supplier_col].unique().tolist()
    for s in suppliers:
        idx = df_copy[df_copy[supplier_col] == s].index
        if len(idx) == 0:
            supplier_probs[s] = 0.5
            continue
        Xs = scaler.transform(feats.loc[idx].values)
        probs = model.predict_proba(Xs)[:, 1]
        supplier_probs[s] = float(np.mean(probs))
    return model, supplier_probs


# ---- survival model (robust fit) ----


def fit_survival_model(df: pd.DataFrame, supplier_col: str, date_col: Optional[str], lead_time_col: Optional[str], ontime_col: Optional[str]) -> Tuple[Optional[Any], Optional[Dict[str, float]]]:
    """
    Attempt to fit a CoxPH survival model with defensive preprocessing:
     - drop constant columns
     - use small penalizer to improve numerical stability
     - catch convergence errors and return (None, None) so pipeline falls back gracefully
    """
    if not _HAS_LIFELINES:
        return None, None

    if ontime_col is None and lead_time_col is None and date_col is None:
        return None, None

    df_copy = df.copy()

    # duration
    if lead_time_col and lead_time_col in df_copy.columns:
        df_copy["duration"] = pd.to_numeric(df_copy[lead_time_col], errors="coerce")
    elif date_col and date_col in df_copy.columns:
        try:
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
            df_copy = df_copy.sort_values([supplier_col, date_col])
            df_copy["duration"] = df_copy.groupby(supplier_col)[date_col].diff().dt.days.fillna(0)
        except Exception:
            df_copy["duration"] = 0.0
    else:
        df_copy["duration"] = 1.0

    # event: late deliveries
    if ontime_col and ontime_col in df_copy.columns:
        df_copy["event"] = (pd.to_numeric(df_copy[ontime_col], errors="coerce") == 0).astype(int)
    else:
        df_copy["event"] = 0

    # covariates
    covars = pd.DataFrame(index=df_copy.index)
    if lead_time_col and lead_time_col in df_copy.columns:
        covars["lead_time"] = pd.to_numeric(df_copy[lead_time_col], errors="coerce").fillna(0.0)
    else:
        covars["lead_time"] = 0.0

    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    for dropc in ["duration", "event"]:
        if dropc in numeric_cols:
            numeric_cols.remove(dropc)
    for c in numeric_cols:
        if c not in covars.columns and c != lead_time_col:
            covars["num_" + str(c)] = pd.to_numeric(df_copy[c], errors="coerce").fillna(0.0)
            break

    covars["supplier_code"] = pd.Categorical(df_copy[supplier_col]).codes

    cph_df = pd.concat([df_copy[["duration", "event"]], covars], axis=1).dropna()

    # drop constant columns
    nunique = cph_df.nunique()
    keep_cols = nunique[nunique > 1].index.tolist()
    cph_df = cph_df[keep_cols]

    if cph_df.empty or ("duration" not in cph_df.columns) or ("event" not in cph_df.columns):
        return None, None

    try:
        cph = CoxPHFitter(penalizer=0.1)
    except TypeError:
        cph = CoxPHFitter()

    try:
        cph.fit(cph_df, duration_col="duration", event_col="event", show_progress=False)
    except Exception as e:
        logger.exception("Survival model fit failed: %s", e)
        return None, None

    suppliers = df_copy[supplier_col].unique().tolist()
    hazard_scores: Dict[str, float] = {}
    for s in suppliers:
        sub_idx = df_copy[df_copy[supplier_col] == s].index
        if len(sub_idx) == 0:
            hazard_scores[s] = 0.5
            continue
        sub_df = cph_df.loc[cph_df.index.intersection(sub_idx)]
        if sub_df.empty:
            hazard_scores[s] = 0.5
            continue
        ph = cph.predict_partial_hazard(sub_df).mean()
        hazard_scores[s] = float(ph)

    vals = np.array(list(hazard_scores.values()), dtype=float)
    if np.nanmax(vals) - np.nanmin(vals) > 0:
        norm = (vals - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals))
        for i, s in enumerate(list(hazard_scores.keys())):
            hazard_scores[s] = float(norm[i])
    else:
        for s in hazard_scores.keys():
            hazard_scores[s] = 0.5

    return cph, hazard_scores


# ---- optimization: supplier allocation MILP (with pulp) ----


def solve_supplier_allocation_mip(
    demand_df: pd.DataFrame,
    supplier_list: List[str],
    supplier_capacity: Dict[str, float],
    supplier_unit_cost: Dict[str, float],
    supplier_risk_score: Dict[str, float],
    demand_col: Optional[str],
    supplier_col: str,
    customer_col: Optional[str] = None,
    max_assign_per_customer: Optional[int] = None,
    integer: bool = True,
    risk_weight: float = 1.0,
    timeout: Optional[int] = 30,
) -> Dict[str, Any]:
    """
    MILP formulation:
     - decision x[i,s] = units of demand row i assigned to supplier s
     - constraints:
         sum_s x[i,s] == demand_i  (all demand satisfied)
         sum_i x[i,s] <= cap_s      (capacity)
     - objective: minimize sum_i,s (unit_cost_s * x[i,s] + risk_weight * risk_s * x[i,s])
    Returns allocation per supplier {supplier: {row_idx: qty}}, and solver status.
    Requires pulp.
    """
    if not _HAS_PULP:
        raise RuntimeError("pulp not installed - MILP solver unavailable")

    demand_df = demand_df.reset_index(drop=True)
    N = demand_df.shape[0]
    S = supplier_list
    prob = pl.LpProblem("supplier_allocation", pl.LpMinimize)
    cat = "Integer" if integer else "Continuous"
    x = pl.LpVariable.dicts("x", (list(range(N)), S), lowBound=0, cat=cat)

    # objective using safe numeric conversion
    obj_terms = []
    for i in range(N):
        d_i = 1.0 if demand_col is None else _to_float_safe(demand_df.loc[i, demand_col], default=0.0)
        for s in S:
            unit = float(supplier_unit_cost.get(s, 0.0))
            risk = float(supplier_risk_score.get(s, 0.0))
            obj_terms.append((unit + risk_weight * risk) * x[i][s])
    prob += pl.lpSum(obj_terms)

    # constraints: satisfy demand for each row
    for i in range(N):
        d_i = 1.0 if demand_col is None else _to_float_safe(demand_df.loc[i, demand_col], default=0.0)
        if d_i > 0:  # Skip zero demands
            prob += pl.lpSum([x[i][s] for s in S]) == d_i

    # capacity constraints
    for s in S:
        cap = float(supplier_capacity.get(s, float("inf")))
        prob += pl.lpSum([x[i][s] for i in range(N)]) <= cap

    # solve
    solver = pl.PULP_CBC_CMD(msg=False, timeLimit=timeout)
    status = prob.solve(solver)
    status_str = pl.LpStatus[status]
    logger.info("Supplier allocation MILP status: %s", status_str)

    allocation: Dict[str, Dict[int, float]] = {s: {} for s in S}
    for i in range(N):
        for s in S:
            try:
                q = float(pl.value(x[i][s]) or 0.0)
            except Exception:
                q = 0.0
            if q > 1e-8:
                allocation[s].setdefault(int(i), 0.0)
                allocation[s][int(i)] += q
    return {"status": status_str, "allocation": allocation}


# ---- heuristic fallback allocator ----


def heuristic_supplier_allocator(
    demand_df: pd.DataFrame,
    supplier_list: List[str],
    supplier_capacity: Dict[str, float],
    supplier_unit_cost: Dict[str, float],
    supplier_risk_score: Dict[str, float],
    demand_col: Optional[str],
) -> Dict[str, Any]:
    """
    Greedy assign each demand row to the supplier with minimal (unit_cost + risk_score) that has remaining capacity.
    If none have capacity, assign proportionally ignoring capacity (best-effort).
    Returns allocation per supplier {supplier: {row_idx: qty}}
    """
    allocation: Dict[str, Dict[int, float]] = {s: {} for s in supplier_list}
    remaining_cap = {s: float(supplier_capacity.get(s, float("inf"))) for s in supplier_list}
    score = {s: float(supplier_unit_cost.get(s, 0.0)) + float(supplier_risk_score.get(s, 0.0)) for s in supplier_list}
    for i in range(demand_df.shape[0]):
        d_i = 1.0 if demand_col is None else _to_float_safe(demand_df.iloc[i][demand_col], default=0.0)
        if d_i <= 0:
            continue  # Skip zero demands
        ordered = sorted(supplier_list, key=lambda s: score[s])
        assigned = False
        for s in ordered:
            if remaining_cap[s] >= d_i:
                allocation[s].setdefault(int(i), 0.0)
                allocation[s][int(i)] += d_i
                remaining_cap[s] -= d_i
                assigned = True
                break
        if not assigned:
            pos_caps = {s: remaining_cap[s] for s in supplier_list if remaining_cap[s] > 0}
            if pos_caps:
                total_pos = sum(pos_caps.values())
                for s, cap in pos_caps.items():
                    frac = cap / total_pos if total_pos > 0 else 0
                    q = d_i * frac
                    if q > 1e-8:
                        allocation[s].setdefault(int(i), 0.0)
                        allocation[s][int(i)] += q
                        remaining_cap[s] -= q
            else:
                s0 = ordered[0]
                allocation[s0].setdefault(int(i), 0.0)
                allocation[s0][int(i)] += d_i
    return {"status": "heuristic_allocated", "allocation": allocation}


# ---- simple VRP / routing heuristic (nearest neighbor clusters) ----


def haversine_distance(lat1, lon1, lat2, lon2):
    # approximate haversine distance in km
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
    customer_loc_cols: Tuple[Optional[str], Optional[str]],
    supplier_locations: Dict[str, Tuple[float, float]],
    vehicle_capacity: float = 100.0,
    max_vehicles_per_supplier: int = 10,
) -> Dict[str, List[List[int]]]:
    """
    For each supplier, build routes (list of customer row indices) using a capacity-constrained nearest neighbor heuristic.
    If no location data, create single-customer routes.
    Returns routes per supplier: {supplier: [ [row_idx, ...], [...], ... ]}
    """
    lat_col, lon_col = customer_loc_cols if isinstance(customer_loc_cols, (list, tuple)) else (None, None)
    has_locations = lat_col is not None and lon_col is not None and lat_col in demand_df.columns and lon_col in demand_df.columns
    routes_per_supplier: Dict[str, List[List[int]]] = {}

    for s, alloc in allocation.items():
        rows = list(alloc.keys())
        if not rows:
            routes_per_supplier[s] = []
            continue

        if not has_locations:
            # No locations: single routes per customer
            routes_per_supplier[s] = [[int(r)] for r in sorted(rows)]
            continue

        nodes = []
        for r in rows:
            try:
                lat = float(demand_df.loc[r, lat_col])
                lon = float(demand_df.loc[r, lon_col])
            except Exception:
                lat, lon = (0.0, 0.0)
            q = float(alloc.get(r, 0.0))
            nodes.append({"row": int(r), "demand": q, "lat": lat, "lon": lon})

        s_loc = supplier_locations.get(s, (0.0, 0.0))
        remaining_nodes = nodes.copy()
        routes = []
        vehicles_used = 0
        while remaining_nodes and vehicles_used < max_vehicles_per_supplier:
            cap_left = vehicle_capacity
            route = []
            cur_lat, cur_lon = s_loc
            while remaining_nodes and cap_left > 0:
                feasible = [n for n in remaining_nodes if n["demand"] <= cap_left]
                if not feasible:
                    nearest = min(remaining_nodes, key=lambda n: haversine_distance(cur_lat, cur_lon, n["lat"], n["lon"]))
                    serve = min(nearest["demand"], cap_left)
                    if serve <= 0:
                        break
                    route.append(int(nearest["row"]))
                    nearest["demand"] -= serve
                    cap_left -= serve
                    if nearest["demand"] <= 1e-8:
                        remaining_nodes.remove(nearest)
                    cur_lat, cur_lon = nearest["lat"], nearest["lon"]
                else:
                    nearest = min(feasible, key=lambda n: haversine_distance(cur_lat, cur_lon, n["lat"], n["lon"]))
                    route.append(int(nearest["row"]))
                    cap_left -= nearest["demand"]
                    remaining_nodes.remove(nearest)
                    cur_lat, cur_lon = nearest["lat"], nearest["lon"]
            if route:
                routes.append(route)
            vehicles_used += 1

        if remaining_nodes:
            leftover_route = [int(n["row"]) for n in remaining_nodes]
            routes.append(leftover_route)
        routes_per_supplier[s] = routes
    return routes_per_supplier


# ---- JSON sanitize helper ----


def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert objects that json can't handle into JSON-friendly types.
    """
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


# ---- orchestration ----


def run_supplier_risk_and_routing_pipeline(
    file_path: str,
    gemini_analysis: Optional[Dict] = None,
    supplier_col_hint: Optional[str] = None,
    demand_col_hint: Optional[str] = None,
    date_col_hint: Optional[str] = None,
    default_unit_cost: float = 1.0,
    default_capacity: float = 1e9,
    default_lead_time: int = 0,
    use_milp: bool = True,
    models_dir: str = "models",
    risk_weight: float = 10.0,
    vehicle_capacity: float = 100.0,
    time_horizon_limit: Optional[int] = None,
    milp_timeout_seconds: int = 30,
) -> Dict[str, Any]:
    """
    Top-level pipeline:
     - Detect columns
     - Compute stats per supplier
     - Optionally train ML & survival models for risk
     - Compute combined supplier risk score
     - Solve allocation via MILP or heuristic
     - Compute VRP-style routes per supplier (heuristic)
     - Persist artifacts and metadata
    """
    df = read_csv(file_path)
    columns = list(df.columns)

    supplier_col = supplier_col_hint or detect_supplier_column(columns)
    demand_col = demand_col_hint or detect_demand_column(columns)
    date_col = date_col_hint or detect_date_column(columns)
    lead_time_col = detect_leadtime_column(columns)
    capacity_col = detect_capacity_column(columns)
    cost_col = detect_cost_column(columns)
    lat_col = detect_lat_column(columns)
    lon_col = detect_lon_column(columns)
    ontime_col = detect_ontime_column(columns)
    defect_col = detect_defect_column(columns)
    customer_col = detect_customer_column(columns)

    if not supplier_col:
        raise ValueError("Could not detect supplier column. Provide hints or ensure file has supplier/carrier columns.")

    # Check if demand_col is useful
    if demand_col:
        numeric_demands = pd.to_numeric(df[demand_col], errors='coerce').dropna()
        if len(numeric_demands) == 0 or numeric_demands.sum() == 0:
            logger.info("Demand column detected but all zero/NaN, treating as unit demands (no demand_col)")
            demand_col = None

    suppliers = df[supplier_col].unique().tolist()
    logger.info(f"Detected {len(suppliers)} unique suppliers")

    # Set historical attributes from full df
    supplier_unit_cost: Dict[str, float] = {}
    supplier_lead_time: Dict[str, int] = {}
    supplier_locations: Dict[str, Tuple[float, float]] = {}

    for s in suppliers:
        supplier_unit_cost[s] = float(df.loc[df[supplier_col] == s, cost_col].dropna().iloc[0]) if cost_col and not df.loc[df[supplier_col] == s, cost_col].dropna().empty else float(default_unit_cost)
        lt_val = int(df.loc[df[supplier_col] == s, lead_time_col].dropna().iloc[0]) if lead_time_col and not df.loc[df[supplier_col] == s, lead_time_col].dropna().empty else int(default_lead_time)
        supplier_lead_time[s] = max(0, lt_val)
        try:
            lat = float(df.loc[df[supplier_col] == s, lat_col].dropna().iloc[0]) if lat_col and not df.loc[df[supplier_col] == s, lat_col].dropna().empty else 0.0
            lon = float(df.loc[df[supplier_col] == s, lon_col].dropna().iloc[0]) if lon_col and not df.loc[df[supplier_col] == s, lon_col].dropna().empty else 0.0
        except Exception:
            lat, lon = 0.0, 0.0
        supplier_locations[s] = (lat, lon)

    # Filter demand_df for "next" demands: most recent
    demand_df = df.copy()
    original_len = len(demand_df)
    if time_horizon_limit:
        if date_col and date_col in demand_df.columns:
            try:
                demand_df[date_col] = pd.to_datetime(demand_df[date_col], errors="coerce")
                valid_dates = demand_df.dropna(subset=[date_col])
                if not valid_dates.empty:
                    demand_df = valid_dates.sort_values(date_col).tail(time_horizon_limit)
                else:
                    logger.warning("No valid dates, falling back to tail")
                    demand_df = demand_df.tail(time_horizon_limit)
                logger.info(f"Filtered to {len(demand_df)} recent rows using date column '{date_col}'")
            except Exception as e:
                logger.warning(f"Date filtering failed: {e}, falling back to tail({time_horizon_limit})")
                demand_df = demand_df.tail(time_horizon_limit)
        else:
            demand_df = demand_df.tail(time_horizon_limit)
            logger.info(f"Filtered to {len(demand_df)} recent rows (no date column)")
    else:
        logger.info(f"No time horizon limit, using all {len(demand_df)} rows")

    if len(demand_df) == 0:
        logger.warning(f"Demand DF empty after filtering (original: {original_len}). Falling back to full DF.")
        demand_df = df.copy()

    # Compute total future demand for default capacity
    total_demand = 0.0
    for i in range(len(demand_df)):
        d_i = 1.0 if demand_col is None else _to_float_safe(demand_df.iloc[i][demand_col], default=0.0)
        total_demand += d_i
    default_cap = max(1.0, total_demand / len(suppliers)) if suppliers else 1.0  # Ensure at least 1 unit per supplier
    logger.info(f"Total demand for allocation: {total_demand}, default cap per supplier: {default_cap}")

    # Set capacities (historical if available, else balanced default)
    supplier_capacity: Dict[str, float] = {}
    for s in suppliers:
        if capacity_col and not df.loc[df[supplier_col] == s, capacity_col].dropna().empty:
            cap_val = float(df.loc[df[supplier_col] == s, capacity_col].dropna().iloc[0])
            supplier_capacity[s] = max(1.0, cap_val)  # Ensure min 1
            logger.info(f"Using historical capacity for {s}: {cap_val}")
        else:
            supplier_capacity[s] = default_cap
            logger.info(f"Using default capacity for {s}: {default_cap}")

    stats = compute_basic_supplier_stats(df, supplier_col, demand_col, lead_time_col, ontime_col, defect_col)
    stat_risk = compute_statistical_risk_scores(stats)

    ml_model, ml_probs = train_ml_risk_model(df, supplier_col, demand_col, lead_time_col, ontime_col, defect_col)
    ml_used = ml_model is not None and ml_probs is not None

    surv_model, surv_scores = fit_survival_model(df, supplier_col, date_col, lead_time_col, ontime_col)
    surv_used = surv_model is not None and surv_scores is not None

    combined_risk: Dict[str, float] = {}
    for s in suppliers:
        parts = []
        parts.append(stat_risk.get(s, 0.5))
        if ml_used:
            parts.append(ml_probs.get(s, 0.5))
        if surv_used:
            parts.append(surv_scores.get(s, 0.5))
        combined = float(np.nanmean(parts))
        combined_risk[s] = float(max(0.0, min(1.0, combined)))

    allocation_result = None
    solver_used = None
    solver_status = None
    allocation = None

    if use_milp and _HAS_PULP:
        try:
            allocation_result = solve_supplier_allocation_mip(
                demand_df=demand_df,
                supplier_list=suppliers,
                supplier_capacity=supplier_capacity,
                supplier_unit_cost=supplier_unit_cost,
                supplier_risk_score=combined_risk,
                demand_col=demand_col,
                supplier_col=supplier_col,
                customer_col=customer_col,
                integer=False,
                risk_weight=risk_weight,
                timeout=milp_timeout_seconds,
            )
            solver_used = "MILP_pulp"
            solver_status = allocation_result.get("status")
            allocation = allocation_result.get("allocation")
            logger.info(f"MILP allocation: {sum(len(v) for v in allocation.values())} assignments")
        except Exception as e:
            logger.exception("MILP failed, falling back to heuristic: %s", e)
            allocation_result = None
            solver_used = "MILP_failed_fallback_heuristic"
            solver_status = str(e)

    if allocation_result is None or not allocation:
        heuristic_alloc = heuristic_supplier_allocator(
            demand_df=demand_df,
            supplier_list=suppliers,
            supplier_capacity=supplier_capacity,
            supplier_unit_cost=supplier_unit_cost,
            supplier_risk_score=combined_risk,
            demand_col=demand_col,
        )
        allocation = heuristic_alloc.get("allocation")
        solver_status = heuristic_alloc.get("status")
        solver_used = "heuristic_allocator"
        logger.info(f"Heuristic allocation: {sum(len(v) for v in allocation.values())} assignments")

    # compute VRP routes per supplier if lat/lon for customers exist
    if lat_col and lon_col and lat_col in demand_df.columns and lon_col in demand_df.columns:
        customer_loc_cols = (lat_col, lon_col)
    else:
        customer_loc_cols = (None, None)

    try:
        routes = simple_vrp_routes(allocation, demand_df, customer_loc_cols, supplier_locations, vehicle_capacity=vehicle_capacity)
        logger.info(f"Generated routes for {sum(len(r) for r in routes.values())} vehicles")
    except Exception as e:
        logger.exception("VRP failed: %s", e)
        routes = {s: [] for s in suppliers}

    results: Dict[str, Any] = {
        "success": True,
        "suppliers": suppliers,
        "stats": stats,
        "stat_risk": stat_risk,
        "ml_used": bool(ml_used),
        "surv_used": bool(surv_used),
        "ml_probs": ml_probs if ml_used else None,
        "surv_scores": surv_scores if surv_used else None,
        "combined_risk": combined_risk,
        "supplier_unit_cost": supplier_unit_cost,
        "supplier_capacity": supplier_capacity,
        "supplier_lead_time": supplier_lead_time,
        "supplier_locations": supplier_locations,
        "solver_used": solver_used,
        "solver_status": solver_status,
        "allocation": allocation,
        "routes": routes,
    }

    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model": "logistics & supplier risk model",
        "solver_used": solver_used,
        "suppliers": suppliers,
        "risk_weight": risk_weight,
        "vehicle_capacity": vehicle_capacity,
        "input_file": os.path.basename(file_path),
    }
    try:
        artifact_obj = {
            "results": results,
            "metadata": metadata,
        }
        saved_model_path = save_model_artifact(artifact_obj, "Supplier_Risk_Routing_Plan", models_dir=models_dir)
        saved_meta_path = save_metadata(metadata, saved_model_path, models_dir=models_dir)
        results["artifact"] = {"model_path": saved_model_path, "meta_path": saved_meta_path}
    except Exception as e:
        logger.exception("Failed to persist artifact: %s", e)
        results["artifact_persist_error"] = str(e)

    try:
        results = _sanitize_for_json(results)
    except Exception:
        results = {"success": False, "error": "Failed to sanitize results for JSON. Check logs."}

    return results


def analyze_file_and_run_pipeline(file_path: str, gemini_response: Dict, models_dir: str = "models", **kwargs) -> Dict:
    """
    Entry point expected by gemini_analyser. Runs this pipeline only if gemini_response.analysis.model_type
    indicates "logistics & supplier risk model" (case-insensitive).
    """
    try:
        model_type = gemini_response.get("analysis", {}).get("model_type")
    except Exception:
        model_type = None

    if model_type and model_type.strip().lower() == "logistics & supplier risk model":
        pipeline_result = run_supplier_risk_and_routing_pipeline(file_path, gemini_analysis=gemini_response, models_dir=models_dir, **kwargs)
        return {"success": True, "gemini_analysis": gemini_response, "pipeline": pipeline_result}
    else:
        return {"success": False, "error": "Gemini did not indicate logistics & supplier risk model in strict mode."}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--gemini-json", required=False, help="Path to gemini analysis json (optional)")
    parser.add_argument("--models-dir", required=False, default="models", help="Directory to save models/artifacts")
    parser.add_argument("--no-milp", action="store_true", help="Disable MILP solver even if pulp is installed")
    parser.add_argument("--vehicle-capacity", required=False, type=float, default=100.0, help="Vehicle capacity for VRP heuristic")
    parser.add_argument("--risk-weight", required=False, type=float, default=10.0, help="Risk weight in allocation objective")
    parser.add_argument("--time-horizon", required=False, type=int, default=None, help="Limit time horizon (rows) to consider")
    parser.add_argument("--milp-timeout", required=False, type=int, default=30, help="MILP solver timeout seconds")
    args = parser.parse_args()

    gemini_resp = None
    if args.gemini_json and os.path.exists(args.gemini_json):
        with open(args.gemini_json, "r", encoding="utf-8") as f:
            gemini_resp = json.load(f)

    out = analyze_file_and_run_pipeline(
        args.csv,
        gemini_resp or {},
        models_dir=args.models_dir,
        use_milp=not args.no_milp,
        vehicle_capacity=args.vehicle_capacity,
        risk_weight=args.risk_weight,
        time_horizon_limit=args.time_horizon,
        milp_timeout_seconds=args.milp_timeout,
    )
    print(json.dumps(out, indent=2, default=str))