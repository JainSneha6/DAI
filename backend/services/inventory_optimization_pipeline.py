# services/inventory_optimization_pipeline.py
# STRICT-but-robust Inventory & Replenishment Optimization pipeline (full file).
# - Implements EOQ, POQ heuristics for inventory optimization per SKU
# - Uses Linear Programming (LP) and Mixed-Integer Linear Programming (MILP) for replenishment scheduling
#   via pulp if available. If pulp is not installed, falls back to a deterministic heuristic scheduler.
# - Deterministic preprocessing, per-SKU demand aggregation by period (daily/weekly/monthly).
# - Persists chosen plan/artifact and metadata, returns ONLY JSON-serializable results.
#
# Usage:
#   from services.inventory_optimization_pipeline import analyze_file_and_run_pipeline
#   analyze_file_and_run_pipeline(csv_path, gemini_response, models_dir="models")
#
# Notes:
# - The pipeline expects a CSV that contains at minimum:
#     * SKU identifier column (detected heuristically)
#     * Demand / quantity consumed column (detected heuristically)
#     * Optional date/time column to aggregate demand into periods
#     * Optional cost columns: holding_cost_per_unit_per_period, ordering_cost_per_order, unit_cost
#     * Optional lead_time column (in periods)
# - If pulp is not available the LP/MILP solvers won't run; the pipeline still runs EOQ/POQ + heuristic scheduling.

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

# Optional pulp dependency for LP / MILP
try:
    import pulp as pl

    _HAS_PULP = True
except Exception:
    _HAS_PULP = False

# ---- persistence helpers (same style as other pipelines) ----


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
_SKEWED_SKU_KEYWORDS = ["sku", "item", "product", "part", "asin", "sku_id", "product_id"]
_DEMAND_KEYWORDS = ["demand", "quantity", "qty", "sales", "ship", "units", "consumed", "demand_qty"]
_LEADTIME_KEYWORDS = ["lead_time", "leadtime", "lt", "lead-time"]
_ORDER_COST_KEYWORDS = ["ordering_cost", "order_cost", "fixed_order_cost", "co"]
_HOLDING_COST_KEYWORDS = ["holding_cost", "carry_cost", "holding", "carry"]
_UNIT_COST_KEYWORDS = ["unit_cost", "cost", "unitprice", "unit_price", "price"]


def read_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def detect_columns_by_keywords(columns: List[str], keywords: List[str]) -> Optional[str]:
    low = [c.lower() for c in columns]
    for kw in keywords:
        for i, c in enumerate(low):
            if kw in c:
                return columns[i]
    return None


def detect_sku_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _SKEWED_SKU_KEYWORDS)


def detect_demand_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _DEMAND_KEYWORDS)


def detect_date_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, COMMON_DATE_KEYWORDS)


def detect_leadtime_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _LEADTIME_KEYWORDS)


def detect_order_cost_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _ORDER_COST_KEYWORDS)


def detect_holding_cost_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _HOLDING_COST_KEYWORDS)


def detect_unit_cost_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _UNIT_COST_KEYWORDS)


# ---- EOQ / POQ formulas ----


def compute_eoq(D: float, Co: float, Ch: float) -> float:
    """
    Economic Order Quantity: EOQ = sqrt(2 * D * Co / Ch)
    D: demand rate (units per period)
    Co: ordering cost per order
    Ch: holding cost per unit per period
    """
    if D <= 0 or Co <= 0 or Ch <= 0:
        return 0.0
    return math.sqrt((2.0 * D * Co) / Ch)


def compute_poq(D: float, Co: float, Ch: float, p: Optional[float]) -> float:
    """
    Production Order Quantity (POQ) when production rate p > demand rate D:
      POQ = sqrt( (2 * D * Co) / (Ch * (1 - D/p)) )
    If p is None or <= D, returns EOQ fallback.
    """
    if p is None or p <= D or D <= 0 or Co <= 0 or Ch <= 0:
        return compute_eoq(D, Co, Ch)
    denom = Ch * (1.0 - (D / p))
    if denom <= 0:
        return compute_eoq(D, Co, Ch)
    return math.sqrt((2.0 * D * Co) / denom)


# ---- demand aggregation helpers ----


def aggregate_demand(df: pd.DataFrame, date_col: Optional[str], sku_col: str, demand_col: str, freq: Optional[str] = None) -> Tuple[pd.DataFrame, List[Any]]:
    """
    Returns a DataFrame with index = period (dates or integer period) and columns = SKUs containing demand per period.
    If date_col is present and freq provided (e.g., 'W', 'M', 'D'), aggregates by that frequency.
    Otherwise uses the row order as periods.
    """
    if date_col and date_col in df.columns:
        # try to parse dates
        try:
            dts = pd.to_datetime(df[date_col], errors="coerce")
            if dts.isna().all():
                # fallback to ordered
                raise ValueError("date parse failed")
            df_copy = df.copy()
            df_copy[date_col] = dts
            if not freq:
                # choose weekly if span > 90 days else daily
                span_days = (dts.max() - dts.min()).days
                freq = "W" if span_days > 90 else "D"
            grouped = df_copy.groupby([pd.Grouper(key=date_col, freq=freq), sku_col])[demand_col].sum().reset_index()
            pivot = grouped.pivot_table(index=date_col, columns=sku_col, values=demand_col, aggfunc="sum", fill_value=0.0)
            pivot = pivot.sort_index()
            periods = list(pivot.index.astype(str))
            return pivot, periods
        except Exception:
            # fallback below
            pass

    # fallback: treat each row as a period; aggregate by row index
    df_copy = df.copy()
    df_copy["_period_idx"] = df_copy.index
    grouped = df_copy.groupby(["_period_idx", sku_col])[demand_col].sum().reset_index()
    pivot = grouped.pivot_table(index="_period_idx", columns=sku_col, values=demand_col, aggfunc="sum", fill_value=0.0)
    pivot = pivot.sort_index()
    periods = list(map(int, pivot.index.tolist()))
    return pivot, periods


# ---- LP / MILP setup ----


def solve_replenishment_mip(
    demand_pivot: pd.DataFrame,
    holding_costs: Dict[str, float],
    ordering_costs: Dict[str, float],
    unit_costs: Optional[Dict[str, float]] = None,
    lead_times: Optional[Dict[str, int]] = None,
    max_inventory: Optional[Dict[str, float]] = None,
    integer: bool = True,
    time_horizon_limit: Optional[int] = None,
    timeout: Optional[int] = 30,
) -> Dict[str, Any]:
    """
    Solve a replenishment planning MILP:
    - decision vars: order_qty[s,t] (>=0), order_binary[s,t] (if integer=True)
    - inv[s,t] inventory balance constraints:
         inv[s,t] = inv[s,t-1] + order_qty[s,t] - demand[s,t]
      with inv >= 0
    - objective: minimize sum_t sum_s (holding_cost * inv[s,t] + ordering_cost * order_binary[s,t] + unit_cost * order_qty[s,t])
    Returns a JSON-serializable schedule: {sku: {period_idx: qty, ...}, ...}
    Requires pulp installed; otherwise raises RuntimeError.
    """

    if not _HAS_PULP:
        raise RuntimeError("pulp not installed - MILP solver unavailable")

    # dims
    SKUS = list(demand_pivot.columns)
    T = demand_pivot.shape[0]
    if time_horizon_limit:
        T = min(T, time_horizon_limit)
        demand_pivot = demand_pivot.iloc[:T, :]

    # create problem
    prob = pl.LpProblem("inventory_replenishment_mip", pl.LpMinimize)

    # variables
    order_qty = pl.LpVariable.dicts("Q", (SKUS, list(range(T))), lowBound=0, cat="Integer" if integer else "Continuous")
    inv = pl.LpVariable.dicts("I", (SKUS, list(range(T))), lowBound=0, cat="Continuous")
    order_bin = None
    BIG_M = max(1.0, float(demand_pivot.values.max()) * 10.0)
    if integer:
        order_bin = pl.LpVariable.dicts("Y", (SKUS, list(range(T))), lowBound=0, upBound=1, cat="Binary")

    # objective terms
    obj_terms = []
    for s in SKUS:
        for t in range(T):
            h_cost = float(holding_costs.get(s, 0.0))
            o_cost = float(ordering_costs.get(s, 0.0))
            u_cost = float((unit_costs or {}).get(s, 0.0))
            obj_terms.append(h_cost * inv[s][t])
            obj_terms.append(u_cost * order_qty[s][t])
            if integer:
                obj_terms.append(o_cost * order_bin[s][t])
            else:
                # ordering cost is ignored for continuous fallback (approximation)
                pass
    prob += pl.lpSum(obj_terms)

    # constraints
    for s in SKUS:
        for t in range(T):
            demand_val = float(demand_pivot.iloc[t][s]) if s in demand_pivot.columns else 0.0
            if t == 0:
                # initial inventory assume 0
                prob += inv[s][t] == order_qty[s][t] - demand_val
            else:
                prob += inv[s][t] == inv[s][t - 1] + order_qty[s][t] - demand_val

            if max_inventory and s in max_inventory:
                prob += inv[s][t] <= float(max_inventory[s])

            if integer:
                # link order_qty and binary: order_qty <= BIG_M * y
                prob += order_qty[s][t] <= BIG_M * order_bin[s][t]

    # Solve
    solver = pl.PULP_CBC_CMD(msg=False, timeLimit=timeout)
    status = prob.solve(solver)

    status_str = pl.LpStatus.get(status, "Unknown")
    logger.info("MILP solve status: %s", status_str)

    # build schedule
    schedule: Dict[str, Dict[int, float]] = {s: {} for s in SKUS}
    inv_matrix: Dict[str, List[float]] = {s: [] for s in SKUS}
    for s in SKUS:
        for t in range(T):
            qval = float(pl.value(order_qty[s][t]) or 0.0)
            ival = float(pl.value(inv[s][t]) or 0.0)
            if qval > 1e-8:
                schedule[s][int(t)] = qval
            inv_matrix[s].append(ival)

    return {"status": status_str, "schedule": schedule, "inventory_projection": inv_matrix}


# ---- heuristic scheduler fallback ----


def heuristic_replenishment_scheduler(
    demand_pivot: pd.DataFrame,
    eoq_map: Dict[str, float],
    lead_times: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Greedy heuristic:
      - For each SKU, simulate periods; when projected inventory to next period goes below zero,
        place an order equal to EOQ (or remaining demand) with lead_time applied.
    Returns schedule: {sku: {period: qty}} and inventory projection.
    """
    SKUS = list(demand_pivot.columns)
    T = demand_pivot.shape[0]
    schedule: Dict[str, Dict[int, float]] = {s: {} for s in SKUS}
    inv_proj: Dict[str, List[float]] = {s: [0.0] * T for s in SKUS}
    lead_times = lead_times or {}

    for s in SKUS:
        inv = 0.0
        pending_orders: Dict[int, float] = {}  # arrival_period -> qty
        for t in range(T):
            # receive orders arriving this period
            if t in pending_orders:
                inv += pending_orders.pop(t)

            demand_val = float(demand_pivot.iloc[t][s])
            inv -= demand_val
            inv_proj[s][t] = inv

            if inv < 0:
                # need order; order arrives after lead_time (in periods)
                lt = int(lead_times.get(s, 0))
                arrival = min(T - 1, t + lt)
                q = float(eoq_map.get(s, 0.0))
                if q <= 0:
                    # fallback: order exactly missing demand for next period
                    q = max(0.0, demand_val - inv)
                # schedule
                schedule[s].setdefault(int(arrival), 0.0)
                schedule[s][int(arrival)] += float(q)
                # assume it will arrive and remedy current negative inventory for projection
                inv += q
                inv_proj[s][t] = inv

    return {"status": "heuristic_applied", "schedule": schedule, "inventory_projection": inv_proj}


# ---- JSON sanitize helper (solve Timestamp / numpy scalar keys/values) ----


def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert objects that json can't handle into JSON-friendly types:
     - pandas/numpy scalar -> python scalar via .item()
     - pandas Timestamp / numpy.datetime64 -> ISO string
     - dict keys that are Timestamp/numpy.datetime64 -> str(...)
     - numpy ints/floats -> python ints/floats
    """
    # dict => sanitize keys and values
    if isinstance(obj, dict):
        new: Dict[Any, Any] = {}
        for k, v in obj.items():
            # sanitize key
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
    # list/tuple
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_sanitize_for_json(v) for v in obj)
    # pandas / numpy scalars & timestamps
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


def run_inventory_optimization_pipeline(
    file_path: str,
    gemini_analysis: Optional[Dict] = None,
    sku_col_hint: Optional[str] = None,
    demand_col_hint: Optional[str] = None,
    date_col_hint: Optional[str] = None,
    period_freq: Optional[str] = None,
    default_order_cost: float = 50.0,
    default_holding_cost: float = 0.1,
    default_unit_cost: float = 1.0,
    default_lead_time_periods: int = 0,
    use_milp: bool = True,
    models_dir: str = "models",
    time_horizon_limit: Optional[int] = None,
    milp_timeout_seconds: int = 30,
) -> Dict[str, Any]:
    """
    Top-level pipeline:
      - Detect columns
      - Aggregate demand into periods
      - Compute EOQ/POQ per SKU
      - Attempt MILP (if pulp installed and use_milp=True), fallback to heuristic
      - Persist resulting schedule metadata
    """

    df = read_csv(file_path)
    columns = list(df.columns)

    sku_col = sku_col_hint or detect_sku_column(columns)
    demand_col = demand_col_hint or detect_demand_column(columns)
    date_col = date_col_hint or detect_date_column(columns)
    lead_time_col = detect_leadtime_column(columns)
    order_cost_col = detect_order_cost_column(columns)
    holding_cost_col = detect_holding_cost_column(columns)
    unit_cost_col = detect_unit_cost_column(columns)

    if not sku_col or not demand_col:
        raise ValueError("Could not detect SKU column or Demand column. Provide hints or ensure columns include sku/item and demand/quantity keywords.")

    # read or default cost maps
    SKUS = df[sku_col].unique().tolist()

    # per-SKU costs defaulted from columns or global defaults
    ordering_costs: Dict[str, float] = {}
    holding_costs: Dict[str, float] = {}
    unit_costs: Dict[str, float] = {}
    lead_times: Dict[str, int] = {}

    for s in SKUS:
        ordering_costs[s] = float(df.loc[df[sku_col] == s, order_cost_col].dropna().iloc[0]) if order_cost_col and not df.loc[df[sku_col] == s, order_cost_col].dropna().empty else float(default_order_cost)
        holding_costs[s] = float(df.loc[df[sku_col] == s, holding_cost_col].dropna().iloc[0]) if holding_cost_col and not df.loc[df[sku_col] == s, holding_cost_col].dropna().empty else float(default_holding_cost)
        unit_costs[s] = float(df.loc[df[sku_col] == s, unit_cost_col].dropna().iloc[0]) if unit_cost_col and not df.loc[df[sku_col] == s, unit_cost_col].dropna().empty else float(default_unit_cost)
        lt_val = int(df.loc[df[sku_col] == s, lead_time_col].dropna().iloc[0]) if lead_time_col and not df.loc[df[sku_col] == s, lead_time_col].dropna().empty else int(default_lead_time_periods)
        lead_times[s] = max(0, lt_val)

    # aggregate demand
    demand_pivot, periods = aggregate_demand(df, date_col, sku_col, demand_col, freq=period_freq)
    # ensure demand_pivot has columns for all SKUS (some SKUs may be missing in some periods)
    for s in SKUS:
        if s not in demand_pivot.columns:
            demand_pivot[s] = 0.0
    # order columns by SKUS list for deterministic output
    demand_pivot = demand_pivot[SKUS]

    # compute D (demand rate per period) as average demand across horizon
    eoq_map: Dict[str, float] = {}
    poq_map: Dict[str, float] = {}
    demand_rate_map: Dict[str, float] = {}
    for s in SKUS:
        D = float(demand_pivot[s].sum())  # total demand across horizon
        periods_count = max(1, demand_pivot.shape[0])
        D_rate = D / periods_count
        demand_rate_map[s] = float(D_rate)
        Co = float(ordering_costs.get(s, default_order_cost))
        Ch = float(holding_costs.get(s, default_holding_cost))
        eoq = compute_eoq(D=D_rate, Co=Co, Ch=Ch)
        # POQ requires production rate p if present as column 'production_rate' per SKU - not implemented; fallback to EOQ
        poq = compute_poq(D=D_rate, Co=Co, Ch=Ch, p=None)
        eoq_map[s] = float(round(eoq, 6))
        poq_map[s] = float(round(poq, 6))

    # Build results
    results: Dict[str, Any] = {
        "success": False,
        "skus": SKUS,
        "periods": periods,
        "demand_rate_per_period": demand_rate_map,
        "EOQ": eoq_map,
        "POQ": poq_map,
        "ordering_costs": ordering_costs,
        "holding_costs": holding_costs,
        "unit_costs": unit_costs,
        "lead_times": lead_times,
    }

    # Provide a safe preview of the demand pivot (stringify index -> orient=index)
    try:
        dp_head = demand_pivot.head(5).copy()
        # convert index values (possibly Timestamps) to strings
        dp_head.index = dp_head.index.map(lambda x: str(x) if not isinstance(x, str) else x)
        results["demand_pivot_preview"] = dp_head.to_dict(orient="index")
    except Exception:
        # best-effort fallback
        results["demand_pivot_preview"] = {}

    # attempt MILP if requested
    schedule_result = None
    if use_milp and _HAS_PULP:
        try:
            schedule_result = solve_replenishment_mip(
                demand_pivot,
                holding_costs=holding_costs,
                ordering_costs=ordering_costs,
                unit_costs=unit_costs,
                lead_times=lead_times,
                integer=True,
                time_horizon_limit=time_horizon_limit,
                timeout=milp_timeout_seconds,
            )
            results["solver_used"] = "MILP_pulp"
            results["solver_status"] = schedule_result.get("status")
            results["schedule"] = schedule_result.get("schedule")
            results["inventory_projection"] = schedule_result.get("inventory_projection")
            results["success"] = True
        except Exception as e:
            logger.exception("MILP failed, falling back to heuristic: %s", e)
            results["solver_used"] = "MILP_failed_fallback_heuristic"
            results["solver_error"] = str(e)
            schedule_result = None

    # fallback to heuristic if MILP not run or failed
    if schedule_result is None:
        heuristic = heuristic_replenishment_scheduler(demand_pivot, eoq_map, lead_times=lead_times)
        results["solver_used"] = "heuristic_scheduler"
        results["solver_status"] = heuristic.get("status")
        results["schedule"] = heuristic.get("schedule")
        results["inventory_projection"] = heuristic.get("inventory_projection")
        results["success"] = True

    # persist chosen schedule + metadata for traceability
    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model": "Inventory & Replenishment Optimization Model",
        "solver_used": results.get("solver_used"),
        "skus": SKUS,
        "EOQ": eoq_map,
        "POQ": poq_map,
        "ordering_costs_used": ordering_costs,
        "holding_costs_used": holding_costs,
        "unit_costs_used": unit_costs,
        "lead_times_used": lead_times,
        "input_file": os.path.basename(file_path),
    }
    try:
        artifact_obj = {
            "schedule": results.get("schedule"),
            "inventory_projection": results.get("inventory_projection"),
            "metadata": metadata,
        }
        saved_model_path = save_model_artifact(artifact_obj, "Inventory_Replenishment_Plan", models_dir=models_dir)
        saved_meta_path = save_metadata(metadata, saved_model_path, models_dir=models_dir)
        results["artifact"] = {"model_path": saved_model_path, "meta_path": saved_meta_path}
    except Exception as e:
        logger.exception("Failed to persist artifact: %s", e)
        results["artifact_persist_error"] = str(e)

    # Sanitize results for JSON (convert Timestamp/numpy scalars/keys -> serializable)
    try:
        results = _sanitize_for_json(results)
    except Exception:
        # As a last resort, ensure a minimal response
        results = {
            "success": False,
            "error": "Failed to sanitize results for JSON. Check pipeline logs.",
        }

    return results


def analyze_file_and_run_pipeline(file_path: str, gemini_response: Dict, models_dir: str = "models", **kwargs) -> Dict:
    """
    Entry point expected by gemini_analyser. Runs this pipeline only if gemini_response.analysis.model_type
    indicates "Inventory & Replenishment Optimization Model".
    """
    try:
        model_type = gemini_response.get("analysis", {}).get("model_type")
    except Exception:
        model_type = None

    if model_type and model_type.strip().lower() == "inventory & replenishment optimization model":
        pipeline_result = run_inventory_optimization_pipeline(file_path, gemini_analysis=gemini_response, models_dir=models_dir, **kwargs)
        return {"success": True, "gemini_analysis": gemini_response, "pipeline": pipeline_result}
    else:
        return {"success": False, "error": "Gemini did not indicate Inventory & Replenishment Optimization Model in strict mode."}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--gemini-json", required=False, help="Path to gemini analysis json (optional)")
    parser.add_argument("--models-dir", required=False, default="models", help="Directory to save models/artifacts")
    parser.add_argument("--no-milp", action="store_true", help="Disable MILP solver even if pulp is installed")
    parser.add_argument("--time-horizon", required=False, type=int, default=None, help="Limit time horizon (periods) to consider")
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
        time_horizon_limit=args.time_horizon,
        milp_timeout_seconds=args.milp_timeout,
    )
    print(json.dumps(out, indent=2, default=str))
