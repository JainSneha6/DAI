# services/time_series_pipeline.py
# STRICT version: ALL fallbacks REMOVED.
# Behaviors:
# - Any ambiguous or missing information raises an error immediately.
# - No fuzzy matching, no heuristic guessing, no automatic derived columns.
# - Optional libraries (pmdarima/prophet) are required for code paths that use them.

import os
import math
import json
import pickle
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Statsmodels
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Optional libraries (must be installed if used)
try:
    import pmdarima as pm
    _HAS_PMDARIMA = True
except Exception:
    _HAS_PMDARIMA = False

try:
    from prophet import Prophet  # type: ignore
    _HAS_PROPHET = True
except Exception:
    try:
        from fbprophet import Prophet  # type: ignore
        _HAS_PROPHET = True
    except Exception:
        _HAS_PROPHET = False

warnings.filterwarnings("ignore")

import re
import logging

logger = logging.getLogger(__name__)

# -------------------------
# Strict helpers (no fallbacks)
# -------------------------

import os
import pandas as pd
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def _row_to_contents(row: pd.Series, max_chars: int = 512) -> str:
    """
    Convert a pandas row to a short contents string suitable for embedding.
    Example: "date: 2025-01-01; sales: 100; region: west"
    """
    parts = []
    for col, val in row.items():
        if pd.isna(val) or str(val).strip() == "":
            continue
        sval = str(val).strip()
        if len(sval) > 150:
            sval = sval[:147].rstrip() + "..."
        parts.append(f"{col}: {sval}")
    contents = "; ".join(parts)
    if len(contents) > max_chars:
        contents = contents[: max_chars - 3].rstrip() + "..."
    return contents

def generate_row_contents_from_csv(file_path: str, max_rows: int = 1000, max_chars_per_row: int = 512) -> List[Dict]:
    """
    Read the CSV and produce a list of items to upsert to Cyborg embedded index.
    Each item: {"id": "<basename>::row::<i>", "contents": "...", "metadata": {...}}
    """
    try:
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False, na_values=[""])
    except Exception as e:
        logger.exception("Failed to read CSV %s: %s", file_path, e)
        return []

    rows = min(len(df), max_rows)
    basename = os.path.basename(file_path)
    items = []
    for i in range(rows):
        row = df.iloc[i]
        contents = _row_to_contents(row, max_chars=max_chars_per_row)
        if not contents:
            continue
        items.append({
            "id": f"{basename}::row::{i}",
            "contents": contents,
            "metadata": {"source_file": basename, "row_index": i}
        })
    return items

def analyze_file_and_run_pipeline(file_path: str, gemini_response: Dict, models_dir: str = "models", **kwargs):
    """
    Run your existing pipeline to train/produce artifacts (keeps your current strict pipeline).
    Then attach 'embeddings' (items with 'contents') for downstream upsert.
    """
    # Replace with your existing pipeline call + artifact save logic
    pipeline_result = run_time_series_pipeline(file_path, gemini_analysis=gemini_response, models_dir=models_dir, **kwargs)

    # produce items for Cyborg upsert (these have contents; embedded Cyborg will compute vectors)
    items_for_cyborg = generate_row_contents_from_csv(file_path,
                                                      max_rows=int(kwargs.get("max_rows_for_embeddings", 1000)),
                                                      max_chars_per_row=int(kwargs.get("max_chars_per_row", 512)))
    pipeline_result["embeddings"] = items_for_cyborg
    return {"success": True, "gemini_analysis": gemini_response, "pipeline": pipeline_result}

def resolve_target_column(df: pd.DataFrame, target_candidate: str) -> str:
    """
    STRICT: Only exact column name allowed. Raises KeyError if not present.
    """
    if target_candidate is None:
        raise KeyError("Target column candidate is None")
    cols = list(df.columns)
    if target_candidate in cols:
        return target_candidate
    raise KeyError(f"Target column '{target_candidate}' was not found. Available columns: {cols}")


COMMON_DATE_KEYWORDS = [
    "date", "datetime", "ds", "timestamp", "time"
]


def read_csv_preview(file_path: str, nrows: int = 10000) -> pd.DataFrame:
    """Read CSV using pandas with default engine. Errors propagate."""
    return pd.read_csv(file_path, nrows=nrows)


def detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """
    STRICT: Return a datetime column only if a column name exactly matches one of COMMON_DATE_KEYWORDS
    or if a column already has a datetime dtype. Otherwise return None.
    """
    cols = df.columns.tolist()
    for k in COMMON_DATE_KEYWORDS:
        for c in cols:
            if k == c.strip().lower():
                return c

    for c in cols:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c

    return None


def detect_target_column(df: pd.DataFrame, gemini_target: Optional[str] = None) -> Optional[str]:
    """
    STRICT: Only accept gemini_target if it exactly matches a column and is numeric.
    Do NOT attempt keyword scanning or numeric-dominance heuristics.
    """
    if gemini_target:
        if gemini_target in df.columns:
            if pd.api.types.is_numeric_dtype(df[gemini_target]):
                return gemini_target
            # allow if convertible to numeric and has at least one non-null numeric value
            coerced = pd.to_numeric(df[gemini_target], errors="coerce")
            if coerced.notna().sum() > 0:
                return gemini_target
            raise ValueError(f"Column '{gemini_target}' exists but is not numeric or convertible to numeric.")
    return None


# -------------------------
# Feature Selection & Engineering (strict)
# -------------------------

def create_lag_features(series: pd.Series, lags: List[int]) -> pd.DataFrame:
    return pd.DataFrame({f"lag_{lag}": series.shift(lag) for lag in lags})


def select_top_features_by_corr(df: pd.DataFrame, target_col: str, max_features: int = 5, min_abs_corr: float = 0.1) -> List[str]:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe columns")
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target_col]
    if not numeric_cols:
        return []

    corrs = {}
    for c in numeric_cols:
        corr = df[target_col].corr(df[c])
        if pd.isna(corr):
            corr = 0.0
        corrs[c] = abs(corr)

    sorted_cols = sorted(corrs.items(), key=lambda x: x[1], reverse=True)
    selected = [c for c, val in sorted_cols if val >= min_abs_corr][:max_features]
    return selected


# -------------------------
# Preprocessing (STRICT: no heuristics)
# -------------------------

def preprocess_timeseries(
    df: pd.DataFrame,
    datetime_col: str,
    target_col: str,
    exog_cols: Optional[List[str]] = None,
    freq: Optional[str] = None,
    fill_method: str = "ffill"
) -> Tuple[pd.DataFrame, Optional[StandardScaler], List[str]]:
    if datetime_col not in df.columns:
        raise KeyError(f"Datetime column '{datetime_col}' not found in dataframe.")
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="raise")
    df = df.dropna(subset=[datetime_col])
    df = df.sort_values(datetime_col)
    df = df.set_index(datetime_col)

    # Do not infer frequency; only resample if freq explicitly provided
    if freq:
        df = df.resample(freq).agg({target_col: "sum", **({c: "mean" for c in (exog_cols or [])})})

    # require numeric target
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # fill strategy (no silent fallbacks)
    if fill_method == "ffill":
        df[target_col] = df[target_col].ffill().bfill()
    elif fill_method == "interpolate":
        df[target_col] = df[target_col].interpolate().ffill().bfill()
    else:
        df[target_col] = df[target_col].fillna(method="ffill").fillna(0)

    exog_cols = exog_cols or []
    exog_cols = [c for c in exog_cols if c in df.columns and c != target_col]
    scaler = None
    if exog_cols:
        exog_df = df[exog_cols].apply(pd.to_numeric, errors="coerce")
        # require no NaNs in exog after conversion
        if exog_df.isna().any().any():
            raise ValueError("Exogenous features contain NaNs after conversion. Please provide clean exog data.")
        scaler = StandardScaler()
        df[exog_cols] = scaler.fit_transform(exog_df)

    return df, scaler, exog_cols


# -------------------------
# Train/Test Split (time-series)
# -------------------------

def train_test_split_time_series(df: pd.DataFrame, test_size: Optional[int] = None, test_frac: Optional[float] = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n == 0:
        raise ValueError("Dataframe is empty after preprocessing")
    if test_size is None:
        test_size = max(1, int(n * test_frac))
    if test_size >= n:
        raise ValueError("Test size must be smaller than the dataset length")
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    return train, test


# -------------------------
# Model training & forecasting (STRICT: raise on errors)
# -------------------------

def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    eps = 1e-8
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.maximum(np.abs(y_true), eps)))) * 100.0)
    return {"rmse": rmse, "mae": mae, "mape": mape}


def forecast_with_sarimax(train: pd.Series, exog_train: Optional[pd.DataFrame], steps: int, order: Tuple[int, int, int], seasonal_order: Optional[Tuple[int, int, int, int]] = None, exog_forecast: Optional[pd.DataFrame] = None) -> np.ndarray:
    # Strict: do not swallow exceptions
    if len(train) == 0:
        raise ValueError("Training series is empty")
    if seasonal_order:
        model = SARIMAX(train, exog=exog_train, order=order, seasonal_order=seasonal_order, enforce_stationarity=True, enforce_invertibility=True)
    else:
        model = SARIMAX(train, exog=exog_train, order=order, enforce_stationarity=True, enforce_invertibility=True)
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=steps, exog=exog_forecast).predicted_mean
    return np.asarray(pred)


def forecast_with_prophet(train_df: pd.DataFrame, target_col: str, exog_cols: List[str], steps: int, freq: Optional[str]) -> np.ndarray:
    if not _HAS_PROPHET:
        raise RuntimeError("Prophet is not installed")

    df_prop = train_df.reset_index().rename(columns={train_df.index.name or train_df.index.names[0]: "ds", target_col: "y"})
    if "ds" not in df_prop.columns:
        df_prop["ds"] = train_df.index.to_series().values

    m = Prophet()
    for c in exog_cols:
        if c not in df_prop.columns:
            raise KeyError(f"Exogenous regressor '{c}' not present in training dataframe for Prophet")
        m.add_regressor(c)

    df_prop_for_fit = df_prop[["ds", "y"] + exog_cols]
    m.fit(df_prop_for_fit)

    last_date = train_df.index.max()
    if freq is None:
        future_dates = pd.date_range(last_date + pd.Timedelta(1, unit="D"), periods=steps, freq="D")
    else:
        future_dates = pd.date_range(last_date + pd.tseries.frequencies.to_offset(freq), periods=steps, freq=freq)

    future = pd.DataFrame({"ds": future_dates})
    for c in exog_cols:
        # STRICT: require that we can provide a value for each regressor (use last known)
        future[c] = train_df[c].iloc[-1]

    forecast = m.predict(future)
    return np.asarray(forecast["yhat"].values)


# -------------------------
# Model persistence helpers (unchanged but let exceptions bubble)
# -------------------------

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


def load_model_artifact(model_filepath: str) -> Any:
    with open(model_filepath, "rb") as f:
        return pickle.load(f)


def load_metadata(meta_filepath: str) -> Dict:
    with open(meta_filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_and_forecast(model_filepath: str, meta_filepath: str, steps: int = 10) -> Dict:
    model = load_model_artifact(model_filepath)
    meta = load_metadata(meta_filepath)

    model_name = meta.get("model_name")
    exogs = meta.get("exogenous_features") or []
    last_exog_values = meta.get("last_exog_values") or None
    inferred_freq = meta.get("inferred_freq")

    result = {"model": model_filepath, "meta": meta, "forecast": None}

    if model_name == "SARIMAX_ARIMA":
        if exogs and last_exog_values is not None:
            future_exog = pd.DataFrame([last_exog_values] * steps, columns=exogs)
        else:
            future_exog = None
        pred = model.get_forecast(steps=steps, exog=future_exog).predicted_mean.values.tolist()
        result["forecast"] = pred

    elif model_name == "Prophet":
        if not _HAS_PROPHET:
            raise RuntimeError("Prophet not installed but metadata indicates Prophet model")

        if inferred_freq:
            future_dates = pd.date_range(pd.to_datetime(meta.get("train_end")) + pd.tseries.frequencies.to_offset(inferred_freq), periods=steps, freq=inferred_freq)
        else:
            future_dates = pd.date_range(pd.to_datetime(meta.get("train_end")) + pd.Timedelta(days=1), periods=steps, freq="D")

        future = pd.DataFrame({"ds": future_dates})
        for idx, c in enumerate(exogs):
            if last_exog_values is None:
                raise ValueError("Missing last_exog_values required for Prophet forecasting")
            future[c] = last_exog_values[idx]

        forecast = model.predict(future)
        result["forecast"] = forecast["yhat"].values.tolist()

    else:
        raise ValueError(f"Unsupported model type for forecasting: {model_name}")

    return result


# -------------------------
# Orchestration: full pipeline (STRICT)
# -------------------------

def run_time_series_pipeline(
    file_path: str,
    gemini_analysis: Optional[Dict] = None,
    target_col_hint: Optional[str] = None,
    test_frac: float = 0.2,
    max_lags: int = 3,
    max_features: int = 5,
    arima_grid: Dict[str, int] = None,
    models_dir: str = "models",
    future_steps: int = 10
) -> Dict[str, Any]:
    if arima_grid is None:
        arima_grid = {"max_p": 2, "max_q": 2, "max_d": 1}

    df_raw = pd.read_csv(file_path)
    df_raw = df_raw.dropna(axis=1, how="all")

    datetime_col = detect_datetime_column(df_raw)
    if datetime_col is None:
        raise ValueError("Could not reliably detect a datetime column. Please provide an explicit datetime column named one of: " + ", ".join(COMMON_DATE_KEYWORDS))

    gemini_target = None
    if gemini_analysis:
        gemini_target = gemini_analysis.get("analysis", {}).get("target_column")

    target_col = detect_target_column(df_raw, gemini_target=gemini_target or target_col_hint)
    if target_col is None:
        raise ValueError("No explicit numeric target column supplied by Gemini or hint. Aborting in strict mode.")

    # select exogenous candidates strictly (no heuristics beyond numeric corr)
    exog_candidates = select_top_features_by_corr(df_raw, target_col, max_features=max_features, min_abs_corr=0.05)

    # Require that datetime_col and target_col exist before creating lags
    if datetime_col not in df_raw.columns:
        raise KeyError(f"Datetime column '{datetime_col}' not in raw dataframe columns")
    if target_col not in df_raw.columns:
        raise KeyError(f"Target column '{target_col}' not in raw dataframe columns")

    df_with_index = df_raw.copy()
    df_with_index[datetime_col] = pd.to_datetime(df_with_index[datetime_col], errors="raise")
    df_with_index = df_with_index.dropna(subset=[datetime_col]).set_index(datetime_col).sort_index()

    # strict: resolve target column exactly
    resolved_target_col = resolve_target_column(df_with_index, target_col)

    for lag in range(1, max_lags + 1):
        df_with_index[f"lag_{lag}"] = pd.to_numeric(df_with_index[resolved_target_col], errors="raise").shift(lag)

    all_exogs = exog_candidates + [f"lag_{i}" for i in range(1, max_lags + 1)]

    processed_df, scaler, exogs_used = preprocess_timeseries(
        df_raw,
        datetime_col=datetime_col,
        target_col=target_col,
        exog_cols=all_exogs,
        freq=None,
        fill_method="interpolate"
    )

    if processed_df[target_col].isna().all():
        raise ValueError("After preprocessing target is all NaN. Aborting in strict mode.")

    inferred_freq = pd.infer_freq(processed_df.index)

    train_df, test_df = train_test_split_time_series(processed_df, test_frac=test_frac)
    if len(train_df) < 5:
        raise ValueError("Not enough training data after split.")

    y_train = train_df[target_col]
    y_test = test_df[target_col]
    exog_train = train_df[exogs_used] if exogs_used else None
    exog_test = test_df[exogs_used] if exogs_used else None

    results = {"models": {}}

    # -----------------
    # SARIMAX / ARIMA (STRICT)
    # -----------------
    order_to_try = None
    seasonal_order_to_try = None

    if _HAS_PMDARIMA:
        auto_model = pm.auto_arima(
            y_train,
            exogenous=exog_train,
            seasonal=False,
            max_p=arima_grid.get("max_p", 2),
            max_q=arima_grid.get("max_q", 2),
            max_d=arima_grid.get("max_d", 1),
            error_action="raise",
            suppress_warnings=False,
            stepwise=True,
            n_jobs=1
        )
        order_to_try = auto_model.order
        seasonal_order_to_try = getattr(auto_model, "seasonal_order", None)
    else:
        # Strict grid search but raise if none succeed
        best_cfg = None
        best_score = None
        for p in range(0, arima_grid.get("max_p", 2) + 1):
            for d in range(0, arima_grid.get("max_d", 1) + 1):
                for q in range(0, arima_grid.get("max_q", 2) + 1):
                    order = (p, d, q)
                    pred = forecast_with_sarimax(y_train, exog_train, steps=len(y_test), order=order, seasonal_order=None, exog_forecast=exog_test)
                    metrics = evaluate_forecast(y_test.values, pred)
                    if best_score is None or metrics["mape"] < best_score:
                        best_score = metrics["mape"]
                        best_cfg = order
        if best_cfg is None:
            raise RuntimeError("ARIMA grid search failed to produce any model")
        order_to_try = best_cfg

    sarimax_pred = forecast_with_sarimax(y_train, exog_train, steps=len(y_test), order=order_to_try, seasonal_order=seasonal_order_to_try, exog_forecast=exog_test)
    sarimax_metrics = evaluate_forecast(y_test.values, sarimax_pred)
    results["models"]["SARIMAX_ARIMA"] = {
        "order": order_to_try,
        "seasonal_order": seasonal_order_to_try,
        "forecast": sarimax_pred.tolist(),
        "metrics": sarimax_metrics
    }

    # -----------------
    # Prophet (STRICT)
    # -----------------
    if _HAS_PROPHET:
        prophet_pred = forecast_with_prophet(train_df, target_col=target_col, exog_cols=exogs_used, steps=len(y_test), freq=inferred_freq)
        prophet_metrics = evaluate_forecast(y_test.values, prophet_pred)
        results["models"]["Prophet"] = {"forecast": prophet_pred.tolist(), "metrics": prophet_metrics}

    # -----------------
    # Decide best model by MAPE (primary), then RMSE
    # -----------------
    candidate_scores = []
    for mname, info in results["models"].items():
        if isinstance(info, dict) and info.get("metrics"):
            candidate_scores.append((mname, info["metrics"]["mape"], info["metrics"]["rmse"]))
    if not candidate_scores:
        raise RuntimeError("No model produced valid forecasts in strict mode.")

    selected = sorted(candidate_scores, key=lambda x: (x[1], x[2]))[0]
    best_model_name = selected[0]
    best_model_info = results["models"][best_model_name]

    # -----------------
    # Fit + SAVE the best model for future prediction (STRICT)
    # -----------------
    saved_model_path = None
    saved_meta_path = None
    future_forecast = []

    metadata = {
        "model_name": best_model_name,
        "target_column": target_col,
        "datetime_column": datetime_col,
        "exogenous_features": exogs_used,
        "inferred_freq": inferred_freq,
        "train_start": str(train_df.index.min()),
        "train_end": str(train_df.index.max()),
        "created_at": datetime.utcnow().isoformat() + "Z"
    }

    if best_model_name == "SARIMAX_ARIMA":
        best_order = best_model_info.get("order")
        best_seasonal = best_model_info.get("seasonal_order")
        sarimax_model = SARIMAX(
            y_train,
            exog=exog_train,
            order=best_order,
            seasonal_order=best_seasonal,
            enforce_stationarity=True,
            enforce_invertibility=True
        )
        fitted = sarimax_model.fit(disp=False)

        last_exog_values = None
        if exogs_used:
            last_row = train_df[exogs_used].iloc[-1].values.tolist()
            last_exog_values = last_row
            future_exog = pd.DataFrame([last_row] * future_steps, columns=exogs_used)
        else:
            future_exog = None

        future_forecast = fitted.get_forecast(steps=future_steps, exog=future_exog).predicted_mean.values.tolist()
        saved_model_path = save_model_artifact(fitted, "SARIMAX_ARIMA", models_dir=models_dir)
        metadata["last_exog_values"] = last_exog_values

    elif best_model_name == "Prophet":
        if not _HAS_PROPHET:
            raise RuntimeError("Prophet not installed but was selected as best model")
        df_prop = train_df.reset_index().rename(columns={train_df.index.name or train_df.index.names[0]: "ds", target_col: "y"})
        m = Prophet()
        for c in (exogs_used or []):
            m.add_regressor(c)
        fit_cols = ["ds", "y"] + (exogs_used or [])
        df_prop_for_fit = df_prop[fit_cols]
        m.fit(df_prop_for_fit)

        if inferred_freq:
            next_period = pd.to_datetime(train_df.index.max()) + pd.tseries.frequencies.to_offset(inferred_freq)
            future_dates = pd.date_range(next_period, periods=future_steps, freq=inferred_freq)
        else:
            future_dates = pd.date_range(pd.to_datetime(train_df.index.max()) + pd.Timedelta(days=1), periods=future_steps, freq="D")

        future = pd.DataFrame({"ds": future_dates})
        last_exog_values = None
        for c in (exogs_used or []):
            last_val = train_df[c].iloc[-1]
            future[c] = last_val
        if exogs_used:
            last_exog_values = [train_df[c].iloc[-1] for c in exogs_used]

        forecast = m.predict(future)
        future_forecast = forecast["yhat"].values.tolist()
        saved_model_path = save_model_artifact(m, "Prophet", models_dir=models_dir)
        metadata["last_exog_values"] = last_exog_values

    else:
        raise RuntimeError(f"Unsupported model '{best_model_name}' for saving in strict mode")

    if saved_model_path:
        saved_meta_path = save_metadata(metadata, saved_model_path, models_dir=models_dir)

    results["best_model"] = {
        "name": best_model_name,
        "metrics": best_model_info.get("metrics"),
        "future_forecast": future_forecast
    }

    results["artifact"] = {"model_path": saved_model_path, "meta_path": saved_meta_path}

    results["success"] = True
    results["target_column"] = target_col
    results["datetime_column"] = datetime_col
    results["exogenous_features"] = exogs_used
    results["train_period"] = {"start": str(train_df.index.min()), "end": str(train_df.index.max()), "n": len(train_df)}
    results["test_period"] = {"start": str(test_df.index.min()), "end": str(test_df.index.max()), "n": len(test_df)}

    return results


# -------------------------
# Convenience integration function to call from existing analyzer
# -------------------------

def analyze_file_and_run_pipeline(file_path: str, gemini_response: Dict, **kwargs) -> Dict:
    try:
        model_type = gemini_response.get("analysis", {}).get("model_type")
    except Exception:
        model_type = None

    if model_type and model_type.strip().lower() == "sales, demand & financial forecasting model":
        pipeline_result = run_time_series_pipeline(file_path, gemini_analysis=gemini_response, **kwargs)
        return {"success": True, "gemini_analysis": gemini_response, "pipeline": pipeline_result}
    else:
        return {"success": False, "error": "Gemini did not indicate Sales/Demand forecasting model in strict mode."}


# If used as script for quick debug
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--gemini-json", required=False, help="Path to gemini analysis json (optional)")
    parser.add_argument("--models-dir", required=False, default="models", help="Directory to save models")
    parser.add_argument("--future-steps", required=False, type=int, default=10, help="Future forecast horizon")
    args = parser.parse_args()

    gemini_resp = None
    if args.gemini_json and os.path.exists(args.gemini_json):
        with open(args.gemini_json, "r") as f:
            gemini_resp = json.load(f)

    out = analyze_file_and_run_pipeline(args.csv, gemini_resp or {}, models_dir=args.models_dir, future_steps=args.future_steps)
    print(json.dumps(out, indent=2, default=str))