# services/time_series_pipeline.py
# STRICT-but-robust version (debugged):
# - Deterministic date-format detection.
# - Sanitization + optional imputation for exogenous numeric columns.
# - Robust SARIMAX fitting with deterministic fallbacks to avoid LU decomposition errors.
# - Grid search skips failing ARIMA orders; raises clear errors if none succeed.

import os
import math
import json
import pickle
import warnings
import re
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

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

COMMON_DATE_KEYWORDS = [
    "date", "datetime", "ds", "timestamp", "time"
]

# Candidate formats to try (deterministic)
_CANDIDATE_DATE_FORMATS = [
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d.%m.%Y",
    "%Y.%m.%d",
    "%b %d %Y",
    "%d %b %Y",
    "%d-%b-%Y",
    "%Y-%m-%d %H:%M:%S",
    "%d-%m-%Y %H:%M:%S",
    "%m-%d-%Y %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
]


def _row_to_contents(row: pd.Series, max_chars: int = 512) -> str:
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


def resolve_target_column(df: pd.DataFrame, target_candidate: str) -> str:
    if target_candidate is None:
        raise KeyError("Target column candidate is None")
    cols = list(df.columns)
    if target_candidate in cols:
        return target_candidate
    raise KeyError(f"Target column '{target_candidate}' was not found. Available columns: {cols}")


def read_csv_preview(file_path: str, nrows: int = 10000) -> pd.DataFrame:
    return pd.read_csv(file_path, nrows=nrows)


def detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
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
    if gemini_target:
        if gemini_target in df.columns:
            if pd.api.types.is_numeric_dtype(df[gemini_target]):
                return gemini_target
            coerced = pd.to_numeric(df[gemini_target], errors="coerce")
            if coerced.notna().sum() > 0:
                return gemini_target
            raise ValueError(f"Column '{gemini_target}' exists but is not numeric or convertible to numeric.")
    return None


def _detect_single_strptime_format_for_series(series: pd.Series, sample_limit: int = 200) -> Optional[str]:
    s = series.dropna().astype(str).str.strip()
    if s.empty:
        return None

    if len(s) > sample_limit:
        s_sample = pd.Series(s.iloc[:sample_limit].tolist())
    else:
        s_sample = s

    for fmt in _CANDIDATE_DATE_FORMATS:
        try:
            pd.to_datetime(s_sample, format=fmt, errors="raise")
            return fmt
        except Exception:
            continue

    return None


def parse_datetime_column_strict(df: pd.DataFrame, datetime_col: str) -> Tuple[pd.DatetimeIndex, Optional[str]]:
    if datetime_col not in df.columns:
        raise KeyError(f"Datetime column '{datetime_col}' not found in dataframe.")

    col = df[datetime_col]

    if pd.api.types.is_datetime64_any_dtype(col):
        return pd.to_datetime(col), None

    if pd.api.types.is_integer_dtype(col) or pd.api.types.is_float_dtype(col):
        try:
            parsed = pd.to_datetime(col, unit="s", errors="raise")
            return parsed, "epoch_seconds_or_float"
        except Exception:
            pass

    detected_fmt = _detect_single_strptime_format_for_series(col)
    if detected_fmt:
        parsed = pd.to_datetime(col, format=detected_fmt, errors="raise")
        return parsed, detected_fmt

    non_empty = col.dropna().astype(str).str.strip()
    sample_values = non_empty.iloc[:10].tolist()
    raise ValueError(
        "Could not detect a single consistent datetime string format for column "
        f"'{datetime_col}'. Sample (first up to 10 non-empty) values: {sample_values}. "
        "In strict mode the pipeline requires a single consistent datetime format. "
        "Please either: (a) normalize the CSV datetime column to a consistent format such as 'YYYY-MM-DD' "
        "(e.g. 2023-05-13), or (b) provide a column named exactly one of: "
        + ", ".join(COMMON_DATE_KEYWORDS)
        + ", or (c) convert the datetime column to ISO8601 before running the pipeline."
    )


def create_lag_features(series: pd.Series, lags: List[int]) -> pd.DataFrame:
    return pd.DataFrame({f"lag_{lag}": series.shift(lag) for lag in lags})


def select_top_features_by_corr(df: pd.DataFrame, target_col: str, max_features: int = 5, min_abs_corr: float = 0.1) -> List[str]:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe columns")

    corrs = {}
    target_num = pd.to_numeric(df[target_col], errors="coerce")

    for c in df.columns:
        if c == target_col:
            continue
        feat_num = pd.to_numeric(df[c], errors="coerce")
        valid_mask = feat_num.notna() & target_num.notna()
        if valid_mask.sum() < 3:
            continue
        corr = feat_num[valid_mask].corr(target_num[valid_mask])
        if pd.isna(corr):
            corr = 0.0
        corrs[c] = abs(corr)

    if not corrs:
        return []

    sorted_cols = sorted(corrs.items(), key=lambda x: x[1], reverse=True)
    selected = [c for c, val in sorted_cols if val >= min_abs_corr][:max_features]
    return selected


# -------------------------
# Sanitization helpers for numeric columns
# -------------------------

_NUMERIC_SANITIZE_RE = re.compile(r"[^\d\.\-eE]")  # allow digits, dot, minus, exponent

def _sanitize_string_numeric(s: Any) -> str:
    if s is None:
        return ""
    s2 = str(s).strip()
    if s2.lower() in {"nan", "na", "n/a", "-", "--", ""}:
        return ""
    cleaned = _NUMERIC_SANITIZE_RE.sub("", s2)
    return cleaned


def _drop_constant_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Drop columns with (near) zero variance; return cleaned df + dropped column names."""
    dropped = []
    for col in df.columns:
        # treat numeric and string that convert to single unique value as constant
        vals = df[col].dropna().unique()
        if len(vals) <= 1:
            dropped.append(col)
    return df.drop(columns=dropped, errors="ignore"), dropped


def _ensure_full_rank_by_dropping_collinear(exog: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Attempt to ensure exog matrix is full column-rank by dropping columns.
    Strategy (deterministic):
      - Drop columns with zero or near-zero variance first.
      - If still rank-deficient, iteratively drop the column whose removal increases matrix rank
        (or drop the column with smallest variance if none increases rank).
    Returns (cleaned_exog, dropped_columns_list).
    """
    if exog.shape[1] == 0:
        return exog, []

    X = exog.copy().astype(float)
    dropped = []

    # drop near-constant columns
    variances = X.var(axis=0, ddof=0)
    const_cols = variances[variances <= 1e-12].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)
        dropped.extend(const_cols)

    def current_rank(mat: pd.DataFrame) -> int:
        # numeric rank via SVD
        try:
            return np.linalg.matrix_rank(mat.values)
        except Exception:
            return np.linalg.matrix_rank(np.nan_to_num(mat.values))

    rank = current_rank(X)
    ncols = X.shape[1]
    # If already full rank or no columns, return
    if ncols == 0 or rank == ncols:
        return X, dropped

    # iteratively drop columns that improve rank
    cols = list(X.columns)
    while True:
        rank = current_rank(X)
        ncols = X.shape[1]
        if ncols == 0 or rank == ncols:
            break
        improved = False
        for col in cols:
            candidate = X.drop(columns=[col])
            cand_rank = current_rank(candidate)
            if cand_rank > rank:
                # dropping this column increases rank -> drop it
                X = candidate
                dropped.append(col)
                cols.remove(col)
                improved = True
                break
        if not improved:
            # no single column removal increases rank -> drop the column with smallest variance
            if cols:
                var_series = X.var(axis=0, ddof=0)
                min_col = var_series.idxmin()
                X = X.drop(columns=[min_col])
                dropped.append(min_col)
                cols.remove(min_col)
            else:
                break

    return X, dropped


# -------------------------
# Preprocessing (strict but robust)
# -------------------------

def preprocess_timeseries(
    df: pd.DataFrame,
    datetime_col: str,
    target_col: str,
    exog_cols: Optional[List[str]] = None,
    freq: Optional[str] = None,
    fill_method: str = "ffill",
    exog_sanitize: bool = True,
    exog_allow_impute: bool = True,
    exog_impute_method: str = "ffill"
) -> Tuple[pd.DataFrame, Optional[StandardScaler], List[str]]:
    if datetime_col not in df.columns:
        raise KeyError(f"Datetime column '{datetime_col}' not found in dataframe.")
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    df = df.copy()

    parsed_index, detected_fmt = parse_datetime_column_strict(df, datetime_col)
    df[datetime_col] = parsed_index
    df = df.dropna(subset=[datetime_col])
    df = df.sort_values(datetime_col)
    df = df.set_index(datetime_col)

    if freq:
        df = df.resample(freq).agg({target_col: "sum", **({c: "mean" for c in (exog_cols or [])})})

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

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
        original_exog = df[exog_cols].copy().astype(object)

        if exog_sanitize:
            cleaned = original_exog.astype(str).applymap(_sanitize_string_numeric)
            exog_numeric = cleaned.apply(pd.to_numeric, errors="coerce")
        else:
            exog_numeric = df[exog_cols].apply(pd.to_numeric, errors="coerce")

        # Drop constant columns deterministically
        exog_numeric, dropped_constants = _drop_constant_columns(exog_numeric)
        dropped_cols = list(dropped_constants)

        # If NaNs present, attempt imputation if allowed
        nan_counts = exog_numeric.isna().sum()
        bad_cols = nan_counts[nan_counts > 0]
        if not bad_cols.empty:
            if exog_allow_impute:
                exog_imputed = exog_numeric.copy()
                if exog_impute_method == "ffill":
                    exog_imputed = exog_imputed.ffill().bfill()
                    # remaining NaNs -> median
                    for col in exog_imputed.columns:
                        if exog_imputed[col].isna().any():
                            med = exog_imputed[col].median()
                            exog_imputed[col] = exog_imputed[col].fillna(med)
                elif exog_impute_method == "median":
                    for col in exog_imputed.columns:
                        med = exog_imputed[col].median()
                        exog_imputed[col] = exog_imputed[col].fillna(med)
                elif exog_impute_method == "zero":
                    exog_imputed = exog_imputed.fillna(0)
                else:
                    raise ValueError(f"Unknown exog_impute_method '{exog_impute_method}'")
                exog_numeric = exog_imputed
            else:
                # Construct detailed error
                details = []
                for col, cnt in bad_cols.items():
                    offending_original = original_exog[col].astype(str)[exog_numeric[col].isna()].head(5).tolist()
                    details.append({"column": col, "nan_count": int(cnt), "sample_offending_values": offending_original})
                msg_lines = [
                    "Exogenous features contain NaNs after conversion. Please provide clean exog data.",
                    "Offending columns and diagnostics:"
                ]
                for d in details:
                    msg_lines.append(f" - {d['column']}: {d['nan_count']} NaNs after numeric conversion; sample offending values: {d['sample_offending_values']}")
                msg_lines.append("Possible fixes: ensure these columns are numeric or convertible (remove thousands separators, fix non-numeric cells), or remove them from exogenous features.")
                raise ValueError("\n".join(msg_lines))

        # After cleaning/imputation, attempt to ensure full rank by dropping collinear columns deterministically
        exog_cleaned, dropped_due_to_collinearity = _ensure_full_rank_by_dropping_collinear(exog_numeric)
        dropped_cols.extend(dropped_due_to_collinearity)

        # Final check: if exog_cleaned empty after dropping -> treat as no exog
        final_exogs = list(exog_cleaned.columns)

        if not final_exogs:
            scaler = None
        else:
            scaler = StandardScaler()
            df[final_exogs] = scaler.fit_transform(exog_cleaned)
            exog_cols = final_exogs

        # Report dropped columns to logger for observability
        if dropped_cols:
            logger.info("Dropped exogenous columns during preprocessing (constants/collinear): %s", dropped_cols)

    return df, scaler, exog_cols


# -------------------------
# Train/Test Split
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
# Forecasting helpers with robust SARIMAX fitting
# -------------------------

def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    eps = 1e-8
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.maximum(np.abs(y_true), eps)))) * 100.0)
    return {"rmse": rmse, "mae": mae, "mape": mape}


def _fit_sarimax_with_fallback(train: pd.Series, exog_train: Optional[pd.DataFrame], order: Tuple[int, int, int], seasonal_order: Optional[Tuple[int, int, int, int]] = None, enforce_stationarity=True, enforce_invertibility=True):
    """
    Try fitting SARIMAX robustly:
      1) strict fit (default enforce_stationarity=True, enforce_invertibility=True)
      2) if LinAlgError or similar, try with enforce_stationarity=False,enforce_invertibility=False
      3) if still fails and exog present, try fit without exog (ARIMA)
    Returns fitted_model (statsmodels results object) and used_exog (DataFrame or None).
    Raises Exception if all attempts fail.
    """
    last_exc = None
    used_exog = exog_train
    # attempt 1: strict
    try:
        model = SARIMAX(train, exog=exog_train, order=order, seasonal_order=seasonal_order, enforce_stationarity=enforce_stationarity, enforce_invertibility=enforce_invertibility)
        res = model.fit(disp=False)
        return res, used_exog
    except Exception as e:
        last_exc = e
        logger.warning("SARIMAX strict fit failed: %s", repr(e))

    # attempt 2: relax stationarity/invertibility
    try:
        model = SARIMAX(train, exog=exog_train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        logger.info("SARIMAX fit succeeded with enforce_stationarity=False and enforce_invertibility=False")
        return res, used_exog
    except Exception as e:
        last_exc = e
        logger.warning("SARIMAX relaxed fit failed: %s", repr(e))

    # attempt 3: try ARIMA (no exog) if exog provided
    if exog_train is not None:
        try:
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            logger.info("Fell back to ARIMA (no exog) successfully")
            return res, None
        except Exception as e:
            last_exc = e
            logger.warning("Fallback ARIMA (no exog) fit failed: %s", repr(e))

    # all attempts failed — build an informative message and raise
    msg = [
        "SARIMAX fitting failed after multiple fallback attempts.",
        f"Primary error: {repr(last_exc)}",
        "Possible causes:",
        " - Exogenous matrix is singular/collinear or contains degenerate columns (constant or perfect multicollinearity).",
        " - Sample size too small for the chosen ARIMA/SARIMAX order.",
        " - Non-finite values present in training series or exogenous variables.",
        "Remedies to try:",
        " - Remove or fix collinear/constant exogenous columns (check for duplicate columns or identical transformations).",
        " - Reduce ARIMA order (p,d,q) or disable seasonal terms.",
        " - Ensure exogenous covariates are numeric, finite and have sufficient variability.",
        " - Try running with no exogenous features to isolate the issue."
    ]
    raise RuntimeError("\n".join(msg)) from last_exc


def forecast_with_sarimax(train: pd.Series, exog_train: Optional[pd.DataFrame], steps: int, order: Tuple[int, int, int], seasonal_order: Optional[Tuple[int, int, int, int]] = None, exog_forecast: Optional[pd.DataFrame] = None) -> np.ndarray:
    """
    Produce a forecast array. This uses _fit_sarimax_with_fallback to get a fitted model robustly.
    If exog_train is provided but later dropped by fallback (None), exog_forecast is ignored.
    """
    if len(train) == 0:
        raise ValueError("Training series is empty")

    # check finiteness
    if not np.isfinite(train.values).all():
        raise ValueError("Training series contains non-finite values")

    if exog_train is not None:
        if not np.isfinite(exog_train.values).all():
            raise ValueError("Exogenous training data contains non-finite values")
        # ensure index alignment
        exog_train = exog_train.loc[train.index]

    fitted_res, used_exog = _fit_sarimax_with_fallback(train, exog_train, order=order, seasonal_order=seasonal_order)

    # Build exog for forecasting if used_exog is not None
    if used_exog is not None and exog_forecast is not None:
        # Make sure columns match
        if list(exog_forecast.columns) != list(used_exog.columns):
            # try to reindex columns (if some dropped earlier, align)
            exog_forecast = exog_forecast.reindex(columns=used_exog.columns, fill_value=None)
        # If any NaNs, try fill with last known values
        if exog_forecast.isna().any().any():
            exog_forecast = exog_forecast.fillna(method="ffill").fillna(method="bfill")
            if exog_forecast.isna().any().any():
                # fill remaining with column medians
                for c in exog_forecast.columns:
                    if exog_forecast[c].isna().any():
                        exog_forecast[c] = exog_forecast[c].fillna(used_exog[c].median())
    else:
        exog_forecast = None

    pred = fitted_res.get_forecast(steps=steps, exog=exog_forecast).predicted_mean
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
        future[c] = train_df[c].iloc[-1]

    forecast = m.predict(future)
    return np.asarray(forecast["yhat"].values)


# -------------------------
# Model persistence helpers
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
# Orchestration: full pipeline
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
    future_steps: int = 10,
    exog_sanitize: bool = True,
    exog_allow_impute: bool = True,
    exog_impute_method: str = "ffill"
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

    exog_candidates = select_top_features_by_corr(df_raw, target_col, max_features=max_features, min_abs_corr=0.05)

    if datetime_col not in df_raw.columns:
        raise KeyError(f"Datetime column '{datetime_col}' not in raw dataframe columns")
    if target_col not in df_raw.columns:
        raise KeyError(f"Target column '{target_col}' not in raw dataframe columns")

    # Strict parse for creating lag features
    df_with_index = df_raw.copy()
    parsed_index, detected_fmt = parse_datetime_column_strict(df_with_index, datetime_col)
    df_with_index[datetime_col] = parsed_index
    df_with_index = df_with_index.dropna(subset=[datetime_col]).set_index(datetime_col).sort_index()

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
        fill_method="interpolate",
        exog_sanitize=exog_sanitize,
        exog_allow_impute=exog_allow_impute,
        exog_impute_method=exog_impute_method
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

    # If pmdarima is available, use it; otherwise grid search but skip failing orders
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
        best_cfg = None
        best_score = None
        orders_tried = []
        for p in range(0, arima_grid.get("max_p", 2) + 1):
            for d in range(0, arima_grid.get("max_d", 1) + 1):
                for q in range(0, arima_grid.get("max_q", 2) + 1):
                    order = (p, d, q)
                    try:
                        pred = forecast_with_sarimax(y_train, exog_train, steps=len(y_test), order=order, seasonal_order=None, exog_forecast=exog_test)
                        metrics = evaluate_forecast(y_test.values, pred)
                        orders_tried.append((order, metrics["mape"]))
                        if best_score is None or metrics["mape"] < best_score:
                            best_score = metrics["mape"]
                            best_cfg = order
                    except Exception as e:
                        # skip failing order but log
                        logger.warning("Skipping ARIMA order %s due to error: %s", order, repr(e))
                        continue
        if best_cfg is None:
            raise RuntimeError("ARIMA grid search failed to produce any usable model. Check exogenous variables, data size, and ARIMA grid.")
        order_to_try = best_cfg

    # attempt final forecast with chosen order
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
        # Fit robustly and save fitted model (using same fallback strategy)
        fitted_res, used_exog = _fit_sarimax_with_fallback(y_train, exog_train, order=best_order, seasonal_order=best_seasonal)
        # Prepare future exog if needed and if used_exog is not None
        last_exog_values = None
        if used_exog is not None and exogs_used:
            last_row = train_df[exogs_used].iloc[-1].values.tolist()
            last_exog_values = last_row
            future_exog = pd.DataFrame([last_row] * future_steps, columns=exogs_used)
            # if used_exog had some columns dropped earlier, reindex to used_exog columns
            future_exog = future_exog.reindex(columns=used_exog.columns, fill_value=None)
            # fill possible NaNs deterministically
            future_exog = future_exog.fillna(method="ffill").fillna(method="bfill")
            for c in future_exog.columns:
                if future_exog[c].isna().any():
                    future_exog[c] = future_exog[c].fillna(train_df[c].median())
        else:
            future_exog = None

        future_forecast = fitted_res.get_forecast(steps=future_steps, exog=future_exog).predicted_mean.values.tolist()
        saved_model_path = save_model_artifact(fitted_res, "SARIMAX_ARIMA", models_dir=models_dir)
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

    results["detected_datetime_format"] = detected_fmt if detected_fmt is not None else None

    return results


def analyze_file_and_run_pipeline(file_path: str, gemini_response: Dict, models_dir: str = "models", **kwargs) -> Dict:
    try:
        model_type = gemini_response.get("analysis", {}).get("model_type")
    except Exception:
        model_type = None

    if model_type and model_type.strip().lower() == "sales, demand & financial forecasting model":
        pipeline_result = run_time_series_pipeline(file_path, gemini_analysis=gemini_response, models_dir=models_dir, **kwargs)
        return {"success": True, "gemini_analysis": gemini_response, "pipeline": pipeline_result}
    else:
        return {"success": False, "error": "Gemini did not indicate Sales/Demand forecasting model in strict mode."}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--gemini-json", required=False, help="Path to gemini analysis json (optional)")
    parser.add_argument("--models-dir", required=False, default="models", help="Directory to save models")
    parser.add_argument("--future-steps", required=False, type=int, default=10, help="Future forecast horizon")
    parser.add_argument("--no-exog-sanitize", action="store_true", help="Disable exogenous sanitization")
    parser.add_argument("--no-exog-impute", action="store_true", help="Disable exogenous imputation")
    args = parser.parse_args()

    gemini_resp = None
    if args.gemini_json and os.path.exists(args.gemini_json):
        with open(args.gemini_json, "r") as f:
            gemini_resp = json.load(f)

    out = analyze_file_and_run_pipeline(
        args.csv,
        gemini_resp or {},
        models_dir=args.models_dir,
        future_steps=args.future_steps,
        exog_sanitize=not args.no_exog_sanitize,
        exog_allow_impute=not args.no_exog_impute
    )
    print(json.dumps(out, indent=2, default=str))
