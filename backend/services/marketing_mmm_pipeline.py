# services/marketing_mmm_pipeline.py
# STRICT-but-robust Marketing ROI & Attribution pipeline.
# - Implements ElasticNet (with CV), Ridge (with CV), Lasso (with CV).
# - Deterministic preprocessing and adstock transform helper.
# - Robust handling of categorical channels, missing values, and model persistence.
# - Designed to be called from gemini_analyser (expects gemini_analysis with model_type "Marketing ROI & Attribution Model").
#
# Usage:
#   from services.marketing_mmm_pipeline import analyze_file_and_run_pipeline
#   analyze_file_and_run_pipeline(csv_path, gemini_response, models_dir="models")

import os
import re
import math
import json
import pickle
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, ElasticNet, RidgeCV, LassoCV, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Candidate keywords for datetime detection (if available)
COMMON_DATE_KEYWORDS = [
    "date", "datetime", "ds", "timestamp", "time"
]

# Regex heuristics to find marketing spend/channel columns
_SPEND_COL_REGEX = re.compile(
    r"(spend|cost|budget|investment|ad[_\s]?spend|media[_\s]?spend|impress|click|cpc|cpa|ctr)|(^tv$)|(^search$)|(^social$)",
    flags=re.I,
)


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


# -------------------------
# Basic I/O + detection
# -------------------------


def read_csv_preview(file_path: str, nrows: int = 100000) -> pd.DataFrame:
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


def extract_spend_and_media_columns(columns: List[str]) -> List[str]:
    """
    Heuristic: pick columns whose names match spend/channel patterns.
    """
    picked = []
    for c in columns:
        if _SPEND_COL_REGEX.search(c):
            picked.append(c)
    return picked


def detect_target_column(df: pd.DataFrame, gemini_target: Optional[str] = None) -> Optional[str]:
    """
    If gemini_target present and numeric or convertible, accept it.
    Otherwise, attempt to find typical revenue/sales columns.
    """
    if gemini_target:
        if gemini_target in df.columns:
            if pd.api.types.is_numeric_dtype(df[gemini_target]):
                return gemini_target
            coerced = pd.to_numeric(df[gemini_target], errors="coerce")
            if coerced.notna().sum() > 0:
                return gemini_target
            raise ValueError(f"Column '{gemini_target}' exists but is not numeric or convertible to numeric.")

    # heuristics: look for revenue/sales/total_sales/total_revenue/order_value columns
    for c in df.columns:
        if re.search(
            r"(revenue|sales|total[_\s]?sales|total[_\s]?revenue|order[_\s]?value|gmv|net[_\s]?sales|total)",
            c,
            flags=re.I,
        ):
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().sum() > 0:
                return c
    return None


# -------------------------
# Adstock / transformation helpers
# -------------------------


def adstock_transform(series: pd.Series, decay: float = 0.5) -> pd.Series:
    """
    Simple adstock (geometric decay) transform applied to 1D spend series.
    result[t] = spend[t] + decay * result[t-1]
    """
    out = np.zeros(len(series))
    prev = 0.0
    for i, v in enumerate(series.fillna(0.0).astype(float)):
        prev = v + decay * prev
        out[i] = prev
    return pd.Series(out, index=series.index)


def saturation_transform(series: pd.Series, alpha: float = 1.0) -> pd.Series:
    """
    Diminishing returns via power transform / concave transform.
    For simplicity we use x^(alpha) where 0 < alpha <= 1 creates concavity.
    """
    arr = np.array(series.fillna(0.0).astype(float))
    if alpha <= 0:
        alpha = 1.0
    return pd.Series(np.sign(arr) * (np.abs(arr) ** alpha), index=series.index)


# -------------------------
# Preprocessing
# -------------------------


def prepare_marketing_matrix(
    df: pd.DataFrame,
    spend_cols: List[str],
    other_exog_cols: Optional[List[str]] = None,
    apply_adstock: bool = True,
    adstock_decay: float = 0.5,
    saturation_alpha: float = 0.7,
) -> Tuple[pd.DataFrame, List[str], StandardScaler]:
    """
    Build a matrix X of marketing inputs suitable for regression:
    - For each spend column apply adstock (optional) and saturation transform.
    - Encode categorical columns (one-hot) for non-spend exogenous categorical variables.
    - Fill numeric NaNs deterministically.
    Returns processed DataFrame (aligned with original index), list of final feature names, and fitted StandardScaler.
    """
    df_proc = df.copy()
    features: List[str] = []

    # Ensure spend_cols exist
    spend_cols = [c for c in spend_cols if c in df_proc.columns]
    for c in spend_cols:
        s = pd.to_numeric(df_proc[c], errors="coerce").fillna(0.0)
        if apply_adstock:
            s = adstock_transform(s, decay=adstock_decay)
        if saturation_alpha is not None and saturation_alpha != 1.0:
            s = saturation_transform(s, alpha=saturation_alpha)
        df_proc[f"_m_{c}"] = s
        features.append(f"_m_{c}")

    # Additional exogenous numeric columns
    other_exog_cols = other_exog_cols or []
    numeric_exogs = []
    categorical_exogs = []
    for c in other_exog_cols:
        if c not in df_proc.columns or c in spend_cols:
            continue
        coerced = pd.to_numeric(df_proc[c], errors="coerce")
        if coerced.notna().sum() > 0:
            numeric_exogs.append(c)
        else:
            categorical_exogs.append(c)

    # numeric exogs
    for c in numeric_exogs:
        df_proc[f"_n_{c}"] = pd.to_numeric(df_proc[c], errors="coerce").fillna(0.0)
        features.append(f"_n_{c}")

    # categorical exogs -> one-hot (drop_first=True deterministic)
    for c in categorical_exogs:
        dummies = pd.get_dummies(df_proc[c].astype(str).fillna(""), prefix=f"_c_{c}", drop_first=True)
        if not dummies.empty:
            df_proc = pd.concat([df_proc, dummies], axis=1)
            features.extend(list(dummies.columns))

    # Final feature matrix
    X = df_proc[features].copy()
    # If any constant columns, drop them deterministically
    const_cols = [col for col in X.columns if X[col].nunique(dropna=True) <= 1]
    if const_cols:
        X = X.drop(columns=const_cols)
        logger.info("Dropped constant features: %s", const_cols)

    # Fill remaining NaNs deterministically
    X = X.fillna(0.0)

    # Standardize features for regression
    scaler = StandardScaler()
    if X.shape[1] > 0:
        X_scaled = pd.DataFrame(scaler.fit_transform(X.values), index=X.index, columns=X.columns)
    else:
        X_scaled = X.copy()

    return X_scaled, list(X_scaled.columns), scaler


# -------------------------
# Evaluation helpers
# -------------------------


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    eps = 1e-8
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs((y_true - y_pred) / (np.maximum(np.abs(y_true), eps)))) * 100.0)
    r2 = float(r2_score(y_true, y_pred))
    return {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape}


def compute_attribution_from_coefs(
    coefs: np.ndarray,
    feature_names: List[str],
    spend_cols_original: List[str],
    df_original: pd.DataFrame,
    adstock_decay: float = 0.5,
    saturation_alpha: float = 0.7,
) -> Dict[str, Any]:
    """
    Best-effort mapping of coefficients back to spend channel contributions and ROI.
    Note: this is approximate because coefficients were learned on standardized features.
    We recompute adstock + saturation on raw spends and multiply by coefficient to produce
    an approximate channel contribution.
    """
    mapping: Dict[str, Any] = {}
    coef_map = {n: float(c) for n, c in zip(feature_names, coefs)}
    for orig in spend_cols_original:
        feat_name = f"_m_{orig}"
        coef = float(coef_map.get(feat_name, 0.0))
        # Recompute transformed spend from raw column
        raw_spend = pd.to_numeric(df_original[orig], errors="coerce").fillna(0.0) if orig in df_original.columns else pd.Series([0.0] * len(df_original))
        transformed = adstock_transform(raw_spend, decay=adstock_decay)
        if saturation_alpha is not None and saturation_alpha != 1.0:
            transformed = saturation_transform(transformed, alpha=saturation_alpha)
        total_transformed = float(transformed.sum())
        total_spend = float(raw_spend.sum())
        approx_contribution = float(coef * total_transformed)
        approx_roi = float(approx_contribution / total_spend) if total_spend != 0 else None
        mapping[orig] = {
            "coef": coef,
            "total_spend": total_spend,
            "total_transformed": total_transformed,
            "approx_contribution": approx_contribution,
            "approx_roi": approx_roi,
        }
    return mapping


# -------------------------
# Core pipeline
# -------------------------


def run_marketing_mmm_pipeline(
    file_path: str,
    gemini_analysis: Optional[Dict] = None,
    target_col_hint: Optional[str] = None,
    test_frac: float = 0.2,
    cv_folds: int = 5,
    apply_adstock: bool = True,
    adstock_decay: float = 0.5,
    saturation_alpha: float = 0.7,
    models_dir: str = "models",
    future_periods: int = 10,
    # removed Bayesian configuration (pymc) per request
) -> Dict[str, Any]:
    """
    Orchestrates the Marketing MMM:
    - Detect target (revenue) column via gemini_analysis or heuristics
    - Identify spend/media columns heuristically
    - Build feature matrix with adstock + saturation
    - Train ElasticNetCV, RidgeCV, LassoCV
    - Evaluate on test set and pick best model by R2 (primary) then RMSE
    - Persist best model + metadata and return detailed results
    """
    df_raw = pd.read_csv(file_path)
    df_raw = df_raw.dropna(axis=1, how="all")

    # derive gemini_target from model_targets map (if present)
    gemini_target = None
    if gemini_analysis:
        analysis = gemini_analysis.get("analysis", {}) or {}
        model_targets = analysis.get("model_targets") or {}
        model_key = "Marketing ROI & Attribution Model"
        if isinstance(model_targets, dict):
            target_val = model_targets.get(model_key)
            if isinstance(target_val, dict):
                gemini_target = target_val.get("target_column") or target_val.get("target") or None
            else:
                gemini_target = target_val

    if not gemini_target:
        legacy_target = gemini_analysis.get("analysis", {}).get("target_column") if gemini_analysis else None
        if legacy_target:
            gemini_target = legacy_target

    if not gemini_target:
        gemini_target = target_col_hint

    target_col = detect_target_column(df_raw, gemini_target=gemini_target)
    if target_col is None:
        raise ValueError("No explicit numeric target column supplied by Gemini or hint. Aborting in strict mode.")

    # Identify spend/media columns heuristically from header + gemini key_columns if present
    header_cols = list(df_raw.columns)
    spend_cols = extract_spend_and_media_columns(header_cols)

    # Also accept explicit gemini hint (e.g. gemini may provide key_columns)
    if gemini_analysis:
        key_cols = gemini_analysis.get("analysis", {}).get("key_columns", []) or []
        for kc in key_cols:
            if kc in header_cols and kc not in spend_cols and _SPEND_COL_REGEX.search(kc):
                spend_cols.append(kc)

    # If still no spend columns, heuristically pick top-N numeric columns excluding target
    if not spend_cols:
        numeric_candidates = []
        for c in header_cols:
            if c == target_col:
                continue
            coerced = pd.to_numeric(df_raw[c], errors="coerce")
            if coerced.notna().sum() > 0:
                numeric_candidates.append((c, float(coerced.sum())))
        numeric_candidates = sorted(numeric_candidates, key=lambda x: x[1], reverse=True)
        spend_cols = [c for c, _ in numeric_candidates[:6]]

    # Additional exog candidates: everything else except target and spend
    other_exogs = [c for c in header_cols if c not in spend_cols and c != target_col]

    if len(spend_cols) == 0:
        raise ValueError("Could not detect any marketing spend / channel columns; pipeline requires at least one spend column in strict mode.")

    # Build transformed features (adstock + saturation) and scaler
    X_all, feature_names, scaler = prepare_marketing_matrix(
        df_raw,
        spend_cols,
        other_exog_cols=other_exogs,
        apply_adstock=apply_adstock,
        adstock_decay=adstock_decay,
        saturation_alpha=saturation_alpha,
    )

    if X_all.shape[1] == 0:
        raise RuntimeError("After preprocessing there are no valid features to train on.")

    # Build y
    y = pd.to_numeric(df_raw[target_col], errors="coerce").fillna(method="ffill").fillna(method="bfill")
    if y.isna().all():
        raise ValueError("Target column contains only NaNs after coercion. Aborting.")

    # Align indices
    X_all = X_all.loc[y.index]

    # Train/test split: if a datetime column exists, split by time to preserve causality
    datetime_col = detect_datetime_column(df_raw)
    if datetime_col:
        try:
            idx_sorted = pd.to_datetime(df_raw[datetime_col], errors="coerce").sort_values().index
            ord_df = pd.DataFrame({"_ord_index": idx_sorted})
            n = len(ord_df)
            test_size = max(1, int(n * test_frac))
            train_idx = ord_df.index[:-test_size]
            test_idx = ord_df.index[-test_size:]
            X_train = X_all.loc[train_idx]
            X_test = X_all.loc[test_idx]
            y_train = y.loc[train_idx]
            y_test = y.loc[test_idx]
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=test_frac, random_state=42, shuffle=True)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=test_frac, random_state=42, shuffle=True)

    if len(X_train) < 5:
        raise ValueError("Not enough training samples after split. At least 5 required in strict mode.")

    results: Dict[str, Any] = {"models": {}}

    # -----------------
    # ElasticNetCV (robust)
    # -----------------
    try:
        enet_cv = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], cv=min(cv_folds, max(2, int(len(X_train) / 5))), n_jobs=1, random_state=42)
        enet_cv.fit(X_train.values, y_train.values)
        enet_pred = enet_cv.predict(X_test.values)
        enet_metrics = evaluate_regression(y_test.values, enet_pred)
        results["models"]["ElasticNetCV"] = {
            "alpha": float(enet_cv.alpha_),
            "l1_ratio": float(enet_cv.l1_ratio_),
            "metrics": enet_metrics,
            "pred": enet_pred.tolist(),
            "coefs": dict(zip(feature_names, [float(x) for x in enet_cv.coef_.tolist()])),
            "intercept": float(enet_cv.intercept_) if hasattr(enet_cv, "intercept_") else None,
        }
    except Exception as e:
        logger.exception("ElasticNetCV failed: %s", e)
        results["models"]["ElasticNetCV"] = {"success": False, "error": str(e)}

    # -----------------
    # RidgeCV
    # -----------------
    try:
        candidate_alphas = [0.1, 1.0, 10.0, 100.0]
        if "ElasticNetCV" in results and isinstance(results["models"].get("ElasticNetCV"), dict) and results["models"]["ElasticNetCV"].get("alpha"):
            try:
                a = results["models"]["ElasticNetCV"]["alpha"]
                candidate_alphas = sorted(list({a, 0.1, 1.0, 10.0, 100.0}))
            except Exception:
                pass
        ridge_cv = RidgeCV(alphas=candidate_alphas, cv=min(cv_folds, max(2, int(len(X_train) / 5))))
        ridge_cv.fit(X_train.values, y_train.values)
        ridge_pred = ridge_cv.predict(X_test.values)
        ridge_metrics = evaluate_regression(y_test.values, ridge_pred)
        ridge_coefs = ridge_cv.coef_.tolist() if hasattr(ridge_cv, "coef_") else [0.0] * len(feature_names)
        results["models"]["RidgeCV"] = {
            "alpha": float(getattr(ridge_cv, "alpha_", candidate_alphas[0])),
            "metrics": ridge_metrics,
            "pred": ridge_pred.tolist(),
            "coefs": dict(zip(feature_names, [float(x) for x in ridge_coefs])),
            "intercept": float(getattr(ridge_cv, "intercept_", None)) if getattr(ridge_cv, "intercept_", None) is not None else None,
        }
    except Exception as e:
        logger.exception("RidgeCV failed: %s", e)
        results["models"]["RidgeCV"] = {"success": False, "error": str(e)}

    # -----------------
    # LassoCV
    # -----------------
    try:
        lasso_cv = LassoCV(cv=min(cv_folds, max(2, int(len(X_train) / 5))), random_state=42, n_jobs=1)
        lasso_cv.fit(X_train.values, y_train.values)
        lasso_pred = lasso_cv.predict(X_test.values)
        lasso_metrics = evaluate_regression(y_test.values, lasso_pred)
        results["models"]["LassoCV"] = {
            "alpha": float(getattr(lasso_cv, "alpha_", None)),
            "metrics": lasso_metrics,
            "pred": lasso_pred.tolist(),
            "coefs": dict(zip(feature_names, [float(x) for x in lasso_cv.coef_.tolist()])),
            "intercept": float(getattr(lasso_cv, "intercept_", None)) if getattr(lasso_cv, "intercept_", None) is not None else None,
        }
    except Exception as e:
        logger.exception("LassoCV failed: %s", e)
        results["models"]["LassoCV"] = {"success": False, "error": str(e)}

    # -----------------
    # Choose best model by R2 (primary), then RMSE
    # -----------------
    candidate_scores: List[Tuple[str, float, float]] = []
    for mname, info in results["models"].items():
        if isinstance(info, dict) and info.get("metrics"):
            candidate_scores.append((mname, info["metrics"]["r2"], info["metrics"]["rmse"]))

    if not candidate_scores:
        raise RuntimeError("No model produced valid results in strict mode.")

    selected = sorted(candidate_scores, key=lambda x: (-x[1], x[2]))[0]
    best_model_name = selected[0]
    best_model_info = results["models"][best_model_name]

    # -----------------
    # Fit best model on full dataset for persistence and future attribution
    # -----------------
    saved_model_path = None
    saved_meta_path = None
    final_model_obj = None
    coefs: np.ndarray

    if best_model_name == "ElasticNetCV":
        try:
            chosen_alpha = best_model_info.get("alpha")
            chosen_l1 = best_model_info.get("l1_ratio")
            enet = ElasticNet(alpha=chosen_alpha, l1_ratio=chosen_l1, random_state=42)
            enet.fit(X_all.values, y.values)
            final_model_obj = enet
            coefs = np.asarray(enet.coef_)
        except Exception as e:
            logger.exception("Refit ElasticNet failed: %s", e)
            raise

    elif best_model_name == "RidgeCV":
        try:
            chosen_alpha = best_model_info.get("alpha")
            from sklearn.linear_model import Ridge

            ridge = Ridge(alpha=chosen_alpha if chosen_alpha is not None else 1.0)
            ridge.fit(X_all.values, y.values)
            final_model_obj = ridge
            coefs = np.asarray(ridge.coef_)
        except Exception as e:
            logger.exception("Refit Ridge failed: %s", e)
            raise

    elif best_model_name == "LassoCV":
        try:
            chosen_alpha = best_model_info.get("alpha")
            from sklearn.linear_model import Lasso

            lasso = Lasso(alpha=chosen_alpha if chosen_alpha is not None else 1.0, random_state=42)
            lasso.fit(X_all.values, y.values)
            final_model_obj = lasso
            coefs = np.asarray(lasso.coef_)
        except Exception as e:
            logger.exception("Refit Lasso failed: %s", e)
            raise

    else:
        # Fallback to OLS refit
        try:
            lr = LinearRegression()
            lr.fit(X_all.values, y.values)
            final_model_obj = lr
            coefs = np.asarray(lr.coef_)
        except Exception as e:
            logger.exception("Fallback OLS failed: %s", e)
            raise

    # Save model artifact & metadata
    saved_model_path = save_model_artifact(final_model_obj, best_model_name, models_dir=models_dir)
    metadata = {
        "model_name": best_model_name,
        "target_column": target_col,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "feature_names": feature_names,
        "spend_columns_detected": spend_cols,
        "adstock_applied": apply_adstock,
        "adstock_decay": adstock_decay,
        "saturation_alpha": saturation_alpha,
        "train_n": int(len(X_all)),
        "gemini_model_targets": gemini_analysis.get("analysis", {}).get("model_targets") if gemini_analysis else None,
    }
    saved_meta_path = save_metadata(metadata, saved_model_path, models_dir=models_dir)

    # Attribution mapping (best-effort)
    try:
        attribution = compute_attribution_from_coefs(coefs, feature_names, spend_cols, df_raw, adstock_decay=adstock_decay, saturation_alpha=saturation_alpha)
    except Exception:
        attribution = {}

    results["best_model"] = {
        "name": best_model_name,
        "metrics": best_model_info.get("metrics"),
        "coefs": dict(zip(feature_names, [float(c) for c in coefs.tolist()])),
        "attribution": attribution,
    }

    results["artifact"] = {"model_path": saved_model_path, "meta_path": saved_meta_path}
    results["success"] = True
    results["target_column"] = target_col
    results["feature_names"] = feature_names
    results["spend_columns"] = spend_cols
    results["train_n"] = int(len(X_all))
    results["test_metrics"] = {
        m: results["models"][m]["metrics"]
        for m in results["models"]
        if isinstance(results["models"][m], dict) and results["models"][m].get("metrics")
    }

    return results


def analyze_file_and_run_pipeline(file_path: str, gemini_response: Dict, models_dir: str = "models", **kwargs) -> Dict:
    """
    Entry point compatible with gemini_analyser trigger_models_for_file / analyze_and_run_pipeline pattern.

    Expects gemini_response (the structure produced by analyze_file_with_gemini).
    If gemini_response.analysis.model_type indicates "Marketing ROI & Attribution Model", run pipeline.
    """
    try:
        model_type = gemini_response.get("analysis", {}).get("model_type")
    except Exception:
        model_type = None

    if model_type and model_type.strip().lower() == "marketing roi & attribution model":
        pipeline_result = run_marketing_mmm_pipeline(file_path, gemini_analysis=gemini_response, models_dir=models_dir, **kwargs)
        return {"success": True, "gemini_analysis": gemini_response, "pipeline": pipeline_result}
    else:
        return {"success": False, "error": "Gemini did not indicate Marketing ROI & Attribution Model in strict mode."}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--gemini-json", required=False, help="Path to gemini analysis json (optional)")
    parser.add_argument("--models-dir", required=False, default="models", help="Directory to save models")
    parser.add_argument("--test-frac", required=False, type=float, default=0.2, help="Test fraction")
    parser.add_argument("--no-adstock", action="store_true", help="Disable adstock transform")
    parser.add_argument("--adstock-decay", required=False, type=float, default=0.5, help="Adstock decay")
    parser.add_argument("--saturation-alpha", required=False, type=float, default=0.7, help="Saturation alpha (0<alpha<=1)")
    args = parser.parse_args()

    gemini_resp = None
    if args.gemini_json and os.path.exists(args.gemini_json):
        with open(args.gemini_json, "r", encoding="utf-8") as f:
            gemini_resp = json.load(f)

    out = analyze_file_and_run_pipeline(
        args.csv,
        gemini_resp or {},
        models_dir=args.models_dir,
        test_frac=args.test_frac,
        apply_adstock=not args.no_adstock,
        adstock_decay=args.adstock_decay,
        saturation_alpha=args.saturation_alpha,
    )
    print(json.dumps(out, indent=2, default=str))
