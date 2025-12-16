# services/customer_segmentation_pipeline.py
# Customer Segmentation & Modeling pipeline (full file).
# - Unsupervised clustering (KMeans / GMM / Autoencoder+KMeans) for segmentation
# - RFM and simple LTV calculations for value segmentation
# - Predictive models (Gradient Boosting, XGBoost, or Neural Net) for churn risk / predictive segments
# - Persists artifacts and metadata; returns JSON-serializable results
#
# Usage:
#   from services.customer_segmentation_pipeline import analyze_file_and_run_pipeline
#   analyze_file_and_run_pipeline(csv_path, gemini_response, models_dir="models")
#
# Notes:
# - CSV expected to contain at minimum:
#     * customer identifier column
#     * transaction/order/date column to compute recency/frequency
#     * monetary/amount column (order value)
# - Optional: churn label column for supervised training, features columns for predictive models.

import os
import math
import json
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Optional ML dependencies
try:
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, silhouette_score
    from sklearn.decomposition import PCA
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except Exception:
    _HAS_XGBOOST = False

# Keras / TensorFlow for autoencoder / neural net fallback
try:
    import tensorflow as tf
    from tensorflow import keras
    _HAS_KERAS = True
except Exception:
    _HAS_KERAS = False

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

COMMON_DATE_KEYWORDS = ["date", "datetime", "ds", "order_date", "trans_date", "timestamp", "time"]
_CUSTOMER_KEYWORDS = ["customer", "cust", "buyer", "client", "user", "customer_id"]
_ORDER_ID_KEYWORDS = ["order", "order_id", "transaction", "trans_id", "invoice"]
_MONETARY_KEYWORDS = ["amount", "revenue", "monetary", "total", "price", "order_value", "sales"]
_QUANTITY_KEYWORDS = ["qty", "quantity", "units"]
_CHURN_KEYWORDS = ["churn", "churned", "is_churn", "attrited", "churn_flag"]

def read_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def detect_columns_by_keywords(columns: List[str], keywords: List[str]) -> Optional[str]:
    low = [c.lower() for c in columns]
    for kw in keywords:
        for i, c in enumerate(low):
            if kw in c:
                return columns[i]
    return None


def detect_customer_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _CUSTOMER_KEYWORDS)


def detect_date_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, COMMON_DATE_KEYWORDS)


def detect_monetary_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _MONETARY_KEYWORDS)


def detect_order_col(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _ORDER_ID_KEYWORDS)


def detect_quantity_column(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _QUANTITY_KEYWORDS)


def detect_churn_col(columns: List[str]) -> Optional[str]:
    return detect_columns_by_keywords(columns, _CHURN_KEYWORDS)


# ---- utility: safe numeric conversion ----


def _to_float_safe(val: Any, default: float = 0.0) -> float:
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


# ---- RFM & LTV helpers ----


def compute_rfm(df: pd.DataFrame, customer_col: str, date_col: str, monetary_col: Optional[str] = None, as_of_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Compute Recency, Frequency, Monetary (RFM) table per customer.
    Recency = days since last purchase (smaller is better)
    Frequency = count of purchases
    Monetary = total spend (or average)
    """
    if as_of_date is None:
        try:
            # pick max date in data + 1 day
            as_of_date = pd.to_datetime(df[date_col], errors="coerce").max() + pd.Timedelta(days=1)
        except Exception:
            as_of_date = pd.Timestamp.utcnow()

    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
    grouped = df_copy.groupby(customer_col).agg(
        recency_days=(date_col, lambda d: (as_of_date - d.max()).days if not d.isnull().all() else np.nan),
        frequency=(date_col, "count")
    )
    if monetary_col and monetary_col in df_copy.columns:
        grouped["monetary"] = df_copy.groupby(customer_col)[monetary_col].sum()
    else:
        grouped["monetary"] = df_copy.groupby(customer_col).size() * 0.0

    # fillna
    grouped = grouped.fillna({"recency_days": grouped["recency_days"].max() if not grouped["recency_days"].empty else 0.0, "frequency": 0, "monetary": 0.0})
    grouped = grouped.reset_index()
    return grouped


def rfm_score(rfm_df: pd.DataFrame, recency_bins: int = 5, frequency_bins: int = 5, monetary_bins: int = 5) -> pd.DataFrame:
    """
    Compute RFM quintile-like scores (1..recency_bins where higher is better for recency/frequency/monetary).
    For recency we invert (smaller recency better -> higher score).
    """
    df = rfm_df.copy()
    # recency: smaller is better, so rank ascending then invert
    df["r_rank"] = pd.qcut(df["recency_days"].rank(method="first"), recency_bins, labels=False, duplicates="drop") if len(df) >= recency_bins else pd.cut(df["recency_days"].rank(method="first"), recency_bins, labels=False)
    # invert so 5 means most recent
    try:
        df["r_score"] = recency_bins - df["r_rank"]
    except Exception:
        df["r_score"] = 0
    # frequency & monetary: larger is better
    for col, bins, label in [("frequency", frequency_bins, "f_score"), ("monetary", monetary_bins, "m_score")]:
        try:
            df[label.replace("_score", "") + "_rank"] = pd.qcut(df[col].rank(method="first"), bins, labels=False, duplicates="drop") if len(df) >= bins else pd.cut(df[col].rank(method="first"), bins, labels=False)
            df[label] = df[label.replace("_score", "") + "_rank"] + 1
        except Exception:
            df[label] = 0

    # ensure numeric
    df["r_score"] = pd.to_numeric(df.get("r_score", 0), errors="coerce").fillna(0).astype(float)
    df["f_score"] = pd.to_numeric(df.get("f_score", 0), errors="coerce").fillna(0).astype(float)
    df["m_score"] = pd.to_numeric(df.get("m_score", 0), errors="coerce").fillna(0).astype(float)

    df["RFM_score"] = df["r_score"] + df["f_score"] + df["m_score"]
    return df[[c for c in df.columns if c not in ["r_rank", "frequency_rank", "monetary_rank"]]]


def compute_simple_ltv(rfm_df: pd.DataFrame, period_days: float = 30.0, expected_life_periods: float = 12.0) -> Dict[str, float]:
    """
    Simple LTV estimator: avg_order_value * purchases_per_period * expected_life_periods
    avg_order_value = monetary / frequency (handle frequency=0)
    purchases_per_period = frequency / (total_periods) approximated by frequency per period_days
    For simplicity, assume period_days and expected_life_periods provided.
    Returns per-customer LTV map.
    """
    ltv_map: Dict[str, float] = {}
    for _, row in rfm_df.iterrows():
        cust = row.get(rfm_df.columns[0])  # assume first column is customer id
        freq = _to_float_safe(row.get("frequency", 0.0), default=0.0)
        monetary = _to_float_safe(row.get("monetary", 0.0), default=0.0)
        avg_order = monetary / freq if freq > 0 else 0.0
        purchases_per_period = (freq / max(1.0, (period_days)))  # rough normalization; user can provide better
        ltv = avg_order * purchases_per_period * expected_life_periods
        ltv_map[str(cust)] = float(max(0.0, ltv))
    return ltv_map


# ---- clustering models ----


def run_kmeans(X: np.ndarray, n_clusters: int = 4, random_state: int = 42) -> Tuple[Any, np.ndarray]:
    if not _HAS_SKLEARN:
        raise RuntimeError("scikit-learn not available for KMeans")
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(X)
    return model, labels


def run_gmm(X: np.ndarray, n_components: int = 4, random_state: int = 42) -> Tuple[Any, np.ndarray]:
    if not _HAS_SKLEARN:
        raise RuntimeError("scikit-learn not available for GMM")
    model = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = model.fit_predict(X)
    return model, labels


def run_autoencoder_kmeans(X: np.ndarray, n_clusters: int = 4, latent_dim: int = 8, epochs: int = 50, batch_size: int = 64) -> Tuple[Optional[Any], Optional[Any], Optional[np.ndarray]]:
    """
    Safe autoencoder + KMeans:
      - Lazy-imports TensorFlow/Keras and forces CPU if env var or TF not configured.
      - On any failure falls back to PCA + KMeans.
    Returns (autoencoder_model or None, encoder_model or None, labels or None).
    """
    # quick guard: if keras not available, fallback to PCA
    if not _HAS_KERAS:
        try:
            # PCA fallback using sklearn (if available)
            if _HAS_SKLEARN:
                pca = PCA(n_components=min(latent_dim, X.shape[1]))
                Z = pca.fit_transform(X)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(Z)
                return None, None, labels
            else:
                return None, None, None
        except Exception:
            return None, None, None

    # If user environment has GPUs but they are misconfigured, forcing CPU can avoid nvlink/compilation issues.
    try:
        import os
        # If not explicitly set, force CPU for autoencoder training to avoid GPU compilation errors on systems
        # where CUDA toolchain is missing or mismatched. You can set this env var externally to use GPU.
        if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Now import TF/Keras lazily
        import tensorflow as tf
        from tensorflow import keras

        tf.keras.backend.clear_session()
        input_dim = X.shape[1]
        inputs = keras.Input(shape=(input_dim,))
        x = keras.layers.Dense(int(max(16, input_dim // 2)), activation="relu")(inputs)
        x = keras.layers.Dense(latent_dim, activation="relu")(x)
        encoded = x
        x = keras.layers.Dense(int(max(16, input_dim // 2)), activation="relu")(encoded)
        outputs = keras.layers.Dense(input_dim, activation="linear")(x)
        autoencoder = keras.Model(inputs, outputs)
        encoder = keras.Model(inputs, encoded)
        autoencoder.compile(optimizer="adam", loss="mse")

        # Train quietly and defensively
        autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=0)
        Z = encoder.predict(X, verbose=0)
        # then cluster
        if _HAS_SKLEARN:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(Z)
            return autoencoder, encoder, labels
        else:
            # no sklearn — return encoder output only
            return autoencoder, encoder, None

    except Exception as e:
        logger.exception("Autoencoder train/predict failed; falling back to PCA+KMeans: %s", e)
        # fallback: PCA + KMeans (if sklearn installed)
        try:
            if _HAS_SKLEARN:
                pca = PCA(n_components=min(latent_dim, X.shape[1]))
                Z = pca.fit_transform(X)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(Z)
                return None, None, labels
        except Exception as ex2:
            logger.exception("PCA fallback also failed: %s", ex2)
        return None, None, None



# ---- predictive churn / segment models ----


def train_predictive_churn_model(X: np.ndarray, y: np.ndarray, method: str = "gbm", random_state: int = 42) -> Tuple[Optional[Any], Dict[str, float]]:
    """
    Train a predictive model for churn risk.
    method: 'gbm' -> sklearn GradientBoostingClassifier (or xgboost if available), 'nn' -> simple Keras NN
    Returns (model, metrics)
    """
    metrics: Dict[str, float] = {}
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y if len(np.unique(y))>1 else None)
    except Exception:
        X_train, X_test, y_train, y_test = X, X, y, y

    if method == "gbm":
        # prefer xgboost if available
        if _HAS_XGBOOST:
            try:
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dtest = xgb.DMatrix(X_test, label=y_test)
                params = {"objective": "binary:logistic", "eval_metric": "auc", "seed": random_state}
                bst = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, "eval")], verbose_eval=False)
                preds = bst.predict(dtest)
                auc = float(roc_auc_score(y_test, preds)) if len(np.unique(y_test)) > 1 else 0.5
                metrics["auc"] = auc
                return bst, metrics
            except Exception:
                logger.exception("XGBoost training failed, falling back to sklearn GBM")
        # sklearn GBM fallback
        if _HAS_SKLEARN:
            clf = GradientBoostingClassifier(random_state=random_state)
            try:
                clf.fit(X_train, y_train)
                preds = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X_test)
                auc = float(roc_auc_score(y_test, preds)) if len(np.unique(y_test)) > 1 else 0.5
                metrics["auc"] = auc
                return clf, metrics
            except Exception:
                logger.exception("Sklearn GBM training failed")
                return None, {}
        else:
            return None, {}
    elif method == "nn":
        if not _HAS_KERAS:
            return None, {}
        try:
            tf.keras.backend.clear_session()
            model = keras.Sequential([
                keras.layers.Input(shape=(X_train.shape[1],)),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid")
            ])
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
            model.fit(X_train, y_train, epochs=30, batch_size=64, verbose=0)
            preds = model.predict(X_test, verbose=0).ravel()
            auc = float(roc_auc_score(y_test, preds)) if len(np.unique(y_test)) > 1 else 0.5
            metrics["auc"] = auc
            return model, metrics
        except Exception:
            logger.exception("Keras NN training failed")
            return None, {}
    else:
        return None, {}


# ---- sanitize helper (JSON-friendly) ----


def _sanitize_for_json(obj: Any) -> Any:
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
        # Keras / sklearn models cannot be serialized; return their class name
        if hasattr(obj, "__class__") and (obj.__class__.__module__.startswith("sklearn") or obj.__class__.__module__.startswith("xgboost") or obj.__class__.__module__.startswith("tensorflow") or obj.__class__.__module__.startswith("keras")):
            return {"model_class": obj.__class__.__name__, "module": obj.__class__.__module__}
        return obj


# ---- orchestration ----


def run_customer_segmentation_pipeline(
    file_path: str,
    gemini_analysis: Optional[Dict] = None,
    customer_col_hint: Optional[str] = None,
    date_col_hint: Optional[str] = None,
    monetary_col_hint: Optional[str] = None,
    segmentation_methods: List[str] = None,  # e.g., ["kmeans", "gmm", "autoencoder"]
    n_segments: int = 4,
    run_predictive: bool = True,
    predictive_method: str = "gbm",  # 'gbm' or 'nn'
    churn_label_col_hint: Optional[str] = None,
    models_dir: str = "models",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Top-level pipeline:
      - detect columns, compute RFM/LTV
      - run segmentation (KMeans / GMM / Autoencoder+KMeans)
      - optionally train predictive churn model
      - persist artifact & metadata
    """
    segmentation_methods = segmentation_methods or ["kmeans", "gmm", "autoencoder"]
    df = read_csv(file_path)
    columns = list(df.columns)

    customer_col = customer_col_hint or detect_customer_column(columns)
    date_col = date_col_hint or detect_date_column(columns)
    monetary_col = monetary_col_hint or detect_monetary_column(columns)
    churn_col = churn_label_col_hint or detect_churn_col(columns)

    if not customer_col or not date_col:
        raise ValueError("Could not detect customer column or date column. Provide hints or ensure CSV contains customer id and date columns.")

    # Compute RFM
    rfm = compute_rfm(df, customer_col, date_col, monetary_col)
    rfm_scored = rfm_score(rfm)

    # compute simple LTV
    ltv_map = compute_simple_ltv(rfm)

    # Feature matrix for clustering: use normalized recency (inverse), frequency, monetary, and optionally PCA
    feat_df = rfm_scored.set_index(customer_col)[["recency_days", "frequency", "monetary", "RFM_score"]].copy()
    # transform recency to inverse recency (higher=more recent)
    max_rec = feat_df["recency_days"].max() if not feat_df["recency_days"].empty else 1.0
    feat_df["recency_inv"] = max_rec - feat_df["recency_days"]
    X_raw = feat_df[["recency_inv", "frequency", "monetary", "RFM_score"]].fillna(0.0).values

    # scale features
    scaler = StandardScaler() if _HAS_SKLEARN else None
    if scaler:
        try:
            X = scaler.fit_transform(X_raw)
        except Exception:
            X = X_raw
    else:
        X = X_raw

    segmentation_results: Dict[str, Any] = {}
    clustering_models: Dict[str, Any] = {}

    # KMeans
    if "kmeans" in segmentation_methods:
        try:
            if _HAS_SKLEARN:
                km_model, km_labels = run_kmeans(X, n_clusters=n_segments, random_state=random_state)
                segmentation_results["kmeans_labels"] = {str(cust): int(lab) for cust, lab in zip(feat_df.index.astype(str).tolist(), km_labels.tolist())}
                clustering_models["kmeans_model"] = km_model
                # silhouette
                try:
                    segmentation_results["kmeans_silhouette"] = float(silhouette_score(X, km_labels)) if len(set(km_labels)) > 1 else None
                except Exception:
                    segmentation_results["kmeans_silhouette"] = None
            else:
                segmentation_results["kmeans_error"] = "sklearn not available"
        except Exception as e:
            logger.exception("KMeans failed: %s", e)
            segmentation_results["kmeans_error"] = str(e)

    # GMM
    if "gmm" in segmentation_methods:
        try:
            if _HAS_SKLEARN:
                gmm_model, gmm_labels = run_gmm(X, n_components=n_segments, random_state=random_state)
                segmentation_results["gmm_labels"] = {str(cust): int(lab) for cust, lab in zip(feat_df.index.astype(str).tolist(), gmm_labels.tolist())}
                clustering_models["gmm_model"] = gmm_model
                try:
                    segmentation_results["gmm_silhouette"] = float(silhouette_score(X, gmm_labels)) if len(set(gmm_labels)) > 1 else None
                except Exception:
                    segmentation_results["gmm_silhouette"] = None
            else:
                segmentation_results["gmm_error"] = "sklearn not available"
        except Exception as e:
            logger.exception("GMM failed: %s", e)
            segmentation_results["gmm_error"] = str(e)

    # Autoencoder + KMeans
    if "autoencoder" in segmentation_methods:
        try:
            ae_model, ae_encoder, ae_labels = run_autoencoder_kmeans(X, n_clusters=n_segments, latent_dim=min(8, X.shape[1]), epochs=50, batch_size=64)
            if ae_labels is not None:
                segmentation_results["autoencoder_kmeans_labels"] = {str(cust): int(lab) for cust, lab in zip(feat_df.index.astype(str).tolist(), ae_labels.tolist())}
                clustering_models["autoencoder_model"] = ae_model
                clustering_models["autoencoder_encoder"] = ae_encoder
            else:
                segmentation_results["autoencoder_note"] = "autoencoder not run (Keras not available or training failed)"
        except Exception as e:
            logger.exception("Autoencoder+KMeans failed: %s", e)
            segmentation_results["autoencoder_error"] = str(e)

    # Build a feature table per customer including LTV and segment assignments
    customers = feat_df.index.astype(str).tolist()
    customer_summary = []
    for cust in customers:
        entry = {
            "customer_id": str(cust),
            "recency_days": float(feat_df.loc[cust, "recency_days"]) if cust in feat_df.index else None,
            "frequency": float(feat_df.loc[cust, "frequency"]) if cust in feat_df.index else None,
            "monetary": float(feat_df.loc[cust, "monetary"]) if cust in feat_df.index else None,
            "RFM_score": float(feat_df.loc[cust, "RFM_score"]) if cust in feat_df.index else None,
            "LTV": float(ltv_map.get(str(cust), 0.0)),
        }
        # attach labels if present
        for k, vmap in [("kmeans", segmentation_results.get("kmeans_labels")), ("gmm", segmentation_results.get("gmm_labels")), ("autoencoder", segmentation_results.get("autoencoder_kmeans_labels"))]:
            if isinstance(vmap, dict):
                entry[f"{k}_segment"] = int(vmap.get(str(cust))) if vmap.get(str(cust)) is not None else None
        customer_summary.append(entry)

    # Predictive churn model (optional)
    predictive_result: Dict[str, Any] = {"trained": False}
    if run_predictive:
        # assemble supervised dataset: features from RFM + segments, label from churn_col if exists
        if churn_col and churn_col in df.columns:
            # build per-transaction label; convert to per-customer label by any churn occurrence (or last known)
            df_copy = df.copy()
            # ensure churn label is numeric 0/1
            df_copy[churn_col] = pd.to_numeric(df_copy[churn_col], errors="coerce").fillna(0).astype(int)
            # per-customer label: max(churn)
            y_series = df_copy.groupby(customer_col)[churn_col].max()
            # build X features aligned to customers
            X_list = []
            y_list = []
            custs = []
            for cust in customers:
                cust_row = feat_df.loc[cust] if cust in feat_df.index else None
                if cust_row is None:
                    continue
                features = [
                    float(cust_row["recency_days"]),
                    float(cust_row["frequency"]),
                    float(cust_row["monetary"]),
                    float(cust_row["RFM_score"]),
                    float(ltv_map.get(str(cust), 0.0))
                ]
                # append segment encodings if available (one-hot could be better)
                kseg = segmentation_results.get("kmeans_labels", {}).get(str(cust))
                gseg = segmentation_results.get("gmm_labels", {}).get(str(cust))
                aseg = segmentation_results.get("autoencoder_kmeans_labels", {}).get(str(cust))
                features += [float(kseg) if kseg is not None else -1.0, float(gseg) if gseg is not None else -1.0, float(aseg) if aseg is not None else -1.0]
                # label (default 0)
                lab = int(y_series.get(cust, 0)) if cust in y_series.index else 0
                X_list.append(features)
                y_list.append(lab)
                custs.append(cust)
            if len(X_list) > 0:
                X_arr = np.array(X_list, dtype=float)
                y_arr = np.array(y_list, dtype=int)
                # scale features
                if _HAS_SKLEARN:
                    try:
                        feat_scaler = StandardScaler()
                        X_scaled = feat_scaler.fit_transform(X_arr)
                    except Exception:
                        X_scaled = X_arr
                else:
                    X_scaled = X_arr
                # choose method
                model, metrics = train_predictive_churn_model(X_scaled, y_arr, method=predictive_method)
                if model is not None:
                    predictive_result["trained"] = True
                    predictive_result["model"] = model
                    predictive_result["metrics"] = metrics
                    # predict churn probability per customer in custs
                    try:
                        if _HAS_XGBOOST and model.__class__.__module__.startswith("xgboost"):
                            dmat = xgb.DMatrix(X_scaled)
                            probs = model.predict(dmat)
                        elif hasattr(model, "predict_proba"):
                            probs = model.predict_proba(X_scaled)[:, 1]
                        elif _HAS_KERAS and isinstance(model, keras.Model):
                            probs = model.predict(X_scaled, verbose=0).ravel()
                        else:
                            # fallback: model predict
                            preds = model.predict(X_scaled)
                            probs = np.asarray(preds).ravel()
                        churn_probs = {cust: float(p) for cust, p in zip(custs, probs)}
                        predictive_result["churn_probability_per_customer"] = churn_probs
                    except Exception:
                        predictive_result["churn_probability_per_customer_error"] = "prediction_failed"
                else:
                    predictive_result["trained"] = False
                    predictive_result["error"] = "training_failed_or_no_model"
            else:
                predictive_result["trained"] = False
                predictive_result["note"] = "no training rows (no customers with churn labels found)"
        else:
            predictive_result["trained"] = False
            predictive_result["note"] = "churn_label column not provided; pass churn_label_col_hint to train supervised model"

    # persist artifact
    results = {
        "success": True,
        "customers_count": len(customers),
        "rfm_sample": rfm_scored.head(5).to_dict(orient="records") if not rfm_scored.empty else [],
        "ltv_sample": {k: ltv_map.get(k) for k in list(ltv_map.keys())[:10]},
        "segmentation": segmentation_results,
        "predictive": predictive_result,
        "customer_summary_sample": customer_summary[:20],
    }

    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model": "Customer Segmentation & Modeling Pipeline",
        "segmentation_methods": segmentation_methods,
        "n_segments": n_segments,
        "predictive_method": predictive_method,
        "input_file": os.path.basename(file_path),
    }

    try:
        artifact_obj = {
            "results": results,
            "metadata": metadata,
            "clustering_models": {k: _sanitize_for_json(v) for k, v in clustering_models.items()},
        }
        saved_model_path = save_model_artifact(artifact_obj, "Customer_Segmentation_Plan", models_dir=models_dir)
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
    indicates "Customer Segmentation & Modeling Pipeline".
    """
    try:
        model_type = gemini_response.get("analysis", {}).get("model_type")
    except Exception:
        model_type = None

    if model_type and model_type.strip().lower() == "customer segmentation & modeling":
        pipeline_result = run_customer_segmentation_pipeline(file_path, gemini_analysis=gemini_response, models_dir=models_dir, **kwargs)
        return {"success": True, "gemini_analysis": gemini_response, "pipeline": pipeline_result}
    else:
        return {"success": False, "error": "Gemini did not indicate Customer Segmentation & Modeling Pipeline in strict mode."}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--gemini-json", required=False, help="Path to gemini analysis json (optional)")
    parser.add_argument("--models-dir", required=False, default="models", help="Directory to save models/artifacts")
    parser.add_argument("--customer-col", required=False, help="Customer id column hint")
    parser.add_argument("--date-col", required=False, help="Date column hint")
    parser.add_argument("--monetary-col", required=False, help="Monetary column hint")
    parser.add_argument("--no-predictive", action="store_true", help="Disable predictive churn training")
    parser.add_argument("--n-segments", required=False, type=int, default=4, help="Number of segments for clustering")
    args = parser.parse_args()

    gemini_resp = None
    if args.gemini_json and os.path.exists(args.gemini_json):
        with open(args.gemini_json, "r", encoding="utf-8") as f:
            gemini_resp = json.load(f)

    out = analyze_file_and_run_pipeline(
        args.csv,
        gemini_resp or {},
        models_dir=args.models_dir,
        customer_col_hint=args.customer_col,
        date_col_hint=args.date_col,
        monetary_col_hint=args.monetary_col,
        segmentation_methods=["kmeans", "gmm", "autoencoder"],
        n_segments=args.n_segments,
        run_predictive=(not args.no_predictive),
    )
    print(json.dumps(out, indent=2, default=str))
