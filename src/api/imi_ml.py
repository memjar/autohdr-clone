"""
IMI ML Engine — scikit-learn powered analytics for Klaus IMI.
Provides clustering, prediction, segmentation, trend forecasting, and scenario analysis.
"""

import os
import glob
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("imi-ml")

CSV_DIR = os.path.expanduser("~/.axe/klaus_data/csv")
os.makedirs(CSV_DIR, exist_ok=True)


def _load_dataset(dataset: str) -> pd.DataFrame:
    """Load a CSV dataset by name (without .csv extension)."""
    path = os.path.join(CSV_DIR, f"{dataset}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset '{dataset}' not found at {path}")
    return pd.read_csv(path)


def _numeric_cols(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Filter to only numeric columns, raise if none remain."""
    valid = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not valid:
        raise ValueError(f"No numeric columns found among: {columns}. Available numeric: {list(df.select_dtypes(include='number').columns)}")
    return valid


def list_datasets() -> list[dict]:
    """List available CSV datasets with their numeric columns."""
    results = []
    for path in sorted(glob.glob(os.path.join(CSV_DIR, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            df = pd.read_csv(path, nrows=5)
            num_cols = list(df.select_dtypes(include="number").columns)
            all_cols = list(df.columns)
            results.append({
                "dataset": name,
                "rows": sum(1 for _ in open(path)) - 1,
                "columns": all_cols,
                "numeric_columns": num_cols,
            })
        except Exception as e:
            results.append({"dataset": name, "error": str(e)})
    return results


def cluster_data(dataset: str, columns: list[str], n_clusters: int = 3) -> dict:
    """KMeans clustering on specified columns."""
    df = _load_dataset(dataset)
    cols = _numeric_cols(df, columns)
    X = df[cols].dropna()

    if len(X) < n_clusters:
        raise ValueError(f"Not enough rows ({len(X)}) for {n_clusters} clusters")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)

    # Build profile summary per cluster
    X_with_labels = X.copy()
    X_with_labels["cluster"] = labels
    profiles = []
    for k in range(n_clusters):
        subset = X_with_labels[X_with_labels["cluster"] == k][cols]
        profiles.append({
            "cluster": k,
            "size": int((labels == k).sum()),
            "mean": subset.mean().round(3).to_dict(),
            "std": subset.std().round(3).to_dict(),
        })

    centroids_original = scaler.inverse_transform(km.cluster_centers_)

    return {
        "n_clusters": n_clusters,
        "columns_used": cols,
        "total_rows": len(X),
        "cluster_assignments": labels.tolist(),
        "centroids": np.round(centroids_original, 3).tolist(),
        "profiles": profiles,
        "inertia": round(float(km.inertia_), 3),
    }


def predict(dataset: str, target_column: str, feature_columns: list[str]) -> dict:
    """Linear regression or RandomForest prediction."""
    df = _load_dataset(dataset)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available: {list(df.columns)}")
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError(f"Target column '{target_column}' is not numeric")

    feat_cols = _numeric_cols(df, feature_columns)
    subset = df[feat_cols + [target_column]].dropna()

    X = subset[feat_cols].values
    y = subset[target_column].values

    if len(subset) < 5:
        raise ValueError(f"Not enough rows ({len(subset)}) for prediction")

    # Use RandomForest for more features, Linear for simple cases
    if len(feat_cols) <= 2 and len(subset) < 50:
        model = LinearRegression()
        model_type = "LinearRegression"
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
        model_type = "RandomForest"

    model.fit(X, y)
    preds = model.predict(X)
    r2 = r2_score(y, preds)

    if model_type == "RandomForest":
        importances = dict(zip(feat_cols, np.round(model.feature_importances_, 4).tolist()))
    else:
        importances = dict(zip(feat_cols, np.round(np.abs(model.coef_), 4).tolist()))

    return {
        "model": model_type,
        "target": target_column,
        "features": feat_cols,
        "r2_score": round(float(r2), 4),
        "feature_importances": importances,
        "predictions": np.round(preds, 3).tolist(),
        "actual": np.round(y, 3).tolist(),
        "n_samples": len(subset),
    }


def segment(dataset: str, columns: Optional[list[str]] = None) -> dict:
    """Auto-segmentation: find optimal k via silhouette score (k=2..6)."""
    df = _load_dataset(dataset)

    if columns:
        cols = _numeric_cols(df, columns)
    else:
        cols = list(df.select_dtypes(include="number").columns)
        if not cols:
            raise ValueError("No numeric columns in dataset")

    X = df[cols].dropna()
    if len(X) < 3:
        raise ValueError(f"Not enough rows ({len(X)}) for segmentation")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    max_k = min(6, len(X) - 1)
    best_k, best_score = 2, -1
    scores = {}

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        s = silhouette_score(X_scaled, labels)
        scores[k] = round(float(s), 4)
        if s > best_score:
            best_k, best_score = k, s

    # Final clustering with best k
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)

    X_with_labels = X.copy()
    X_with_labels["segment"] = labels
    profiles = []
    for k in range(best_k):
        subset = X_with_labels[X_with_labels["segment"] == k][cols]
        profiles.append({
            "segment": k,
            "size": int((labels == k).sum()),
            "pct": round(float((labels == k).sum()) / len(X) * 100, 1),
            "mean": subset.mean().round(3).to_dict(),
        })

    return {
        "optimal_k": best_k,
        "silhouette_score": round(float(best_score), 4),
        "silhouette_scores_by_k": scores,
        "columns_used": cols,
        "total_rows": len(X),
        "segments": profiles,
    }


def trend_forecast(dataset: str, column: str, periods: int = 4) -> dict:
    """Simple linear extrapolation of a numeric column."""
    df = _load_dataset(dataset)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric")

    values = df[column].dropna().values
    if len(values) < 3:
        raise ValueError(f"Not enough data points ({len(values)}) for trend forecast")

    x = np.arange(len(values)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, values)

    future_x = np.arange(len(values), len(values) + periods).reshape(-1, 1)
    forecast = model.predict(future_x)

    return {
        "column": column,
        "n_historical": len(values),
        "periods_forecast": periods,
        "slope": round(float(model.coef_[0]), 4),
        "intercept": round(float(model.intercept_), 4),
        "r2": round(float(r2_score(values, model.predict(x))), 4),
        "historical": np.round(values, 3).tolist(),
        "forecast": np.round(forecast, 3).tolist(),
    }


def scenario_analysis(dataset: str, column: str, change_pct: float) -> dict:
    """What-if: apply % change to column, show downstream correlations affected."""
    df = _load_dataset(dataset)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric")

    numeric_df = df.select_dtypes(include="number").dropna()
    if column not in numeric_df.columns:
        raise ValueError(f"Column '{column}' has no numeric data after dropping NaN")

    original_mean = float(numeric_df[column].mean())
    new_mean = original_mean * (1 + change_pct / 100)

    # Compute correlations with other numeric columns
    other_cols = [c for c in numeric_df.columns if c != column]
    correlations = {}
    impacts = {}

    for c in other_cols:
        corr = float(numeric_df[column].corr(numeric_df[c]))
        correlations[c] = round(corr, 4)
        # Estimated impact: correlation * change_pct (simplified linear assumption)
        estimated_change = corr * change_pct
        impacts[c] = {
            "correlation": round(corr, 4),
            "estimated_change_pct": round(estimated_change, 2),
            "original_mean": round(float(numeric_df[c].mean()), 3),
            "estimated_new_mean": round(float(numeric_df[c].mean()) * (1 + estimated_change / 100), 3),
        }

    return {
        "column": column,
        "change_pct": change_pct,
        "original_mean": round(original_mean, 3),
        "new_mean": round(new_mean, 3),
        "correlated_impacts": impacts,
        "strongest_positive": max(correlations, key=correlations.get) if correlations else None,
        "strongest_negative": min(correlations, key=correlations.get) if correlations else None,
    }
