from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
TARGET = "is_fraud"
FEATURES = [
    "amount",
    "hour",
    "merchant_risk",
    "customer_age_days",
    "velocity_1h",
    "velocity_24h",
    "distance_from_home_km",
    "device_age_days",
    "payment_channel",
    "country_match",
]
NUMERIC_FEATURES = [
    "amount",
    "hour",
    "merchant_risk",
    "customer_age_days",
    "velocity_1h",
    "velocity_24h",
    "distance_from_home_km",
    "device_age_days",
]
CATEGORICAL_FEATURES = ["payment_channel", "country_match"]


def make_transaction_data(rows: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    channel = rng.choice(["card_present", "ecommerce", "wallet", "bank_transfer"], rows, p=[0.42, 0.34, 0.16, 0.08])
    amount = rng.lognormal(mean=3.55, sigma=0.95, size=rows)
    hour = rng.integers(0, 24, rows)
    merchant_risk = rng.beta(1.5, 8.0, rows)
    customer_age_days = rng.gamma(5.0, 130.0, rows).astype(int)
    velocity_1h = rng.poisson(0.7 + (channel == "ecommerce") * 0.35, rows)
    velocity_24h = velocity_1h + rng.poisson(2.5, rows)
    distance = rng.gamma(1.6, 18.0, rows)
    device_age_days = rng.gamma(2.0, 70.0, rows).astype(int)
    country_match = rng.choice(["yes", "no"], rows, p=[0.93, 0.07])

    night = ((hour <= 5) | (hour >= 23)).astype(int)
    logit = (
        -6.2
        + 1.45 * (channel == "ecommerce")
        + 1.15 * (country_match == "no")
        + 0.95 * night
        + 1.10 * (merchant_risk > 0.35)
        + 0.34 * np.log1p(amount)
        + 0.23 * velocity_1h
        + 0.08 * velocity_24h
        + 0.018 * np.minimum(distance, 120)
        - 0.003 * np.minimum(customer_age_days, 1000)
        - 0.004 * np.minimum(device_age_days, 365)
    )
    fraud_probability = 1 / (1 + np.exp(-logit))
    is_fraud = rng.binomial(1, fraud_probability)
    return pd.DataFrame(
        {
            "amount": amount.round(2),
            "hour": hour,
            "merchant_risk": merchant_risk.round(4),
            "customer_age_days": customer_age_days,
            "velocity_1h": velocity_1h,
            "velocity_24h": velocity_24h,
            "distance_from_home_km": distance.round(2),
            "device_age_days": device_age_days,
            "payment_channel": channel,
            "country_match": country_match,
            TARGET: is_fraud,
        }
    )


def build_pipeline(model) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", RobustScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessing = ColumnTransformer([("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)])
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def best_threshold(y_true: pd.Series, score: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    f1_values = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    index = int(np.nanargmax(f1_values[:-1]))
    return float(thresholds[index]), float(f1_values[index])


def evaluate(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
    score = pipeline.predict_proba(X_test)[:, 1]
    threshold, f1_at_threshold = best_threshold(y_test, score)
    prediction = (score >= threshold).astype(int)
    return {
        "model": name,
        "threshold": threshold,
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_at_threshold),
        "average_precision": float(average_precision_score(y_test, score)),
        "roc_auc": float(roc_auc_score(y_test, score)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fraud detection models and choose decision thresholds.")
    parser.add_argument("--rows", type=int, default=45000)
    parser.add_argument("--random-state", type=int, default=23)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data = make_transaction_data(args.rows, args.random_state)
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=args.random_state)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1200, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=320, min_samples_leaf=8, class_weight="balanced_subsample", random_state=args.random_state, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=220, learning_rate=0.06, random_state=args.random_state),
    }

    metrics = []
    for name, model in candidates.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, MODELS_DIR / f"{name}.joblib")
        metrics.append(evaluate(name, pipeline, X_test, y_test))

    best = max(metrics, key=lambda row: (row["average_precision"], row["f1"]))
    with open(MODELS_DIR / "comparison.json", "w", encoding="utf-8") as handle:
        json.dump({"metrics": metrics, "best_model": best["model"], "features": FEATURES}, handle, indent=2)

    print(pd.DataFrame(metrics).sort_values("average_precision", ascending=False).to_string(index=False))
    print(f"\nBest model: {best['model']} at threshold {best['threshold']:.4f}")


if __name__ == "__main__":
    main()
