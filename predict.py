from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODEL_FILES = {
    "logistic": "logistic_regression.joblib",
    "forest": "random_forest.joblib",
    "boosting": "hist_gradient_boosting.joblib",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a transaction for fraud risk.")
    parser.add_argument("--model", choices=MODEL_FILES.keys(), default="boosting")
    parser.add_argument("--amount", type=float, required=True)
    parser.add_argument("--hour", type=int, choices=range(24), required=True)
    parser.add_argument("--merchant-risk", type=float, required=True)
    parser.add_argument("--customer-age-days", type=int, required=True)
    parser.add_argument("--velocity-1h", type=int, required=True)
    parser.add_argument("--velocity-24h", type=int, required=True)
    parser.add_argument("--distance-from-home-km", type=float, required=True)
    parser.add_argument("--device-age-days", type=int, required=True)
    parser.add_argument("--payment-channel", choices=["card_present", "ecommerce", "wallet", "bank_transfer"], required=True)
    parser.add_argument("--country-match", choices=["yes", "no"], required=True)
    return parser.parse_args()


def load_threshold(model_file: str) -> float:
    comparison_path = MODELS_DIR / "comparison.json"
    if not comparison_path.exists():
        return 0.5
    with open(comparison_path, "r", encoding="utf-8") as handle:
        comparison = json.load(handle)
    model_name = model_file.removesuffix(".joblib")
    for row in comparison.get("metrics", []):
        if row.get("model") == model_name:
            return float(row.get("threshold", 0.5))
    return 0.5


def main() -> None:
    args = parse_args()
    model_file = MODEL_FILES[args.model]
    model_path = MODELS_DIR / model_file
    if not model_path.exists():
        raise SystemExit(f"Model artifact not found: {model_path}. Run python train.py first.")

    row = pd.DataFrame(
        [
            {
                "amount": args.amount,
                "hour": args.hour,
                "merchant_risk": args.merchant_risk,
                "customer_age_days": args.customer_age_days,
                "velocity_1h": args.velocity_1h,
                "velocity_24h": args.velocity_24h,
                "distance_from_home_km": args.distance_from_home_km,
                "device_age_days": args.device_age_days,
                "payment_channel": args.payment_channel,
                "country_match": args.country_match,
            }
        ]
    )
    pipeline = joblib.load(model_path)
    probability = float(pipeline.predict_proba(row)[0, 1])
    threshold = load_threshold(model_file)
    print(f"fraud_probability={probability:.4f}")
    print(f"decision_threshold={threshold:.4f}")
    print(f"prediction={int(probability >= threshold)}")


if __name__ == "__main__":
    main()
