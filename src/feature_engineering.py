"""
feature_engineering.py
Loads the analytical_mart table and prepares features for model training.
Run from project root: python src/feature_engineering.py
"""

from pathlib import Path
import sqlite3
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT    = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "processed" / "zepto.db"


# --- functions ---
def load_mart():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM analytical_mart", conn)
    conn.close()
    return df


def prepare_features(df=None):
    if df is None:
        df = load_mart()

    df = df.copy()

    # Impute nulls
    df["avg_rating"]             = df["avg_rating"].fillna(df["avg_rating"].median())
    df["min_rating"]             = df["min_rating"].fillna(3)
    df["avg_delivery_delta"]     = df["avg_delivery_delta"].fillna(0)
    df["max_delivery_delta"]     = df["max_delivery_delta"].fillna(0)
    df["pct_late_orders"]        = df["pct_late_orders"].fillna(0)
    df["days_since_last_order"]  = df["days_since_last_order"].fillna(
        df["days_since_registration"]
    )
    df["pct_cancelled_returned"] = df["pct_cancelled_returned"].fillna(0)

    # Engineer order frequency feature
    df["orders_per_month"] = (
        df["total_orders"] / (df["days_since_registration"].clip(lower=1) / 30)
    )

    # LabelEncode city
    le = LabelEncoder()
    df["city_encoded"] = le.fit_transform(df["city"].astype(str))
    df = df.drop(columns=["city"])

    # Drop non-feature columns + leaky feature (churn_flag is defined from days_since_last_order)
    drop_cols = ["customer_id", "first_order_date", "last_order_date", "state", "clv_tier",
                 "days_since_last_order"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    y = df["churn_flag"]
    X = df.drop(columns=["churn_flag"])

    return X, y


# --- main ---
if __name__ == "__main__":
    X, y = prepare_features()
    print(f"Features shape : {X.shape}")
    print(f"Churn rate     : {y.mean() * 100:.1f}%")
    print(f"Columns        : {X.columns.tolist()}")
