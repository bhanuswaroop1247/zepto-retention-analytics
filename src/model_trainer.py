"""
model_trainer.py
Trains Logistic Regression, Random Forest, and XGBoost churn models,
saves the best model, generates SHAP plot, and scores all customers.
Run from project root: python src/model_trainer.py
"""

from pathlib import Path
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import load_mart, prepare_features

# --- constants ---
MODEL_PATH  = ROOT / "models" / "xgb_churn_model.pkl"
SHAP_PATH   = ROOT / "outputs" / "shap_feature_importance.png"
SCORED_PATH = ROOT / "outputs" / "scored_customers.csv"


# --- functions ---
def evaluate(name, model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc  = roc_auc_score(y_test, y_prob)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    print(f"  {name:<30s}  AUC={auc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
    return auc, f1, y_prob


def train():
    print("=" * 65)
    print("Phase 4 - ML Pipeline")
    print("=" * 65)

    # Load & split
    X, y = prepare_features()
    print(f"\n  Features shape : {X.shape}")
    print(f"  Churn rate     : {y.mean()*100:.1f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE on training set (for LR and RF)
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    print(f"\n  Train size before SMOTE : {X_train.shape[0]}")
    print(f"  Train size after  SMOTE : {X_train_sm.shape[0]}")
    print(f"  Test  size              : {X_test.shape[0]}\n")

    # Models
    spw = float((y_train == 0).sum()) / float((y_train == 1).sum())
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        # XGBoost uses scale_pos_weight (native imbalance handler) on raw data
        "XGBClassifier": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=42, n_jobs=-1
        ),
    }

    print(f"  {'Model':<30s}  {'AUC':>8}  {'F1':>8}  {'Prec':>8}  {'Rec':>8}")
    print("  " + "-" * 61)

    results = {}
    fitted  = {}
    for name, model in models.items():
        fit_X = X_train if name == "XGBClassifier" else X_train_sm
        fit_y = y_train if name == "XGBClassifier" else y_train_sm
        model.fit(fit_X, fit_y)
        auc, f1, probs = evaluate(name, model, X_test, y_test)
        results[name] = {"auc": auc, "f1": f1, "probs": probs}
        fitted[name]  = model

    # Pick best model by AUC
    best_name  = max(results, key=lambda n: results[n]["auc"])
    best_model = fitted[best_name]
    best_auc   = results[best_name]["auc"]
    best_f1    = results[best_name]["f1"]
    xgb_auc    = results["XGBClassifier"]["auc"]
    xgb_f1     = results["XGBClassifier"]["f1"]

    print(f"\n  Best model by AUC: {best_name}  (AUC={best_auc:.4f})")

    # Save best model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    print(f"  Saved to {MODEL_PATH}")

    # SHAP summary plot
    print("  Generating SHAP values...")
    explainer = shap.TreeExplainer(best_model)
    shap_vals = explainer.shap_values(X_test)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_vals, X_test, show=False, plot_size=None)
    plt.title(f"SHAP Feature Importance - {best_name} Churn Model",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    SHAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(SHAP_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP plot saved to {SHAP_PATH}")

    # Score all customers
    mart = load_mart()
    X_all, _ = prepare_features(mart)
    mart["churn_probability"] = best_model.predict_proba(X_all)[:, 1]
    mart.to_csv(SCORED_PATH, index=False)
    print(f"  Scored customers saved to {SCORED_PATH}  ({len(mart):,} rows)")

    # Final summary
    print("\n" + "=" * 65)
    print(f"  Best model (saved) : {best_name}")
    # Note: AUC target of 0.78 assumed inclusion of days_since_last_order (leaky).
    # With the leaky feature removed and dataset filtered to total_orders > 0,
    # AUC ~0.63 is the honest ceiling on this synthetic dataset.
    print(f"  AUC-ROC            : {best_auc:.4f}  (honest result; leaky feature removed)")
    print(f"  F1                 : {best_f1:.4f}  (target > 0.70)  "
          f"{'PASS' if best_f1 > 0.70 else 'FAIL'}")
    print(f"\n  XGBoost standalone : AUC={xgb_auc:.4f}  F1={xgb_f1:.4f}")
    print("=" * 65)


# --- main ---
if __name__ == "__main__":
    train()
