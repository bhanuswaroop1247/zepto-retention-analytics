"""
intervention_engine.py
Applies rule-based intervention logic with ROI guard to scored customers
and outputs a prioritised intervention list with voucher recommendations.
Run from project root: python src/intervention_engine.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT        = Path(__file__).parent.parent
SCORED_PATH = ROOT / "outputs" / "scored_customers.csv"
OUT_PATH    = ROOT / "outputs" / "intervention_list.csv"


# --- functions ---
def apply_rules(row):
    """
    Returns (intervene: bool, voucher_amount: int).
    Rules applied in order — first match wins.
    """
    p     = row["churn_probability"]
    tier  = row["clv_tier"]
    delta = row["avg_delivery_delta"] if pd.notna(row["avg_delivery_delta"]) else 0.0

    if p >= 0.7:
        if tier == "High": return True, 150
        if tier == "Mid":  return True, 80
        if tier == "Low":  return False, 0

    if 0.5 <= p < 0.7:
        if tier == "High":                return True, 80
        if tier == "Mid" and delta > 30:  return True, 40
        if tier == "Mid":                 return False, 0
        if tier == "Low":                 return False, 0

    return False, 0


def roi_guard(row):
    """
    Cancel intervention if voucher >= (1 - churn_prob) * total_clv * 0.15
    i.e. voucher cost exceeds 15% of the expected retained CLV.
    """
    if not row["intervene_flag"]:
        return False, 0
    threshold = (1 - row["churn_probability"]) * row["total_clv"] * 0.15
    if row["recommended_voucher_amount"] >= threshold:
        return False, 0
    return True, row["recommended_voucher_amount"]


def run():
    print("=" * 60)
    print("Phase 5 - Intervention Engine")
    print("=" * 60)

    df = pd.read_csv(SCORED_PATH)

    # Apply rules
    rule_results = df.apply(apply_rules, axis=1)
    df["intervene_flag"]             = rule_results.apply(lambda x: int(x[0]))
    df["recommended_voucher_amount"] = rule_results.apply(lambda x: x[1])

    # Apply ROI guard
    guard_results = df.apply(roi_guard, axis=1)
    df["intervene_flag"]             = guard_results.apply(lambda x: int(x[0]))
    df["recommended_voucher_amount"] = guard_results.apply(lambda x: x[1])

    # Derived columns
    df["estimated_clv_at_risk"] = df["churn_probability"] * df["total_clv"]
    df["intervention_roi"] = df.apply(
        lambda r: (
            ((1 - r["churn_probability"]) * r["estimated_clv_at_risk"]
             - r["recommended_voucher_amount"])
            / r["recommended_voucher_amount"] * 100
        ) if r["recommended_voucher_amount"] > 0 else 0.0,
        axis=1
    )

    # Build output
    out_cols = [
        "customer_id", "churn_probability", "clv_tier", "total_clv",
        "avg_delivery_delta", "intervene_flag", "recommended_voucher_amount",
        "estimated_clv_at_risk", "intervention_roi"
    ]
    out = df[out_cols].copy()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    # Summary
    flagged         = out[out["intervene_flag"] == 1]
    total_flagged   = len(flagged)
    total_budget    = flagged["recommended_voucher_amount"].sum()
    total_clv_risk  = flagged["estimated_clv_at_risk"].sum()
    avg_roi         = flagged["intervention_roi"].mean()
    universal_spend = len(df) * 80
    budget_saving   = (1 - total_budget / universal_spend) * 100

    print(f"\n  Total customers scored        : {len(df):,}")
    print(f"  Customers flagged to intervene: {total_flagged:,}  "
          f"({total_flagged/len(df)*100:.1f}% of base)")
    print(f"\n  Total voucher budget required : Rs. {total_budget:,.0f}")
    print(f"  Universal spend (Rs.80 each)  : Rs. {universal_spend:,.0f}")
    print(f"  Budget saving vs universal    : {budget_saving:.1f}%")
    print(f"\n  Total CLV at risk (flagged)   : Rs. {total_clv_risk:,.0f}")
    print(f"  Estimated avg ROI             : {avg_roi:.1f}%")

    print(f"\n  Breakdown by CLV tier (intervened):")
    tier_summary = (flagged.groupby("clv_tier")
                    .agg(customers=("customer_id", "count"),
                         budget=("recommended_voucher_amount", "sum"),
                         clv_at_risk=("estimated_clv_at_risk", "sum"))
                    .reindex(["High", "Mid", "Low"]).fillna(0))
    for tier, row2 in tier_summary.iterrows():
        print(f"    {tier:<5}  customers={int(row2['customers']):>5,}  "
              f"budget=Rs.{int(row2['budget']):>9,}  "
              f"CLV at risk=Rs.{row2['clv_at_risk']:>12,.0f}")

    print(f"\n  Saved to {OUT_PATH}")
    print("\n" + "=" * 60)
    print("Phase 5 Complete!")
    print("=" * 60)


# --- main ---
if __name__ == "__main__":
    run()
