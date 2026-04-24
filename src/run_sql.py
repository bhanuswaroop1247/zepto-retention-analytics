"""
run_sql.py
Executes sql/02_feature_engineering.sql against the SQLite database
and exports the analytical_mart table to outputs/analytical_mart.csv.
Run from project root: python src/run_sql.py
"""

from pathlib import Path
import sqlite3
import pandas as pd

ROOT     = Path(__file__).parent.parent
DB_PATH  = ROOT / "data" / "processed" / "zepto.db"
SQL_PATH = ROOT / "sql" / "02_feature_engineering.sql"
OUT_PATH = ROOT / "outputs" / "analytical_mart.csv"


# --- functions ---
def run():
    print("=" * 50)
    print("Running SQL feature engineering...")
    print("=" * 50)

    sql = SQL_PATH.read_text()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Execute each statement separately, stripping leading comment lines
    for raw in sql.split(";"):
        lines = [ln for ln in raw.splitlines() if not ln.strip().startswith("--")]
        stmt = "\n".join(lines).strip()
        if stmt:
            cursor.execute(stmt)
    conn.commit()

    # Row count
    count = cursor.execute("SELECT COUNT(*) FROM analytical_mart").fetchone()[0]
    print(f"\n  analytical_mart row count: {count:,}")

    # First 3 rows
    df = pd.read_sql("SELECT * FROM analytical_mart LIMIT 3", conn)
    print("\n  First 3 rows:")
    print(df.to_string(index=False))

    # Export CSV
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full_df = pd.read_sql("SELECT * FROM analytical_mart", conn)
    full_df.to_csv(OUT_PATH, index=False)
    print(f"\n  Exported to {OUT_PATH}  ({len(full_df):,} rows)")

    conn.close()
    print("\n" + "=" * 50)
    print("Phase 2 Complete!")
    print("=" * 50)


# --- main ---
if __name__ == "__main__":
    run()
