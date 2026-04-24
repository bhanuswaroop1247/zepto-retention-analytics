"""
data_loader.py
Loads all 6 sheets from Zepto_Dataset.xlsx into a local SQLite database.
Run from project root: python src/data_loader.py
"""

from pathlib import Path
import pandas as pd
import sqlite3

ROOT      = Path(__file__).parent.parent
XLSX_PATH = ROOT / "data" / "raw" / "Zepto_Dataset.xlsx"
DB_PATH   = ROOT / "data" / "processed" / "zepto.db"

# --- constants ---
SHEETS = ["customer", "product", "order", "transaction", "rating", "delivery"]

EXPECTED_ROWS = {
    "customer":    10_000,
    "product":      1_200,
    "order":       20_000,
    "transaction": 50_000,
    "rating":      20_000,
    "delivery":    20_000,
}


# --- functions ---
def load_all_sheets():
    if not XLSX_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {XLSX_PATH}. "
            "Please place Zepto_Dataset.xlsx in data/raw/"
        )

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    print("=" * 50)
    print("Loading Zepto_Dataset.xlsx into SQLite...")
    print("=" * 50)

    total_rows = 0
    for sheet in SHEETS:
        df = pd.read_excel(XLSX_PATH, sheet_name=sheet)
        df.to_sql(sheet, conn, if_exists="replace", index=False)
        expected = EXPECTED_ROWS.get(sheet, "?")
        status = "OK" if len(df) == expected else "CHECK"
        print(f"  {status:<6} {sheet:<15} {len(df):>7,} rows  (expected: {expected:,})")
        total_rows += len(df)

    conn.close()
    print("=" * 50)
    print(f"  Total rows loaded : {total_rows:,}")
    print(f"  Database saved at : {DB_PATH}")
    print("=" * 50)


# --- main ---
if __name__ == "__main__":
    load_all_sheets()
