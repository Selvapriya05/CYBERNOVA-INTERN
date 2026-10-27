
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
RESULTS_DIR = PROJECT_DIR / "output" / "analysis_results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("E-COMMERCE DATA QUALITY ANALYSIS")
print("=" * 70)

# Expected primary or candidate keys for each dataset
key_columns = {
    "olist_customers_dataset.csv": ["customer_id"],
    "olist_orders_dataset.csv": ["order_id"],
    "olist_products_dataset.csv": ["product_id"],
    "olist_sellers_dataset.csv": ["seller_id"],
    "olist_order_items_dataset.csv": ["order_id", "order_item_id"],
    "olist_order_payments_dataset.csv": ["order_id", "payment_sequential"],
    "olist_order_reviews_dataset.csv": ["review_id"],
    "olist_geolocation_dataset.csv": None,
    "product_category_name_translation.csv": ["product_category_name"],
}

quality_results = []

for filename, keys in key_columns.items():
    file_path = DATASET_DIR / filename

    if not file_path.exists():
        print(f"\nWARNING: File not found: {filename}")
        continue

    df = pd.read_csv(file_path)

    print(f"\n{'-' * 70}")
    print(f"FILE: {filename}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Exact duplicate rows: {df.duplicated().sum():,}")
    print(f"Total missing cells: {df.isna().sum().sum():,}")

    if keys:
        print(f"Key columns: {keys}")
        print(f"Rows with missing key: {df[keys].isna().any(axis=1).sum():,}")
        print(f"Duplicate key combinations: {df.duplicated(subset=keys).sum():,}")

    # Detailed missing values
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        print("\nColumns with missing values:")
        print(missing.to_string())
    else:
        print("No missing values.")

    # Save one summary record for this file
    quality_results.append({
        "file_name": filename,
        "row_count": len(df),
        "column_count": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_key_combinations": (
            int(df.duplicated(subset=keys).sum()) if keys else None
        ),
    })

# Save the combined quality report
quality_df = pd.DataFrame(quality_results)
report_path = RESULTS_DIR / "data_quality_summary.csv"
quality_df.to_csv(report_path, index=False)

print("\n" + "=" * 70)
print("DATA QUALITY SUMMARY")
print("=" * 70)
print(quality_df.to_string(index=False))
print(f"\nReport saved to: {report_path}")
print("\nData quality analysis completed.")