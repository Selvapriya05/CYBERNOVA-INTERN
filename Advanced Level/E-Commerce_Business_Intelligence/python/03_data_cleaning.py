
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
CLEAN_DIR = PROJECT_DIR / "output" / "cleaned_data"
RESULTS_DIR = PROJECT_DIR / "output" / "analysis_results"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("E-COMMERCE DATA CLEANING PIPELINE")
print("=" * 70)

files = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

cleaning_report = []


def save_cleaned(name, df, duplicates_removed=0, notes=""):
    output_path = CLEAN_DIR / f"{name}_cleaned.csv"
    df.to_csv(output_path, index=False)

    cleaning_report.append({
        "dataset": name,
        "rows_after_cleaning": len(df),
        "columns": len(df.columns),
        "missing_cells_remaining": int(df.isna().sum().sum()),
        "exact_duplicates_removed": duplicates_removed,
        "notes": notes,
    })

    print(f"\n{name}")
    print(f"Rows saved: {len(df):,}")
    print(f"Columns saved: {len(df.columns)}")
    print(f"Missing cells remaining: {df.isna().sum().sum():,}")
    print(f"Saved to: {output_path.name}")


# -------------------------------------------------------
# 1. CUSTOMERS
# -------------------------------------------------------
customers = pd.read_csv(DATASET_DIR / files["customers"])
before = len(customers)
customers = customers.drop_duplicates().copy()
removed = before - len(customers)

save_cleaned("customers", customers, removed)


# -------------------------------------------------------
# 2. ORDERS
# -------------------------------------------------------
orders = pd.read_csv(DATASET_DIR / files["orders"])
before = len(orders)
orders = orders.drop_duplicates().copy()
removed = before - len(orders)

date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

for col in date_columns:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")

# Keep undelivered/canceled orders. Missing dates can be meaningful.
save_cleaned(
    "orders",
    orders,
    removed,
    "Missing operational dates preserved where unavailable."
)


# -------------------------------------------------------
# 3. PRODUCTS
# -------------------------------------------------------
products = pd.read_csv(DATASET_DIR / files["products"])
before = len(products)
products = products.drop_duplicates().copy()
removed = before - len(products)

# Correct original dataset spelling in column names.
products = products.rename(columns={
    "product_name_lenght": "product_name_length",
    "product_description_lenght": "product_description_length",
})

# Missing category is labeled, not guessed.
products["product_category_name"] = (
    products["product_category_name"].fillna("unknown")
)

# Keep unknown dimensions as missing; do not invent measurements.
save_cleaned(
    "products",
    products,
    removed,
    "Unknown category labeled; missing physical dimensions preserved."
)


# -------------------------------------------------------
# 4. SELLERS
# -------------------------------------------------------
sellers = pd.read_csv(DATASET_DIR / files["sellers"])
before = len(sellers)
sellers = sellers.drop_duplicates().copy()
removed = before - len(sellers)

save_cleaned("sellers", sellers, removed)


# -------------------------------------------------------
# 5. ORDER ITEMS
# -------------------------------------------------------
items = pd.read_csv(DATASET_DIR / files["order_items"])
before = len(items)
items = items.drop_duplicates().copy()
removed = before - len(items)

items["shipping_limit_date"] = pd.to_datetime(
    items["shipping_limit_date"], errors="coerce"
)

save_cleaned("order_items", items, removed)


# -------------------------------------------------------
# 6. PAYMENTS
# -------------------------------------------------------
payments = pd.read_csv(DATASET_DIR / files["payments"])
before = len(payments)
payments = payments.drop_duplicates().copy()
removed = before - len(payments)

save_cleaned("payments", payments, removed)


# -------------------------------------------------------
# 7. REVIEWS
# -------------------------------------------------------
reviews = pd.read_csv(DATASET_DIR / files["reviews"])
before = len(reviews)

# Remove only fully identical records.
reviews = reviews.drop_duplicates().copy()
removed = before - len(reviews)

reviews["review_creation_date"] = pd.to_datetime(
    reviews["review_creation_date"], errors="coerce"
)
reviews["review_answer_timestamp"] = pd.to_datetime(
    reviews["review_answer_timestamp"], errors="coerce"
)

# Empty comments are allowed; missing review text is not a missing score.
for col in ["review_comment_title", "review_comment_message"]:
    reviews[col] = reviews[col].fillna("")

duplicate_review_ids = int(
    reviews["review_id"].duplicated(keep=False).sum()
)

save_cleaned(
    "reviews",
    reviews,
    removed,
    f"Rows with repeated review_id flagged for investigation: "
    f"{duplicate_review_ids}. No records removed based only on review_id."
)


# -------------------------------------------------------
# 8. GEOLOCATION
# -------------------------------------------------------
geo = pd.read_csv(DATASET_DIR / files["geolocation"])
before = len(geo)

# Exact repeated records can be removed from this reference table.
geo = geo.drop_duplicates().copy()
removed = before - len(geo)

save_cleaned(
    "geolocation",
    geo,
    removed,
    "Exact duplicate rows removed; multiple coordinates per postal code retained."
)


# -------------------------------------------------------
# 9. CATEGORY TRANSLATION
# -------------------------------------------------------
translation = pd.read_csv(
    DATASET_DIR / files["category_translation"]
)
before = len(translation)
translation = translation.drop_duplicates().copy()
removed = before - len(translation)

save_cleaned("category_translation", translation, removed)


# -------------------------------------------------------
# SAVE CLEANING REPORT
# -------------------------------------------------------
report = pd.DataFrame(cleaning_report)

report_path = RESULTS_DIR / "data_cleaning_report.csv"
report.to_csv(report_path, index=False)

print("\n" + "=" * 70)
print("CLEANING PIPELINE COMPLETED")
print("=" * 70)
print(report.to_string(index=False))
print(f"\nCleaning report saved to: {report_path}")
print(f"Cleaned datasets folder: {CLEAN_DIR}")