
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_DIR / "output" / "cleaned_data"
RESULTS_DIR = PROJECT_DIR / "output" / "analysis_results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data(filename):
    return pd.read_csv(CLEAN_DIR / filename, low_memory=False)

print("=" * 70)
print("ADVANCED FEATURE ENGINEERING")
print("=" * 70)

# -------------------------------------------------------
# 1. LOAD CLEANED DATA
# -------------------------------------------------------
orders = load_data("orders_cleaned.csv")
customers = load_data("customers_cleaned.csv")
items = load_data("order_items_cleaned.csv")
payments = load_data("payments_cleaned.csv")
products = load_data("products_cleaned.csv")
reviews = load_data("reviews_cleaned.csv")
sellers = load_data("sellers_cleaned.csv")
translation = load_data("category_translation_cleaned.csv")

date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

for col in date_columns:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")

items["shipping_limit_date"] = pd.to_datetime(
    items["shipping_limit_date"], errors="coerce"
)

reviews["review_creation_date"] = pd.to_datetime(
    reviews["review_creation_date"], errors="coerce"
)

# -------------------------------------------------------
# 2. ORDER-ITEM FEATURES
# -------------------------------------------------------
items["item_revenue"] = items["price"]
items["freight_cost"] = items["freight_value"]
items["item_total_value"] = items["price"] + items["freight_value"]

# Join product categories and seller information
products["product_category_name"] = (
    products["product_category_name"].fillna("unknown")
)

products = products.merge(
    translation,
    on="product_category_name",
    how="left",
    validate="many_to_one"
)

products["product_category_name_english"] = (
    products["product_category_name_english"]
    .fillna(products["product_category_name"])
)

items = items.merge(
    products[["product_id", "product_category_name_english"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

items = items.merge(
    sellers[["seller_id", "seller_city", "seller_state"]],
    on="seller_id",
    how="left",
    validate="many_to_one"
)

items_path = RESULTS_DIR / "order_items_featured.csv"
items.to_csv(items_path, index=False)

# -------------------------------------------------------
# 3. ORDER-LEVEL FEATURES
# -------------------------------------------------------
item_summary = (
    items.groupby("order_id", as_index=False)
    .agg(
        item_count=("order_item_id", "count"),
        unique_products=("product_id", "nunique"),
        unique_sellers=("seller_id", "nunique"),
        item_revenue=("item_revenue", "sum"),
        freight_cost=("freight_cost", "sum"),
        order_item_total=("item_total_value", "sum"),
    )
)

payment_summary = (
    payments.groupby("order_id", as_index=False)
    .agg(
        payment_total=("payment_value", "sum"),
        payment_methods=("payment_type", "nunique"),
        payment_installments_max=("payment_installments", "max"),
    )
)

order_features = orders.merge(
    item_summary, on="order_id", how="left", validate="one_to_one"
)

order_features = order_features.merge(
    payment_summary, on="order_id", how="left", validate="one_to_one"
)

order_features["item_count"] = (
    order_features["item_count"].fillna(0).astype(int)
)

order_features["item_revenue"] = order_features["item_revenue"].fillna(0)
order_features["freight_cost"] = order_features["freight_cost"].fillna(0)
order_features["order_item_total"] = (
    order_features["order_item_total"].fillna(0)
)

order_features["payment_total"] = order_features["payment_total"].fillna(0)

# Payment difference is a reconciliation check, not profit.
order_features["payment_item_difference"] = (
    order_features["payment_total"] - order_features["order_item_total"]
)

order_features["purchase_year"] = (
    order_features["order_purchase_timestamp"].dt.year
)
order_features["purchase_month"] = (
    order_features["order_purchase_timestamp"].dt.month
)
order_features["purchase_year_month"] = (
    order_features["order_purchase_timestamp"].dt.to_period("M").astype(str)
)
order_features["purchase_day_of_week"] = (
    order_features["order_purchase_timestamp"].dt.day_name()
)
order_features["purchase_hour"] = (
    order_features["order_purchase_timestamp"].dt.hour
)

order_features["delivery_days"] = (
    order_features["order_delivered_customer_date"]
    - order_features["order_purchase_timestamp"]
).dt.total_seconds() / 86400

order_features["estimated_delivery_days"] = (
    order_features["order_estimated_delivery_date"]
    - order_features["order_purchase_timestamp"]
).dt.total_seconds() / 86400

order_features["delivery_delay_days"] = (
    order_features["order_delivered_customer_date"]
    - order_features["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

# Only delivered orders with both actual and estimated dates are classified.
valid_delivery = (
    order_features["order_status"].eq("delivered")
    & order_features["order_delivered_customer_date"].notna()
    & order_features["order_estimated_delivery_date"].notna()
)

order_features["delivery_performance"] = "not_evaluated"
order_features.loc[
    valid_delivery & (order_features["delivery_delay_days"] > 0),
    "delivery_performance"
] = "late"

order_features.loc[
    valid_delivery & (order_features["delivery_delay_days"] <= 0),
    "delivery_performance"
] = "on_time_or_early"

order_features["is_late_delivery"] = np.nan
order_features.loc[valid_delivery, "is_late_delivery"] = (
    order_features.loc[valid_delivery, "delivery_delay_days"] > 0
).astype(int)

order_features["is_delivered"] = (
    order_features["order_status"].eq("delivered").astype(int)
)

order_features_path = RESULTS_DIR / "orders_featured.csv"
order_features.to_csv(order_features_path, index=False)

# -------------------------------------------------------
# 4. CUSTOMER-LEVEL FEATURES
# -------------------------------------------------------
customer_features = customers.merge(
    order_features[
        [
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
            "item_revenue",
            "freight_cost",
            "payment_total",
        ]
    ],
    on="customer_id",
    how="left",
    validate="one_to_many"
)

customer_summary = (
    customer_features.groupby(
        ["customer_id", "customer_unique_id",
         "customer_city", "customer_state"],
        as_index=False
    )
    .agg(
        total_orders=("order_id", "nunique"),
        delivered_orders=(
            "order_status",
            lambda x: x.eq("delivered").sum()
        ),
        customer_item_revenue=("item_revenue", "sum"),
        customer_freight_cost=("freight_cost", "sum"),
        customer_payment_total=("payment_total", "sum"),
        first_order_date=("order_purchase_timestamp", "min"),
        last_order_date=("order_purchase_timestamp", "max"),
    )
)

# Unique customer identity is better represented by customer_unique_id.
# Order history is initially summarized by customer_id here; repeat behavior
# will be computed across customer_unique_id in the next analysis stage.
customer_summary_path = RESULTS_DIR / "customer_features.csv"
customer_summary.to_csv(customer_summary_path, index=False)

# -------------------------------------------------------
# 5. REVIEW FEATURES
# -------------------------------------------------------
reviews["rating_group"] = pd.cut(
    reviews["review_score"],
    bins=[0, 2, 3, 5],
    labels=["low", "medium", "high"],
    include_lowest=True
)

reviews["has_review_comment"] = (
    reviews["review_comment_message"].fillna("").str.strip().ne("")
).astype(int)

reviews_path = RESULTS_DIR / "reviews_featured.csv"
reviews.to_csv(reviews_path, index=False)

# -------------------------------------------------------
# 6. SUMMARY
# -------------------------------------------------------
print("\nFEATURE ENGINEERING COMPLETED")
print("-" * 70)
print(f"Orders featured: {len(order_features):,}")
print(f"Order items featured: {len(items):,}")
print(f"Customer records summarized: {len(customer_summary):,}")
print(f"Review records featured: {len(reviews):,}")

print("\nOrder status distribution:")
print(order_features["order_status"].value_counts().to_string())

print("\nDelivery performance:")
print(order_features["delivery_performance"].value_counts().to_string())

print("\nMain outputs saved:")
print(order_features_path)
print(items_path)
print(customer_summary_path)
print(reviews_path)