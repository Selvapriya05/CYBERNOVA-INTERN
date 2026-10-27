
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "output" / "analysis_results"
CLEAN_DIR = PROJECT_DIR / "output" / "cleaned_data"
CHART_DIR = PROJECT_DIR / "output" / "charts"

CHART_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

print("=" * 70)
print("ADVANCED E-COMMERCE EXPLORATORY DATA ANALYSIS")
print("=" * 70)

orders = pd.read_csv(DATA_DIR / "orders_featured.csv")
items = pd.read_csv(DATA_DIR / "order_items_featured.csv")
customers = pd.read_csv(CLEAN_DIR / "customers_cleaned.csv")
payments = pd.read_csv(CLEAN_DIR / "payments_cleaned.csv")
reviews = pd.read_csv(DATA_DIR / "reviews_featured.csv")

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"], errors="coerce"
)
orders["order_delivered_customer_date"] = pd.to_datetime(
    orders["order_delivered_customer_date"], errors="coerce"
)
orders["order_estimated_delivery_date"] = pd.to_datetime(
    orders["order_estimated_delivery_date"], errors="coerce"
)

# Use delivered orders for completed-order sales trends.
delivered = orders[orders["order_status"] == "delivered"].copy()
delivered["purchase_month"] = (
    delivered["order_purchase_timestamp"].dt.to_period("M").astype(str)
)

# -------------------------------------------------------
# 1. MONTHLY REVENUE AND ORDER TRENDS
# -------------------------------------------------------
monthly = (
    delivered.groupby("purchase_month", as_index=False)
    .agg(
        orders=("order_id", "nunique"),
        merchandise_revenue=("item_revenue", "sum"),
        freight_charged=("freight_cost", "sum"),
    )
)

monthly.to_csv(DATA_DIR / "monthly_sales_analysis.csv", index=False)

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(
    monthly["purchase_month"],
    monthly["merchandise_revenue"],
    marker="o",
    label="Merchandise revenue",
)
ax1.set_ylabel("Merchandise revenue")
ax1.set_xlabel("Purchase month")
ax1.tick_params(axis="x", rotation=75)
ax1.set_title("Monthly Merchandise Revenue — Delivered Orders")
ax2 = ax1.twinx()
ax2.plot(
    monthly["purchase_month"],
    monthly["orders"],
    marker="s",
    linestyle="--",
    label="Delivered orders",
)
ax2.set_ylabel("Delivered orders")
fig.tight_layout()
fig.savefig(CHART_DIR / "01_monthly_sales_trend.png", dpi=160)
plt.close(fig)

# -------------------------------------------------------
# 2. PRODUCT CATEGORY PERFORMANCE
# -------------------------------------------------------
category = (
    items.groupby("product_category_name_english", as_index=False)
    .agg(
        merchandise_revenue=("item_revenue", "sum"),
        freight_charged=("freight_cost", "sum"),
        units_sold=("order_item_id", "count"),
        unique_products=("product_id", "nunique"),
    )
    .sort_values("merchandise_revenue", ascending=False)
)

category.to_csv(DATA_DIR / "category_performance.csv", index=False)
top_categories = category.head(15).sort_values("merchandise_revenue")

plt.figure(figsize=(12, 7))
sns.barplot(
    data=top_categories,
    x="merchandise_revenue",
    y="product_category_name_english",
)
plt.title("Top 15 Product Categories by Merchandise Revenue")
plt.xlabel("Merchandise revenue")
plt.ylabel("Product category")
plt.tight_layout()
plt.savefig(CHART_DIR / "02_top_product_categories.png", dpi=160)
plt.close()

# -------------------------------------------------------
# 3. STATE-LEVEL CUSTOMER AND ORDER ANALYSIS
# -------------------------------------------------------
customer_orders = customers.merge(
    orders[["customer_id", "order_id", "order_status", "item_revenue"]],
    on="customer_id",
    how="left",
    validate="one_to_many",
)

state_summary = (
    customer_orders.groupby("customer_state", as_index=False)
    .agg(
        unique_customers=("customer_unique_id", "nunique"),
        orders=("order_id", "nunique"),
        merchandise_revenue=("item_revenue", "sum"),
    )
    .sort_values("merchandise_revenue", ascending=False)
)

state_summary.to_csv(DATA_DIR / "state_performance.csv", index=False)

top_states = state_summary.head(15).sort_values("merchandise_revenue")
plt.figure(figsize=(10, 7))
sns.barplot(
    data=top_states,
    x="merchandise_revenue",
    y="customer_state",
)
plt.title("Top 15 States by Merchandise Revenue")
plt.xlabel("Merchandise revenue")
plt.ylabel("Customer state")
plt.tight_layout()
plt.savefig(CHART_DIR / "03_state_performance.png", dpi=160)
plt.close()

# -------------------------------------------------------
# 4. REPEAT PURCHASE BEHAVIOR
# Use customer_unique_id, not customer_id, to identify people.
# -------------------------------------------------------
identity = customers[["customer_id", "customer_unique_id"]]

customer_order_history = orders.merge(
    identity, on="customer_id", how="left", validate="many_to_one"
)

customer_order_history = customer_order_history[
    customer_order_history["order_status"] == "delivered"
].copy()

repeat = (
    customer_order_history.groupby("customer_unique_id", as_index=False)
    .agg(
        delivered_orders=("order_id", "nunique"),
        first_purchase=("order_purchase_timestamp", "min"),
        last_purchase=("order_purchase_timestamp", "max"),
    )
)

repeat["customer_type"] = np.where(
    repeat["delivered_orders"] >= 2,
    "Repeat customer",
    "One-time customer",
)

repeat_summary = (
    repeat["customer_type"]
    .value_counts()
    .rename_axis("customer_type")
    .reset_index(name="customers")
)

repeat_summary.to_csv(DATA_DIR / "repeat_purchase_analysis.csv", index=False)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=repeat_summary,
    x="customer_type",
    y="customers",
)
plt.title("Delivered-Order Customer Purchase Frequency")
plt.xlabel("Customer type")
plt.ylabel("Unique customers")
plt.tight_layout()
plt.savefig(CHART_DIR / "04_repeat_purchase_analysis.png", dpi=160)
plt.close()

# -------------------------------------------------------
# 5. DELIVERY PERFORMANCE
# Evaluate delivered orders with valid actual and estimated dates.
# -------------------------------------------------------
delivery = orders[
    (orders["order_status"] == "delivered")
    & orders["order_delivered_customer_date"].notna()
    & orders["order_estimated_delivery_date"].notna()
].copy()

delivery["delay_days"] = (
    delivery["order_delivered_customer_date"]
    - delivery["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

delivery["delivery_result"] = np.where(
    delivery["delay_days"] > 0, "Late", "On time or early"
)

delivery_summary = (
    delivery["delivery_result"]
    .value_counts()
    .rename_axis("delivery_result")
    .reset_index(name="orders")
)

delivery_summary["percentage"] = (
    delivery_summary["orders"] / delivery_summary["orders"].sum() * 100
)

delivery_summary.to_csv(DATA_DIR / "delivery_performance_analysis.csv", index=False)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=delivery_summary,
    x="delivery_result",
    y="percentage",
)
plt.title("Delivery Performance for Eligible Delivered Orders")
plt.xlabel("Delivery result")
plt.ylabel("Percentage of eligible delivered orders")
plt.tight_layout()
plt.savefig(CHART_DIR / "05_delivery_performance.png", dpi=160)
plt.close()

# -------------------------------------------------------
# 6. PAYMENT METHOD ANALYSIS
# -------------------------------------------------------
payment_summary = (
    payments.groupby("payment_type", as_index=False)
    .agg(
        payment_records=("order_id", "size"),
        unique_orders=("order_id", "nunique"),
        total_payment_value=("payment_value", "sum"),
    )
    .sort_values("total_payment_value", ascending=False)
)

payment_summary.to_csv(DATA_DIR / "payment_method_analysis.csv", index=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=payment_summary,
    x="total_payment_value",
    y="payment_type",
)
plt.title("Payment Value by Payment Method")
plt.xlabel("Recorded payment value")
plt.ylabel("Payment method")
plt.tight_layout()
plt.savefig(CHART_DIR / "06_payment_method_analysis.png", dpi=160)
plt.close()

# -------------------------------------------------------
# 7. REVIEW SCORE VS DELIVERY DELAY
# Join review and order-level delivery data.
# Multiple reviews may exist for an order, so preserve review rows.
# -------------------------------------------------------
review_delivery = reviews.merge(
    delivery[["order_id", "delay_days"]],
    on="order_id",
    how="inner",
    validate="many_to_one",
)

review_delay_summary = (
    review_delivery.groupby("review_score", as_index=False)
    .agg(
        reviews=("review_id", "count"),
        average_delay_days=("delay_days", "mean"),
        median_delay_days=("delay_days", "median"),
    )
)

review_delay_summary.to_csv(
    DATA_DIR / "review_vs_delivery_delay.csv", index=False
)

plt.figure(figsize=(9, 6))
sns.boxplot(
    data=review_delivery,
    x="review_score",
    y="delay_days",
)
plt.axhline(0, linestyle="--")
plt.title("Delivery Delay Distribution by Review Score")
plt.xlabel("Review score")
plt.ylabel("Days after estimated delivery (negative = early)")
plt.tight_layout()
plt.savefig(CHART_DIR / "07_review_vs_delivery_delay.png", dpi=160)
plt.close()

# -------------------------------------------------------
# 8. EXECUTIVE SUMMARY METRICS
# -------------------------------------------------------
eligible_count = len(delivery)
late_count = int((delivery["delay_days"] > 0).sum())

metrics = {
    "total_orders_all_statuses": int(orders["order_id"].nunique()),
    "delivered_orders": int(delivered["order_id"].nunique()),
    "delivered_merchandise_revenue": round(
        float(delivered["item_revenue"].sum()), 2
    ),
    "delivered_freight_charged": round(
        float(delivered["freight_cost"].sum()), 2
    ),
    "eligible_delivered_orders_for_delay_rate": eligible_count,
    "late_delivered_orders": late_count,
    "late_delivery_rate_pct": round(
        late_count / eligible_count * 100, 2
    ) if eligible_count else None,
    "average_delivery_days": round(
        float(delivery["delivery_days"].mean()), 2
    ) if "delivery_days" in delivery.columns else round(
        float(
            (
                delivery["order_delivered_customer_date"]
                - delivery["order_purchase_timestamp"]
            ).dt.total_seconds().div(86400).mean()
        ), 2
    ),
    "review_records_joined_to_eligible_deliveries": len(review_delivery),
}

pd.DataFrame([metrics]).to_csv(
    DATA_DIR / "executive_eda_metrics.csv", index=False
)

print("\nEDA COMPLETED")
print("-" * 70)
for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nAnalysis files saved in:", DATA_DIR)
print("Charts saved in:", CHART_DIR)