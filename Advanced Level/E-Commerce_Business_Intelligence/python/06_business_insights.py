
from pathlib import Path
import pandas as pd

# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = BASE_DIR / "output" / "analysis_results"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# 2. LOAD ANALYSIS FILES
# ==================================================

monthly = pd.read_csv(ANALYSIS_DIR / "monthly_sales_analysis.csv")
categories = pd.read_csv(ANALYSIS_DIR / "category_performance.csv")
states = pd.read_csv(ANALYSIS_DIR / "state_performance.csv")
repeat = pd.read_csv(ANALYSIS_DIR / "repeat_purchase_analysis.csv")
delivery = pd.read_csv(ANALYSIS_DIR / "delivery_performance_analysis.csv")
payments = pd.read_csv(ANALYSIS_DIR / "payment_method_analysis.csv")

insights = []


def add_insight(area, finding, recommendation):
    insights.append({
        "Business_Area": area,
        "Finding": finding,
        "Recommended_Action": recommendation
    })


# ==================================================
# 3. MONTHLY SALES INSIGHTS
# ==================================================

if not monthly.empty and "merchandise_revenue" in monthly.columns:
    monthly["purchase_month"] = monthly["purchase_month"].astype(str)

    best_month = monthly.loc[monthly["merchandise_revenue"].idxmax()]
    lowest_month = monthly.loc[monthly["merchandise_revenue"].idxmin()]

    add_insight(
        "Monthly Sales",
        f"Highest-revenue month: {best_month['purchase_month']}, "
        f"with {best_month['merchandise_revenue']:,.2f} BRL.",
        "Investigate order volume, seasonal demand, and popular categories "
        "during the highest-revenue month."
    )

    add_insight(
        "Monthly Sales",
        f"Lowest-revenue month: {lowest_month['purchase_month']}, "
        f"with {lowest_month['merchandise_revenue']:,.2f} BRL.",
        "Check whether the result reflects seasonality, fewer orders, "
        "or an incomplete month."
    )


# ==================================================
# 4. PRODUCT CATEGORY INSIGHTS
# ==================================================

if not categories.empty and "merchandise_revenue" in categories.columns:
    valid_categories = categories.dropna(
        subset=["product_category_name_english"]
    )

    if not valid_categories.empty:
        top_category = valid_categories.loc[
            valid_categories["merchandise_revenue"].idxmax()
        ]

        add_insight(
            "Product Categories",
            f"Highest-revenue category: "
            f"{top_category['product_category_name_english']}, "
            f"with {top_category['merchandise_revenue']:,.2f} BRL.",
            "Review demand and inventory availability for this category "
            "before planning stock levels."
        )


# ==================================================
# 5. STATE PERFORMANCE INSIGHTS
# ==================================================

if not states.empty and "merchandise_revenue" in states.columns:
    top_state = states.loc[states["merchandise_revenue"].idxmax()]

    add_insight(
        "Geographic Performance",
        f"Highest-revenue state: {top_state['customer_state']}, "
        f"with {top_state['merchandise_revenue']:,.2f} BRL.",
        "Compare order volume, delivery performance, and category demand "
        "in this state before making regional investment decisions."
    )


# ==================================================
# 6. REPEAT PURCHASE INSIGHTS
# ==================================================

if not repeat.empty and {"customer_type", "customers"}.issubset(repeat.columns):
    customer_counts = repeat["customers"].sum()

    for _, row in repeat.iterrows():
        count = int(row["customers"])
        percentage = (
            count / customer_counts * 100 if customer_counts else 0
        )

        add_insight(
            "Customer Retention",
            f"{row['customer_type']}: {count:,} customers "
            f"({percentage:.2f}% of customers in this analysis).",
            "Use customer purchase history to understand retention "
            "and identify opportunities to encourage repeat orders."
        )


# ==================================================
# 7. DELIVERY PERFORMANCE INSIGHTS
# ==================================================

if not delivery.empty and {
    "delivery_result", "orders", "percentage"
}.issubset(delivery.columns):

    for _, row in delivery.iterrows():
        add_insight(
            "Delivery Performance",
            f"{row['delivery_result']}: {int(row['orders']):,} orders "
            f"({row['percentage']:.2f}%).",
            "Investigate delayed deliveries by region and seller to find "
            "possible logistics bottlenecks."
        )


# ==================================================
# 8. PAYMENT METHOD INSIGHTS
# ==================================================

if not payments.empty and {
    "payment_type", "payment_records", "total_payment_value"
}.issubset(payments.columns):

    most_used = payments.loc[payments["payment_records"].idxmax()]

    add_insight(
        "Payment Methods",
        f"Most frequent payment type: {most_used['payment_type']}, "
        f"with {int(most_used['payment_records']):,} payment records.",
        "Compare payment frequency and total payment value to understand "
        "customer payment preferences."
    )

    highest_value = payments.loc[payments["total_payment_value"].idxmax()]

    add_insight(
        "Payment Value",
        f"Highest total payment value: {highest_value['payment_type']}, "
        f"with {highest_value['total_payment_value']:,.2f} BRL.",
        "Review payment values by method when evaluating payment options."
    )


# ==================================================
# 9. SAVE BUSINESS INSIGHTS REPORT
# ==================================================

insights_df = pd.DataFrame(insights)

output_file = REPORT_DIR / "business_insights.csv"

insights_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 65)
print("BUSINESS INSIGHTS REPORT")
print("=" * 65)

if insights_df.empty:
    print("No insights were generated. Check the analysis CSV files.")
else:
    for number, row in insights_df.iterrows():
        print(f"\n{number + 1}. {row['Business_Area']}")
        print(f"Finding: {row['Finding']}")
        print(f"Recommendation: {row['Recommended_Action']}")

print("\nReport saved successfully:")
print(output_file)