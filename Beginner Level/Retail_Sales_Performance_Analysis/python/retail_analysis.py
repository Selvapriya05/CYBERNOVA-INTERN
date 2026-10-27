# ============================================================
# RETAIL SALES PERFORMANCE ANALYSIS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs("../output/charts", exist_ok=True)
os.makedirs("../output/analysis_results", exist_ok=True)
os.makedirs("../output/cleaned_data", exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

file_path = "../dataset/Retail_Sales.csv"

df = pd.read_csv(file_path, encoding="latin1")

print("=" * 60)
print("RETAIL SALES PERFORMANCE ANALYSIS")
print("=" * 60)


# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nDATASET INFO")
print(df.info())

print("\nCOLUMN NAMES")
print(df.columns.tolist())


# ============================================================
# 4. DATA CLEANING
# ============================================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Convert date columns
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

# Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Remove duplicate rows
duplicate_count = df.duplicated().sum()

print("\nDUPLICATE ROWS:", duplicate_count)

df = df.drop_duplicates()

# Check dates after conversion
print("\nINVALID ORDER DATES:", df["Order Date"].isnull().sum())
print("INVALID SHIP DATES:", df["Ship Date"].isnull().sum())


# ============================================================
# 5. CREATE NEW ANALYSIS COLUMNS
# ============================================================

# Shipping duration
df["Shipping Days"] = (
    df["Ship Date"] - df["Order Date"]
).dt.days

# Year
df["Year"] = df["Order Date"].dt.year

# Month number
df["Month Number"] = df["Order Date"].dt.month

# Month name
df["Month"] = df["Order Date"].dt.strftime("%B")

# Year-Month
df["Year-Month"] = df["Order Date"].dt.to_period("M").astype(str)

# Quarter
df["Quarter"] = df["Order Date"].dt.to_period("Q").astype(str)

# Profit Margin
df["Profit Margin"] = np.where(
    df["Sales"] != 0,
    (df["Profit"] / df["Sales"]) * 100,
    0
)


# ============================================================
# 6. FINAL DATASET INFORMATION
# ============================================================

print("\nDATA TYPES AFTER CLEANING")
print(df.dtypes)

print("\nFINAL SHAPE")
print(df.shape)

print("\nFINAL MISSING VALUES")
print(df.isnull().sum())


# ============================================================
# 7. STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe())

df.describe().to_csv(
    "../output/analysis_results/statistical_summary.csv"
)


# ============================================================
# 8. KEY BUSINESS KPIs
# ============================================================

total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
total_quantity = df["Quantity"].sum()
total_orders = df["Order ID"].nunique()
total_customers = df["Customer ID"].nunique()
total_products = df["Product ID"].nunique()

profit_margin = (total_profit / total_sales) * 100

average_order_value = total_sales / total_orders

average_profit_per_order = total_profit / total_orders


print("\n" + "=" * 60)
print("KEY BUSINESS KPIs")
print("=" * 60)

print(f"Total Sales           : ${total_sales:,.2f}")
print(f"Total Profit          : ${total_profit:,.2f}")
print(f"Total Quantity        : {total_quantity:,}")
print(f"Total Orders          : {total_orders:,}")
print(f"Total Customers       : {total_customers:,}")
print(f"Total Products        : {total_products:,}")
print(f"Profit Margin         : {profit_margin:.2f}%")
print(f"Average Order Value   : ${average_order_value:,.2f}")
print(f"Average Profit/Order  : ${average_profit_per_order:,.2f}")


# Save KPI results

kpi_data = pd.DataFrame({
    "Metric": [
        "Total Sales",
        "Total Profit",
        "Total Quantity",
        "Total Orders",
        "Total Customers",
        "Total Products",
        "Profit Margin",
        "Average Order Value",
        "Average Profit Per Order"
    ],
    "Value": [
        total_sales,
        total_profit,
        total_quantity,
        total_orders,
        total_customers,
        total_products,
        profit_margin,
        average_order_value,
        average_profit_per_order
    ]
})

kpi_data.to_csv(
    "../output/analysis_results/business_kpis.csv",
    index=False
)


# ============================================================
# 9. SALES BY REGION
# ============================================================

region_analysis = (
    df.groupby("Region")
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("Order ID", "nunique"),
        Customers=("Customer ID", "nunique")
    )
    .sort_values("Sales", ascending=False)
)

print("\n" + "=" * 60)
print("REGIONAL PERFORMANCE")
print("=" * 60)

print(region_analysis)

region_analysis.to_csv(
    "../output/analysis_results/region_analysis.csv"
)


# ============================================================
# 10. SALES BY STATE
# ============================================================

state_analysis = (
    df.groupby("State")
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("Order ID", "nunique")
    )
    .sort_values("Sales", ascending=False)
)

print("\nTOP 10 STATES BY SALES")
print(state_analysis.head(10))

state_analysis.to_csv(
    "../output/analysis_results/state_analysis.csv"
)


# ============================================================
# 11. TOP 10 CITIES
# ============================================================

city_analysis = (
    df.groupby("City")
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Orders=("Order ID", "nunique")
    )
    .sort_values("Sales", ascending=False)
)

print("\nTOP 10 CITIES BY SALES")
print(city_analysis.head(10))

city_analysis.to_csv(
    "../output/analysis_results/city_analysis.csv"
)


# ============================================================
# 12. CATEGORY ANALYSIS
# ============================================================

category_analysis = (
    df.groupby("Category")
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("Order ID", "nunique")
    )
    .sort_values("Sales", ascending=False)
)

print("\n" + "=" * 60)
print("CATEGORY PERFORMANCE")
print("=" * 60)

print(category_analysis)

category_analysis.to_csv(
    "../output/analysis_results/category_analysis.csv"
)


# ============================================================
# 13. SUB-CATEGORY ANALYSIS
# ============================================================

subcategory_analysis = (
    df.groupby("Sub-Category")
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum")
    )
    .sort_values("Sales", ascending=False)
)

print("\nTOP SUB-CATEGORIES BY SALES")
print(subcategory_analysis)

subcategory_analysis.to_csv(
    "../output/analysis_results/subcategory_analysis.csv"
)


# ============================================================
# 14. TOP 10 PRODUCTS BY SALES
# ============================================================

product_sales = (
    df.groupby(["Product ID", "Product Name"])
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum")
    )
    .sort_values("Sales", ascending=False)
)

print("\nTOP 10 PRODUCTS BY SALES")
print(product_sales.head(10))

product_sales.head(10).to_csv(
    "../output/analysis_results/top_10_products.csv"
)


# ============================================================
# 15. TOP 10 PRODUCTS BY PROFIT
# ============================================================

product_profit = product_sales.sort_values(
    "Profit",
    ascending=False
)

print("\nTOP 10 PRODUCTS BY PROFIT")
print(product_profit.head(10))

product_profit.head(10).to_csv(
    "../output/analysis_results/top_10_products_profit.csv"
)


# ============================================================
# 16. LOSS-MAKING PRODUCTS
# ============================================================

loss_products = (
    product_sales[product_sales["Profit"] < 0]
    .sort_values("Profit")
)

print("\nTOP 10 LOSS-MAKING PRODUCTS")
print(loss_products.head(10))

loss_products.head(10).to_csv(
    "../output/analysis_results/loss_making_products.csv"
)


# ============================================================
# 17. CUSTOMER SEGMENT ANALYSIS
# ============================================================

segment_analysis = (
    df.groupby("Segment")
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("Order ID", "nunique"),
        Customers=("Customer ID", "nunique")
    )
    .sort_values("Sales", ascending=False)
)

print("\n" + "=" * 60)
print("CUSTOMER SEGMENT PERFORMANCE")
print("=" * 60)

print(segment_analysis)

segment_analysis.to_csv(
    "../output/analysis_results/segment_analysis.csv"
)


# ============================================================
# 18. SHIP MODE ANALYSIS
# ============================================================

ship_mode_analysis = (
    df.groupby("Ship Mode")
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Orders=("Order ID", "nunique"),
        Average_Shipping_Days=("Shipping Days", "mean")
    )
    .sort_values("Sales", ascending=False)
)

print("\nSHIP MODE ANALYSIS")
print(ship_mode_analysis)

ship_mode_analysis.to_csv(
    "../output/analysis_results/ship_mode_analysis.csv"
)


# ============================================================
# 19. DISCOUNT ANALYSIS
# ============================================================

discount_analysis = (
    df.groupby("Discount")
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum")
    )
    .sort_index()
)

print("\nDISCOUNT ANALYSIS")
print(discount_analysis)

discount_analysis.to_csv(
    "../output/analysis_results/discount_analysis.csv"
)


# ============================================================
# 20. SALES AND PROFIT BY YEAR
# ============================================================

year_analysis = (
    df.groupby("Year")
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("Order ID", "nunique")
    )
)

print("\n" + "=" * 60)
print("YEARLY PERFORMANCE")
print("=" * 60)

print(year_analysis)

year_analysis.to_csv(
    "../output/analysis_results/year_analysis.csv"
)


# ============================================================
# 21. MONTHLY SALES TREND
# ============================================================

monthly_analysis = (
    df.groupby("Year-Month")
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum")
    )
    .reset_index()
)

print("\nMONTHLY SALES TREND")
print(monthly_analysis.head())

monthly_analysis.to_csv(
    "../output/analysis_results/monthly_analysis.csv",
    index=False
)


# ============================================================
# 22. QUARTERLY ANALYSIS
# ============================================================

quarter_analysis = (
    df.groupby("Quarter")
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum")
    )
)

print("\nQUARTERLY ANALYSIS")
print(quarter_analysis)

quarter_analysis.to_csv(
    "../output/analysis_results/quarter_analysis.csv"
)


# ============================================================
# 23. CUSTOMER PERFORMANCE
# ============================================================

customer_analysis = (
    df.groupby(["Customer ID", "Customer Name"])
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("Order ID", "nunique")
    )
    .sort_values("Sales", ascending=False)
)

print("\nTOP 10 CUSTOMERS BY SALES")
print(customer_analysis.head(10))

customer_analysis.head(10).to_csv(
    "../output/analysis_results/top_customers.csv"
)


# ============================================================
# 24. PROFITABLE AND UNPROFITABLE STATES
# ============================================================

profitable_states = state_analysis.sort_values(
    "Profit",
    ascending=False
)

unprofitable_states = state_analysis.sort_values(
    "Profit",
    ascending=True
)

print("\nTOP 10 PROFITABLE STATES")
print(profitable_states.head(10))

print("\nTOP 10 LOSS-MAKING STATES")
print(unprofitable_states.head(10))

profitable_states.head(10).to_csv(
    "../output/analysis_results/top_profitable_states.csv"
)

unprofitable_states.head(10).to_csv(
    "../output/analysis_results/top_loss_making_states.csv"
)


# ============================================================
# 25. CORRELATION ANALYSIS
# ============================================================

numeric_columns = [
    "Sales",
    "Quantity",
    "Discount",
    "Profit",
    "Shipping Days"
]

correlation = df[numeric_columns].corr()

print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)

print(correlation)

correlation.to_csv(
    "../output/analysis_results/correlation_matrix.csv"
)


# ============================================================
# 26. CHART 1 - SALES BY REGION
# ============================================================

plt.figure(figsize=(10, 6))

region_analysis["Sales"].plot(
    kind="bar"
)

plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "../output/charts/01_sales_by_region.png",
    dpi=300
)

plt.close()


# ============================================================
# 27. CHART 2 - PROFIT BY REGION
# ============================================================

plt.figure(figsize=(10, 6))

region_analysis["Profit"].plot(
    kind="bar"
)

plt.title("Profit by Region")
plt.xlabel("Region")
plt.ylabel("Profit")

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "../output/charts/02_profit_by_region.png",
    dpi=300
)

plt.close()


# ============================================================
# 28. CHART 3 - SALES BY CATEGORY
# ============================================================

plt.figure(figsize=(8, 6))

category_analysis["Sales"].plot(
    kind="bar"
)

plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "../output/charts/03_sales_by_category.png",
    dpi=300
)

plt.close()


# ============================================================
# 29. CHART 4 - PROFIT BY CATEGORY
# ============================================================

plt.figure(figsize=(8, 6))

category_analysis["Profit"].plot(
    kind="bar"
)

plt.title("Profit by Category")
plt.xlabel("Category")
plt.ylabel("Profit")

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "../output/charts/04_profit_by_category.png",
    dpi=300
)

plt.close()


# ============================================================
# 30. CHART 5 - SALES BY SEGMENT
# ============================================================

plt.figure(figsize=(8, 6))

segment_analysis["Sales"].plot(
    kind="bar"
)

plt.title("Sales by Customer Segment")
plt.xlabel("Segment")
plt.ylabel("Sales")

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "../output/charts/05_sales_by_segment.png",
    dpi=300
)

plt.close()


# ============================================================
# 31. CHART 6 - PROFIT BY SEGMENT
# ============================================================

plt.figure(figsize=(8, 6))

segment_analysis["Profit"].plot(
    kind="bar"
)

plt.title("Profit by Customer Segment")
plt.xlabel("Segment")
plt.ylabel("Profit")

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "../output/charts/06_profit_by_segment.png",
    dpi=300
)

plt.close()


# ============================================================
# 32. CHART 7 - MONTHLY SALES TREND
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    monthly_analysis["Year-Month"],
    monthly_analysis["Sales"]
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig(
    "../output/charts/07_monthly_sales_trend.png",
    dpi=300
)

plt.close()


# ============================================================
# 33. CHART 8 - MONTHLY PROFIT TREND
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    monthly_analysis["Year-Month"],
    monthly_analysis["Profit"]
)

plt.title("Monthly Profit Trend")
plt.xlabel("Month")
plt.ylabel("Profit")

plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig(
    "../output/charts/08_monthly_profit_trend.png",
    dpi=300
)

plt.close()


# ============================================================
# 34. CHART 9 - TOP 10 PRODUCTS
# ============================================================

top_products = product_sales.head(10).sort_values(
    "Sales",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_products.index.get_level_values("Product Name"),
    top_products["Sales"]
)

plt.title("Top 10 Products by Sales")
plt.xlabel("Sales")
plt.ylabel("Product")

plt.tight_layout()

plt.savefig(
    "../output/charts/09_top_10_products.png",
    dpi=300
)

plt.close()


# ============================================================
# 35. CHART 10 - TOP 10 LOSS-MAKING PRODUCTS
# ============================================================

top_losses = loss_products.head(10).sort_values(
    "Profit",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_losses.index.get_level_values("Product Name"),
    top_losses["Profit"]
)

plt.title("Top 10 Loss-Making Products")
plt.xlabel("Profit")
plt.ylabel("Product")

plt.tight_layout()

plt.savefig(
    "../output/charts/10_loss_making_products.png",
    dpi=300
)

plt.close()

# ============================================================
# 36. CHART 11 - TOP STATES BY SALES
# ============================================================

top_states = state_analysis.head(10).sort_values(
    "Sales",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_states.index,
    top_states["Sales"]
)

plt.title("Top 10 States by Sales")
plt.xlabel("Sales")
plt.ylabel("State")

plt.tight_layout()

plt.savefig(
    "../output/charts/11_top_states_sales.png",
    dpi=300
)

plt.close()


# ============================================================
# 37. CHART 12 - PROFIT VS DISCOUNT
# ============================================================

discount_profit = (
    df.groupby("Discount")["Profit"]
    .mean()
    .sort_index()
)

plt.figure(figsize=(10, 6))

plt.plot(
    discount_profit.index,
    discount_profit.values,
    marker="o"
)

plt.title("Average Profit vs Discount")
plt.xlabel("Discount")
plt.ylabel("Average Profit")

plt.tight_layout()

plt.savefig(
    "../output/charts/12_profit_vs_discount.png",
    dpi=300
)

plt.close()


# ============================================================
# 38. CHART 13 - SALES BY SHIP MODE
# ============================================================

plt.figure(figsize=(9, 6))

ship_mode_analysis["Sales"].plot(
    kind="bar"
)

plt.title("Sales by Ship Mode")
plt.xlabel("Ship Mode")
plt.ylabel("Sales")

plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    "../output/charts/13_sales_by_ship_mode.png",
    dpi=300
)

plt.close()


# ============================================================
# 39. CHART 14 - PROFIT BY SUB-CATEGORY
# ============================================================

subcategory_profit = subcategory_analysis.sort_values(
    "Profit",
    ascending=True
)

plt.figure(figsize=(10, 8))

plt.barh(
    subcategory_profit.index,
    subcategory_profit["Profit"]
)

plt.title("Profit by Sub-Category")
plt.xlabel("Profit")
plt.ylabel("Sub-Category")

plt.tight_layout()

plt.savefig(
    "../output/charts/14_profit_by_subcategory.png",
    dpi=300
)

plt.close()


# ============================================================
# 40. SAVE CLEANED DATASET
# ============================================================

df.to_csv(
    "../output/cleaned_data/retail_sales_cleaned.csv",
    index=False
)


# ============================================================
# 41. CREATE FINAL BUSINESS INSIGHTS
# ============================================================

best_region = region_analysis["Sales"].idxmax()
best_category = category_analysis["Sales"].idxmax()
best_segment = segment_analysis["Sales"].idxmax()
best_state = state_analysis["Sales"].idxmax()
best_product = product_sales.index[0][1]

most_profitable_region = region_analysis["Profit"].idxmax()
least_profitable_region = region_analysis["Profit"].idxmin()

most_profitable_category = category_analysis["Profit"].idxmax()
least_profitable_category = category_analysis["Profit"].idxmin()

best_customer = customer_analysis.index[0][1]


print("\n" + "=" * 60)
print("KEY BUSINESS INSIGHTS")
print("=" * 60)

print(f"\n1. Best Region by Sales: {best_region}")

print(f"2. Best Category by Sales: {best_category}")

print(f"3. Best Customer Segment: {best_segment}")

print(f"4. Best State by Sales: {best_state}")

print(f"5. Top Product by Sales: {best_product}")

print(f"6. Most Profitable Region: {most_profitable_region}")

print(f"7. Least Profitable Region: {least_profitable_region}")

print(f"8. Most Profitable Category: {most_profitable_category}")

print(f"9. Least Profitable Category: {least_profitable_category}")

print(f"10. Top Customer by Sales: {best_customer}")


# ============================================================
# 42. SAVE BUSINESS INSIGHTS
# ============================================================

insights = [
    ["Best Region by Sales", best_region],
    ["Best Category by Sales", best_category],
    ["Best Customer Segment", best_segment],
    ["Best State by Sales", best_state],
    ["Top Product by Sales", best_product],
    ["Most Profitable Region", most_profitable_region],
    ["Least Profitable Region", least_profitable_region],
    ["Most Profitable Category", most_profitable_category],
    ["Least Profitable Category", least_profitable_category],
    ["Top Customer by Sales", best_customer]
]

insights_df = pd.DataFrame(
    insights,
    columns=["Business Question", "Answer"]
)

insights_df.to_csv(
    "../output/analysis_results/business_insights.csv",
    index=False
)


# ============================================================
# 43. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nGenerated folders:")
print("1. output/charts/")
print("2. output/analysis_results/")
print("3. output/cleaned_data/")

print("\nYou can now use the CSV files for Power BI.")