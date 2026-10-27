import pandas as pd
import os

# ============================================================
# LOAD CLEANED DATA
# ============================================================

file_path = "../output/cleaned_data/hotel_bookings_cleaned.csv"

df = pd.read_csv(file_path)

# ============================================================
# OUTPUT FOLDER
# ============================================================

output_path = "../output/analysis_results"

os.makedirs(output_path, exist_ok=True)

print("=" * 70)
print("HOTEL BOOKING ANALYSIS - BUSINESS ANALYSIS")
print("=" * 70)

# ============================================================
# 1. MAIN KPIs
# ============================================================

total_bookings = len(df)

cancelled_bookings = df["is_canceled"].sum()

confirmed_bookings = total_bookings - cancelled_bookings

cancellation_rate = (
    cancelled_bookings / total_bookings
) * 100

average_adr = df["adr"].mean()

average_lead_time = df["lead_time"].mean()

total_nights = df["total_nights"].sum()

total_revenue = df["estimated_revenue"].sum()

repeat_guest_rate = (
    df["is_repeated_guest"].mean()
) * 100

print("\nMAIN KPIs")
print("-" * 50)

print(f"Total Bookings       : {total_bookings:,}")
print(f"Confirmed Bookings   : {confirmed_bookings:,}")
print(f"Cancelled Bookings   : {cancelled_bookings:,}")
print(f"Cancellation Rate    : {cancellation_rate:.2f}%")
print(f"Average ADR          : ${average_adr:.2f}")
print(f"Average Lead Time    : {average_lead_time:.2f} days")
print(f"Total Nights         : {total_nights:,}")
print(f"Estimated Revenue    : ${total_revenue:,.2f}")
print(f"Repeat Guest Rate    : {repeat_guest_rate:.2f}%")

# ============================================================
# 2. CANCELLATION BY HOTEL
# ============================================================

hotel_analysis = (
    df.groupby("hotel")
    .agg(
        Total_Bookings=("is_canceled", "count"),
        Cancelled_Bookings=("is_canceled", "sum"),
        Average_ADR=("adr", "mean"),
        Average_Lead_Time=("lead_time", "mean")
    )
    .reset_index()
)

hotel_analysis["Cancellation_Rate"] = (
    hotel_analysis["Cancelled_Bookings"]
    / hotel_analysis["Total_Bookings"]
    * 100
)

hotel_analysis = hotel_analysis.sort_values(
    "Cancellation_Rate",
    ascending=False
)

hotel_analysis.to_csv(
    f"{output_path}/01_cancellation_by_hotel.csv",
    index=False
)

print("\nCANCELLATION BY HOTEL")
print(hotel_analysis)

# ============================================================
# 3. CANCELLATION BY MONTH
# ============================================================

monthly_analysis = (
    df.groupby(
        ["arrival_date_year",
         "arrival_month_number",
         "arrival_date_month"]
    )
    .agg(
        Total_Bookings=("is_canceled", "count"),
        Cancelled_Bookings=("is_canceled", "sum"),
        Average_ADR=("adr", "mean")
    )
    .reset_index()
)

monthly_analysis["Cancellation_Rate"] = (
    monthly_analysis["Cancelled_Bookings"]
    / monthly_analysis["Total_Bookings"]
    * 100
)

monthly_analysis = monthly_analysis.sort_values(
    ["arrival_date_year", "arrival_month_number"]
)

monthly_analysis.to_csv(
    f"{output_path}/02_monthly_cancellation.csv",
    index=False
)

print("\nMONTHLY CANCELLATION ANALYSIS")
print(monthly_analysis)

# ============================================================
# 4. CANCELLATION BY LEAD TIME
# ============================================================

lead_time_analysis = (
    df.groupby("lead_time_group")
    .agg(
        Total_Bookings=("is_canceled", "count"),
        Cancelled_Bookings=("is_canceled", "sum"),
        Average_Lead_Time=("lead_time", "mean")
    )
    .reset_index()
)

lead_order = [
    "0-7 Days",
    "8-30 Days",
    "31-90 Days",
    "91-180 Days",
    "181-365 Days",
    "365+ Days"
]

lead_time_analysis["lead_time_group"] = pd.Categorical(
    lead_time_analysis["lead_time_group"],
    categories=lead_order,
    ordered=True
)

lead_time_analysis = lead_time_analysis.sort_values(
    "lead_time_group"
)

lead_time_analysis["Cancellation_Rate"] = (
    lead_time_analysis["Cancelled_Bookings"]
    / lead_time_analysis["Total_Bookings"]
    * 100
)

lead_time_analysis.to_csv(
    f"{output_path}/03_cancellation_by_lead_time.csv",
    index=False
)

print("\nCANCELLATION BY LEAD TIME")
print(lead_time_analysis)

# ============================================================
# 5. CANCELLATION BY MARKET SEGMENT
# ============================================================

market_analysis = (
    df.groupby("market_segment")
    .agg(
        Total_Bookings=("is_canceled", "count"),
        Cancelled_Bookings=("is_canceled", "sum"),
        Average_ADR=("adr", "mean")
    )
    .reset_index()
)

market_analysis["Cancellation_Rate"] = (
    market_analysis["Cancelled_Bookings"]
    / market_analysis["Total_Bookings"]
    * 100
)

market_analysis = market_analysis.sort_values(
    "Cancellation_Rate",
    ascending=False
)

market_analysis.to_csv(
    f"{output_path}/04_cancellation_by_market_segment.csv",
    index=False
)

print("\nCANCELLATION BY MARKET SEGMENT")
print(market_analysis)

# ============================================================
# 6. CANCELLATION BY CUSTOMER TYPE
# ============================================================

customer_analysis = (
    df.groupby("customer_type")
    .agg(
        Total_Bookings=("is_canceled", "count"),
        Cancelled_Bookings=("is_canceled", "sum"),
        Average_ADR=("adr", "mean")
    )
    .reset_index()
)

customer_analysis["Cancellation_Rate"] = (
    customer_analysis["Cancelled_Bookings"]
    / customer_analysis["Total_Bookings"]
    * 100
)

customer_analysis = customer_analysis.sort_values(
    "Cancellation_Rate",
    ascending=False
)

customer_analysis.to_csv(
    f"{output_path}/05_cancellation_by_customer_type.csv",
    index=False
)

print("\nCANCELLATION BY CUSTOMER TYPE")
print(customer_analysis)

# ============================================================
# 7. CANCELLATION BY DEPOSIT TYPE
# ============================================================

deposit_analysis = (
    df.groupby("deposit_type")
    .agg(
        Total_Bookings=("is_canceled", "count"),
        Cancelled_Bookings=("is_canceled", "sum")
    )
    .reset_index()
)

deposit_analysis["Cancellation_Rate"] = (
    deposit_analysis["Cancelled_Bookings"]
    / deposit_analysis["Total_Bookings"]
    * 100
)

deposit_analysis = deposit_analysis.sort_values(
    "Cancellation_Rate",
    ascending=False
)

deposit_analysis.to_csv(
    f"{output_path}/06_cancellation_by_deposit_type.csv",
    index=False
)

print("\nCANCELLATION BY DEPOSIT TYPE")
print(deposit_analysis)

# ============================================================
# 8. CANCELLATION BY REPEAT GUEST
# ============================================================

repeat_analysis = (
    df.groupby("guest_type")
    .agg(
        Total_Bookings=("is_canceled", "count"),
        Cancelled_Bookings=("is_canceled", "sum")
    )
    .reset_index()
)

repeat_analysis["Cancellation_Rate"] = (
    repeat_analysis["Cancelled_Bookings"]
    / repeat_analysis["Total_Bookings"]
    * 100
)

repeat_analysis.to_csv(
    f"{output_path}/07_cancellation_by_guest_type.csv",
    index=False
)

print("\nCANCELLATION BY GUEST TYPE")
print(repeat_analysis)

# ============================================================
# 9. CANCELLATION BY ADR GROUP
# ============================================================

adr_analysis = (
    df.groupby("adr_group")
    .agg(
        Total_Bookings=("is_canceled", "count"),
        Cancelled_Bookings=("is_canceled", "sum"),
        Average_ADR=("adr", "mean")
    )
    .reset_index()
)

adr_order = [
    "Below 50",
    "50-100",
    "100-150",
    "150-200",
    "200+"
]

adr_analysis["adr_group"] = pd.Categorical(
    adr_analysis["adr_group"],
    categories=adr_order,
    ordered=True
)

adr_analysis = adr_analysis.sort_values("adr_group")

adr_analysis["Cancellation_Rate"] = (
    adr_analysis["Cancelled_Bookings"]
    / adr_analysis["Total_Bookings"]
    * 100
)

adr_analysis.to_csv(
    f"{output_path}/08_cancellation_by_adr.csv",
    index=False
)

print("\nCANCELLATION BY ADR")
print(adr_analysis)

# ============================================================
# 10. CANCELLATION BY DISTRIBUTION CHANNEL
# ============================================================

channel_analysis = (
    df.groupby("distribution_channel")
    .agg(
        Total_Bookings=("is_canceled", "count"),
        Cancelled_Bookings=("is_canceled", "sum")
    )
    .reset_index()
)

channel_analysis["Cancellation_Rate"] = (
    channel_analysis["Cancelled_Bookings"]
    / channel_analysis["Total_Bookings"]
    * 100
)

channel_analysis = channel_analysis.sort_values(
    "Cancellation_Rate",
    ascending=False
)

channel_analysis.to_csv(
    f"{output_path}/09_cancellation_by_channel.csv",
    index=False
)

print("\nCANCELLATION BY DISTRIBUTION CHANNEL")
print(channel_analysis)

# ============================================================
# 11. TOP COUNTRIES BY BOOKINGS
# ============================================================

country_analysis = (
    df.groupby("country")
    .agg(
        Total_Bookings=("is_canceled", "count"),
        Cancelled_Bookings=("is_canceled", "sum")
    )
    .reset_index()
)

country_analysis["Cancellation_Rate"] = (
    country_analysis["Cancelled_Bookings"]
    / country_analysis["Total_Bookings"]
    * 100
)

country_analysis = country_analysis.sort_values(
    "Total_Bookings",
    ascending=False
)

country_analysis.to_csv(
    f"{output_path}/10_country_analysis.csv",
    index=False
)

print("\nTOP 10 COUNTRIES BY BOOKINGS")
print(country_analysis.head(10))

# ============================================================
# 12. FINAL CORRELATION ANALYSIS
# ============================================================

correlation_columns = [
    "is_canceled",
    "lead_time",
    "adr",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "total_of_special_requests",
    "total_nights"
]

correlation = (
    df[correlation_columns]
    .corr()["is_canceled"]
    .sort_values(ascending=False)
)

correlation.to_csv(
    f"{output_path}/11_cancellation_correlations.csv"
)

print("\nCORRELATION WITH CANCELLATION")
print(correlation)

# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("BUSINESS ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nAnalysis files saved in:")
print("../output/analysis_results/")

import matplotlib.pyplot as plt

# ============================================================
# STEP 6 - PYTHON VISUALIZATIONS
# ============================================================

chart_path = "../output/charts"
os.makedirs(chart_path, exist_ok=True)

# ============================================================
# CHART 1 - CANCELLATION RATE BY HOTEL
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    hotel_analysis["hotel"],
    hotel_analysis["Cancellation_Rate"]
)

plt.title("Cancellation Rate by Hotel")
plt.xlabel("Hotel")
plt.ylabel("Cancellation Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    f"{chart_path}/01_cancellation_rate_by_hotel.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 2 - MONTHLY CANCELLATION TREND
# ============================================================

monthly_analysis["Year_Month"] = (
    monthly_analysis["arrival_date_year"].astype(str)
    + "-"
    + monthly_analysis["arrival_month_number"].astype(str).str.zfill(2)
)

plt.figure(figsize=(12, 5))

plt.plot(
    monthly_analysis["Year_Month"],
    monthly_analysis["Cancellation_Rate"],
    marker="o"
)

plt.title("Monthly Cancellation Rate Trend")
plt.xlabel("Year-Month")
plt.ylabel("Cancellation Rate (%)")
plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    f"{chart_path}/02_monthly_cancellation_trend.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 3 - CANCELLATION RATE BY LEAD TIME
# ============================================================

plt.figure(figsize=(10, 5))

plt.bar(
    lead_time_analysis["lead_time_group"].astype(str),
    lead_time_analysis["Cancellation_Rate"]
)

plt.title("Cancellation Rate by Lead Time")
plt.xlabel("Lead Time")
plt.ylabel("Cancellation Rate (%)")
plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    f"{chart_path}/03_cancellation_by_lead_time.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 4 - CANCELLATION RATE BY MARKET SEGMENT
# ============================================================

plt.figure(figsize=(10, 5))

plt.bar(
    market_analysis["market_segment"],
    market_analysis["Cancellation_Rate"]
)

plt.title("Cancellation Rate by Market Segment")
plt.xlabel("Market Segment")
plt.ylabel("Cancellation Rate (%)")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    f"{chart_path}/04_cancellation_by_market_segment.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 5 - CANCELLATION RATE BY CUSTOMER TYPE
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    customer_analysis["customer_type"],
    customer_analysis["Cancellation_Rate"]
)

plt.title("Cancellation Rate by Customer Type")
plt.xlabel("Customer Type")
plt.ylabel("Cancellation Rate (%)")
plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    f"{chart_path}/05_cancellation_by_customer_type.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 6 - CANCELLATION RATE BY DEPOSIT TYPE
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    deposit_analysis["deposit_type"],
    deposit_analysis["Cancellation_Rate"]
)

plt.title("Cancellation Rate by Deposit Type")
plt.xlabel("Deposit Type")
plt.ylabel("Cancellation Rate (%)")

plt.tight_layout()

plt.savefig(
    f"{chart_path}/06_cancellation_by_deposit_type.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 7 - CANCELLATION RATE BY GUEST TYPE
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    repeat_analysis["guest_type"],
    repeat_analysis["Cancellation_Rate"]
)

plt.title("Cancellation Rate by Guest Type")
plt.xlabel("Guest Type")
plt.ylabel("Cancellation Rate (%)")

plt.tight_layout()

plt.savefig(
    f"{chart_path}/07_cancellation_by_guest_type.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 8 - CANCELLATION RATE BY ADR
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    adr_analysis["adr_group"].astype(str),
    adr_analysis["Cancellation_Rate"]
)

plt.title("Cancellation Rate by ADR Range")
plt.xlabel("ADR Range")
plt.ylabel("Cancellation Rate (%)")

plt.tight_layout()

plt.savefig(
    f"{chart_path}/08_cancellation_by_adr.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 9 - BOOKINGS BY HOTEL
# ============================================================

hotel_bookings = (
    df.groupby("hotel")
    .size()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))

plt.bar(
    hotel_bookings.index,
    hotel_bookings.values
)

plt.title("Total Bookings by Hotel")
plt.xlabel("Hotel")
plt.ylabel("Number of Bookings")

plt.tight_layout()

plt.savefig(
    f"{chart_path}/09_bookings_by_hotel.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 10 - TOP 10 COUNTRIES BY BOOKINGS
# ============================================================

top_countries = (
    country_analysis
    .head(10)
    .sort_values("Total_Bookings")
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_countries["country"],
    top_countries["Total_Bookings"]
)

plt.title("Top 10 Countries by Bookings")
plt.xlabel("Total Bookings")
plt.ylabel("Country")

plt.tight_layout()

plt.savefig(
    f"{chart_path}/10_top_countries.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 11 - CANCELLATION VS LEAD TIME
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["lead_time"],
    df["is_canceled"],
    alpha=0.05
)

plt.title("Cancellation vs Lead Time")
plt.xlabel("Lead Time (Days)")
plt.ylabel("Cancellation Status")

plt.tight_layout()

plt.savefig(
    f"{chart_path}/11_cancellation_vs_lead_time.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 12 - ADR VS CANCELLATION
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["adr"],
    df["is_canceled"],
    alpha=0.05
)

plt.title("ADR vs Cancellation")
plt.xlabel("Average Daily Rate")
plt.ylabel("Cancellation Status")

plt.tight_layout()

plt.savefig(
    f"{chart_path}/12_adr_vs_cancellation.png",
    dpi=300
)

plt.close()


print("\n" + "=" * 70)
print("PYTHON VISUALIZATIONS COMPLETED")
print("=" * 70)

print("\nCharts saved in:")
print("../output/charts/")