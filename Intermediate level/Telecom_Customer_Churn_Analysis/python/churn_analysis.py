import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# STEP 6: BUSINESS DRIVER ANALYSIS + CHARTS
# ============================================================

file_path = "../output/cleaned_data/telecom_customer_churn_featured.csv"
df = pd.read_csv(file_path)

# Create output folder
chart_path = "../output/charts"
os.makedirs(chart_path, exist_ok=True)

print("=" * 70)
print("TELECOM CUSTOMER CHURN - BUSINESS DRIVER ANALYSIS")
print("=" * 70)


# ============================================================
# 1. CHURN RATE BY CONTRACT
# ============================================================

contract = (
    df.groupby("Contract")["churn_flag"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 5))
contract.plot(kind="bar")
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    f"{chart_path}/01_churn_rate_by_contract.png",
    dpi=300
)
plt.close()


# ============================================================
# 2. CHURN RATE BY TENURE GROUP
# ============================================================

tenure_order = [
    "0-6 Months",
    "7-12 Months",
    "13-24 Months",
    "25-48 Months",
    "49+ Months"
]

tenure = (
    df.groupby("tenure_group")["churn_flag"]
    .mean()
    .mul(100)
    .reindex(tenure_order)
)

plt.figure(figsize=(9, 5))
tenure.plot(kind="bar")
plt.title("Churn Rate by Customer Tenure")
plt.xlabel("Tenure Group")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    f"{chart_path}/02_churn_rate_by_tenure.png",
    dpi=300
)
plt.close()


# ============================================================
# 3. CHURN RATE BY INTERNET SERVICE
# ============================================================

internet = (
    df.groupby("InternetService")["churn_flag"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 5))
internet.plot(kind="bar")
plt.title("Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    f"{chart_path}/03_churn_rate_by_internet_service.png",
    dpi=300
)
plt.close()


# ============================================================
# 4. CHURN RATE BY PAYMENT METHOD
# ============================================================

payment = (
    df.groupby("PaymentMethod")["churn_flag"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 5))
payment.plot(kind="bar")
plt.title("Churn Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

plt.savefig(
    f"{chart_path}/04_churn_rate_by_payment_method.png",
    dpi=300
)
plt.close()


# ============================================================
# 5. CHURN RATE BY MONTHLY CHARGE GROUP
# ============================================================

charge_order = [
    "Low",
    "Medium",
    "High",
    "Very High"
]

charge = (
    df.groupby("monthly_charge_group")["churn_flag"]
    .mean()
    .mul(100)
    .reindex(charge_order)
)

plt.figure(figsize=(9, 5))
charge.plot(kind="bar")
plt.title("Churn Rate by Monthly Charge Group")
plt.xlabel("Monthly Charge Group")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    f"{chart_path}/05_churn_rate_by_monthly_charge.png",
    dpi=300
)
plt.close()


# ============================================================
# 6. CHURN RATE BY NUMBER OF SERVICES
# ============================================================

services = (
    df.groupby("service_count")["churn_flag"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(9, 5))
services.plot(kind="bar")
plt.title("Churn Rate by Number of Services")
plt.xlabel("Number of Services")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    f"{chart_path}/06_churn_rate_by_service_count.png",
    dpi=300
)
plt.close()


# ============================================================
# 7. CHURN RATE BY CUSTOMER VALUE
# ============================================================

value = (
    df.groupby("is_high_value_customer")["churn_flag"]
    .mean()
    .mul(100)
)

value.index = [
    "Standard Customer",
    "High-Value Customer"
]

plt.figure(figsize=(9, 5))
value.plot(kind="bar")
plt.title("Churn Rate: Standard vs High-Value Customers")
plt.xlabel("Customer Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    f"{chart_path}/07_churn_rate_high_value_customers.png",
    dpi=300
)
plt.close()


# ============================================================
# 8. RISK SEGMENT ANALYSIS
# ============================================================

risk_order = [
    "High Risk",
    "Medium Risk",
    "Watchlist",
    "Low Risk"
]

risk = (
    df.groupby("risk_segment")["customerID"]
    .count()
    .reindex(risk_order)
)

plt.figure(figsize=(9, 5))
risk.plot(kind="bar")
plt.title("Customers by Risk Segment")
plt.xlabel("Risk Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    f"{chart_path}/08_customers_by_risk_segment.png",
    dpi=300
)
plt.close()


# ============================================================
# 9. MONTHLY CHARGES VS CHURN
# ============================================================

plt.figure(figsize=(9, 5))

df.boxplot(
    column="MonthlyCharges",
    by="Churn"
)

plt.title("Monthly Charges Distribution by Churn Status")
plt.suptitle("")
plt.xlabel("Churn Status")
plt.ylabel("Monthly Charges")

plt.tight_layout()

plt.savefig(
    f"{chart_path}/09_monthly_charges_vs_churn.png",
    dpi=300
)
plt.close()


# ============================================================
# 10. TENURE VS MONTHLY CHARGES
# ============================================================

plt.figure(figsize=(9, 5))

for churn_status in ["No", "Yes"]:
    subset = df[df["Churn"] == churn_status]

    plt.scatter(
        subset["tenure"],
        subset["MonthlyCharges"],
        alpha=0.25,
        label=f"Churn = {churn_status}"
    )

plt.title("Tenure vs Monthly Charges by Churn Status")
plt.xlabel("Tenure (Months)")
plt.ylabel("Monthly Charges")
plt.legend()

plt.tight_layout()

plt.savefig(
    f"{chart_path}/10_tenure_vs_monthly_charges.png",
    dpi=300
)
plt.close()


# ============================================================
# 11. CHURN RATE BY CONTRACT + INTERNET SERVICE
# ============================================================

contract_internet = (
    df.groupby(
        ["Contract", "InternetService"]
    )["churn_flag"]
    .mean()
    .mul(100)
    .unstack()
)

contract_internet.plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Churn Rate by Contract and Internet Service")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.legend(title="Internet Service")

plt.tight_layout()

plt.savefig(
    f"{chart_path}/11_contract_vs_internet_churn.png",
    dpi=300
)
plt.close()


# ============================================================
# 12. CHURN RATE BY CONTRACT + TENURE
# ============================================================

contract_tenure = (
    df.groupby(
        ["Contract", "tenure_group"]
    )["churn_flag"]
    .mean()
    .mul(100)
    .unstack()
)

contract_tenure.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Churn Rate by Contract and Tenure")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.legend(title="Tenure Group")

plt.tight_layout()

plt.savefig(
    f"{chart_path}/12_contract_vs_tenure_churn.png",
    dpi=300
)
plt.close()


# ============================================================
# PRINT KEY BUSINESS FINDINGS
# ============================================================

print("\n" + "=" * 70)
print("KEY BUSINESS FINDINGS")
print("=" * 70)

print(
    f"\nOverall Churn Rate: "
    f"{df['churn_flag'].mean() * 100:.2f}%"
)

print(
    f"\nHighest Contract Churn: "
    f"{contract.idxmax()} "
    f"({contract.max():.2f}%)"
)

print(
    f"\nHighest Tenure Churn: "
    f"{tenure.idxmax()} "
    f"({tenure.max():.2f}%)"
)

print(
    f"\nHighest Internet Service Churn: "
    f"{internet.idxmax()} "
    f"({internet.max():.2f}%)"
)

print(
    f"\nHighest Payment Method Churn: "
    f"{payment.idxmax()} "
    f"({payment.max():.2f}%)"
)

print(
    f"\nHighest Monthly Charge Group Churn: "
    f"{charge.idxmax()} "
    f"({charge.max():.2f}%)"
)

print(
    f"\nHighest Risk Segment: "
    f"{risk.idxmax()} "
    f"({risk.max():,.0f} customers)"
)

print("\n" + "=" * 70)
print("12 CHARTS CREATED SUCCESSFULLY")
print("=" * 70)

print(f"\nCharts saved to: {chart_path}")