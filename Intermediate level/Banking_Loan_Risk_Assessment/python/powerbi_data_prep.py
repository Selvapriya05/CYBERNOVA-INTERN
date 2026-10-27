
import os
import pandas as pd

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

INPUT_FILE = "../output/cleaned_data/loan_risk_metrics_featured.csv"
OUTPUT_DIR = "../output/powerbi"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("POWER BI DATA PREPARATION")
print("=" * 60)

# --------------------------------------------------
# 2. VALIDATE DATA
# --------------------------------------------------

required_columns = [
    "loan_status",
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# In this dataset, 1 = default and 0 = non-default
df["loan_status"] = pd.to_numeric(
    df["loan_status"], errors="raise"
).astype(int)

if not set(df["loan_status"].unique()).issubset({0, 1}):
    raise ValueError("Unexpected loan_status values.")

df["default_flag"] = df["loan_status"]

df["loan_outcome"] = df["loan_status"].map({
    0: "Non-Default",
    1: "Default"
})

# Create percentage version for clear Power BI formatting.
df["loan_to_income_percent"] = (
    df["loan_percent_income"] * 100
).round(2)

# Ensure numeric columns are numeric.
numeric_columns = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "loan_to_income_percent",
    "cb_person_cred_hist_length"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# --------------------------------------------------
# 3. SAVE MAIN DASHBOARD DATASET
# --------------------------------------------------

main_file = os.path.join(
    OUTPUT_DIR, "loan_risk_powerbi.csv"
)

df.to_csv(main_file, index=False)

# --------------------------------------------------
# 4. CREATE CATEGORY SUMMARY TABLE
# --------------------------------------------------

category_features = [
    "loan_grade",
    "loan_intent",
    "person_home_ownership",
    "cb_person_default_on_file",
    "age_group",
    "income_band",
    "employment_band",
    "credit_history_band",
    "loan_to_income_group",
    "interest_rate_group",
    "loan_amount_group"
]

summary_tables = []

for col in category_features:
    if col not in df.columns:
        continue

    temp = df[[col, "loan_status"]].copy()
    temp[col] = temp[col].fillna("Unknown").astype(str)

    summary = temp.groupby(col).agg(
        total_loans=("loan_status", "count"),
        defaulted_loans=("loan_status", "sum")
    ).reset_index()

    summary["default_rate_percent"] = (
        summary["defaulted_loans"]
        / summary["total_loans"] * 100
    ).round(2)

    summary = summary.rename(
        columns={col: "category"}
    )
    summary.insert(0, "risk_factor", col)

    summary_tables.append(summary)

if summary_tables:
    category_summary = pd.concat(
        summary_tables, ignore_index=True
    )

    category_summary.to_csv(
        os.path.join(
            OUTPUT_DIR, "loan_risk_category_summary.csv"
        ),
        index=False
    )

# --------------------------------------------------
# 5. CREATE LOAN GRADE VS PURPOSE MATRIX
# --------------------------------------------------

grade_purpose = pd.pivot_table(
    df,
    index="loan_grade",
    columns="loan_intent",
    values="loan_status",
    aggfunc="mean",
    observed=True
) * 100

grade_purpose.to_csv(
    os.path.join(
        OUTPUT_DIR, "loan_grade_purpose_matrix.csv"
    )
)

# --------------------------------------------------
# 6. CREATE KPI SUMMARY
# --------------------------------------------------

total_loans = len(df)
defaulted_loans = int(df["default_flag"].sum())
non_defaulted_loans = total_loans - defaulted_loans

kpis = pd.DataFrame({
    "KPI": [
        "Total Loans",
        "Defaulted Loans",
        "Non-Defaulted Loans",
        "Default Rate (%)",
        "Non-Default Rate (%)",
        "Average Loan Amount",
        "Average Interest Rate (%)",
        "Average Annual Income",
        "Average Loan-to-Income (%)",
        "Average Customer Age"
    ],
    "Value": [
        total_loans,
        defaulted_loans,
        non_defaulted_loans,
        round(defaulted_loans / total_loans * 100, 2),
        round(non_defaulted_loans / total_loans * 100, 2),
        round(df["loan_amnt"].mean(), 2),
        round(df["loan_int_rate"].mean(), 2),
        round(df["person_income"].mean(), 2),
        round(df["loan_to_income_percent"].mean(), 2),
        round(df["person_age"].mean(), 2)
    ]
})

kpis.to_csv(
    os.path.join(OUTPUT_DIR, "loan_risk_kpi_summary.csv"),
    index=False
)

print("\nPOWER BI FILES CREATED")
print("----------------------------------------")
print("1. loan_risk_powerbi.csv")
print("2. loan_risk_category_summary.csv")
print("3. loan_grade_purpose_matrix.csv")
print("4. loan_risk_kpi_summary.csv")
print("\nMain dataset shape:", df.shape)
print("Total loans:", total_loans)
print("Defaulted loans:", defaulted_loans)
print("Default rate:", round(
    defaulted_loans / total_loans * 100, 2
), "%")
print("\nOutput folder:", OUTPUT_DIR)