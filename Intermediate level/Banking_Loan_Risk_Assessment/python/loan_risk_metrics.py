
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# 1. Paths
INPUT_FILE = "../output/cleaned_data/german_credit_risk_featured.csv"
RESULT_DIR = "../output/analysis_results"
CHART_DIR = "../output/charts"
CLEANED_DIR = "../output/cleaned_data"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("BANKING LOAN RISK - RISK METRICS")
print("=" * 60)

# 2. Validate target
if "loan_status" not in df.columns:
    raise ValueError("The loan_status column was not found.")

df["loan_status"] = pd.to_numeric(
    df["loan_status"], errors="raise"
).astype(int)

if not set(df["loan_status"].unique()).issubset({0, 1}):
    raise ValueError("Expected loan_status values to be 0 and 1.")

# In this dataset: 1 = default, 0 = non-default
df["default_flag"] = df["loan_status"]

# 3. Feature engineering using actual dataset columns
df["age_group"] = pd.cut(
    df["person_age"],
    bins=[0, 25, 35, 45, 55, float("inf")],
    labels=["18-25", "26-35", "36-45", "46-55", "56+"],
    right=True
)

df["income_group"] = pd.cut(
    df["person_income"],
    bins=[0, 25000, 50000, 75000, 100000, float("inf")],
    labels=["Up to 25k", "25k-50k", "50k-75k",
            "75k-100k", "Above 100k"],
    right=False
)

df["loan_amount_group"] = pd.cut(
    df["loan_amnt"],
    bins=[0, 5000, 10000, 20000, 35000, float("inf")],
    labels=["Up to 5k", "5k-10k", "10k-20k",
            "20k-35k", "Above 35k"],
    right=False
)

df["interest_rate_group"] = pd.cut(
    df["loan_int_rate"],
    bins=[0, 8, 12, 16, float("inf")],
    labels=["Up to 8%", "8%-12%", "12%-16%", "Above 16%"],
    right=False
)

df["employment_group"] = pd.cut(
    df["person_emp_length"],
    bins=[-0.01, 1, 5, 10, float("inf")],
    labels=["0-1 years", "2-5 years",
            "6-10 years", "Above 10 years"],
    right=True
)

df["credit_history_group"] = pd.cut(
    df["cb_person_cred_hist_length"],
    bins=[-1, 2, 5, 10, float("inf")],
    labels=["0-2 years", "3-5 years",
            "6-10 years", "Above 10 years"],
    right=True
)

# Save the dataset with new features
featured_output = os.path.join(
    CLEANED_DIR, "loan_risk_metrics_featured.csv"
)
df.to_csv(featured_output, index=False)

# 4. Overall metrics
total_loans = len(df)
default_count = int(df["default_flag"].sum())
non_default_count = int((df["default_flag"] == 0).sum())

metrics = {
    "total_loan_records": total_loans,
    "defaulted_loan_records": default_count,
    "non_defaulted_loan_records": non_default_count,
    "default_rate_percent": round(
        default_count / total_loans * 100, 2
    ) if total_loans else 0,
    "non_default_rate_percent": round(
        non_default_count / total_loans * 100, 2
    ) if total_loans else 0,
    "average_loan_amount": round(df["loan_amnt"].mean(), 2),
    "average_interest_rate_percent": round(
        df["loan_int_rate"].mean(), 2
    ),
    "average_income": round(df["person_income"].mean(), 2),
    "average_loan_to_income_percent": round(
        df["loan_percent_income"].mean() * 100, 2
    ),
    "average_age": round(df["person_age"].mean(), 2)
}

with open(
    os.path.join(RESULT_DIR, "loan_risk_metrics.json"), "w"
) as file:
    json.dump(metrics, file, indent=4)

pd.DataFrame(
    list(metrics.items()), columns=["Metric", "Value"]
).to_csv(
    os.path.join(RESULT_DIR, "loan_risk_metrics.csv"),
    index=False
)

print("\nOVERALL METRICS")
for name, value in metrics.items():
    print(f"{name}: {value}")

# 5. Default rate by category
categorical_features = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
    "age_group",
    "income_group",
    "loan_amount_group",
    "interest_rate_group",
    "employment_group",
    "credit_history_group"
]

all_tables = []

for col in categorical_features:
    temp = df[[col, "default_flag"]].copy()
    temp[col] = temp[col].astype("object").fillna("Unknown")

    grouped = temp.groupby(
        col, observed=False
    )["default_flag"].agg(
        loan_records="count",
        defaulted_records="sum",
        default_rate="mean"
    ).reset_index()

    grouped["default_rate_percent"] = (
        grouped["default_rate"] * 100
    ).round(2)

    grouped = grouped.rename(columns={col: "category"})
    grouped.insert(0, "feature", col)
    all_tables.append(grouped)

default_analysis = pd.concat(all_tables, ignore_index=True)

default_analysis.to_csv(
    os.path.join(RESULT_DIR, "default_rate_by_category.csv"),
    index=False
)

# 6. Create charts for important business factors
chart_features = [
    ("loan_grade", "Default Rate by Loan Grade"),
    ("loan_intent", "Default Rate by Loan Purpose"),
    ("person_home_ownership", "Default Rate by Home Ownership"),
    ("interest_rate_group", "Default Rate by Interest Rate Group"),
    ("loan_amount_group", "Default Rate by Loan Amount Group")
]

for i, (col, title) in enumerate(chart_features, start=1):
    chart_data = default_analysis[
        default_analysis["feature"] == col
    ].copy()

    chart_data = chart_data.dropna(
        subset=["default_rate_percent"]
    ).sort_values("default_rate_percent")

    if chart_data.empty:
        continue

    plt.figure(figsize=(9, 5))
    plt.barh(
        chart_data["category"].astype(str),
        chart_data["default_rate_percent"]
    )
    plt.title(title)
    plt.xlabel("Default Rate (%)")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(
        os.path.join(CHART_DIR, f"risk_factor_{i:02d}.png"),
        dpi=300
    )
    plt.close()

print("\nFEATURE ENGINEERING AND RISK METRICS COMPLETED")
print("Featured dataset:", featured_output)
print("Results folder:", RESULT_DIR)
print("Charts folder:", CHART_DIR)