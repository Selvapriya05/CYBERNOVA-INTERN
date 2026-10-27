import os
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

INPUT_FILE = "../output/cleaned_data/loan_risk_metrics_featured.csv"
RESULT_DIR = "../output/analysis_results"
CHART_DIR = "../output/charts"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

required = [
    "loan_status", "loan_amnt", "person_income",
    "loan_percent_income", "loan_int_rate",
    "person_age", "person_emp_length",
    "cb_person_cred_hist_length", "loan_grade",
    "loan_intent"
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["loan_status"] = pd.to_numeric(
    df["loan_status"], errors="raise"
)

# Dataset definition: 1 = default, 0 = non-default
overall_default_rate = df["loan_status"].mean() * 100

print("=" * 60)
print("STEP 7: DEEP LOAN RISK ANALYSIS")
print("=" * 60)
print(f"Overall default rate: {overall_default_rate:.2f}%")

# --------------------------------------------------
# 2. CREATE ANALYSIS GROUPS
# --------------------------------------------------

df["loan_to_income_group"] = pd.cut(
    df["loan_percent_income"],
    bins=[-0.001, 0.10, 0.20, 0.30, 0.40, float("inf")],
    labels=[
        "Up to 10%", "10%-20%", "20%-30%",
        "30%-40%", "Above 40%"
    ],
    right=True
)

df["income_band"] = pd.cut(
    df["person_income"],
    bins=[0, 25000, 50000, 75000, 100000, float("inf")],
    labels=[
        "Below 25k", "25k-50k", "50k-75k",
        "75k-100k", "Above 100k"
    ]
)

df["employment_band"] = pd.cut(
    df["person_emp_length"],
    bins=[-0.01, 1, 5, 10, float("inf")],
    labels=[
        "0-1 years", "Over 1-5 years",
        "Over 5-10 years", "Over 10 years"
    ]
)

df["credit_history_band"] = pd.cut(
    df["cb_person_cred_hist_length"],
    bins=[-1, 2, 5, 10, float("inf")],
    labels=[
        "0-2 years", "Over 2-5 years",
        "Over 5-10 years", "Over 10 years"
    ]
)

# --------------------------------------------------
# 3. DEFAULT RATE BY BUSINESS CATEGORY
# --------------------------------------------------

features = [
    "loan_grade",
    "loan_intent",
    "person_home_ownership",
    "cb_person_default_on_file",
    "loan_to_income_group",
    "income_band",
    "employment_band",
    "credit_history_band",
    "age_group",
    "interest_rate_group",
    "loan_amount_group"
]

all_results = []

for feature in features:
    if feature not in df.columns:
        continue

    temp = df[[feature, "loan_status"]].copy()
    temp[feature] = temp[feature].astype("object").fillna("Unknown")

    grouped = temp.groupby(
        feature, observed=True
    )["loan_status"].agg(
        loan_count="count",
        default_count="sum",
        default_rate="mean"
    ).reset_index()

    grouped["default_rate_percent"] = (
        grouped["default_rate"] * 100
    ).round(2)

    grouped["difference_from_average_pp"] = (
        grouped["default_rate_percent"] - overall_default_rate
    ).round(2)

    grouped = grouped.rename(columns={feature: "category"})
    grouped.insert(0, "feature", feature)

    all_results.append(grouped)

category_results = pd.concat(all_results, ignore_index=True)

category_results.to_csv(
    os.path.join(RESULT_DIR, "deep_default_analysis.csv"),
    index=False
)

# Only flag categories with at least 100 records.
# This avoids highlighting categories with very small sample sizes.
reliable_categories = category_results[
    category_results["loan_count"] >= 100
].copy()

higher_risk = reliable_categories[
    reliable_categories["default_rate_percent"] > overall_default_rate
].sort_values("difference_from_average_pp", ascending=False)

higher_risk.to_csv(
    os.path.join(RESULT_DIR, "above_average_default_categories.csv"),
    index=False
)

print("\nCATEGORIES ABOVE THE OVERALL DEFAULT RATE")
print("-" * 60)

if higher_risk.empty:
    print("No categories exceeded the overall rate with 100+ records.")
else:
    print(
        higher_risk[
            [
                "feature", "category", "loan_count",
                "default_rate_percent",
                "difference_from_average_pp"
            ]
        ].to_string(index=False)
    )

# --------------------------------------------------
# 4. NUMERIC COMPARISON: DEFAULT VS NON-DEFAULT
# --------------------------------------------------

numeric_features = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length"
]

comparison = df.groupby("loan_status")[numeric_features].mean().T
comparison = comparison.rename(
    columns={0: "non_default_average", 1: "default_average"}
)

if "non_default_average" in comparison.columns and \
   "default_average" in comparison.columns:

    comparison["difference_default_minus_non_default"] = (
        comparison["default_average"]
        - comparison["non_default_average"]
    )

comparison.to_csv(
    os.path.join(RESULT_DIR, "default_vs_non_default_comparison.csv")
)

print("\nAVERAGE PROFILE: DEFAULT VS NON-DEFAULT")
print("-" * 60)
print(comparison.round(2).to_string())

# --------------------------------------------------
# 5. CROSS-ANALYSIS: LOAN GRADE VS LOAN PURPOSE
# --------------------------------------------------

cross_table = pd.pivot_table(
    df,
    values="loan_status",
    index="loan_grade",
    columns="loan_intent",
    aggfunc="mean",
    observed=True
) * 100

cross_table.to_csv(
    os.path.join(RESULT_DIR, "grade_vs_intent_default_rate.csv")
)

print("\nDEFAULT RATE BY LOAN GRADE AND PURPOSE (%)")
print(cross_table.round(2).to_string())

# Heatmap
if not cross_table.empty:
    fig, ax = plt.subplots(figsize=(11, 6))

    image = ax.imshow(
        cross_table.values,
        aspect="auto",
        cmap="YlOrRd"
    )

    ax.set_xticks(range(len(cross_table.columns)))
    ax.set_xticklabels(
        cross_table.columns.astype(str),
        rotation=35,
        ha="right"
    )

    ax.set_yticks(range(len(cross_table.index)))
    ax.set_yticklabels(cross_table.index.astype(str))

    ax.set_title("Default Rate: Loan Grade vs Loan Purpose")
    ax.set_xlabel("Loan Purpose")
    ax.set_ylabel("Loan Grade")

    fig.colorbar(image, ax=ax, label="Default Rate (%)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(CHART_DIR, "deep_grade_vs_intent_heatmap.png"),
        dpi=300
    )
    plt.close()

# --------------------------------------------------
# 6. CHART: DEFAULT RATE BY LOAN-TO-INCOME GROUP
# --------------------------------------------------

feature = "loan_to_income_group"

chart_data = category_results[
    category_results["feature"] == feature
].copy()

chart_data = chart_data.dropna(
    subset=["default_rate_percent"]
)

if not chart_data.empty:
    chart_data = chart_data.sort_values("default_rate_percent")

    plt.figure(figsize=(9, 5))
    plt.barh(
        chart_data["category"].astype(str),
        chart_data["default_rate_percent"]
    )
    plt.axvline(
        overall_default_rate,
        linestyle="--",
        label=f"Overall: {overall_default_rate:.2f}%"
    )
    plt.title("Default Rate by Loan-to-Income Group")
    plt.xlabel("Default Rate (%)")
    plt.ylabel("Loan-to-Income Group")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(CHART_DIR, "deep_default_by_loan_income.png"),
        dpi=300
    )
    plt.close()

# --------------------------------------------------
# 7. CHART: LOAN INTEREST RATE BY DEFAULT STATUS
# --------------------------------------------------

interest_data = [
    df.loc[df["loan_status"] == 0, "loan_int_rate"].dropna(),
    df.loc[df["loan_status"] == 1, "loan_int_rate"].dropna()
]

if all(len(values) > 0 for values in interest_data):
    plt.figure(figsize=(8, 5))
    plt.boxplot(
        interest_data,
        tick_labels=["Non-default", "Default"]
    )
    plt.title("Interest Rate Distribution by Loan Status")
    plt.ylabel("Interest Rate (%)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(CHART_DIR, "deep_interest_rate_by_status.png"),
        dpi=300
    )
    plt.close()

# --------------------------------------------------
# 8. SAVE FINAL ANALYSIS DATASET
# --------------------------------------------------

df.to_csv(
    os.path.join(RESULT_DIR, "deep_analysis_dataset.csv"),
    index=False
)

print("Results saved to:", RESULT_DIR)
print("Charts saved to:", CHART_DIR)