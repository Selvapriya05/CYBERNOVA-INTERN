import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. SET PROJECT PATHS
# --------------------------------------------------

FEATURED_FILE = "../output/cleaned_data/german_credit_risk_featured.csv"
CLEANED_FILE = "../output/cleaned_data/german_credit_risk_cleaned.csv"

CHART_DIR = "../output/charts"
RESULT_DIR = "../output/analysis_results"

os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Use the featured file if Step 4 created it.
if os.path.exists(FEATURED_FILE):
    file_path = FEATURED_FILE
elif os.path.exists(CLEANED_FILE):
    file_path = CLEANED_FILE
else:
    raise FileNotFoundError(
        "Cleaned dataset not found. Run Step 3 first."
    )

df = pd.read_csv(file_path)

print("=" * 60)
print("BANKING LOAN RISK ASSESSMENT - EDA")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# --------------------------------------------------
# 2. IDENTIFY THE TARGET COLUMN
# --------------------------------------------------

target_candidates = [
    "risk", "class", "credit_risk", "loan_status"
]

target_col = next(
    (col for col in target_candidates if col in df.columns),
    None
)

if target_col is None:
    raise ValueError(
        "Risk target column not found. Check the dataset columns."
    )

print("\nTarget column:", target_col)
print("\nTarget distribution:")
print(df[target_col].value_counts(dropna=False))

# --------------------------------------------------
# 3. IDENTIFY GOOD AND BAD CREDIT RISK
# --------------------------------------------------

target_text = df[target_col].astype(str).str.strip().str.lower()

# Known labels from common German Credit dataset formats.
good_labels = {"good", "good risk", "1"}
bad_labels = {"bad", "bad risk", "2"}

unique_labels = set(target_text.dropna().unique())

if unique_labels.issubset(good_labels | bad_labels):
    df["bad_credit_risk_flag"] = target_text.isin(
        bad_labels
    ).astype(int)

    risk_mapping_available = True
else:
    # Do not guess the meaning of unfamiliar target labels.
    risk_mapping_available = False
    print(
        "\nWARNING: Target labels are not recognized as "
        "good/bad or 1/2. Bad-risk metrics will not be calculated."
    )

# --------------------------------------------------
# 4. GENERAL DATA SUMMARY
# --------------------------------------------------

summary = pd.DataFrame({
    "Metric": [
        "Total Customers / Records",
        "Total Columns",
        "Duplicate Rows",
        "Missing Values"
    ],
    "Value": [
        len(df),
        len(df.columns),
        int(df.duplicated().sum()),
        int(df.isna().sum().sum())
    ]
})

summary.to_csv(
    os.path.join(RESULT_DIR, "dataset_overview.csv"),
    index=False
)

numeric_summary = df.select_dtypes(
    include="number"
).describe().T

numeric_summary.to_csv(
    os.path.join(RESULT_DIR, "numeric_summary.csv")
)

categorical_summary = pd.DataFrame({
    "Column": df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns,
})

if not categorical_summary.empty:
    categorical_summary["Unique_Values"] = (
        categorical_summary["Column"]
        .map(df.nunique(dropna=False))
    )

    categorical_summary.to_csv(
        os.path.join(RESULT_DIR, "categorical_summary.csv"),
        index=False
    )

# --------------------------------------------------
# 5. KEY BANKING RISK METRICS
# --------------------------------------------------

metrics = {
    "total_records": int(len(df)),
    "total_columns": int(len(df.columns)),
    "target_column": target_col
}

if risk_mapping_available:
    bad_count = int(df["bad_credit_risk_flag"].sum())
    good_count = int(len(df) - bad_count)

    metrics.update({
        "good_risk_customers": good_count,
        "bad_risk_customers": bad_count,
        "bad_risk_rate_percent": round(
            bad_count / len(df) * 100, 2
        ) if len(df) else 0,
        "good_risk_rate_percent": round(
            good_count / len(df) * 100, 2
        ) if len(df) else 0
    })

    print("\nKEY RISK METRICS")
    print("-" * 40)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    risk_distribution = (
        df["bad_credit_risk_flag"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
        .rename_axis("bad_credit_risk_flag")
        .reset_index(name="customer_count")
    )

    risk_distribution["risk_label"] = (
        risk_distribution["bad_credit_risk_flag"]
        .map({0: "Good Risk", 1: "Bad Risk"})
    )

    risk_distribution.to_csv(
        os.path.join(RESULT_DIR, "risk_distribution.csv"),
        index=False
    )

else:
    print("\nTarget counts:")
    print(df[target_col].value_counts())

with open(
    os.path.join(RESULT_DIR, "key_risk_metrics.json"),
    "w"
) as file:
    json.dump(metrics, file, indent=4)

# --------------------------------------------------
# 6. TARGET DISTRIBUTION CHART
# --------------------------------------------------

target_counts = df[target_col].value_counts()

plt.figure(figsize=(8, 5))
target_counts.plot(kind="bar")
plt.title("Credit Risk Target Distribution")
plt.xlabel("Risk Category")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    os.path.join(CHART_DIR, "01_risk_target_distribution.png"),
    dpi=300
)
plt.close()

# --------------------------------------------------
# 7. NUMERIC FEATURE DISTRIBUTIONS
# --------------------------------------------------

numeric_cols = df.select_dtypes(
    include="number"
).columns.tolist()

# Exclude the engineered binary target from general distributions.
numeric_cols = [
    col for col in numeric_cols
    if col != "bad_credit_risk_flag"
]

for i, col in enumerate(numeric_cols[:8], start=2):
    values = df[col].dropna()

    if values.empty or values.nunique() <= 1:
        continue

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=20, edgecolor="black")
    plt.title(f"Distribution of {col.replace('_', ' ').title()}")
    plt.xlabel(col.replace("_", " ").title())
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            CHART_DIR,
            f"{i:02d}_{col[:40]}_distribution.png"
        ),
        dpi=300
    )
    plt.close()

# --------------------------------------------------
# 8. BAD-RISK RATE BY CATEGORICAL FEATURES
# --------------------------------------------------

if risk_mapping_available:

    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    categorical_cols = [
        col for col in categorical_cols
        if col != target_col
        and df[col].nunique(dropna=False) <= 20
    ]

    all_risk_tables = []

    for col in categorical_cols:
        temp = df[[col, "bad_credit_risk_flag"]].copy()
        temp[col] = temp[col].fillna("Unknown").astype(str)

        grouped = temp.groupby(
            col, dropna=False
        )["bad_credit_risk_flag"].agg(
            customer_count="count",
            bad_risk_count="sum",
            bad_risk_rate="mean"
        ).reset_index()

        grouped["bad_risk_rate_percent"] = (
            grouped["bad_risk_rate"] * 100
        ).round(2)

        grouped = grouped.rename(
            columns={col: "category"}
        )
        grouped.insert(0, "feature", col)

        all_risk_tables.append(grouped)

    if all_risk_tables:
        risk_by_feature = pd.concat(
            all_risk_tables,
            ignore_index=True
        )

        risk_by_feature.to_csv(
            os.path.join(
                RESULT_DIR,
                "bad_risk_rate_by_category.csv"
            ),
            index=False
        )

        # Chart the first categorical feature with multiple categories.
        chart_feature = next(
            (
                col for col in categorical_cols
                if df[col].nunique(dropna=False) > 1
            ),
            None
        )

        if chart_feature:
            chart_data = risk_by_feature[
                risk_by_feature["feature"] == chart_feature
            ].sort_values("bad_risk_rate_percent")

            plt.figure(figsize=(9, 5))
            plt.barh(
                chart_data["category"].astype(str),
                chart_data["bad_risk_rate_percent"]
            )
            plt.title(
                f"Bad Credit Risk Rate by {chart_feature.replace('_', ' ').title()}"
            )
            plt.xlabel("Bad Risk Rate (%)")
            plt.ylabel(chart_feature.replace("_", " ").title())
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    CHART_DIR,
                    "10_bad_risk_rate_by_category.png"
                ),
                dpi=300
            )
            plt.close()

# --------------------------------------------------
# 9. SAVE EDA DATASET
# --------------------------------------------------

df.to_csv(
    os.path.join(RESULT_DIR, "eda_dataset.csv"),
    index=False
)

print("\nEDA completed successfully.")
print("\nSaved results in:", RESULT_DIR)
print("Saved charts in:", CHART_DIR)