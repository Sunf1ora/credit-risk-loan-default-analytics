import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# Project 3 Day 2
# Clean credit risk dataset and generate initial default rate summaries
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

input_path = DATA_DIR / "credit_risk_dataset.csv"
cleaned_output_path = DATA_DIR / "credit_risk_cleaned.csv"

# Read raw dataset
df = pd.read_csv(input_path)

print("Raw dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print()

# ------------------------------------------------------------
# 1. Standardise column names
# ------------------------------------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("Column names:")
print(list(df.columns))
print()

# ------------------------------------------------------------
# 2. Basic data quality checks
# ------------------------------------------------------------

print("Missing values before cleaning:")
print(df.isna().sum())
print()

print("Duplicate rows before cleaning:", df.duplicated().sum())
print()

# Remove duplicate rows
df = df.drop_duplicates().copy()

# ------------------------------------------------------------
# 3. Ensure correct data types
# ------------------------------------------------------------

numeric_cols = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_status",
    "loan_percent_income",
    "cb_person_cred_hist_length"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

categorical_cols = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file"
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper()

# ------------------------------------------------------------
# 4. Handle missing values
# ------------------------------------------------------------

# For employment length and interest rate, use median by loan grade if possible.
if "loan_grade" in df.columns:
    if "person_emp_length" in df.columns:
        df["person_emp_length"] = df.groupby("loan_grade")["person_emp_length"].transform(
            lambda x: x.fillna(x.median())
        )
        df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())

    if "loan_int_rate" in df.columns:
        df["loan_int_rate"] = df.groupby("loan_grade")["loan_int_rate"].transform(
            lambda x: x.fillna(x.median())
        )
        df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())

# If any categorical values are missing or became "NAN", label as UNKNOWN.
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].replace("NAN", "UNKNOWN")
        df[col] = df[col].fillna("UNKNOWN")

# ------------------------------------------------------------
# 5. Remove or cap unrealistic outliers
# ------------------------------------------------------------

# This dataset may contain very high ages or incomes.
# For portfolio analysis, we cap extreme values rather than deleting many rows.

if "person_age" in df.columns:
    df = df[df["person_age"].between(18, 100)].copy()

if "person_income" in df.columns:
    income_cap = df["person_income"].quantile(0.99)
    df["person_income_capped"] = np.where(
        df["person_income"] > income_cap,
        income_cap,
        df["person_income"]
    )

if "loan_amnt" in df.columns:
    loan_cap = df["loan_amnt"].quantile(0.99)
    df["loan_amnt_capped"] = np.where(
        df["loan_amnt"] > loan_cap,
        loan_cap,
        df["loan_amnt"]
    )

# ------------------------------------------------------------
# 6. Create analytical bands
# ------------------------------------------------------------

# Income bands
if "person_income_capped" in df.columns:
    df["income_band"] = pd.cut(
        df["person_income_capped"],
        bins=[0, 30000, 50000, 75000, 100000, np.inf],
        labels=["<30k", "30k-50k", "50k-75k", "75k-100k", "100k+"],
        include_lowest=True
    )

# Age bands
if "person_age" in df.columns:
    df["age_band"] = pd.cut(
        df["person_age"],
        bins=[17, 25, 35, 45, 60, 100],
        labels=["18-25", "26-35", "36-45", "46-60", "60+"],
        include_lowest=True
    )

# Loan amount bands
if "loan_amnt_capped" in df.columns:
    df["loan_amount_band"] = pd.cut(
        df["loan_amnt_capped"],
        bins=[0, 5000, 10000, 15000, 25000, np.inf],
        labels=["<5k", "5k-10k", "10k-15k", "15k-25k", "25k+"],
        include_lowest=True
    )

# Interest rate bands
if "loan_int_rate" in df.columns:
    df["interest_rate_band"] = pd.cut(
        df["loan_int_rate"],
        bins=[0, 8, 12, 16, 20, np.inf],
        labels=["<8%", "8%-12%", "12%-16%", "16%-20%", "20%+"],
        include_lowest=True
    )

# ------------------------------------------------------------
# 7. Create portfolio-level risk indicators
# ------------------------------------------------------------

df["default_flag"] = df["loan_status"]

df["high_loan_to_income_flag"] = np.where(
    df["loan_percent_income"] >= 0.30,
    1,
    0
)

df["previous_default_flag"] = np.where(
    df["cb_person_default_on_file"] == "Y",
    1,
    0
)

df["high_interest_flag"] = np.where(
    df["loan_int_rate"] >= 16,
    1,
    0
)

# ------------------------------------------------------------
# 8. Summary helper function
# ------------------------------------------------------------

def default_rate_summary(data, group_col):
    summary = (
        data
        .groupby(group_col, observed=True)
        .agg(
            total_loans=("loan_status", "count"),
            default_loans=("default_flag", "sum"),
            total_loan_amount=("loan_amnt", "sum"),
            average_income=("person_income", "mean"),
            average_loan_amount=("loan_amnt", "mean"),
            average_interest_rate=("loan_int_rate", "mean"),
            average_loan_percent_income=("loan_percent_income", "mean")
        )
        .reset_index()
    )

    summary["default_rate"] = summary["default_loans"] / summary["total_loans"]

    summary = summary.sort_values(
        by=["default_rate", "total_loans"],
        ascending=[False, False]
    )

    return summary

# ------------------------------------------------------------
# 9. Export cleaned dataset and summaries
# ------------------------------------------------------------

df.to_csv(cleaned_output_path, index=False, encoding="utf-8-sig")

summary_outputs = {
    "default_rate_by_loan_grade.csv": default_rate_summary(df, "loan_grade"),
    "default_rate_by_home_ownership.csv": default_rate_summary(df, "person_home_ownership"),
    "default_rate_by_loan_intent.csv": default_rate_summary(df, "loan_intent"),
    "default_rate_by_income_band.csv": default_rate_summary(df, "income_band"),
    "default_rate_by_age_band.csv": default_rate_summary(df, "age_band"),
    "default_rate_by_interest_rate_band.csv": default_rate_summary(df, "interest_rate_band"),
    "default_rate_by_loan_amount_band.csv": default_rate_summary(df, "loan_amount_band")
}

for filename, summary_df in summary_outputs.items():
    summary_df.to_csv(DATA_DIR / filename, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 10. Print results
# ------------------------------------------------------------

print("Cleaned dataset saved successfully.")
print(f"File saved to: {cleaned_output_path}")
print()

print("Rows after cleaning:", len(df))
print("Columns after cleaning:", len(df.columns))
print()

print("Missing values after cleaning:")
print(df.isna().sum())
print()

print("Overall default rate:")
overall_default_rate = df["default_flag"].mean()
print(f"{overall_default_rate:.2%}")
print()

print("Default rate by loan grade:")
print(summary_outputs["default_rate_by_loan_grade.csv"][[
    "loan_grade",
    "total_loans",
    "default_loans",
    "default_rate"
]])
print()

print("Default rate by home ownership:")
print(summary_outputs["default_rate_by_home_ownership.csv"][[
    "person_home_ownership",
    "total_loans",
    "default_loans",
    "default_rate"
]])
print()

print("Output files created:")
for filename in summary_outputs.keys():
    print("-", DATA_DIR / filename)

print("-", cleaned_output_path)