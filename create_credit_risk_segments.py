import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# Project 3 Day 3
# Create borrower credit risk segmentation and portfolio summaries
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

input_path = DATA_DIR / "credit_risk_cleaned.csv"

segmented_output_path = DATA_DIR / "credit_risk_segmented.csv"
portfolio_summary_path = DATA_DIR / "portfolio_risk_summary.csv"
risk_level_summary_path = DATA_DIR / "risk_level_summary.csv"
high_risk_borrowers_path = DATA_DIR / "high_risk_borrowers.csv"

df = pd.read_csv(input_path)

print("Cleaned credit risk dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print()

# ------------------------------------------------------------
# 1. Ensure key columns are numeric
# ------------------------------------------------------------

numeric_cols = [
    "person_age",
    "person_income",
    "person_income_capped",
    "person_emp_length",
    "loan_amnt",
    "loan_amnt_capped",
    "loan_int_rate",
    "loan_status",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "default_flag",
    "high_loan_to_income_flag",
    "previous_default_flag",
    "high_interest_flag"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

categorical_cols = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
    "income_band",
    "age_band",
    "loan_amount_band",
    "interest_rate_band"
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper()

# ------------------------------------------------------------
# 2. Create rule-based credit risk score
# This is a portfolio demonstration score, not an official credit score.
# ------------------------------------------------------------

# Loan grade risk points
grade_score_map = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7
}

df["loan_grade_score"] = df["loan_grade"].map(grade_score_map).fillna(3)

# Home ownership risk points
home_score_map = {
    "OWN": 0,
    "MORTGAGE": 1,
    "RENT": 3,
    "OTHER": 3
}

df["home_ownership_score"] = df["person_home_ownership"].map(home_score_map).fillna(2)

# Loan-to-income risk points
df["loan_to_income_score"] = np.select(
    [
        df["loan_percent_income"] < 0.10,
        df["loan_percent_income"].between(0.10, 0.20, inclusive="left"),
        df["loan_percent_income"].between(0.20, 0.30, inclusive="left"),
        df["loan_percent_income"] >= 0.30
    ],
    [0, 1, 2, 4],
    default=2
)

# Interest rate risk points
df["interest_rate_score"] = np.select(
    [
        df["loan_int_rate"] < 8,
        df["loan_int_rate"].between(8, 12, inclusive="left"),
        df["loan_int_rate"].between(12, 16, inclusive="left"),
        df["loan_int_rate"] >= 16
    ],
    [0, 1, 2, 4],
    default=2
)

# Previous default risk points
df["previous_default_score"] = np.where(
    df["previous_default_flag"] == 1,
    4,
    0
)

# Employment length risk points
df["employment_length_score"] = np.select(
    [
        df["person_emp_length"] >= 5,
        df["person_emp_length"].between(2, 5, inclusive="left"),
        df["person_emp_length"] < 2
    ],
    [0, 1, 2],
    default=1
)

# Credit history length risk points
df["credit_history_score"] = np.select(
    [
        df["cb_person_cred_hist_length"] >= 8,
        df["cb_person_cred_hist_length"].between(3, 8, inclusive="left"),
        df["cb_person_cred_hist_length"] < 3
    ],
    [0, 1, 2],
    default=1
)

# Total rule-based risk score
df["credit_risk_score"] = (
    df["loan_grade_score"]
    + df["home_ownership_score"]
    + df["loan_to_income_score"]
    + df["interest_rate_score"]
    + df["previous_default_score"]
    + df["employment_length_score"]
    + df["credit_history_score"]
)

# ------------------------------------------------------------
# 3. Assign credit risk level
# ------------------------------------------------------------

def assign_credit_risk_level(score):
    if score >= 15:
        return "Very High"
    elif score >= 10:
        return "High"
    elif score >= 6:
        return "Medium"
    else:
        return "Low"

df["credit_risk_level"] = df["credit_risk_score"].apply(assign_credit_risk_level)

# ------------------------------------------------------------
# 4. Create monitoring flags
# ------------------------------------------------------------

df["high_risk_borrower_flag"] = np.where(
    df["credit_risk_level"].isin(["High", "Very High"]),
    1,
    0
)

df["watchlist_flag"] = np.where(
    (df["credit_risk_level"].isin(["High", "Very High"]))
    | (df["previous_default_flag"] == 1)
    | (df["high_loan_to_income_flag"] == 1)
    | (df["high_interest_flag"] == 1),
    1,
    0
)

df["model_priority"] = np.where(
    (df["credit_risk_level"] == "Very High") | (df["default_flag"] == 1),
    "Priority Review",
    np.where(
        df["credit_risk_level"] == "High",
        "Enhanced Monitoring",
        "Routine Monitoring"
    )
)

# ------------------------------------------------------------
# 5. Portfolio-level summaries
# ------------------------------------------------------------

portfolio_summary = pd.DataFrame([{
    "total_loans": len(df),
    "total_defaults": df["default_flag"].sum(),
    "overall_default_rate": df["default_flag"].mean(),
    "total_loan_amount": df["loan_amnt"].sum(),
    "average_loan_amount": df["loan_amnt"].mean(),
    "average_income": df["person_income"].mean(),
    "average_interest_rate": df["loan_int_rate"].mean(),
    "average_loan_percent_income": df["loan_percent_income"].mean(),
    "average_credit_risk_score": df["credit_risk_score"].mean(),
    "high_risk_borrowers": df["high_risk_borrower_flag"].sum(),
    "watchlist_borrowers": df["watchlist_flag"].sum()
}])

risk_level_summary = (
    df
    .groupby("credit_risk_level")
    .agg(
        total_loans=("loan_status", "count"),
        default_loans=("default_flag", "sum"),
        total_loan_amount=("loan_amnt", "sum"),
        average_loan_amount=("loan_amnt", "mean"),
        average_income=("person_income", "mean"),
        average_interest_rate=("loan_int_rate", "mean"),
        average_loan_percent_income=("loan_percent_income", "mean"),
        average_credit_risk_score=("credit_risk_score", "mean")
    )
    .reset_index()
)

risk_level_summary["default_rate"] = (
    risk_level_summary["default_loans"] / risk_level_summary["total_loans"]
)

risk_order = {
    "Very High": 4,
    "High": 3,
    "Medium": 2,
    "Low": 1
}

risk_level_summary["risk_order"] = risk_level_summary["credit_risk_level"].map(risk_order)

risk_level_summary = risk_level_summary.sort_values(
    by="risk_order",
    ascending=False
)

high_risk_borrowers = df[
    df["credit_risk_level"].isin(["High", "Very High"])
].copy()

high_risk_borrowers = high_risk_borrowers.sort_values(
    by=["credit_risk_score", "loan_amnt", "loan_int_rate"],
    ascending=[False, False, False]
)

# ------------------------------------------------------------
# 6. Export results
# ------------------------------------------------------------

df.to_csv(segmented_output_path, index=False, encoding="utf-8-sig")
portfolio_summary.to_csv(portfolio_summary_path, index=False, encoding="utf-8-sig")
risk_level_summary.to_csv(risk_level_summary_path, index=False, encoding="utf-8-sig")
high_risk_borrowers.to_csv(high_risk_borrowers_path, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 7. Print results
# ------------------------------------------------------------

print("Credit risk segmentation completed successfully.")
print()

print("Files saved:")
print("-", segmented_output_path)
print("-", portfolio_summary_path)
print("-", risk_level_summary_path)
print("-", high_risk_borrowers_path)
print()

print("Portfolio summary:")
print(portfolio_summary.T)
print()

print("Risk level summary:")
print(risk_level_summary[[
    "credit_risk_level",
    "total_loans",
    "default_loans",
    "default_rate",
    "average_credit_risk_score"
]])
print()

print("High-risk borrower count:", len(high_risk_borrowers))
print()

print("Preview of high-risk borrowers:")
print(high_risk_borrowers[[
    "person_age",
    "person_income",
    "person_home_ownership",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "credit_risk_score",
    "credit_risk_level",
    "model_priority"
]].head(10))