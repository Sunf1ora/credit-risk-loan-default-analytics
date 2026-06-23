import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# Project 3 Day 5
# Prepare Tableau-ready dashboard datasets for credit risk analytics
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

scored_input_path = DATA_DIR / "credit_risk_model_scored.csv"
performance_input_path = DATA_DIR / "model_performance_summary.csv"
coefficients_input_path = DATA_DIR / "model_feature_coefficients.csv"

overview_output_path = DATA_DIR / "credit_dashboard_overview.csv"
probability_band_output_path = DATA_DIR / "default_probability_band_summary.csv"
model_risk_band_output_path = DATA_DIR / "model_risk_band_summary.csv"
loan_grade_model_output_path = DATA_DIR / "loan_grade_model_summary.csv"
high_probability_loans_output_path = DATA_DIR / "high_probability_loans.csv"
model_performance_dashboard_output_path = DATA_DIR / "model_performance_dashboard.csv"
top_features_output_path = DATA_DIR / "top_model_features.csv"

# ------------------------------------------------------------
# 1. Load model-scored dataset
# ------------------------------------------------------------

df = pd.read_csv(scored_input_path)
performance_df = pd.read_csv(performance_input_path)
coef_df = pd.read_csv(coefficients_input_path)

print("Model-scored credit risk dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print()

# Ensure numeric fields
numeric_cols = [
    "person_age",
    "person_income",
    "person_income_capped",
    "loan_amnt",
    "loan_amnt_capped",
    "loan_int_rate",
    "loan_status",
    "loan_percent_income",
    "default_flag",
    "credit_risk_score",
    "predicted_default_probability",
    "predicted_default_flag"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ------------------------------------------------------------
# 2. Dashboard overview KPI table
# ------------------------------------------------------------

overview = pd.DataFrame([{
    "total_loans": len(df),
    "actual_default_loans": df["default_flag"].sum(),
    "actual_default_rate": df["default_flag"].mean(),
    "predicted_default_loans": df["predicted_default_flag"].sum(),
    "predicted_default_rate": df["predicted_default_flag"].mean(),
    "average_predicted_default_probability": df["predicted_default_probability"].mean(),
    "total_loan_amount": df["loan_amnt"].sum(),
    "average_loan_amount": df["loan_amnt"].mean(),
    "average_interest_rate": df["loan_int_rate"].mean(),
    "average_loan_percent_income": df["loan_percent_income"].mean(),
    "high_probability_loans_50pct_plus": (df["predicted_default_probability"] >= 0.50).sum(),
    "very_high_probability_loans_75pct_plus": (df["predicted_default_probability"] >= 0.75).sum()
}])

overview.to_csv(overview_output_path, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 3. Default probability band summary
# ------------------------------------------------------------

prob_band_summary = (
    df
    .groupby("model_risk_band", observed=True)
    .agg(
        total_loans=("loan_status", "count"),
        actual_default_loans=("default_flag", "sum"),
        predicted_default_loans=("predicted_default_flag", "sum"),
        total_loan_amount=("loan_amnt", "sum"),
        average_loan_amount=("loan_amnt", "mean"),
        average_predicted_default_probability=("predicted_default_probability", "mean"),
        average_credit_risk_score=("credit_risk_score", "mean")
    )
    .reset_index()
)

prob_band_summary["actual_default_rate"] = (
    prob_band_summary["actual_default_loans"] / prob_band_summary["total_loans"]
)

prob_band_summary["predicted_default_rate"] = (
    prob_band_summary["predicted_default_loans"] / prob_band_summary["total_loans"]
)

prob_band_summary.to_csv(probability_band_output_path, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 4. Rule-based credit risk level summary
# ------------------------------------------------------------

risk_level_summary = (
    df
    .groupby("credit_risk_level")
    .agg(
        total_loans=("loan_status", "count"),
        actual_default_loans=("default_flag", "sum"),
        predicted_default_loans=("predicted_default_flag", "sum"),
        total_loan_amount=("loan_amnt", "sum"),
        average_loan_amount=("loan_amnt", "mean"),
        average_predicted_default_probability=("predicted_default_probability", "mean"),
        average_credit_risk_score=("credit_risk_score", "mean")
    )
    .reset_index()
)

risk_level_summary["actual_default_rate"] = (
    risk_level_summary["actual_default_loans"] / risk_level_summary["total_loans"]
)

risk_level_summary["predicted_default_rate"] = (
    risk_level_summary["predicted_default_loans"] / risk_level_summary["total_loans"]
)

risk_order = {
    "VERY HIGH": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
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

risk_level_summary.to_csv(model_risk_band_output_path, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 5. Loan grade and model summary
# ------------------------------------------------------------

loan_grade_summary = (
    df
    .groupby("loan_grade")
    .agg(
        total_loans=("loan_status", "count"),
        actual_default_loans=("default_flag", "sum"),
        predicted_default_loans=("predicted_default_flag", "sum"),
        total_loan_amount=("loan_amnt", "sum"),
        average_loan_amount=("loan_amnt", "mean"),
        average_interest_rate=("loan_int_rate", "mean"),
        average_predicted_default_probability=("predicted_default_probability", "mean"),
        average_credit_risk_score=("credit_risk_score", "mean")
    )
    .reset_index()
)

loan_grade_summary["actual_default_rate"] = (
    loan_grade_summary["actual_default_loans"] / loan_grade_summary["total_loans"]
)

loan_grade_summary["predicted_default_rate"] = (
    loan_grade_summary["predicted_default_loans"] / loan_grade_summary["total_loans"]
)

loan_grade_summary = loan_grade_summary.sort_values(by="loan_grade")

loan_grade_summary.to_csv(loan_grade_model_output_path, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 6. High probability loan list
# ------------------------------------------------------------

high_probability_loans = df[
    df["predicted_default_probability"] >= 0.75
].copy()

high_probability_loans = high_probability_loans.sort_values(
    by=["predicted_default_probability", "credit_risk_score", "loan_amnt"],
    ascending=[False, False, False]
)

# Add display priority
high_probability_loans["review_priority"] = np.where(
    high_probability_loans["predicted_default_probability"] >= 0.90,
    "Critical Review",
    "Priority Review"
)

# Limit columns for dashboard table
high_probability_columns = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "credit_risk_score",
    "credit_risk_level",
    "predicted_default_probability",
    "model_risk_band",
    "review_priority"
]

high_probability_columns = [
    col for col in high_probability_columns if col in high_probability_loans.columns
]

high_probability_loans[high_probability_columns].to_csv(
    high_probability_loans_output_path,
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# 7. Model performance dashboard table
# ------------------------------------------------------------

performance_long = performance_df.melt(
    id_vars=["model_name"],
    value_vars=[
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ],
    var_name="metric",
    value_name="value"
)

performance_long.to_csv(
    model_performance_dashboard_output_path,
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# 8. Top model features
# ------------------------------------------------------------

top_features = coef_df.head(20).copy()
top_features.to_csv(top_features_output_path, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 9. Print output summary
# ------------------------------------------------------------

print("Credit risk dashboard datasets created successfully.")
print()

print("Files saved:")
print("-", overview_output_path)
print("-", probability_band_output_path)
print("-", model_risk_band_output_path)
print("-", loan_grade_model_output_path)
print("-", high_probability_loans_output_path)
print("-", model_performance_dashboard_output_path)
print("-", top_features_output_path)
print()

print("Dashboard overview:")
print(overview.T)
print()

print("Default probability band summary:")
print(prob_band_summary)
print()

print("Loan grade model summary:")
print(loan_grade_summary[[
    "loan_grade",
    "total_loans",
    "actual_default_rate",
    "predicted_default_rate",
    "average_predicted_default_probability"
]])
print()

print("High probability loans:", len(high_probability_loans))
print()

print("Model performance metrics:")
print(performance_long)
