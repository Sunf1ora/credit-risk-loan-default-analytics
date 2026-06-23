import pandas as pd
import numpy as np
from pathlib import Path
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------------------
# Project 3 Day 4
# Train a baseline logistic regression model for loan default prediction
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

input_path = DATA_DIR / "credit_risk_segmented.csv"

scored_output_path = DATA_DIR / "credit_risk_model_scored.csv"
performance_output_path = DATA_DIR / "model_performance_summary.csv"
coefficients_output_path = DATA_DIR / "model_feature_coefficients.csv"
model_output_path = MODEL_DIR / "logistic_regression_default_model.pkl"

# ------------------------------------------------------------
# 1. Load segmented dataset
# ------------------------------------------------------------

df = pd.read_csv(input_path)

print("Segmented credit risk dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print()

# ------------------------------------------------------------
# 2. Define target and features
# ------------------------------------------------------------

target_col = "default_flag"

numeric_features = [
    "person_age",
    "person_income_capped",
    "person_emp_length",
    "loan_amnt_capped",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_risk_score"
]

categorical_features = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
    "credit_risk_level"
]

# Keep only available columns
numeric_features = [col for col in numeric_features if col in df.columns]
categorical_features = [col for col in categorical_features if col in df.columns]

model_cols = numeric_features + categorical_features + [target_col]

model_df = df[model_cols].copy()

# Drop rows with missing target or features
model_df = model_df.dropna().copy()

X = model_df[numeric_features + categorical_features]
y = model_df[target_col]

print("Model dataset prepared.")
print("Rows used for modelling:", len(model_df))
print("Default rate in modelling data:", f"{y.mean():.2%}")
print()
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)
print()

# ------------------------------------------------------------
# 3. Train/test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Train/test split completed.")
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print()

# ------------------------------------------------------------
# 4. Build preprocessing + logistic regression pipeline
# ------------------------------------------------------------

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# ------------------------------------------------------------
# 5. Train model
# ------------------------------------------------------------

clf.fit(X_train, y_train)

print("Logistic regression model trained successfully.")
print()

# ------------------------------------------------------------
# 6. Predict and evaluate
# ------------------------------------------------------------

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

performance_summary = pd.DataFrame([{
    "model_name": "Logistic Regression",
    "target": target_col,
    "rows_used": len(model_df),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "default_rate": y.mean(),
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "true_negative": tn,
    "false_positive": fp,
    "false_negative": fn,
    "true_positive": tp
}])

performance_summary.to_csv(performance_output_path, index=False, encoding="utf-8-sig")

print("Model performance:")
print(performance_summary.T)
print()

print("Classification report:")
print(classification_report(y_test, y_pred))
print()

print("Confusion matrix:")
print(cm)
print()

# ------------------------------------------------------------
# 7. Extract model coefficients
# ------------------------------------------------------------

# Get feature names after preprocessing
preprocessor_fitted = clf.named_steps["preprocessor"]
model_fitted = clf.named_steps["model"]

feature_names = []

# Numeric feature names
feature_names.extend(numeric_features)

# Categorical one-hot feature names
if categorical_features:
    onehot = preprocessor_fitted.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(onehot.get_feature_names_out(categorical_features))
    feature_names.extend(cat_feature_names)

coefficients = model_fitted.coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
})

coef_df["absolute_coefficient"] = coef_df["coefficient"].abs()

coef_df = coef_df.sort_values(
    by="absolute_coefficient",
    ascending=False
)

coef_df.to_csv(coefficients_output_path, index=False, encoding="utf-8-sig")

print("Top model coefficients:")
print(coef_df.head(20))
print()

# ------------------------------------------------------------
# 8. Score full dataset
# ------------------------------------------------------------

full_X = df[numeric_features + categorical_features].copy()

df["predicted_default_probability"] = clf.predict_proba(full_X)[:, 1]
df["predicted_default_flag"] = clf.predict(full_X)

df["model_risk_band"] = pd.cut(
    df["predicted_default_probability"],
    bins=[0, 0.10, 0.25, 0.50, 0.75, 1.00],
    labels=["<10%", "10%-25%", "25%-50%", "50%-75%", "75%+"],
    include_lowest=True
)

df["model_priority"] = np.where(
    df["predicted_default_probability"] >= 0.75,
    "Priority Review",
    np.where(
        df["predicted_default_probability"] >= 0.50,
        "Enhanced Monitoring",
        "Routine Monitoring"
    )
)

df.to_csv(scored_output_path, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 9. Save model
# ------------------------------------------------------------

with open(model_output_path, "wb") as f:
    pickle.dump(clf, f)

print("Model scoring completed.")
print("Files saved:")
print("-", performance_output_path)
print("-", coefficients_output_path)
print("-", scored_output_path)
print("-", model_output_path)
print()

print("Predicted default probability summary:")
print(df["predicted_default_probability"].describe())
print()

print("Model risk band summary:")
print(df["model_risk_band"].value_counts().sort_index())
