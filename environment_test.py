import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

print("Project 3 environment is ready.")
print("Base directory:", BASE_DIR)
print("Data directory:", DATA_DIR)
print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)

input_path = DATA_DIR / "credit_risk_dataset.csv"

if input_path.exists():
    df = pd.read_csv(input_path)
    print("credit_risk_dataset.csv loaded successfully.")
    print("Number of rows:", len(df))
    print("Number of columns:", len(df.columns))
    print("Columns:", list(df.columns))
    print()
    print("Preview:")
    print(df.head())
    print()
    print("Loan status distribution:")
    print(df["loan_status"].value_counts())
else:
    print("credit_risk_dataset.csv not found. Please place the dataset in the data folder.")