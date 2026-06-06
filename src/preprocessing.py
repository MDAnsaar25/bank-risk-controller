"""
Step 3: Preprocessing for Bank Risk Controller Systems.
Lean, transparent feature set. Produces a clean customer-level dataset.
Splits on SK_ID_CURR (no leakage), imputes + encodes on train only.
Saves processed data and fitted transformers to disk.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_PATH = "data/raw/loan_data.csv"
PROC_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---- Lean feature set ----
ID_COL = "SK_ID_CURR"
TARGET = "TARGET"

NUMERIC_RAW = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "AMT_INCOME_TOTAL", "AMT_CREDIT_x", "AMT_ANNUITY_x", "AMT_GOODS_PRICE_x",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
]
CATEGORICAL = [
    "CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE",
    "NAME_FAMILY_STATUS", "NAME_CONTRACT_TYPE_x",
    "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
]

USE_COLS = [ID_COL, TARGET] + NUMERIC_RAW + CATEGORICAL


def load_data():
    """Read only the lean columns, then deduplicate to one row per customer."""
    print("Loading lean columns from full CSV (this reads the file once)...")
    df = pd.read_csv(RAW_PATH, usecols=USE_COLS)
    print("Raw rows:", len(df))
    # One row per customer — keep first application row
    df = df.drop_duplicates(subset=ID_COL, keep="first").reset_index(drop=True)
    print("After dedup to one row per customer:", len(df))
    return df


def clean(df):
    """Fix known data issues."""
    # DAYS_EMPLOYED sentinel for pensioners/unemployed -> NaN
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    # CODE_GENDER has a few 'XNA' values -> treat as missing then most-frequent
    df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)
    return df


def engineer(df):
    """Domain ratio features known to help on this dataset."""
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT_x"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY_x"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["CREDIT_TERM"] = df["AMT_ANNUITY_x"] / (df["AMT_CREDIT_x"] + 1)
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).round(1)
    df["YEARS_EMPLOYED"] = (-df["DAYS_EMPLOYED"] / 365).round(1)
    return df


ENGINEERED_NUMERIC = [
    "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM",
    "AGE_YEARS", "YEARS_EMPLOYED",
]


def build():
    df = load_data()
    df = clean(df)
    df = engineer(df)

    # Final feature lists
    numeric_features = [
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "AMT_INCOME_TOTAL", "AMT_CREDIT_x", "AMT_ANNUITY_x", "AMT_GOODS_PRICE_x",
        "CNT_CHILDREN", "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT",
    ] + ENGINEERED_NUMERIC
    cat_features = CATEGORICAL

    X = df[numeric_features + cat_features].copy()
    y = df[TARGET].copy()

    # ---- Split on customers (rows are already 1-per-customer) ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train default rate: {y_train.mean():.4f}")

    # ---- Impute numerics (median, fit on train) ----
    medians = X_train[numeric_features].median()
    X_train[numeric_features] = X_train[numeric_features].fillna(medians)
    X_test[numeric_features] = X_test[numeric_features].fillna(medians)

    # ---- Impute categoricals (mode, fit on train) ----
    modes = X_train[cat_features].mode().iloc[0]
    X_train[cat_features] = X_train[cat_features].fillna(modes)
    X_test[cat_features] = X_test[cat_features].fillna(modes)

    # ---- One-hot encode categoricals ----
    X_train = pd.get_dummies(X_train, columns=cat_features)
    X_test = pd.get_dummies(X_test, columns=cat_features)
    # align columns (test may miss some dummy categories)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
    feature_columns = list(X_train.columns)

    # ---- Scale numerics (fit on train) ----
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])

    # ---- Save everything ----
    X_train.to_parquet(f"{PROC_DIR}/X_train.parquet")
    X_test.to_parquet(f"{PROC_DIR}/X_test.parquet")
    y_train.to_frame().to_parquet(f"{PROC_DIR}/y_train.parquet")
    y_test.to_frame().to_parquet(f"{PROC_DIR}/y_test.parquet")

    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(medians, f"{MODEL_DIR}/medians.pkl")
    joblib.dump(modes, f"{MODEL_DIR}/modes.pkl")
    joblib.dump(numeric_features, f"{MODEL_DIR}/numeric_features.pkl")
    joblib.dump(feature_columns, f"{MODEL_DIR}/feature_columns.pkl")

    # also save a small clean sample for the Streamlit "Data" sidebar
    df.sample(min(5000, len(df)), random_state=42).to_csv(
        f"{PROC_DIR}/clean_sample.csv", index=False)

    print("\nSaved processed data and transformers.")
    print("Final feature count:", len(feature_columns))


if __name__ == "__main__":
    build()