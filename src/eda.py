"""
EDA for Bank Risk Controller Systems (Home Credit data).
Memory-safe: works on a stratified sample of the 1.2GB CSV.
Saves plots and summary tables to reports/eda/.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RAW_PATH = "data/raw/loan_data.csv"
OUT_DIR = "reports/eda"
os.makedirs(OUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

SAMPLE_ROWS = 150_000

# Columns that actually carry signal — what we'll visualize
KEY_NUMERIC = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "AMT_INCOME_TOTAL", "AMT_CREDIT_x", "AMT_ANNUITY_x", "AMT_GOODS_PRICE_x",
]
KEY_CATEGORICAL = [
    "CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE",
    "NAME_FAMILY_STATUS", "NAME_CONTRACT_TYPE_x",
]


def load_sample():
    """Stratified sample by TARGET, read in chunks to limit memory."""
    target = pd.read_csv(RAW_PATH, usecols=["TARGET"])
    n_total = len(target)
    frac = min(SAMPLE_ROWS / n_total, 1.0)
    sample_idx = (
        target.groupby("TARGET", group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=42))
        .index
    )
    sample_set = set(sample_idx)

    chunks, start = [], 0
    for chunk in pd.read_csv(RAW_PATH, chunksize=200_000):
        idx = range(start, start + len(chunk))
        chunk.index = idx
        chunks.append(chunk[chunk.index.isin(sample_set)])
        start += len(chunk)
    df = pd.concat(chunks)
    print(f"Loaded sample: {df.shape}")
    return df


def basic_overview(df):
    print("\n=== SHAPE ===", df.shape)
    print("=== UNIQUE CUSTOMERS ===", df["SK_ID_CURR"].nunique())
    print("=== ROWS PER CUSTOMER (avg) ===",
          round(len(df) / df["SK_ID_CURR"].nunique(), 2))
    print("\n=== TARGET BALANCE ===")
    print(df["TARGET"].value_counts(normalize=True).round(4))

    miss = (df.isnull().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0].round(2)
    miss.to_csv(f"{OUT_DIR}/missing_values.csv", header=["pct_missing"])
    print(f"\nColumns with missing values: {len(miss)} "
          f"(top saved to missing_values.csv)")
    print(miss.head(10))


def plot_target(df):
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x="TARGET", data=df)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():,}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    plt.title("Target Distribution (0=Repaid, 1=Default)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/target_distribution.png", dpi=120)
    plt.close()


def plot_numeric(df):
    for col in KEY_NUMERIC:
        if col not in df.columns:
            continue
        data = df[[col, "TARGET"]].dropna()
        # clip extreme outliers for readable plots (income especially)
        upper = data[col].quantile(0.99)
        data = data[data[col] <= upper]
        plt.figure(figsize=(7, 4))
        sns.kdeplot(data=data, x=col, hue="TARGET",
                    common_norm=False, fill=True, alpha=0.4)
        plt.title(f"{col} by TARGET")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/dist_{col}.png", dpi=120)
        plt.close()


def plot_categorical(df):
    for col in KEY_CATEGORICAL:
        if col not in df.columns:
            continue
        # default rate per category
        rates = df.groupby(col)["TARGET"].mean().sort_values(ascending=False)
        plt.figure(figsize=(8, 4))
        sns.barplot(x=rates.values, y=rates.index)
        plt.xlabel("Default rate")
        plt.title(f"Default rate by {col}")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/defaultrate_{col}.png", dpi=120)
        plt.close()


def plot_age(df):
    """DAYS_BIRTH is negative days; convert to years."""
    if "DAYS_BIRTH" not in df.columns:
        return
    tmp = df[["DAYS_BIRTH", "TARGET"]].copy()
    tmp["AGE_YEARS"] = (-tmp["DAYS_BIRTH"] / 365).round(1)
    plt.figure(figsize=(7, 4))
    sns.kdeplot(data=tmp, x="AGE_YEARS", hue="TARGET",
                common_norm=False, fill=True, alpha=0.4)
    plt.title("Age (years) by TARGET")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/dist_AGE.png", dpi=120)
    plt.close()


def correlation_with_target(df):
    num = df.select_dtypes(include=[np.number])
    corr = num.corr()["TARGET"].drop("TARGET").sort_values(key=abs, ascending=False)
    corr.head(30).round(4).to_csv(f"{OUT_DIR}/top_corr_with_target.csv",
                                  header=["correlation"])
    print("\n=== TOP 15 FEATURES CORRELATED WITH TARGET ===")
    print(corr.head(15).round(4))


if __name__ == "__main__":
    df = load_sample()
    basic_overview(df)
    plot_target(df)
    plot_numeric(df)
    plot_categorical(df)
    plot_age(df)
    correlation_with_target(df)
    print("\nEDA artifacts saved to", OUT_DIR)