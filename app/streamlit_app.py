"""
Bank Risk Controller Systems - Streamlit Dashboard
Tabs: Data | EDA - Visual | Prediction (NLP, Object Detection, GenAI added later)
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Bank Risk Controller", layout="wide")

PROC_DIR = "data/processed"
MODEL_DIR = "models"
EDA_DIR = "reports/eda"
MODEL_REPORT_DIR = "reports/model"


# ---------- Cached loaders ----------
@st.cache_data
def load_sample():
    return pd.read_csv(f"{PROC_DIR}/clean_sample.csv")

@st.cache_data
def load_metrics():
    return pd.read_csv(f"{MODEL_DIR}/metrics.csv")

@st.cache_resource
def load_model_bundle():
    return {
        "model": joblib.load(f"{MODEL_DIR}/best_model.pkl"),
        "threshold": joblib.load(f"{MODEL_DIR}/best_threshold.pkl"),
        "scaler": joblib.load(f"{MODEL_DIR}/scaler.pkl"),
        "medians": joblib.load(f"{MODEL_DIR}/medians.pkl"),
        "modes": joblib.load(f"{MODEL_DIR}/modes.pkl"),
        "numeric_features": joblib.load(f"{MODEL_DIR}/numeric_features.pkl"),
        "feature_columns": joblib.load(f"{MODEL_DIR}/feature_columns.pkl"),
    }


# ---------- Sidebar nav ----------
st.sidebar.title("Bank Risk Controller")
page = st.sidebar.radio(
    "Navigate",
    ["Data", "EDA - Visual", "Prediction"],
)


# ====================================================
# TAB 1: DATA
# ====================================================
if page == "Data":
    st.title("📊 Data")
    st.subheader("Dataset used for model building (sample)")
    df = load_sample()
    st.write(f"Showing a {len(df):,}-row sample of the cleaned, "
             f"customer-level dataset (full set: ~291k customers).")
    st.dataframe(df.head(200), use_container_width=True)

    st.subheader("Model Performance Metrics")
    metrics = load_metrics()
    st.dataframe(metrics, use_container_width=True)
    st.caption(
        "Metrics computed on the untouched test set (real 8% default rate). "
        "LightGBM is the best model. Note: this is a heavily imbalanced problem "
        "— see documentation for why these are strong, honest results."
    )


# ====================================================
# TAB 2: EDA - VISUAL
# ====================================================
elif page == "EDA - Visual":
    st.title("📈 EDA - Visual")
    df = load_sample()

    st.subheader("Target Distribution")
    target_counts = df["TARGET"].value_counts().rename(
        {0: "Repaid (0)", 1: "Default (1)"})
    fig = px.bar(x=target_counts.index, y=target_counts.values,
                 labels={"x": "Class", "y": "Count"},
                 title="Loan Default Distribution")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age vs Default")
        if "AGE_YEARS" in df.columns:
            fig = px.histogram(df, x="AGE_YEARS", color="TARGET",
                               barmode="overlay", nbins=40,
                               title="Age distribution by default status")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("External Score 3 vs Default")
        if "EXT_SOURCE_3" in df.columns:
            fig = px.histogram(df.dropna(subset=["EXT_SOURCE_3"]),
                               x="EXT_SOURCE_3", color="TARGET",
                               barmode="overlay", nbins=40,
                               title="EXT_SOURCE_3 by default status")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Default Rate by Education")
    if "NAME_EDUCATION_TYPE" in df.columns:
        rate = (df.groupby("NAME_EDUCATION_TYPE")["TARGET"]
                .mean().sort_values(ascending=False).reset_index())
        fig = px.bar(rate, x="TARGET", y="NAME_EDUCATION_TYPE",
                     orientation="h", labels={"TARGET": "Default rate"},
                     title="Default rate by education level")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Income vs Credit (sampled)")
    sample = df.sample(min(2000, len(df)), random_state=1)
    fig = px.scatter(sample, x="AMT_INCOME_TOTAL", y="AMT_CREDIT_x",
                     color="TARGET", opacity=0.5,
                     title="Income vs Credit amount")
    fig.update_xaxes(range=[0, sample["AMT_INCOME_TOTAL"].quantile(0.99)])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance (LightGBM)")
    fi_path = f"{MODEL_REPORT_DIR}/feature_importance.png"
    if os.path.exists(fi_path):
        st.image(fi_path)


# ====================================================
# TAB 3: PREDICTION
# ====================================================
elif page == "Prediction":
    st.title("🔮 Loan Default Prediction")
    st.write("Enter applicant details to predict default risk.")
    bundle = load_model_bundle()

    col1, col2, col3 = st.columns(3)
    with col1:
        ext1 = st.number_input("External Score 1 (0-1)", 0.0, 1.0, 0.5)
        ext2 = st.number_input("External Score 2 (0-1)", 0.0, 1.0, 0.5)
        ext3 = st.number_input("External Score 3 (0-1)", 0.0, 1.0, 0.5)
        income = st.number_input("Annual Income", 0.0, value=150000.0, step=10000.0)
    with col2:
        credit = st.number_input("Credit Amount", 0.0, value=500000.0, step=10000.0)
        annuity = st.number_input("Annuity", 0.0, value=25000.0, step=1000.0)
        goods = st.number_input("Goods Price", 0.0, value=450000.0, step=10000.0)
        age = st.number_input("Age (years)", 18, 100, 40)
    with col3:
        emp_years = st.number_input("Years Employed", 0.0, 50.0, 5.0)
        children = st.number_input("Number of Children", 0, 20, 0)
        fam = st.number_input("Family Members", 1, 20, 2)
        region = st.selectbox("Region Rating", [1, 2, 3], index=1)

    col4, col5 = st.columns(2)
    with col4:
        gender = st.selectbox("Gender", ["M", "F"])
        education = st.selectbox("Education", [
            "Secondary / secondary special", "Higher education",
            "Incomplete higher", "Lower secondary", "Academic degree"])
        income_type = st.selectbox("Income Type", [
            "Working", "Commercial associate", "Pensioner",
            "State servant", "Unemployed", "Student"])
    with col5:
        family_status = st.selectbox("Family Status", [
            "Married", "Single / not married", "Civil marriage",
            "Separated", "Widow"])
        contract = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
        own_car = st.selectbox("Owns Car", ["Y", "N"])
        own_realty = st.selectbox("Owns Realty", ["Y", "N"])

    if st.button("Predict", type="primary"):
        # Build a single-row dataframe matching training features
        row = {
            "EXT_SOURCE_1": ext1, "EXT_SOURCE_2": ext2, "EXT_SOURCE_3": ext3,
            "AMT_INCOME_TOTAL": income, "AMT_CREDIT_x": credit,
            "AMT_ANNUITY_x": annuity, "AMT_GOODS_PRICE_x": goods,
            "CNT_CHILDREN": children, "CNT_FAM_MEMBERS": fam,
            "REGION_RATING_CLIENT": region,
            # engineered
            "CREDIT_INCOME_RATIO": credit / (income + 1),
            "ANNUITY_INCOME_RATIO": annuity / (income + 1),
            "CREDIT_TERM": annuity / (credit + 1),
            "AGE_YEARS": age, "YEARS_EMPLOYED": emp_years,
            # categoricals
            "CODE_GENDER": gender, "NAME_EDUCATION_TYPE": education,
            "NAME_INCOME_TYPE": income_type,
            "NAME_FAMILY_STATUS": family_status,
            "NAME_CONTRACT_TYPE_x": contract,
            "FLAG_OWN_CAR": own_car, "FLAG_OWN_REALTY": own_realty,
        }
        X = pd.DataFrame([row])

        # One-hot encode, align to training columns
        cat_cols = ["CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE",
                    "NAME_FAMILY_STATUS", "NAME_CONTRACT_TYPE_x",
                    "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]
        X = pd.get_dummies(X, columns=cat_cols)
        X = X.reindex(columns=bundle["feature_columns"], fill_value=0)

        # Scale numerics
        num = bundle["numeric_features"]
        X[num] = bundle["scaler"].transform(X[num])

        proba = bundle["model"].predict_proba(X)[0, 1]
        pred = int(proba >= bundle["threshold"])

        st.divider()
        if pred == 1:
            st.error(f"⚠️ Predicted: DEFAULT  (risk probability: {proba:.1%})")
        else:
            st.success(f"✅ Predicted: NO DEFAULT  (risk probability: {proba:.1%})")
        st.caption(f"Decision threshold: {bundle['threshold']:.2f} "
                   f"(tuned for best F1 on imbalanced data)")