# Bank Risk Controller Systems

A machine learning project that predicts whether a bank customer will **default on a loan** (`TARGET`), served through an interactive **Streamlit** dashboard. Alongside the core default-prediction model, the application includes EDA, NLP, object detection, and a GenAI chatbot.

> **Domain:** Banking &nbsp;|&nbsp; **Skills:** Python, Analytics, Statistics, Plotting, Streamlit, Machine Learning, Deep Learning, GenAI

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Reproducing the Results](#reproducing-the-results)
- [Model & Results](#model--results)
- [Tech Stack](#tech-stack)
- [Notes & Design Decisions](#notes--design-decisions)

---

## Overview

The goal is to assess the credit-default risk of loan applicants to support risk management, credit scoring, and loan-approval decisions. The dataset is the Home Credit default-risk data (`application` merged with `previous_application`), containing ~1.4M rows that resolve to ~291K unique customers, with a heavily imbalanced target (~8.7% defaulters).

The project follows a full ML workflow: data preprocessing, EDA, feature engineering, model selection and tuning, evaluation, and deployment via a multi-page Streamlit app.

---

## Features

The Streamlit app is organised into sidebar menus:

| # | Sidebar | Description |
|---|---------|-------------|
| 1 | **Data** | Displays the cleaned dataset sample and the model performance-metrics table. |
| 2 | **EDA – Visual** | Interactive Plotly charts: target distribution, age vs default, external-score distributions, default rate by education, income vs credit, and feature importance. |
| 3 | **Prediction** | Input form for an applicant's details; predicts default / no-default using the trained LightGBM model. |
| 4 | **NLP** | Text preprocessing, sentiment analysis with a plot, and next-word prediction using an LSTM. |
| 5 | **Object Detection** | Detects and counts **only humans** in an uploaded image using a pretrained YOLOv8 model. |
| 6 | **Chat – GenAI** | A RAG chatbot that answers bank-customer queries grounded in bank PDF documents (Groq LLM + FAISS). |

*(Sidebar 7, Recommendation, is optional and not implemented.)*

---

## Project Structure

```
bank-risk-controller/
├── data/
│   ├── raw/                  # Loan_data.csv, Data_dictionary.csv (not in Git)
│   ├── processed/            # cleaned/encoded train-test parquet files (not in Git)
│   └── bank_docs/            # bank PDFs for the chatbot (not in Git)
├── models/                   # trained model + transformers (not in Git)
│   ├── best_model.pkl
│   ├── best_threshold.pkl
│   ├── scaler.pkl
│   ├── lstm_nextword.h5
│   ├── faiss_index/
│   └── metrics.csv
├── reports/
│   ├── eda/                  # saved EDA plots & summaries
│   └── model/                # ROC curve, feature importance
├── src/
│   ├── eda.py                # exploratory data analysis
│   ├── preprocessing.py      # cleaning, feature engineering, train/test split
│   ├── train_model.py        # model training, evaluation, threshold tuning
│   ├── train_lstm.py         # trains the next-word LSTM
│   ├── nlp_module.py         # NLP helpers (preprocess, sentiment, predict)
│   ├── object_detection.py   # YOLOv8 human detection
│   └── chatbot.py            # RAG chatbot logic
├── app/
│   └── streamlit_app.py      # main Streamlit dashboard (all tabs)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

The dataset is **not stored in this repository** due to its size (~1.2 GB). Download it from the project's shared Drive folder and place the files as follows:

| File | Location |
|------|----------|
| `loan_data.csv` | `data/raw/loan_data.csv` |
| `data_dictionary.csv` | `data/raw/data_dictionary.csv` |

The trained model files and FAISS index are also excluded from Git. Either download them from the shared Drive folder into `models/`, or regenerate them by following [Reproducing the Results](#reproducing-the-results).

---

## Installation

**Requirements:** Python **3.10 or 3.11** (TensorFlow and other libraries do not yet support 3.13).

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/bank-risk-controller.git
cd bank-risk-controller

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

If you only need the core ML pipeline (not the deep-learning/GenAI tabs), the heavy packages (`tensorflow`, `ultralytics`, `langchain`, etc.) can be installed later.

---

## How to Run

From the **project root**:

```bash
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501`.

**Tab-specific notes:**
- **NLP tab:** run `python src/train_lstm.py` once beforehand to create the LSTM model.
- **Object Detection tab:** the YOLOv8 weights download automatically on first use (~6 MB, needs internet once).
- **Chat – GenAI tab:** requires a free [Groq API key](https://console.groq.com). Paste it into the app, add at least one PDF to `data/bank_docs/`, then click **Build / Rebuild Document Index**.

---

## Reproducing the Results

Run the pipeline scripts in order from the project root:

```bash
# 1. Exploratory data analysis (saves plots to reports/eda/)
python src/eda.py

# 2. Preprocessing + feature engineering (saves processed data + transformers)
python src/preprocessing.py

# 3. Train & evaluate models (saves best_model.pkl + metrics.csv)
python src/train_model.py

# 4. Train the NLP LSTM (saves lstm_nextword.h5)
python src/train_lstm.py
```

Then launch the Streamlit app as shown above.

---

## Model & Results

Three models were compared on the **untouched test set** (preserving the real ~8% default rate): Logistic Regression, Random Forest, and LightGBM. Class imbalance was handled with **class weights** (not resampling), and the decision threshold was tuned for best F1.

**Best model: LightGBM**

| Metric | Score |
|--------|-------|
| ROC-AUC | ~0.76 |
| Accuracy | ~0.85 (tuned threshold) |
| Precision (default class) | ~0.26 |
| Recall (default class) | ~0.42 |
| F1 (default class) | ~0.32 |

The strongest predictors are the external credit scores (`EXT_SOURCE_1/2/3`) followed by the engineered credit-to-income and annuity-to-income ratios.

> **On the metrics:** This is a heavily imbalanced problem (~8.7% defaulters). For reference, the best-performing solutions on this well-known dataset reach roughly **0.80 ROC-AUC**. The results here are produced with an honest, leakage-free methodology — split at the customer level, with metrics measured on the true class distribution rather than on a resampled set. See the documentation report for a full discussion.

---

## Tech Stack

- **Data & ML:** pandas, NumPy, scikit-learn, LightGBM, imbalanced-learn
- **Visualisation:** Plotly, seaborn, matplotlib
- **App:** Streamlit
- **NLP / Deep Learning:** NLTK, TextBlob, TensorFlow/Keras (LSTM)
- **Object Detection:** Ultralytics YOLOv8
- **GenAI (RAG):** Groq API, LangChain, FAISS, sentence-transformers, pypdf

---

## Notes & Design Decisions

- **Lean, transparent feature set.** The prediction model uses ~22 interpretable application-level features (plus engineered ratios) that map one-to-one to the prediction form — no hidden defaults — so every input is explainable.
- **Customer-level split.** Because customers appear in multiple rows, the data is deduplicated to one row per customer and split on `SK_ID_CURR` to prevent leakage between train and test.
- **Honest evaluation.** No SMOTE-on-test inflation; metrics reflect the real default rate.
- **Large files excluded from Git.** The raw dataset, processed data, model binaries, FAISS index, and bank PDFs are gitignored and shared separately via Drive.

---

*This project was developed as a final/capstone project. The bank policy document used for the chatbot demo is fictional ("Meridian Bank") and created for demonstration purposes only.*