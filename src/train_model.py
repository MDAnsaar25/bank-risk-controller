"""
Step 4: Train and evaluate models for Bank Risk Controller Systems.
Honest approach: class weights for imbalance, evaluation on the real
(untouched) test distribution, threshold tuning for F1.
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve,
)

PROC_DIR = "data/processed"
MODEL_DIR = "models"
REPORT_DIR = "reports/model"
os.makedirs(REPORT_DIR, exist_ok=True)


def load():
    X_train = pd.read_parquet(f"{PROC_DIR}/X_train.parquet")
    X_test = pd.read_parquet(f"{PROC_DIR}/X_test.parquet")
    y_train = pd.read_parquet(f"{PROC_DIR}/y_train.parquet")["TARGET"]
    y_test = pd.read_parquet(f"{PROC_DIR}/y_test.parquet")["TARGET"]
    return X_train, X_test, y_train, y_test


def metrics_row(name, y_true, y_pred, y_proba):
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred), 4),
        "ROC_AUC": round(roc_auc_score(y_true, y_proba), 4),
    }


def main():
    X_train, X_test, y_train, y_test = load()
    results = []
    models = {}

    # ---- 1. Logistic Regression (baseline) ----
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    p = lr.predict_proba(X_test)[:, 1]
    results.append(metrics_row("LogisticRegression", y_test,
                               (p >= 0.5).astype(int), p))
    models["LogisticRegression"] = lr

    # ---- 2. Random Forest ----
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, n_jobs=-1,
        class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    p = rf.predict_proba(X_test)[:, 1]
    results.append(metrics_row("RandomForest", y_test,
                               (p >= 0.5).astype(int), p))
    models["RandomForest"] = rf

    # ---- 3. LightGBM ----
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    lgbm = LGBMClassifier(
        n_estimators=500, learning_rate=0.02, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, random_state=42, n_jobs=-1)
    lgbm.fit(X_train, y_train)
    p_lgbm = lgbm.predict_proba(X_test)[:, 1]
    results.append(metrics_row("LightGBM", y_test,
                               (p_lgbm >= 0.5).astype(int), p_lgbm))
    models["LightGBM"] = lgbm

    # ---- Results table (at 0.5 threshold) ----
    df_res = pd.DataFrame(results)
    print("\n=== METRICS @ threshold 0.5 (real test distribution) ===")
    print(df_res.to_string(index=False))

    # ---- Pick best by ROC-AUC ----
    best_name = df_res.sort_values("ROC_AUC", ascending=False).iloc[0]["Model"]
    best_model = models[best_name]
    best_proba = best_model.predict_proba(X_test)[:, 1]
    print(f"\nBest model by ROC-AUC: {best_name}")

    # ---- Threshold tuning for best F1 ----
    prec, rec, thr = precision_recall_curve(y_test, best_proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1s)
    best_thr = thr[best_idx] if best_idx < len(thr) else 0.5
    print(f"Best F1 threshold: {best_thr:.3f}")

    y_pred_tuned = (best_proba >= best_thr).astype(int)
    tuned = metrics_row(f"{best_name} (tuned thr={best_thr:.2f})",
                        y_test, y_pred_tuned, best_proba)
    df_res = pd.concat([df_res, pd.DataFrame([tuned])], ignore_index=True)

    print("\n=== Best model, tuned threshold ===")
    print(pd.DataFrame([tuned]).to_string(index=False))
    print("\nConfusion matrix (tuned):")
    print(confusion_matrix(y_test, y_pred_tuned))
    print("\nClassification report (tuned):")
    print(classification_report(y_test, y_pred_tuned, digits=4))

    # ---- Save metrics dataset (for Streamlit Data sidebar) ----
    df_res.to_csv(f"{MODEL_DIR}/metrics.csv", index=False)

    # ---- ROC curve plot ----
    fpr, tpr, _ = roc_curve(y_test, best_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{best_name} (AUC={tuned['ROC_AUC']})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/roc_curve.png", dpi=120)
    plt.close()

    # ---- Feature importance (if available) ----
    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_,
                       index=X_train.columns).sort_values(ascending=False)
        fi.head(20).to_csv(f"{REPORT_DIR}/feature_importance.csv")
        plt.figure(figsize=(8, 6))
        fi.head(20).iloc[::-1].plot.barh()
        plt.title(f"Top 20 Feature Importances ({best_name})")
        plt.tight_layout()
        plt.savefig(f"{REPORT_DIR}/feature_importance.png", dpi=120)
        plt.close()

    # ---- Save best model + threshold ----
    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(float(best_thr), f"{MODEL_DIR}/best_threshold.pkl")
    print(f"\nSaved best model ({best_name}) and threshold to {MODEL_DIR}/")


if __name__ == "__main__":
    main()