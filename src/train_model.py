"""
src/train_model.py
เทรนและเปรียบเทียบ 3 โมเดล แล้วเลือกตัวที่ดีที่สุดอัตโนมัติ

โมเดลที่เทียบ:
  1. Logistic Regression  (baseline เข้าใจง่าย ตีความได้)
  2. Random Forest         (ต้นไม้หลายต้นโหวตกัน)
  3. XGBoost               (gradient boosting มักแม่นสุดกับข้อมูลตาราง)

วัดผลด้วย: Accuracy, Precision, Recall, F1, ROC-AUC
เลือกโมเดลที่ ROC-AUC สูงสุด (เหมาะกับข้อมูล imbalanced มากกว่า Accuracy)

รันจากโฟลเดอร์รากของโปรเจกต์:  python src/train_model.py
"""

import sys
from pathlib import Path

# เพิ่ม src เข้า path เพื่อ import preprocessing ได้ ไม่ว่าจะรันจากที่ไหน
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

from preprocessing import prepare_for_training, DEFAULT_DATA_PATH


def train_and_compare(df):
    """เทรนทั้ง 3 โมเดล คืนตารางผล + โมเดลที่เทรนแล้ว + ข้อมูลทดสอบ"""
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Logistic Regression ต้อง scale ข้อมูลก่อน (โมเดลอื่นไม่ต้อง)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=42,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        ),
    }

    results = []
    trained = {}

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        })
        trained[name] = model

    results_df = pd.DataFrame(results).set_index("Model")
    return results_df, trained, X, scaler, (X_test, y_test, X_test_scaled)


def main():
    print("[1/4] loading and cleaning data...")
    df = prepare_for_training(DEFAULT_DATA_PATH)
    print(f"      ready: {df.shape[0]} rows, {df.shape[1]} columns")

    print("[2/4] training 3 models...")
    results_df, trained, X, scaler, test_data = train_and_compare(df)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results_df.round(4).to_string())

    # เลือกโมเดลที่ ROC-AUC สูงสุด
    best_name = results_df["ROC-AUC"].idxmax()
    best_model = trained[best_name]
    print(f"\n[3/4] best model: {best_name} "
          f"(ROC-AUC = {results_df.loc[best_name, 'ROC-AUC']:.4f})")

    # รายงานละเอียดของโมเดลที่ชนะ
    X_test, y_test, X_test_scaled = test_data
    X_eval = X_test_scaled if best_name == "Logistic Regression" else X_test
    y_pred = best_model.predict(X_eval)

    print("\nDetailed report (winning model):")
    print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # บันทึกไฟล์ทั้งหมดที่ dashboard ต้องใช้
    print("\n[4/4] saving artifacts to models/ ...")
    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_model, "models/churn_model.pkl")
    joblib.dump(list(X.columns), "models/model_columns.pkl")

    # เซฟ scaler เฉพาะเมื่อโมเดลที่ชนะเป็น Logistic Regression
    scaler_path = Path("models/scaler.pkl")
    if best_name == "Logistic Regression":
        joblib.dump(scaler, scaler_path)
    elif scaler_path.exists():
        # ถ้ารอบก่อน LR ชนะแล้วรอบนี้ไม่ใช่ ให้ลบ scaler เก่าทิ้งกันสับสน
        scaler_path.unlink()

    results_df.round(4).to_csv("models/model_comparison.csv")
    print("      done. saved churn_model.pkl, model_columns.pkl, model_comparison.csv")


if __name__ == "__main__":
    main()