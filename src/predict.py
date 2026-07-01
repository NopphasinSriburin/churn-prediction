"""
src/predict.py
ฟังก์ชันโหลดโมเดลและทำนาย ใช้โดย dashboard (app/streamlit_app.py)
แยกออกมาเพื่อให้ logic การทำนายอยู่ที่เดียว ทดสอบง่าย และ import ไปใช้ที่ไหนก็ได้
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import joblib

from preprocessing import align_to_model

MODELS_DIR = Path("models")


def load_artifacts():
    """
    โหลดโมเดล + รายชื่อคอลัมน์ + scaler (ถ้ามี)
    scaler จะมีเฉพาะกรณีโมเดลที่ชนะเป็น Logistic Regression
    คืน (model, columns, scaler) โดย scaler อาจเป็น None
    """
    model = joblib.load(MODELS_DIR / "churn_model.pkl")
    columns = joblib.load(MODELS_DIR / "model_columns.pkl")
    scaler_path = MODELS_DIR / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return model, columns, scaler


def _apply_scaler(X, scaler):
    """ถ้ามี scaler ให้ transform ก่อนทำนาย (สำหรับ Logistic Regression)"""
    return scaler.transform(X) if scaler is not None else X


def build_single_row(data: dict, model_columns):
    """
    สร้าง DataFrame 1 แถวจาก dict ที่ผู้ใช้กรอกในฟอร์ม
    เริ่มจากตารางเปล่า (ทุกช่อง 0) ที่มีคอลัมน์ตรงกับตอนเทรน แล้วเติมค่า
    วิธีนี้การันตีว่าโครงสร้างตรงกับตอนเทรนเสมอ
    """
    row = pd.DataFrame(0, index=[0], columns=model_columns)

    # ค่าตัวเลขตรง ๆ
    row["tenure"] = data["tenure"]
    row["MonthlyCharges"] = data["monthly"]
    row["TotalCharges"] = data["total"]
    row["SeniorCitizen"] = data["senior"]

    # ค่าที่ถูกแตกด้วย One-Hot Encoding (เติม 1 เฉพาะช่องที่ตรงและมีอยู่จริง)
    def set_if_exists(col):
        if col in row.columns:
            row[col] = 1

    mapping = {
        ("contract", "One year"): "Contract_One year",
        ("contract", "Two year"): "Contract_Two year",
        ("internet", "Fiber optic"): "InternetService_Fiber optic",
        ("internet", "No"): "InternetService_No",
        ("payment", "Electronic check"): "PaymentMethod_Electronic check",
        ("payment", "Mailed check"): "PaymentMethod_Mailed check",
        ("payment", "Credit card (automatic)"): "PaymentMethod_Credit card (automatic)",
    }
    for (key, value), col in mapping.items():
        if data[key] == value:
            set_if_exists(col)

    if data["paperless"] == 1:
        set_if_exists("PaperlessBilling_Yes")

    return row


def predict_single(data: dict, model, model_columns, scaler=None):
    """ทำนายลูกค้า 1 คน คืนความน่าจะเป็นที่จะเลิกใช้ (0.0 - 1.0)"""
    row = build_single_row(data, model_columns)
    X = _apply_scaler(row, scaler)
    return float(model.predict_proba(X)[0][1])


def predict_batch(df_raw, model, model_columns, scaler=None):
    """
    ทำนายทั้งไฟล์ (โหมด Batch)
    รับ DataFrame ดิบ (แบบเดียวกับ data.csv) คืน numpy array ของความน่าจะเป็น
    """
    proc = df_raw.copy()
    for col in ["customerID", "Churn"]:
        if col in proc.columns:
            proc = proc.drop(col, axis=1)

    proc["TotalCharges"] = pd.to_numeric(proc["TotalCharges"], errors="coerce")
    proc = proc.fillna(0)

    encoded = pd.get_dummies(proc, drop_first=True)
    encoded = align_to_model(encoded, model_columns)

    X = _apply_scaler(encoded, scaler)
    return model.predict_proba(X)[:, 1]


def risk_level(prob: float) -> str:
    """แบ่งระดับความเสี่ยงเป็น 3 ระดับจากความน่าจะเป็น"""
    if prob >= 0.7:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    return "Low"