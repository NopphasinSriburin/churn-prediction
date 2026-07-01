"""
src/preprocessing.py
ฟังก์ชันกลางสำหรับเตรียมข้อมูล ใช้ร่วมกันทั้งตอนเทรนและตอนทำนาย
รวมไว้ที่เดียว (single source of truth) เพื่อกันโค้ดซ้ำและกันข้อมูลไม่ตรงกัน
"""

import pandas as pd

# path เริ่มต้นของไฟล์ข้อมูลดิบ
DEFAULT_DATA_PATH = "data/raw/data.csv"


def load_raw(path=DEFAULT_DATA_PATH):
    """โหลดไฟล์ CSV ดิบเข้ามาเป็น DataFrame"""
    return pd.read_csv(path)


def clean_data(df):
    """
    ทำความสะอาดข้อมูล:
      - ลบคอลัมน์ customerID (ไม่ช่วยทำนาย)
      - แก้ TotalCharges ที่มีช่องว่างปน ให้เป็นตัวเลข
      - ลบแถวที่มีค่าว่าง
      - แปลง Churn (Yes/No) เป็น 1/0
    คืน DataFrame ที่สะอาดแล้ว (ยังไม่ได้ encode)
    """
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # TotalCharges มีบางแถวเป็นช่องว่าง ทำให้ pandas อ่านเป็นข้อความ
    # errors="coerce" = แปลงไม่ได้ให้เป็น NaN แล้วค่อยลบทิ้ง
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    # แปลงเป้าหมายเป็นตัวเลข (ถ้ายังไม่ใช่ตัวเลข)
    # เช็คด้วย is_numeric_dtype เพื่อให้ทนทานทุกเวอร์ชัน pandas
    # (pandas ใหม่อ่านข้อความเป็น str/StringArray ไม่ใช่ object แบบเดิม)
    if not pd.api.types.is_numeric_dtype(df["Churn"]):
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def encode_features(df):
    """
    แปลงคอลัมน์ข้อความเป็นตัวเลขด้วย One-Hot Encoding
    drop_first=True เพื่อเลี่ยงคอลัมน์ซ้ำซ้อน (multicollinearity)
    """
    return pd.get_dummies(df, drop_first=True)


def prepare_for_training(path=DEFAULT_DATA_PATH):
    """
    รวมทุกขั้นตอน: โหลด -> clean -> encode
    คืน DataFrame ที่พร้อมแยก X, y เพื่อเทรนโมเดล
    """
    df = load_raw(path)
    df = clean_data(df)
    df = encode_features(df)
    return df


def align_to_model(df_encoded, model_columns):
    """
    ปรับ DataFrame ที่ encode แล้ว ให้มีคอลัมน์ตรงและเรียงเหมือนตอนเทรนเป๊ะ ๆ
    ใช้ตอนทำนายข้อมูลใหม่ (เช่น โหมด Batch) ที่อาจมีค่าบางประเภทไม่ครบ
    reindex จะเติมคอลัมน์ที่ขาดด้วย 0 และตัดคอลัมน์ที่เกินออก
    """
    return df_encoded.reindex(columns=model_columns, fill_value=0)