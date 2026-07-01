"""
notebooks/02_clean.py
ขั้นที่ 2: ทำความสะอาดข้อมูล (Data Cleaning)

จุดประสงค์: แก้ปัญหาที่เจอในขั้นที่ 1 และเตรียมข้อมูลให้พร้อมวิเคราะห์
ใช้ฟังก์ชันกลางจาก src/preprocessing.py (กันโค้ดซ้ำ)
รันจากโฟลเดอร์รากของโปรเจกต์:  python notebooks/02_clean.py
"""

import sys
from pathlib import Path

# ให้ import src/preprocessing ได้
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing import load_raw, clean_data, encode_features, DEFAULT_DATA_PATH


def main():
    # 1. โหลดข้อมูลดิบ
    df = load_raw(DEFAULT_DATA_PATH)
    print(f"ข้อมูลเริ่มต้น: {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")

    # 2. ทำความสะอาด (แก้ TotalCharges, ลบแถวว่าง, แปลง Churn เป็น 0/1)
    df_clean = clean_data(df)
    print(f"หลังทำความสะอาด: {df_clean.shape[0]} แถว, {df_clean.shape[1]} คอลัมน์")
    print(f"ลบแถวว่างไป: {df.shape[0] - df_clean.shape[0]} แถว")

    # 3. ตรวจว่าไม่มีค่าว่างเหลือแล้ว
    print(f"\nค่าว่างที่เหลือทั้งหมด: {df_clean.isnull().sum().sum()}")

    # 4. แปลงข้อความเป็นตัวเลขด้วย One-Hot Encoding
    df_encoded = encode_features(df_clean)
    print(f"หลัง One-Hot Encoding: {df_encoded.shape[1]} คอลัมน์")

    print("\nตัวอย่างข้อมูลหลังทำความสะอาดและแปลงเป็นตัวเลข:")
    print(df_encoded.head())

    # 5. บันทึกไว้ตรวจสอบ (ไม่บังคับ — pipeline จริงใช้ preprocessing โดยตรง)
    out_path = "data/raw/data_clean.csv"
    df_encoded.to_csv(out_path, index=False)
    print(f"\nบันทึกไฟล์ตรวจสอบที่ {out_path}")


if __name__ == "__main__":
    main()