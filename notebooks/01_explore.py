"""
notebooks/01_explore.py
ขั้นที่ 1: สำรวจข้อมูลเบื้องต้น (Data Exploration)

จุดประสงค์: ทำความรู้จักข้อมูลก่อนลงมือทำความสะอาด
รันจากโฟลเดอร์รากของโปรเจกต์:  python notebooks/01_explore.py
"""

import pandas as pd

DATA_PATH = "data/raw/data.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("5 แถวแรกของข้อมูล")
    print("=" * 60)
    print(df.head())

    print("\n" + "=" * 60)
    print("ขนาดข้อมูล (แถว, คอลัมน์)")
    print("=" * 60)
    print(df.shape)

    print("\n" + "=" * 60)
    print("ชนิดข้อมูลแต่ละคอลัมน์ + ค่าที่หายไป")
    print("=" * 60)
    print(df.info())

    print("\n" + "=" * 60)
    print("สถิติเบื้องต้นของคอลัมน์ตัวเลข")
    print("=" * 60)
    print(df.describe())

    print("\n" + "=" * 60)
    print("สัดส่วนลูกค้าที่เลิกใช้ (Churn)")
    print("=" * 60)
    print(df["Churn"].value_counts())
    print(df["Churn"].value_counts(normalize=True).round(3))

    # จุดสังเกตสำคัญ: TotalCharges ควรเป็นตัวเลข แต่ pandas อ่านเป็น object
    # เพราะมีบางแถวเป็นช่องว่าง " " ปนอยู่ (จะแก้ในขั้นที่ 2)
    print("\n" + "=" * 60)
    print("ตรวจ TotalCharges (ควรเป็นตัวเลข แต่อาจเป็น object)")
    print("=" * 60)
    print(f"ชนิดข้อมูลปัจจุบัน: {df['TotalCharges'].dtype}")
    blank = (df["TotalCharges"].astype(str).str.strip() == "").sum()
    print(f"จำนวนแถวที่เป็นช่องว่าง: {blank}")


if __name__ == "__main__":
    main()