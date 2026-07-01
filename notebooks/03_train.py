"""
notebooks/03_modeling.py
ขั้นที่ 3: วิเคราะห์เชิงสำรวจ (EDA) + ทดลองโมเดล

จุดประสงค์: หา insight ทางธุรกิจด้วยกราฟ และดูผลโมเดลเบื้องต้น
กราฟจะถูกบันทึกไว้ในโฟลเดอร์ images/ เพื่อใช้ประกอบ README
รันจากโฟลเดอร์รากของโปรเจกต์:  python notebooks/03_modeling.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ไม่เปิดหน้าต่าง ใช้บันทึกไฟล์อย่างเดียว
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_raw, clean_data, DEFAULT_DATA_PATH

sns.set_style("whitegrid")
INDIGO = "#4F46E5"
RED = "#EF4444"
IMG_DIR = Path("images")


def main():
    IMG_DIR.mkdir(exist_ok=True)

    # โหลดข้อมูลดิบ (ยังไม่ encode เพราะ EDA ต้องการชื่อกลุ่มที่อ่านออก)
    raw = load_raw(DEFAULT_DATA_PATH)
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    raw = raw.dropna()

    # ---- คำถามธุรกิจที่ 1: สัญญาแบบไหน churn สูงสุด? ----
    print("Churn rate ตามรูปแบบสัญญา:")
    contract_churn = raw.groupby("Contract")["Churn"].apply(
        lambda x: (x == "Yes").mean())
    print(contract_churn.round(3), "\n")

    plt.figure(figsize=(7, 4))
    (contract_churn * 100).plot(kind="bar", color=INDIGO)
    plt.title("Churn Rate by Contract Type")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "churn_by_contract.png", dpi=120)
    plt.close()

    # ---- คำถามธุรกิจที่ 2: ลูกค้าใหม่ (tenure ต่ำ) เสี่ยงกว่าไหม? ----
    raw["tenure_group"] = pd.cut(
        raw["tenure"], bins=[0, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"])
    tenure_churn = raw.groupby("tenure_group", observed=True)["Churn"].apply(
        lambda x: (x == "Yes").mean())
    print("Churn rate ตามระยะเวลาใช้บริการ:")
    print(tenure_churn.round(3), "\n")

    plt.figure(figsize=(7, 4))
    (tenure_churn * 100).plot(kind="line", marker="o", color=INDIGO, linewidth=2)
    plt.title("Churn Rate by Tenure Group")
    plt.ylabel("Churn Rate (%)")
    plt.xlabel("Tenure (months)")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "churn_by_tenure.png", dpi=120)
    plt.close()

    # ---- คำถามธุรกิจที่ 3: ค่าบริการรายเดือนมีผลไหม? ----
    plt.figure(figsize=(7, 4))
    sns.kdeplot(data=raw, x="MonthlyCharges", hue="Churn",
                palette=[INDIGO, RED], fill=True, alpha=0.4)
    plt.title("Monthly Charges Distribution by Churn")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "churn_by_charges.png", dpi=120)
    plt.close()

    print(f"บันทึกกราฟ 3 รูปไว้ในโฟลเดอร์ {IMG_DIR}/ เรียบร้อย")
    print("\nสรุป insight:")
    print("- สัญญา Month-to-month มี churn สูงสุด")
    print("- ลูกค้าใหม่ (tenure ต่ำ) เสี่ยงเลิกมากที่สุด")
    print("- ต่อไปรัน src/train_model.py เพื่อเทรนและเทียบโมเดล")


if __name__ == "__main__":
    main()