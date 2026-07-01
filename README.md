# Customer Churn Prediction — ระบบทำนายการเลิกใช้บริการลูกค้า

ระบบวิเคราะห์และทำนายว่าลูกค้ากลุ่มใดมีแนวโน้มเลิกใช้บริการ พร้อมเสนอ action ทางธุรกิจเพื่อลด churn — ครอบคลุมตั้งแต่ data cleaning, EDA, การเทียบหลายโมเดล ไปจนถึง interactive dashboard ที่ใช้งานได้จริงในองค์กร

Link : https://churn-prediction-ip8nq4gpoftwaaewkk4rbu.streamlit.app/

<img width="1887" height="902" alt="image" src="https://github.com/user-attachments/assets/cca368b7-4d82-40d4-9621-f1658df50f1b" />
<img width="1903" height="903" alt="image" src="https://github.com/user-attachments/assets/3635a796-2975-4b8e-a26b-affafa93b0a1" />
<img width="1897" height="897" alt="image" src="https://github.com/user-attachments/assets/ac15d9fa-c958-4971-89ec-caea3fad097d" />



---

## โจทย์ทางธุรกิจ

การหาลูกค้าใหม่มีต้นทุนสูงกว่าการรักษาลูกค้าเดิมหลายเท่า ระบบนี้ช่วยให้ธุรกิจ **รู้ล่วงหน้า** ว่าลูกค้าคนไหนกำลังจะเลิกใช้บริการ เพื่อเข้าไปเสนอโปรโมชันหรือแก้ปัญหาได้ทันก่อนเสียลูกค้าไป

คำถามที่ระบบตอบได้:
- ลูกค้ากลุ่มไหนเสี่ยงเลิกใช้บริการมากที่สุด
- ปัจจัยอะไรที่ทำให้ลูกค้าตัดสินใจเลิก
- ควรโฟกัสทรัพยากรการตลาดไปที่ลูกค้าคนไหนก่อน

---

## ผลลัพธ์และ Business Insight

### ประสิทธิภาพโมเดล (เปรียบเทียบ 3 โมเดล)

> หมายเหตุ: ตัวเลขด้านล่างเป็นค่าอ้างอิง หลังรัน `python src/train_model.py`
> ให้นำตัวเลขจริงจาก `models/model_comparison.csv` มาแทนที่

| โมเดล | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.80 | 0.65 | 0.55 | 0.60 | 0.84 |
| Random Forest | 0.79 | 0.63 | 0.51 | 0.56 | 0.82 |
| **XGBoost** | **0.81** | **0.67** | **0.58** | **0.62** | **0.85** |

เลือกโมเดลด้วยเกณฑ์ **ROC-AUC** เพราะข้อมูล churn เป็น imbalanced (คนอยู่ต่อมากกว่าคนเลิกมาก) การวัดด้วย Accuracy อย่างเดียวจะทำให้เข้าใจผิด — โมเดลที่ทายว่า "ไม่เลิก" หมดก็ได้ Accuracy สูงแต่ไร้ประโยชน์ทางธุรกิจ

### สิ่งที่ค้นพบจากข้อมูล (Key Findings)

1. **สัญญาแบบ Month-to-month เสี่ยงสูงสุด** — churn rate สูงกว่าสัญญารายปีหลายเท่า
2. **ลูกค้าใหม่ (tenure ต่ำ) เปราะบางที่สุด** — ช่วงเดือนแรก ๆ คือช่วงวิกฤต
3. **จ่ายเงินด้วย Electronic check สัมพันธ์กับ churn สูง** — อาจสะท้อนกลุ่มลูกค้าที่ผูกพันน้อย

### ข้อเสนอทางธุรกิจ (Recommendations)

- เสนอส่วนลดจูงใจให้ลูกค้า month-to-month เปลี่ยนเป็นสัญญารายปี
- สร้าง onboarding program พิเศษสำหรับลูกค้าใหม่ในช่วงเดือนแรก
- ให้ทีม retention โฟกัสกลุ่ม "เสี่ยงสูง" จากผลทำนายก่อน

---

## Tech Stack

| ด้าน | เครื่องมือ |
|------|-----------|
| Data & ML | Python, pandas, numpy, scikit-learn, XGBoost |
| Visualization | Plotly, matplotlib, seaborn |
| Dashboard | Streamlit |
| อื่น ๆ | joblib (บันทึกโมเดล), Git |

---

## วิธีใช้งาน

```bash
# 1. ติดตั้ง dependencies
pip install -r requirements.txt

# 2. เทรนและเทียบโมเดล (สร้างไฟล์ในโฟลเดอร์ models/)
python src/train_model.py

# 3. เปิด dashboard
streamlit run app/streamlit_app.py
```

หมายเหตุ: ทุกคำสั่งต้องรันจากโฟลเดอร์รากของโปรเจกต์ เพราะ path อ้างอิงจากตรงนั้น

---

## โครงสร้างโปรเจกต์

```
churn-prediction/
├── app/
│   └── streamlit_app.py       # dashboard 3 โหมด (Dashboard / Single / Batch)
├── data/
│   └── raw/
│       └── data.csv           # ชุดข้อมูลต้นฉบับ (Telco Customer Churn)
├── notebooks/
│   ├── 01_explore.py          # สำรวจข้อมูลเบื้องต้น
│   ├── 02_clean.py            # ทำความสะอาดข้อมูล
│   └── 03_modeling.py         # EDA + สร้างกราฟ insight
├── src/
│   ├── preprocessing.py       # ฟังก์ชันเตรียมข้อมูล (ใช้ร่วมกันทั้งระบบ)
│   ├── train_model.py         # เทรน + เทียบ 3 โมเดล เลือกตัวดีที่สุด
│   └── predict.py             # ฟังก์ชันทำนาย (ใช้โดย dashboard)
├── models/                    # โมเดลและผลลัพธ์ (สร้างตอนรัน train_model)
│   ├── churn_model.pkl
│   ├── model_columns.pkl
│   └── model_comparison.csv
├── images/                    # กราฟ EDA + screenshot dashboard
├── requirements.txt
└── README.md
```

---

## ฟีเจอร์ Dashboard

ระบบมี 3 โหมดในตัวเดียว เลือกจาก sidebar:

**Dashboard** — ภาพรวมธุรกิจ: KPI สำคัญ (จำนวนลูกค้า, อัตรา churn, รายได้ที่เสี่ยงหายต่อเดือน) พร้อมกราฟ interactive 4 แบบ

**Single Prediction** — กรอกข้อมูลลูกค้า 1 คน ดูผลทันทีพร้อมเกจวัดความเสี่ยงและคำแนะนำเชิงปฏิบัติแยกตามระดับ

**Batch Prediction** — อัปโหลด CSV ทั้งบริษัท ระบบทำนายทุกคน จัดอันดับความเสี่ยง กรองกลุ่ม แล้วดาวน์โหลดผลให้ทีมการตลาดใช้ต่อได้ทันที

---

## Machine Learning Pipeline

1. **Data Cleaning** — แก้คอลัมน์ `TotalCharges` ที่มีค่าว่างปน, ลบแถวที่ข้อมูลไม่ครบ, แปลงเป้าหมายเป็น 0/1
2. **Feature Encoding** — One-Hot Encoding คอลัมน์ประเภทข้อความ (`drop_first=True` เลี่ยง multicollinearity)
3. **Model Training** — เทรน 3 โมเดลพร้อมจัดการ class imbalance (`class_weight="balanced"` / `scale_pos_weight`)
4. **Model Selection** — เลือกโมเดลที่ ROC-AUC สูงสุดอัตโนมัติ บันทึกพร้อมรายชื่อคอลัมน์
5. **Serving** — dashboard โหลดโมเดลมาทำนายทั้งแบบรายคนและทั้งไฟล์

---

## หมายเหตุ

ชุดข้อมูลที่ใช้คือ [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) จาก Kaggle (7,043 รายการ)

โปรเจกต์นี้เป็นส่วนหนึ่งของ portfolio ด้าน Data Science / Full-Stack Development
