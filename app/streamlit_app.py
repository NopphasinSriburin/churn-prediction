"""
app/streamlit_app.py
ระบบทำนายการเลิกใช้บริการลูกค้า (Customer Churn Prediction) — Dashboard 3 โหมด

โหมดที่ 1: Dashboard        ภาพรวมธุรกิจ + KPI
โหมดที่ 2: Single Prediction กรอกข้อมูลลูกค้า 1 คน
โหมดที่ 3: Batch Prediction  อัปโหลดไฟล์ทั้งบริษัท จัดอันดับ ดาวน์โหลด

ใช้ Material icons ของ Streamlit (:material/name:) แทน emoji

รันจากโฟลเดอร์รากของโปรเจกต์:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# เพิ่ม src เข้า path เพื่อ import โมดูลจาก src/ ได้
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from predict import (
    load_artifacts, predict_single, predict_batch, risk_level,
)

# ============================================================
#  ตั้งค่าหน้าเว็บ + ธีมดีไซน์
# ============================================================
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS ปรับให้ดูโปร สะอาด (โทน indigo #4F46E5 เหมือน design system เดิม)
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6e8eb;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label { color: #64748b; font-weight: 500; }
    div[data-testid="stMetricValue"] { color: #0f172a; font-weight: 700; }

    h1 { color: #0f172a; font-weight: 800; letter-spacing: -0.5px; }

    .stButton button[kind="primary"] {
        background: #4F46E5;
        border-radius: 8px;
        font-weight: 600;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

INDIGO = "#4F46E5"
RED = "#EF4444"


# ============================================================
#  โหลดโมเดล (cache)
# ============================================================
@st.cache_resource
def get_model():
    return load_artifacts()


try:
    model, model_columns, scaler = get_model()
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False


# ============================================================
#  Sidebar
# ============================================================
with st.sidebar:
    st.markdown("### :material/analytics: Churn Intelligence")
    st.caption("ระบบทำนายการเลิกใช้บริการลูกค้า")
    st.divider()

    page = st.radio(
        "menu",
        ["Dashboard", "Single Prediction", "Batch Prediction"],
        label_visibility="collapsed",
    )

    st.divider()
    if MODEL_LOADED:
        st.success("โมเดลพร้อมใช้งาน", icon=":material/check_circle:")
    else:
        st.error("ไม่พบไฟล์โมเดล", icon=":material/error:")
        st.caption("รัน python src/train_model.py ก่อน")


if not MODEL_LOADED:
    st.title(":material/warning: ยังไม่พบไฟล์โมเดล")
    st.write("กรุณารัน `python src/train_model.py` ก่อน เพื่อสร้างไฟล์โมเดลใน `models/`")
    st.stop()


# ============================================================
#  โหมดที่ 1: DASHBOARD
# ============================================================
if page == "Dashboard":
    st.title("ภาพรวมธุรกิจ")
    st.caption("สรุปสถานการณ์การเลิกใช้บริการของลูกค้าทั้งหมด")

    # หาไฟล์ข้อมูลดิบ
    data_path = None
    for candidate in ["data/raw/data.csv", "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv", "data.csv"]:
        if Path(candidate).exists():
            data_path = candidate
            break

    if data_path is None:
        st.warning("ไม่พบไฟล์ข้อมูลดิบใน data/raw/ — Dashboard ต้องใช้ข้อมูลเพื่อแสดงภาพรวม")
        st.stop()

    raw = pd.read_csv(data_path)
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    raw = raw.dropna()

    total_customers = len(raw)
    churned = (raw["Churn"] == "Yes").sum()
    churn_rate = churned / total_customers
    revenue_at_risk = raw[raw["Churn"] == "Yes"]["MonthlyCharges"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ลูกค้าทั้งหมด", f"{total_customers:,}")
    c2.metric("ลูกค้าที่เลิกใช้", f"{churned:,}")
    c3.metric("อัตราการเลิกใช้", f"{churn_rate:.1%}")
    c4.metric("รายได้เสี่ยงหาย/เดือน", f"{revenue_at_risk:,.0f} B")

    st.divider()

    g1, g2 = st.columns(2)

    with g1:
        st.subheader("สัดส่วนลูกค้าที่เลิกใช้")
        counts = raw["Churn"].value_counts()
        fig = px.pie(
            values=counts.values, names=["อยู่ต่อ", "เลิกใช้"],
            hole=0.55, color_discrete_sequence=[INDIGO, RED],
        )
        fig.update_layout(showlegend=True, height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        st.subheader("อัตราการเลิกใช้ ตามรูปแบบสัญญา")
        cc = raw.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean()).reset_index()
        cc.columns = ["Contract", "ChurnRate"]
        fig = px.bar(
            cc, x="Contract", y="ChurnRate",
            color="ChurnRate", color_continuous_scale=[INDIGO, RED],
            text=cc["ChurnRate"].apply(lambda x: f"{x:.1%}"),
        )
        fig.update_layout(height=320, margin=dict(t=10, b=10), coloraxis_showscale=False)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    g3, g4 = st.columns(2)

    with g3:
        st.subheader("อัตราการเลิกใช้ ตามระยะเวลาใช้บริการ")
        raw["tenure_group"] = pd.cut(
            raw["tenure"], bins=[0, 12, 24, 48, 72],
            labels=["0-12", "13-24", "25-48", "49-72"],
        )
        tc = raw.groupby("tenure_group", observed=True)["Churn"].apply(
            lambda x: (x == "Yes").mean()).reset_index()
        tc.columns = ["tenure_group", "ChurnRate"]
        fig = px.line(tc, x="tenure_group", y="ChurnRate", markers=True,
                      color_discrete_sequence=[INDIGO])
        fig.update_layout(height=320, margin=dict(t=10, b=10), xaxis_title="เดือน")
        fig.update_yaxes(tickformat=".0%")
        fig.update_traces(line=dict(width=3), marker=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)

    with g4:
        st.subheader("ปัจจัยที่มีผลต่อการเลิกใช้มากที่สุด")
        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame({
                "feature": model_columns,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=True).tail(8)
            fig = px.bar(imp, x="importance", y="feature", orientation="h",
                         color_discrete_sequence=[INDIGO])
            fig.update_layout(height=320, margin=dict(t=10, b=10),
                              yaxis_title="", xaxis_title="ระดับความสำคัญ")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("โมเดลปัจจุบันไม่มีค่า feature importance โดยตรง")


# ============================================================
#  โหมดที่ 2: SINGLE PREDICTION
# ============================================================
elif page == "Single Prediction":
    st.title("ทำนายรายบุคคล")
    st.caption("กรอกข้อมูลลูกค้า 1 คน เพื่อประเมินความเสี่ยงในการเลิกใช้บริการ")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**:material/timeline: ข้อมูลการใช้งาน**")
        tenure = st.slider("ระยะเวลาใช้บริการ (เดือน)", 0, 72, 12)
        monthly = st.number_input("ค่าบริการรายเดือน (B)", 0.0, 200.0, 70.0)
        total = st.number_input("ค่าบริการรวม (B)", 0.0, 10000.0, 1000.0)

    with col2:
        st.markdown("**:material/description: ข้อมูลสัญญา**")
        contract = st.selectbox("รูปแบบสัญญา", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("ประเภทอินเทอร์เน็ต", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox("วิธีชำระเงิน", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])

    with col3:
        st.markdown("**:material/person: ข้อมูลอื่น ๆ**")
        senior = st.radio("ผู้สูงอายุ", ["ไม่ใช่", "ใช่"], horizontal=True)
        paperless = st.radio("รับบิลอิเล็กทรอนิกส์", ["ใช่", "ไม่ใช่"], horizontal=True)

    st.divider()

    if st.button("ทำนายผล", type="primary", use_container_width=True,
                 icon=":material/insights:"):
        data = {
            "tenure": tenure, "monthly": monthly, "total": total,
            "senior": 1 if senior == "ใช่" else 0,
            "contract": contract, "internet": internet, "payment": payment,
            "paperless": 1 if paperless == "ใช่" else 0,
        }
        prob = predict_single(data, model, model_columns, scaler)
        level = risk_level(prob)

        st.divider()
        r1, r2 = st.columns([1, 2])

        with r1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 40}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": INDIGO},
                    "steps": [
                        {"range": [0, 40], "color": "#dcfce7"},
                        {"range": [40, 70], "color": "#fef9c3"},
                        {"range": [70, 100], "color": "#fee2e2"},
                    ],
                },
            ))
            fig.update_layout(height=260, margin=dict(t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with r2:
            st.markdown("### ผลการประเมิน")
            if level == "High":
                st.error(f"ความเสี่ยงสูง — โอกาสเลิกใช้ {prob:.1%}",
                         icon=":material/warning:")
                st.markdown("""
                **คำแนะนำ:**
                - ติดต่อลูกค้าโดยตรงภายใน 7 วัน
                - เสนอส่วนลดหรือแพ็กเกจสัญญาระยะยาว
                - จัดเป็นลูกค้ากลุ่มสำคัญของทีม Retention
                """)
            elif level == "Medium":
                st.warning(f"ความเสี่ยงปานกลาง — โอกาสเลิกใช้ {prob:.1%}",
                           icon=":material/info:")
                st.markdown("""
                **คำแนะนำ:**
                - ติดตามพฤติกรรมการใช้งานอย่างใกล้ชิด
                - ส่งข้อเสนอผ่านอีเมลหรือแอป
                """)
            else:
                st.success(f"ความเสี่ยงต่ำ — โอกาสเลิกใช้ {prob:.1%}",
                           icon=":material/check_circle:")
                st.markdown("""
                **คำแนะนำ:**
                - ลูกค้ามีแนวโน้มอยู่ต่อ
                - เหมาะสำหรับเสนอบริการเสริม (upsell)
                """)


# ============================================================
#  โหมดที่ 3: BATCH PREDICTION
# ============================================================
elif page == "Batch Prediction":
    st.title("ทำนายหมู่จากไฟล์")
    st.caption("อัปโหลดไฟล์ลูกค้าทั้งหมด ระบบจะทำนายและจัดอันดับความเสี่ยงให้อัตโนมัติ")
    st.divider()

    st.markdown("""
    **วิธีใช้:** อัปโหลดไฟล์ CSV ที่มีคอลัมน์เหมือนชุดข้อมูลต้นฉบับ (เช่น `data.csv`)
    ระบบจะประมวลผลทุกแถวและให้ดาวน์โหลดผลลัพธ์กลับไป
    """)

    uploaded = st.file_uploader("เลือกไฟล์ CSV", type=["csv"])

    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        st.success(f"อ่านไฟล์สำเร็จ: {len(df_raw):,} แถว", icon=":material/upload_file:")

        with st.spinner("กำลังทำนาย..."):
            id_col = (df_raw["customerID"] if "customerID" in df_raw.columns
                      else pd.Series(range(len(df_raw))))
            probs = predict_batch(df_raw, model, model_columns, scaler)

        result = pd.DataFrame({
            "customerID": id_col.values,
            "churn_probability_%": (probs * 100).round(1),
            "risk_level": [risk_level(p) for p in probs],
        }).sort_values("churn_probability_%", ascending=False)

        high = (result["risk_level"] == "High").sum()
        mid = (result["risk_level"] == "Medium").sum()
        low = (result["risk_level"] == "Low").sum()

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("เสี่ยงสูง (High)", f"{high:,}")
        m2.metric("เสี่ยงกลาง (Medium)", f"{mid:,}")
        m3.metric("เสี่ยงต่ำ (Low)", f"{low:,}")

        st.divider()

        filter_level = st.multiselect(
            "กรองตามระดับความเสี่ยง",
            ["High", "Medium", "Low"], default=["High"],
        )
        filtered = result[result["risk_level"].isin(filter_level)]
        st.markdown(f"**แสดง {len(filtered):,} รายการ** (เรียงจากเสี่ยงมากไปน้อย)")

        def color_risk(val):
            colors = {"High": "#fee2e2", "Medium": "#fef9c3", "Low": "#dcfce7"}
            return f"background-color: {colors.get(val, '')}"

        st.dataframe(
            filtered.style.map(color_risk, subset=["risk_level"]),
            use_container_width=True, height=400,
        )

        csv = filtered.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "ดาวน์โหลดผลลัพธ์ (CSV)", data=csv,
            file_name="churn_predictions.csv", mime="text/csv",
            type="primary", use_container_width=True,
            icon=":material/download:",
        )