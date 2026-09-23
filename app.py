import streamlit as st
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# ⚙️ CONFIG — แก้ไขค่าต่อไปนี้ให้ตรงกับตอนเทรนโมเดลจริงของคุณ
# ============================================================
MODEL_PATH = "044titanic_svm.joblib"
SCALER_PATH = "titanic_scaler.joblib"

# ค่าความแม่นยำของโมเดล (จากตอนเทรน/ทดสอบ) — แก้เป็นค่าจริงของคุณ
MODEL_ACCURACY = 0.80  # เช่น 0.82 หมายถึง 82%

# ลำดับฟีเจอร์ตามที่ตรวจพบใน scaler จริง (titanic_scaler.joblib):
# Pclass, Sex_female, Age, Fare, FamilySize
FEATURE_ORDER = ["Pclass", "Sex_female", "Age", "Fare", "FamilySize"]

# Age และ Fare ใน scaler มีค่าเฉลี่ยต่ำผิดปกติ (~0.36) ซึ่งบ่งชี้ว่าถูก
# normalize มาก่อนแล้ว (เช่นหารด้วยค่าสูงสุด) ก่อนเข้า StandardScaler อีกที
# ผู้พัฒนายังไม่ได้ยืนยันสูตรที่แน่นอน จึงใช้ค่าประมาณนี้ไปก่อน
# **กรุณาแก้ AGE_MAX / FARE_MAX ให้ตรงกับตอนเทรนจริง เพื่อความแม่นยำสูงสุด**
AGE_MAX = 80.0
FARE_MAX = 512.3292
# ============================================================

st.set_page_config(
    page_title="ระบบทำนายการรอดชีวิตผู้โดยสารไททานิค",
    page_icon="🚢",
    layout="centered",
)

# ---------- Minimal / Clean CSS ----------
st.markdown(
    """
    <style>
        .main {background-color: #FAFAFA;}
        .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 720px;}
        h1 {
            font-size: 1.9rem !important;
            font-weight: 700 !important;
            color: #1F2937 !important;
            text-align: center;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            text-align: center;
            color: #6B7280;
            font-size: 0.95rem;
            margin-bottom: 1.8rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #2563EB;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 0;
            font-weight: 600;
            font-size: 1rem;
        }
        .stButton>button:hover {
            background-color: #1D4ED8;
            color: white;
        }
        .result-card {
            border-radius: 12px;
            padding: 1.4rem;
            text-align: center;
            margin-top: 1.2rem;
        }
        .result-survive {
            background-color: #ECFDF5;
            border: 1px solid #10B981;
            color: #065F46;
        }
        .result-not-survive {
            background-color: #FEF2F2;
            border: 1px solid #EF4444;
            color: #991B1B;
        }
        footer {visibility: hidden;}
        .dev-footer {
            text-align: center;
            color: #9CA3AF;
            font-size: 0.85rem;
            margin-top: 3rem;
            border-top: 1px solid #E5E7EB;
            padding-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load model + scaler ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

try:
    model, scaler = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    load_error = str(e)

# ---------- Header ----------
st.markdown("<h1>🚢 ระบบทำนายการรอดชีวิตผู้โดยสารเรือไททานิค</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>ทำนายโอกาสรอดชีวิตของผู้โดยสาร ด้วยโมเดล Support Vector Machine (SVM)</div>",
    unsafe_allow_html=True,
)

# ---------- Accuracy badge ----------
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.metric(label="ความแม่นยำของระบบ (Accuracy)", value=f"{MODEL_ACCURACY * 100:.2f}%")

st.divider()

if not artifacts_loaded:
    st.error(f"ไม่สามารถโหลดโมเดลหรือ scaler ได้: {load_error}")
    st.stop()

# ---------- Input form ----------
st.subheader("กรอกข้อมูลผู้โดยสาร")

with st.form("prediction_form"):
    c1, c2 = st.columns(2)

    with c1:
        pclass = st.selectbox(
            "ชั้นโดยสาร (Pclass)",
            options=[1, 2, 3],
            format_func=lambda x: f"ชั้น {x}",
        )
        sex = st.radio("เพศ (Sex)", options=["ชาย", "หญิง"], horizontal=True)
        age = st.slider("อายุ (Age)", min_value=0, max_value=90, value=30)

    with c2:
        sibsp = st.number_input(
            "จำนวนพี่น้อง/คู่สมรสที่ร่วมเดินทาง (SibSp)",
            min_value=0, max_value=10, value=0, step=1,
        )
        parch = st.number_input(
            "จำนวนพ่อแม่/ลูกที่ร่วมเดินทาง (Parch)",
            min_value=0, max_value=10, value=0, step=1,
        )
        fare = st.number_input(
            "ค่าโดยสาร (Fare)",
            min_value=0.0, max_value=600.0, value=32.0, step=1.0,
        )

    submitted = st.form_submit_button("🔍 ทำนายผล")

if submitted:
    sex_female = 1 if sex == "หญิง" else 0
    family_size = sibsp + parch + 1
    age_norm = min(age / AGE_MAX, 1.0)
    fare_norm = min(fare / FARE_MAX, 1.0)

    raw_values = {
        "Pclass": pclass,
        "Sex_female": sex_female,
        "Age": age_norm,
        "Fare": fare_norm,
        "FamilySize": family_size,
    }
    features = np.array([[raw_values[f] for f in FEATURE_ORDER]])

    try:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        if prediction == 1:
            st.markdown(
                "<div class='result-card result-survive'>"
                "<h3>✅ รอดชีวิต (Survived)</h3>"
                "<p>ผู้โดยสารรายนี้มีแนวโน้มรอดชีวิตจากเหตุการณ์เรือไททานิคจม</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='result-card result-not-survive'>"
                "<h3>❌ ไม่รอดชีวิต (Not Survived)</h3>"
                "<p>ผู้โดยสารรายนี้มีแนวโน้มไม่รอดชีวิตจากเหตุการณ์เรือไททานิคจม</p>"
                "</div>",
                unsafe_allow_html=True,
            )

        with st.expander("ดูข้อมูลที่ใช้ทำนาย (ก่อน/หลัง scale)"):
            st.write("ค่าดิบก่อนเข้า scaler:")
            st.json(raw_values)
            st.write("ค่าหลังผ่าน scaler:")
            st.json(dict(zip(FEATURE_ORDER, features_scaled[0].tolist())))

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดระหว่างทำนายผล: {e}")

# ---------- Footer ----------
st.markdown(
    "<div class='dev-footer'>พัฒนาโดย: นายอติชาต พันธุ์ขะวงษ์</div>",
    unsafe_allow_html=True,
)
