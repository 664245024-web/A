"""
Streamlit Web Application for Loan Status Prediction
รันด้วยคำสั่ง: streamlit run 2_streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Loan Status Prediction - SVM",
    page_icon="💰",
    layout="wide"
)

# ============================================================
# Load Model
# ============================================================
@st.cache_resource
def load_model():
    """โหลดโมเดลที่บันทึกไว้"""
    model_path = 'svm_loan_model.pkl'
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model = load_model()

# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("💰 Loan Prediction App")
st.sidebar.markdown("---")
st.sidebar.info("""
**โมเดล:** Support Vector Machine (SVM)  
**Kernel:** RBF  
**Target:** Loan Status (0=Approved, 1=Default)
""")

# ============================================================
# Main Title
# ============================================================
st.title("🏦 ระบบทำนายสถานะการกู้ยืม")
st.markdown("ป้อนข้อมูลผู้กู้ยืมเพื่อทำนายว่าสินเชื่อจะได้รับการอนุมัติหรือไม่")

if model is None:
    st.error("❌ ไม่พบไฟล์โมเดล 'svm_loan_model.pkl'")
    st.info("กรุณารันสคริปต์ `1_train_model.py` ก่อนเพื่อสร้างโมเดล")
    st.stop()

# ============================================================
# Input Form
# ============================================================
st.header("📝 ข้อมูลผู้กู้ยืม")

col1, col2, col3 = st.columns(3)

with col1:
    person_age = st.number_input("อายุ (Age)", min_value=18, max_value=100, value=25, step=1)
    person_gender = st.selectbox("เพศ (Gender)", ["male", "female"])
    person_education = st.selectbox(
        "ระดับการศึกษา (Education)",
        ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
    )
    person_income = st.number_input(
        "รายได้ต่อปี (Income)", min_value=0, value=50000, step=1000,
        help="หน่วย: ดอลลาร์"
    )

with col2:
    person_emp_exp = st.number_input(
        "ปีประสบการณ์ทำงาน (Work Experience)",
        min_value=0, max_value=50, value=3, step=1
    )
    person_home_ownership = st.selectbox(
        "สถานะที่อยู่อาศัย (Home Ownership)",
        ["RENT", "MORTGAGE", "OWN", "OTHER"]
    )
    loan_amnt = st.number_input(
        "จำนวนเงินกู้ (Loan Amount)", min_value=0, value=10000, step=500
    )
    loan_intent = st.selectbox(
        "วัตถุประสงค์การกู้ (Loan Intent)",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
         "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
    )

with col3:
    loan_int_rate = st.number_input(
        "อัตราดอกเบี้ย (%)", min_value=0.0, max_value=30.0,
        value=10.0, step=0.1
    )
    loan_percent_income = st.number_input(
        "สัดส่วนเงินกู้ต่อรายได้ (%)", min_value=0.0, max_value=1.0,
        value=0.1, step=0.01,
        help="เช่น 0.1 = 10% ของรายได้"
    )
    cb_person_cred_hist_length = st.number_input(
        "ปีประวัติเครดิต (Credit History Length)",
        min_value=0, max_value=50, value=5, step=1
    )
    credit_score = st.number_input(
        "คะแนนเครดิต (Credit Score)",
        min_value=300, max_value=850, value=650, step=10
    )
    previous_loan_defaults = st.selectbox(
        "เคยผิดนัดชำระหนี้หรือไม่?",
        ["No", "Yes"]
    )

# ============================================================
# Prediction Button
# ============================================================
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict_button = st.button("🔮 ทำนายผล", use_container_width=True, type="primary")

# ============================================================
# Prediction Logic
# ============================================================
if predict_button:
    # สร้าง DataFrame จากข้อมูลผู้ใช้
    input_data = pd.DataFrame({
        'person_age': [person_age],
        'person_gender': [person_gender],
        'person_education': [person_education],
        'person_income': [person_income],
        'person_emp_exp': [person_emp_exp],
        'person_home_ownership': [person_home_ownership],
        'loan_amnt': [loan_amnt],
        'loan_intent': [loan_intent],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length],
        'credit_score': [credit_score],
        'previous_loan_defaults_on_file': [previous_loan_defaults]
    })
    
    with st.spinner("กำลังประมวลผล..."):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
    
    # แสดงผลลัพธ์
    st.markdown("---")
    st.header("📊 ผลการทำนาย")
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        if prediction == 0:
            st.success("✅ สินเชื่อได้รับการอนุมัติ (Approved)")
            st.metric("ความน่าจะเป็นที่จะอนุมัติ", f"{probability[0]*100:.2f}%")
        else:
            st.error("❌ สินเชื่อไม่ได้รับการอนุมัติ / มีแนวโน้มผิดนัด (Default)")
            st.metric("ความน่าจะเป็นที่จะผิดนัด", f"{probability[1]*100:.2f}%")
    
    with col_res2:
        st.subheader("📈 ความน่าจะเป็น")
        prob_df = pd.DataFrame({
            'สถานะ': ['Approved', 'Default'],
            'ความน่าจะเป็น (%)': [probability[0]*100, probability[1]*100]
        })
        st.bar_chart(prob_df.set_index('สถานะ'))
    
    # แสดงข้อมูลที่ใช้ทำนาย
    with st.expander("🔍 ดูข้อมูลที่ใช้ทำนาย"):
        st.dataframe(input_data.T.rename(columns={0: 'ค่า'}))

# ============================================================
# Batch Prediction
# ============================================================
st.markdown("---")
st.header("📂 ทำนายผลจากไฟล์ CSV")

uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV", type=['csv'])

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        st.write(f"📄 ข้อมูลที่อัปโหลด: {df_input.shape[0]} แถว, {df_input.shape[1]} คอลัมน์")
        st.dataframe(df_input.head())
        
        if st.button("🚀 ทำนายทั้งหมด", type="primary"):
            with st.spinner("กำลังประมวลผล..."):
                predictions = model.predict(df_input)
                probabilities = model.predict_proba(df_input)
            
            df_result = df_input.copy()
            df_result['Prediction'] = predictions
            df_result['Pred_Label'] = df_result['Prediction'].map(
                {0: 'Approved', 1: 'Default'}
            )
            df_result['Prob_Approved'] = probabilities[:, 0]
            df_result['Prob_Default'] = probabilities[:, 1]
            
            st.success(f"✅ ทำนายเสร็จสิ้น! ({len(df_result)} รายการ)")
            
            # สรุปผล
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("ทั้งหมด", len(df_result))
            with col_s2:
                approved = (df_result['Prediction'] == 0).sum()
                st.metric("อนุมัติ", approved)
            with col_s3:
                default = (df_result['Prediction'] == 1).sum()
                st.metric("ผิดนัด", default)
            
            st.dataframe(df_result)
            
            # ดาวน์โหลดผล
            csv = df_result.to_csv(index=False).encode('utf-8')
            st.download_button(
                "💾 ดาวน์โหลดผลลัพธ์ (CSV)",
                csv,
                "prediction_results.csv",
                "text/csv"
            )
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {e}")

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <small>Built with ❤️ using Streamlit & Scikit-learn | SVM Model</small>
    </div>
    """,
    unsafe_allow_html=True
)