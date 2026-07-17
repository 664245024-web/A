"""
Streamlit Web Application for Loan Status Prediction (Self-Contained) - Improved Version
รันด้วยคำสั่ง: streamlit run 2_streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Loan Status Prediction - SVM",
    page_icon="💰",
    layout="wide"
)

# ============================================================
# Embedded Data & Model Training Logic
# ============================================================
# ข้อมูลตั้งต้นสำหรับฝึกโมเดลอัตโนมัติ (กรณีไม่มีไฟล์เดิม)
CSV_DATA = """person_age,person_gender,person_education,person_income,person_emp_exp,person_home_ownership,loan_amnt,loan_intent,loan_int_rate,loan_percent_income,cb_person_cred_hist_length,credit_score,previous_loan_defaults_on_file,loan_status
23.0,female,Associate,53395.0,1,OWN,15000.0,EDUCATION,11.01,0.28,3.0,574,Yes,0
23.0,female,Bachelor,60080.0,1,RENT,9000.0,MEDICAL,11.01,0.15,2.0,562,No,1
24.0,male,Master,68121.0,1,RENT,9000.0,HOMEIMPROVEMENT,11.01,0.13,2.0,651,Yes,0
24.0,male,Bachelor,67890.0,0,RENT,9000.0,PERSONAL,7.29,0.13,2.0,548,Yes,0
23.0,female,High School,68170.0,0,RENT,9000.0,EDUCATION,13.35,0.13,3.0,579,Yes,0
23.0,female,High School,69638.0,0,RENT,9000.0,PERSONAL,7.14,0.13,2.0,660,No,0
23.0,male,Bachelor,46345.0,1,MORTGAGE,10000.0,DEBTCONSOLIDATION,9.63,0.22,3.0,646,No,1
24.0,male,Master,37260.0,2,MORTGAGE,6000.0,DEBTCONSOLIDATION,13.57,0.16,2.0,634,No,0
26.0,male,Master,37000.0,4,MORTGAGE,3500.0,VENTURE,8.0,0.09,4.0,569,No,0"""

# รายชื่อ Feature ที่โมเดลคาดหวังตามลำดับ (เพื่อป้องกันมิติข้อมูลสลับที่)
FEATURE_COLUMNS = [
    'person_age', 'person_gender', 'person_education', 'person_income',
    'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
    'credit_score', 'previous_loan_defaults_on_file'
]

def get_or_train_model():
    """โหลดโมเดลจากไฟล์ หรือฝึกใหม่หากไม่พบ/โหลดไม่ได้"""
    model_path = 'svm_loan_model.pkl'
    
    # พยายามโหลดโมเดลที่บันทึกไว้ก่อน
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.warning(f"⚠️ ไม่สามารถโหลดโมเดลเดิมได้ เนื่องจากข้อผิดพลาดทางระบบ ({type(e).__name__}) จะทำการสร้างและฝึกโมเดลขึ้นมาทดแทนใหม่...")
    
    # หากไม่มีไฟล์หรือโหลดไม่ได้ ให้ฝึกโมเดลใหม่จากข้อมูลจำลอง
    with st.spinner("🔧 กำลังเตรียมโครงสร้างข้อมูลและสร้างโมเดล SVM ใหม่เพื่อเริ่มต้นระบบ..."):
        df = pd.read_csv(StringIO(CSV_DATA))
        
        X = df[FEATURE_COLUMNS]
        y = df['loan_status']
        
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # ปรับการทำงานของ OneHotEncoder เพื่อรองรับทั้งเวอร์ชันเก่าและใหม่
        try:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', encoder)
                ]), categorical_features)
            ])
            
        svm_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
        ])
        
        # ใช้ข้อมูลทั้งหมดในการฝึกสอนเนื่องจากจำนวนข้อมูลจำลองยังมีน้อย
        svm_pipeline.fit(X, y)
        
        # บันทึกโมเดลสำหรับใช้ในอนาคต
        joblib.dump(svm_pipeline, model_path)
        st.success("✅ บันทึกโมเดลสำรองลงเครื่องเรียบร้อยแล้ว!")
        
    return svm_pipeline

# โหลดโมเดลผ่านหน่วยความจำแคชเพื่อความเร็วในการแสดงผล
@st.cache_resource
def load_cached_model():
    return get_or_train_model()

model = load_cached_model()

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
    # สร้าง DataFrame โดยเรียงคอลัมน์ให้ตรงกับที่โมเดลเทรน
    input_data = pd.DataFrame({
        'person_age': [float(person_age)],
        'person_gender': [person_gender],
        'person_education': [person_education],
        'person_income': [float(person_income)],
        'person_emp_exp': [float(person_emp_exp)],
        'person_home_ownership': [person_home_ownership],
        'loan_amnt': [float(loan_amnt)],
        'loan_intent': [loan_intent],
        'loan_int_rate': [float(loan_int_rate)],
        'loan_percent_income': [float(loan_percent_income)],
        'cb_person_cred_hist_length': [float(cb_person_cred_hist_length)],
        'credit_score': [float(credit_score)],
        'previous_loan_defaults_on_file': [previous_loan_defaults]
    })[FEATURE_COLUMNS] # บังคับเรียงตำแหน่งของฟีเจอร์ให้ถูกต้องตามที่โมเดลถูกฝึกสอน
    
    with st.spinner("กำลังประมวลผล..."):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
    
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
        
        # ค้นหาว่าในไฟล์ CSV ขาดคอลัมน์ใดที่โมเดลต้องการใช้หรือไม่
        missing_cols = [col for col in FEATURE_COLUMNS if col not in df_input.columns]
        
        if len(missing_cols) > 0:
            st.error(f"❌ โครงสร้างข้อมูลไม่ถูกต้อง ขาดคอลัมน์ที่จำเป็นดังนี้: {', '.join(missing_cols)}")
        else:
            if st.button("🚀 ทำนายทั้งหมด", type="primary"):
                with st.spinner("กำลังประมวลผลการทำนายผลลัพธ์แบบกลุ่ม..."):
                    # เรียงคอลัมน์ข้อมูลนำเข้าเฉพาะคอลัมน์ที่จำเป็นสำหรับโมเดล
                    df_to_predict = df_input[FEATURE_COLUMNS]
                    predictions = model.predict(df_to_predict)
                    probabilities = model.predict_proba(df_to_predict)
                
                df_result = df_input.copy()
                df_result['Prediction'] = predictions
                df_result['Pred_Label'] = df_result['Prediction'].map(
                    {0: 'Approved', 1: 'Default'}
                )
                df_result['Prob_Approved'] = probabilities[:, 0]
                df_result['Prob_Default'] = probabilities[:, 1]
                
                st.success(f"✅ ทำนายเสร็จสิ้น! ({len(df_result)} รายการ)")
                
                # แสดงค่าสรุปสถิติเบื้องต้น
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("จำนวนรายการทั้งหมด", len(df_result))
                with col_s2:
                    approved = (df_result['Prediction'] == 0).sum()
                    st.metric("ได้รับการอนุมัติ", approved)
                with col_s3:
                    default = (df_result['Prediction'] == 1).sum()
                    st.metric("คาดว่าผิดนัดชำระ", default)
                
                st.dataframe(df_result)
                
                # เตรียมฟังก์ชันดาวน์โหลดผลลัพธ์
                csv = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "💾 ดาวน์โหลดผลลัพธ์ (CSV)",
                    csv,
                    "prediction_results.csv",
                    "text/csv"
                )
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในระหว่างจัดเตรียมข้อมูล: {e}")

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