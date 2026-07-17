import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os
from io import StringIO

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Loan Prediction - SVM",
    page_icon="💰",
    layout="wide"
)

# ข้อมูล CSV (จากไฟล์ที่คุณให้มา)
CSV_DATA = """person_age,person_gender,person_education,person_income,person_emp_exp,person_home_ownership,loan_amnt,loan_intent,loan_int_rate,loan_percent_income,cb_person_cred_hist_length,credit_score,previous_loan_defaults_on_file,loan_status
23.0,female,Associate,53395.0,1,OWN,15000.0,EDUCATION,11.01,0.28,3.0,574,Yes,0
23.0,female,Bachelor,60080.0,1,RENT,9000.0,MEDICAL,11.01,0.15,2.0,562,No,1
24.0,male,Master,68121.0,1,RENT,9000.0,HOMEIMPROVEMENT,11.01,0.13,2.0,651,Yes,0
24.0,male,Bachelor,67890.0,0,RENT,9000.0,PERSONAL,7.29,0.13,2.0,548,Yes,0
23.0,female,High School,68170.0,0,RENT,9000.0,EDUCATION,13.35,0.13,3.0,579,Yes,0"""

@st.cache_resource
def train_model():
    """ฝึกโมเดล SVM"""
    try:
        # อ่านข้อมูล
        df = pd.read_csv(StringIO(CSV_DATA))
        
        # แยก features และ target
        X = df.drop('loan_status', axis=1)
        y = df['loan_status']
        
        # แยกประเภทคอลัมน์
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # สร้าง preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # สร้างโมเดล SVM
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
        ])
        
        # แบ่งข้อมูลและฝึกโมเดล
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model.fit(X_train, y_train)
        
        # คำนวณ accuracy
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        
        return model, train_acc, test_acc
        
    except Exception as e:
        st.error(f"ข้อผิดพลาดในการฝึกโมเดล: {e}")
        return None, 0, 0

# Sidebar
st.sidebar.title("💰 Loan Prediction App")
st.sidebar.markdown("---")
st.sidebar.info("""
**โมเดล:** Support Vector Machine (SVM)  
**Kernel:** RBF  
**Target:** Loan Status (0=Approved, 1=Default)
""")

# หัวข้อหลัก
st.title("🏦 ระบบทำนายสถานะการกู้ยืม")
st.markdown("ป้อนข้อมูลผู้กู้ยืมเพื่อทำนายว่าสินเชื่อจะได้รับการอนุมัติหรือไม่")

# ฝึกหรือโหลดโมเดล
with st.spinner(" กำลังเตรียมโมเดล..."):
    model, train_acc, test_acc = train_model()

if model is not None:
    st.success(f"✅ โมเดลพร้อมใช้งาน (Train Accuracy: {train_acc:.2%}, Test Accuracy: {test_acc:.2%})")
    
    st.markdown("---")
    st.header("📝 ข้อมูลผู้กู้ยืม")
    
    # ฟอร์มกรอกข้อมูล
    col1, col2, col3 = st.columns(3)
    
    with col1:
        person_age = st.number_input("อายุ (Age)", min_value=18, max_value=100, value=25, step=1)
        person_gender = st.selectbox("เพศ (Gender)", ["male", "female"])
        person_education = st.selectbox(
            "ระดับการศึกษา (Education)",
            ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
        )
        person_income = st.number_input("รายได้ต่อปี (Income)", min_value=0, value=50000, step=1000)
    
    with col2:
        person_emp_exp = st.number_input("ปีประสบการณ์ทำงาน", min_value=0, max_value=50, value=3, step=1)
        person_home_ownership = st.selectbox("สถานะที่อยู่อาศัย", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        loan_amnt = st.number_input("จำนวนเงินกู้", min_value=0, value=10000, step=500)
        loan_intent = st.selectbox(
            "วัตถุประสงค์การกู้",
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
        )
    
    with col3:
        loan_int_rate = st.number_input("อัตราดอกเบี้ย (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.1)
        loan_percent_income = st.number_input("สัดส่วนเงินกู้ต่อรายได้", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        cb_person_cred_hist_length = st.number_input("ปีประวัติเครดิต", min_value=0, max_value=50, value=5, step=1)
        credit_score = st.number_input("คะแนนเครดิต", min_value=300, max_value=850, value=650, step=10)
        previous_loan_defaults = st.selectbox("เคยผิดนัดชำระหนี้", ["No", "Yes"])
    
    # ปุ่มทำนาย
    st.markdown("---")
    if st.button("🔮 ทำนายผล", type="primary", use_container_width=True):
        # สร้าง DataFrame จากข้อมูล
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
        
        # ทำนาย
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # แสดงผล
        st.markdown("---")
        st.header("📊 ผลการทำนาย")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 0:
                st.success("✅ **สินเชื่อได้รับการอนุมัติ (Approved)**")
                st.metric("ความน่าจะเป็นที่จะอนุมัติ", f"{probability[0]*100:.2f}%")
            else:
                st.error("❌ **สินเชื่อไม่ได้รับการอนุมัติ / มีแนวโน้มผิดนัด (Default)**")
                st.metric("ความน่าจะเป็นที่จะผิดนัด", f"{probability[1]*100:.2f}%")
        
        with col2:
            st.subheader(" ความน่าจะเป็น")
            prob_df = pd.DataFrame({
                'สถานะ': ['Approved', 'Default'],
                'ความน่าจะเป็น (%)': [probability[0]*100, probability[1]*100]
            })
            st.bar_chart(prob_df.set_index('สถานะ'))
        
        # แสดงข้อมูลที่ใช้ทำนาย
        with st.expander("🔍 ดูข้อมูลที่ใช้ทำนาย"):
            st.dataframe(input_data.T.rename(columns={0: 'ค่า'}))

else:
    st.error("❌ ไม่สามารถโหลดโมเดลได้ กรุณาลองใหม่อีกครั้ง")