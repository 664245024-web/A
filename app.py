import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

st.set_page_config(
    page_title="Loan Prediction - SVM",
    page_icon="💰",
    layout="wide"
)

# ตรวจสอบว่ามีไฟล์ CSV หรือไม่
@st.cache_data
def load_training_data():
    """โหลดข้อมูลสำหรับฝึกโมเดล"""
    if os.path.exists('loan_data.csv'):
        return pd.read_csv('loan_data.csv')
    return None

def get_model_hash(model_path):
    """สร้าง hash ของโมเดลเพื่อตรวจสอบความถูกต้อง"""
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    return None

def create_model():
    """สร้างและฝึกโมเดล SVM"""
    df = load_training_data()
    
    if df is None:
        st.error("❌ ไม่พบไฟล์ loan_data.csv")
        return None
    
    # เตรียมข้อมูล
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    # แยกประเภท features
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
        ])
    
    # สร้าง SVM model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(
            kernel='rbf',
            C=1.0,
            probability=True,
            random_state=42,
            max_iter=-1
        ))
    ])
    
    # แบ่งข้อมูลและฝึกโมเดล
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model.fit(X_train, y_train)
    
    # แสดงผลการประเมิน
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    return model, train_score, test_score

# Sidebar
st.sidebar.title("💰 Loan Prediction App")
st.sidebar.markdown("---")
st.sidebar.info("""
**โมเดล:** Support Vector Machine (SVM)  
**Kernel:** RBF  
**Target:** Loan Status (0=Approved, 1=Default)
""")

# Main content
st.title("🏦 ระบบทำนายสถานะการกู้ยืม")
st.markdown("ป้อนข้อมูลผู้กู้ยืมเพื่อทำนายว่าสินเชื่อจะได้รับการอนุมัติหรือไม่")

# โหลดหรือสร้างโมเดล
model_path = 'svm_loan_model.pkl'
model = None

col1, col2 = st.columns(2)

with col1:
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            st.success("✅ โหลดโมเดลจากไฟล์สำเร็จ")
            st.info(f"📁 ไฟล์โมเดล: {model_path}")
        except Exception as e:
            st.warning(f"⚠️ ไม่สามารถโหลดโมเดลได้: {type(e).__name__}")
            st.info("🔄 กำลังฝึกโมเดลใหม่...")
    else:
        st.info("📝 ไม่พบไฟล์โมเดล กำลังฝึกโมเดลใหม่...")

with col2:
    if model is None:
        with st.spinner("🔧 กำลังฝึกโมเดล..."):
            result = create_model()
            if result:
                model, train_score, test_score = result
                
                # บันทึกโมเดล
                joblib.dump(model, model_path)
                st.success("✅ ฝึกโมเดลและบันทึกเรียบร้อยแล้ว!")
                
                # แสดงผลการประเมิน
                st.metric("Training Accuracy", f"{train_score:.2%}")
                st.metric("Test Accuracy", f"{test_score:.2%}")
    else:
        st.success("✓ พร้อมใช้งาน")

# แสดงฟอร์มเฉพาะเมื่อมีโมเดล
if model is not None:
    st.markdown("---")
    st.header(" ข้อมูลผู้กู้ยืม")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        person_age = st.number_input("อายุ (Age)", min_value=18, max_value=100, value=25)
        person_gender = st.selectbox("เพศ (Gender)", ["male", "female"])
        person_education = st.selectbox(
            "ระดับการศึกษา (Education)",
            ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
        )
        person_income = st.number_input("รายได้ต่อปี (Income)", min_value=0, value=50000, step=1000)
    
    with col2:
        person_emp_exp = st.number_input("ปีประสบการณ์ทำงาน", min_value=0, max_value=50, value=3)
        person_home_ownership = st.selectbox("สถานะที่อยู่อาศัย", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        loan_amnt = st.number_input("จำนวนเงินกู้", min_value=0, value=10000, step=500)
        loan_intent = st.selectbox(
            "วัตถุประสงค์การกู้",
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
        )
    
    with col3:
        loan_int_rate = st.number_input("อัตราดอกเบี้ย (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.1)
        loan_percent_income = st.number_input("สัดส่วนเงินกู้ต่อรายได้", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        cb_person_cred_hist_length = st.number_input("ปีประวัติเครดิต", min_value=0, max_value=50, value=5)
        credit_score = st.number_input("คะแนนเครดิต", min_value=300, max_value=850, value=650, step=10)
        previous_loan_defaults = st.selectbox("เคยผิดนัดชำระหนี้", ["No", "Yes"])
    
    # ปุ่มทำนาย
    st.markdown("---")
    if st.button("🔮 ทำนายผล", type="primary", use_container_width=True):
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
        
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            st.markdown("---")
            st.header("📊 ผลการทำนาย")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 0:
                    st.success("✅ **สินเชื่อได้รับการอนุมัติ**")
                    st.metric("ความน่าจะเป็นที่จะอนุมัติ", f"{probability[0]*100:.2f}%")
                else:
                    st.error("❌ **สินเชื่อไม่ได้รับการอนุมัติ / มีแนวโน้มผิดนัด**")
                    st.metric("ความน่าจะเป็นที่จะผิดนัด", f"{probability[1]*100:.2f}%")
            
            with col2:
                st.subheader(" ความน่าจะเป็น")
                prob_df = pd.DataFrame({
                    'สถานะ': ['Approved', 'Default'],
                    'ความน่าจะเป็น (%)': [probability[0]*100, probability[1]*100]
                })
                st.bar_chart(prob_df.set_index('สถานะ'))
                
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดในการทำนาย: {e}")
            st.info("💡 ลองฝึกโมเดลใหม่อีกครั้ง")

else:
    st.warning("️ ไม่มีโมเดลพร้อมใช้งาน กรุณาตรวจสอบไฟล์ loan_data.csv")