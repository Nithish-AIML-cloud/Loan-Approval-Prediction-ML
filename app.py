import streamlit as st
import pandas as pd
import numpy as np
import joblib
model = joblib.load('model.pkl')
st.set_page_config(page_title="Loan Approval predictor"
st.title("Loan Approval perdiction App")                   


st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")

st.title("🏦 Loan Approval Prediction App")
st.markdown("### கடன் ஒப்புதல் கணிப்பு | Fill the details below 👇")
st.markdown("---")

# --- Input fields ---
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("👤 Gender (பாலினம்)", ["Male", "Female"])
    married = st.selectbox("💍 Married (திருமணமானவரா?)", ["Yes", "No"])
    dependents = st.selectbox("👨‍👩‍👧 Dependents (சார்ந்தவர்கள்)", ["0", "1", "2", "3+"])
    education = st.selectbox("🎓 Education (கல்வி)", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("💼 Self Employed (சுயதொழில்)", ["Yes", "No"])
    property_area = st.selectbox("🏡 Property Area (சொத்து பகுதி)", ["Urban", "Semiurban", "Rural"])

with col2:
    applicant_income = st.number_input("💰 Applicant Income (வருமானம்)", min_value=0, value=5000)
    coapplicant_income = st.number_input("💵 Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("🏷️ Loan Amount (in thousands)", min_value=0, value=150)
    loan_amount_term = st.number_input("📅 Loan Amount Term (days)", min_value=0, value=360)
    credit_history = st.selectbox("📊 Credit History (கடன் வரலாறு)", ["Good (1.0)", "Bad (0.0)"])

st.markdown("---")

# --- Predict button ---
if st.button("🔍 Predict Loan Status", use_container_width=True):

    # Inputs-ஐ model expect பண்ற encoding-க்கு convert பண்றோம்
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Yes" else 0
    dependents_val = 3 if dependents == "3+" else int(dependents)
    education_val = 1 if education == "Graduate" else 0
    self_employed_val = 1 if self_employed == "Yes" else 0
    credit_history_val = 1.0 if credit_history == "Good (1.0)" else 0.0
    property_area_val = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

    # Final input array - உங்க training column order-ஐ பொறுத்து இதை மாற்றிக்கோங்க
    input_data = pd.DataFrame([[
        gender_val, married_val, dependents_val, education_val, self_employed_val,
        applicant_income, coapplicant_income, loan_amount, loan_amount_term,
        credit_history_val, property_area_val
    ]], columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                 'Loan_Amount_Term', 'Credit_History', 'Property_Area'])

    # Prediction & probability
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.markdown("---")
    if prediction == 1:
        st.success(f"✅ Loan Approved! | கடன் ஒப்புதல் கிடைத்தது! 🎉")
        st.metric("Approval Probability (ஒப்புதல் வாய்ப்பு)", f"{probability[1]*100:.2f}%")
        st.balloons()
    else:
        st.error(f"❌ Loan Rejected | கடன் நிராகரிக்கப்பட்டது 😔")
        st.metric("Rejection Probability (நிராகரிப்பு வாய்ப்பு)", f"{probability[0]*100:.2f}%")

st.markdown("---")
st.caption("Built with ❤️ using RandomForest | Made by Nithish")
