import streamlit as st
import pickle
import pandas as pd
import numpy as np

try:
    with open("loan_approval_.pkl", "rb") as file:
        pipeline_components = pickle.load(file)
    model = pipeline_components['model']
    preprocessor = pipeline_components['preprocessor']
    scaler = pipeline_components['scaler']
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

Q1_INCOME = 3659.0
Q2_INCOME = 5153.5
Q3_INCOME = 7612.0

def get_income_level(income):
    if income <= Q1_INCOME:
        return 'Q1_low'
    elif income <= Q2_INCOME:
        return 'Q2_medium_low'
    elif income <= Q3_INCOME:
        return 'Q3_medium_high'
    else:
        return 'Q4_high'

st.markdown(
   "<h1 style='text-align: center; padding: 12px; color: #333333; font-family: Arial, sans-serif; font-size: 28px;'>Loan Approval Prediction</h1>",
    unsafe_allow_html=True
)

st.header("Enter Loan Applicant's Details")

requested_loan_amount = st.slider("Requested Loan Amount", min_value=5000, max_value=200000, step=1000)
fico_score = st.slider("FICO Score", min_value=385, max_value=850, step=1)
monthly_gross_income = st.slider("Monthly Gross Income", min_value=500, max_value=20000, step=100)
monthly_housing_payment = st.slider("Monthly Housing Payment", min_value=300, max_value=5000, step=50)

ever_bankrupt_or_foreclose = st.selectbox("Ever Bankrupt or Foreclosed?", options=["Yes", "No"])

reason = st.selectbox(
    "Reason for Loan",
    options=['cover_an_unexpected_cost', 'credit_card_refinancing', 'home_improvement', 'major_purchase', 'debt_conslidation', 'other']
)
employment_status = st.selectbox(
    "Employment Status",
    options=['full_time', 'part_time', 'unemployed']
)
employment_sector = st.selectbox(
    "Employment Sector",
    options=['consumer_discretionary', 'information_technology', 'energy', 'unknown', 'communication_services', 'health_care', 'financials', 'industrials', 'materials', 'utilities', 'real_estate', 'consumer_staples']
)
lender = st.selectbox(
    "Preferred Lender",
    options=['A', 'B', 'C']
)

income_level = get_income_level(monthly_gross_income)

input_data = pd.DataFrame({
    "Reason": [reason],
    "Requested_Loan_Amount": [requested_loan_amount],
    "FICO_score": [fico_score],
    "Employment_Status": [employment_status],
    "Employment_Sector": [employment_sector],
    "Monthly_Gross_Income": [monthly_gross_income],
    "Monthly_Housing_Payment": [monthly_housing_payment],
    "Ever_Bankrupt_or_Foreclose": [ever_bankrupt_or_foreclose],
    "Lender": [lender],
    "Income_Level": [income_level]
})

if st.button("Predict Loan Approval"):
    try:
        input_preprocessed = preprocessor.transform(input_data)
        input_scaled = scaler.transform(input_preprocessed)

        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.success(f"Prediction: **Approved!**(Probability: {prediction_proba:.2f})")
        else:
            st.error(f"Prediction: **Denied.** (Probability: {prediction_proba:.2f})")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
