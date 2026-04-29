
# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import sklearn  # This is needed for the pickle file to load!

# Load the trained model
# --- Put the Model in Drive First---
with open("/content/drive/MyDrive/Colab Notebooks/logistic_regression_final_model.pkl", "rb") as file:
    model = pickle.load(file)

# --- Define quantiles for Income_Level based on training data ---
# These values are derived from df_clean['Monthly_Gross_Income'].quantile([0.25, 0.5, 0.75]) from the notebook
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

# Title for the app
st.markdown(
    "<h1 style='text-align: center; background-color: #e0f2f7; padding: 10px; color: #007bb5;'><b>Loan Approval Prediction</b></h1>",
    unsafe_allow_html=True
)

# Numeric inputs
st.header("Enter Loan Applicant's Details")

# Input fields for numerical features (ranges based on df_clean.describe() after outlier removal)
requested_loan_amount = st.slider("Requested Loan Amount", min_value=5000, max_value=200000, step=1000)
fico_score = st.slider("FICO Score", min_value=385, max_value=850, step=1)
monthly_gross_income = st.slider("Monthly Gross Income", min_value=500, max_value=20000, step=100)
monthly_housing_payment = st.slider("Monthly Housing Payment", min_value=300, max_value=5000, step=50)

# Input field for Ever_Bankrupt_or_Foreclose (0 or 1)
ever_bankrupt_or_foreclose = st.selectbox("Ever Bankrupt or Foreclosed?", options=[0, 1])

# Categorical inputs with options (from df_clean.unique())
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

# Derive Income_Level
income_level = get_income_level(monthly_gross_income)

# Create the input data as a DataFrame
# Ensure column names match those dropped from df_model before preprocessing
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

# --- Prepare Data for Prediction ---
# 1. One-hot encode the user's input.
# The categorical columns must be explicitly listed for pd.get_dummies to function correctly
categorical_cols_for_dummies = [
    'Reason', 'Employment_Status', 'Employment_Sector', 'Lender', 'Income_Level'
]
input_data_encoded = pd.get_dummies(input_data, columns=categorical_cols_for_dummies)

# 2. Add any "missing" columns the model expects (fill with 0) and reorder.
# The model.feature_names_in_ contains the column names after preprocessing.
model_columns = model.feature_names_in_
final_input_data = pd.DataFrame(0, index=[0], columns=model_columns)

# Populate the final_input_data with user's encoded input
for col in input_data_encoded.columns:
    if col in final_input_data.columns:
        final_input_data[col] = input_data_encoded[col]


# Predict button
if st.button("Predict Loan Approval"): # Changed button text
    # Predict using the loaded model
    prediction = model.predict(final_input_data)[0]
    prediction_proba = model.predict_proba(final_input_data)[0][1] # Probability of approval

    # Display result (corrected message as 1 means Approved)
    if prediction == 1:
        st.success(f"Prediction: **Approved!** \ud83d\udcb8 (Probability: {prediction_proba:.2f})") # Green for success
    else:
        st.error(f"Prediction: **Denied.** \ud83d\udeab (Probability: {prediction_proba:.2f})") # Red for denial
