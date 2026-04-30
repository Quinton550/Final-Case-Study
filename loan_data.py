import streamlit as st
import pickle
import pandas as pd
import sklearn  # required for pickle to load correctly


# ── Load the trained pipeline ──────────────────────────────────────────────────
try:
    with open("loan_approval_.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


# ── Income-level helper (matches training quartiles) ───────────────────────────
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


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;background-color:#e0f2f7;"
    "padding:10px;color:#007bb5;'><b>Loan Approval Prediction</b></h1>",
    unsafe_allow_html=True,
)


st.header("Enter Loan Applicant's Details")


requested_loan_amount  = st.slider("Requested Loan Amount",   min_value=5_000,  max_value=200_000, step=1_000)
fico_score             = st.slider("FICO Score",               min_value=385,    max_value=850,     step=1)
monthly_gross_income   = st.slider("Monthly Gross Income",     min_value=500,    max_value=20_000,  step=100)
monthly_housing_payment= st.slider("Monthly Housing Payment",  min_value=300,    max_value=5_000,   step=50)


ever_bankrupt_or_foreclose = st.selectbox("Ever Bankrupt or Foreclosed?", options=["No", "Yes"])


reason = st.selectbox(
    "Reason for Loan",
    options=[
        'cover_an_unexpected_cost', 'credit_card_refinancing',
        'home_improvement', 'major_purchase', 'debt_conslidation', 'other',
    ],
)
employment_status = st.selectbox(
    "Employment Status",
    options=['full_time', 'part_time', 'unemployed'],
)
employment_sector = st.selectbox(
    "Employment Sector",
    options=[
        'consumer_discretionary', 'information_technology', 'energy', 'unknown',
        'communication_services', 'health_care', 'financials', 'industrials',
        'materials', 'utilities', 'real_estate', 'consumer_staples',
    ],
)
lender = st.selectbox("Preferred Lender", options=['A', 'B', 'C'])


# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("Predict Loan Approval"):


    # Convert Yes/No → 1/0  (model was trained with numeric 0/1)
    ever_bankrupt_int = 1 if ever_bankrupt_or_foreclose == "Yes" else 0


    # Derive Income_Level exactly as it was derived at training time
    income_level = get_income_level(monthly_gross_income)


    # Build input DataFrame with EXACTLY the columns the pipeline expects
    # (must match feature_names_in_ of the ColumnTransformer):
    #   Reason, Requested_Loan_Amount, FICO_score, Employment_Status,
    #   Employment_Sector, Monthly_Gross_Income, Monthly_Housing_Payment,
    #   Ever_Bankrupt_or_Foreclose, Lender, Income_Level
    input_df = pd.DataFrame([{
        "Reason":                    reason,
        "Requested_Loan_Amount":     requested_loan_amount,
        "FICO_score":                fico_score,
        "Employment_Status":         employment_status,
        "Employment_Sector":         employment_sector,
        "Monthly_Gross_Income":      monthly_gross_income,
        "Monthly_Housing_Payment":   monthly_housing_payment,
        "Ever_Bankrupt_or_Foreclose": ever_bankrupt_int,
        "Lender":                    lender,
        "Income_Level":              income_level,
    }])


    # The pipeline handles OHE + scaling internally — pass raw data directly
    try:
        prediction       = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]


        if prediction == 1:
            st.success(f"Prediction: **Approved!** 💸  (Probability: {prediction_proba:.2f})")
        else:
            st.error(f"Prediction: **Denied.** 🚫  (Probability: {prediction_proba:.2f})")


    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Debug — columns sent to model:", list(input_df.columns))
        st.write(input_df)
