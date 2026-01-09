"""
Loan Approval Checker - Web App
Predicts if a loan will be approved
Run with: streamlit run app2.py
"""

import pandas as pd
import joblib
import streamlit as st
import os
from log import logger

# Path to the trained model file
MODEL_PATH = "model.joblib"
# Path to the file containing model accuracy
ACCURACY_PATH = "accuracy.txt"
# Path to the training data CSV
TRAIN_DATA_PATH = "train.csv"

# List of feature names used for prediction
FEATURES = ['Married', 'Education', 'ApplicantIncome', 'CoapplicantIncome',
            'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

#  Page Configuration 
st.set_page_config(
    page_title="Loan Approval Checker",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Approval Checker")

logger.info("="*70)
logger.info("LOAN APPROVAL CHECKER APP STARTED")
logger.info("="*70)

#  Load Model 
logger.info(f"Loading model from {MODEL_PATH}...")

if not os.path.exists(MODEL_PATH):
    logger.error(f"Model file not found: {MODEL_PATH}")
    st.info("The system is initializing, please wait.")
    st.error("The model file was not found. Please train the model and save it before running the app.")
    st.stop()

#  Cache Model Loading 
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

model = load_model(MODEL_PATH)
logger.info("Model loaded successfully")

#  Input Form (screenshot-friendly UI) 
with st.form("loan_form"):
    st.subheader("Loan Application Details")

    applicant_income = st.number_input(
        "Applicant Income (monthly)",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income (monthly)",
        min_value=0.0,
        value=0.0,
        step=100.0
    )

    loan_amount = st.number_input(
        "Requested Loan Amount",
        min_value=50000.0,
        max_value=200000.0,
        value=100000.0,
        step=5000.0
    )

    loan_term = st.number_input(
        "Loan Term (months)",
        min_value=1.0,
        value=360.0,
        step=12.0
    )

    credit_history = st.selectbox(
        "Credit History",
        options=[1, 0],
        index=0,
        format_func=lambda x: "Exists (1)" if x == 1 else "Does not exist (0)"
    )

    dict_opt = {"Married": "Married", "Single": "Single"}
    married = st.selectbox(
        "Marital Status",
        options=["Married", "Single"],
        index=0,
        format_func=lambda x: dict_opt[x]
    )

    education = st.selectbox(
        "Education",
        options=["Graduate", "Not Graduate"],
        index=0
    )

    submitted = st.form_submit_button("Check Loan Eligibility")

#  Process Prediction 
if submitted:
    married_val = 1 if married == "Married" else 0
    edu_val = 1 if education == "Graduate" else 0

    total_income = applicant_income + coapplicant_income
    logger.info(f"Loan application - Married: {married}, Education: {education}, Credit: {credit_history}, Income: {total_income}, LoanAmount: {loan_amount}")

    # Check requirements: must be married AND educated AND credit history AND income sufficient
    if married_val == 0:
        logger.warning("Rejected: not married")
        st.error("Sorry, you are not eligible for the loan. Applicants must be married.")
    elif edu_val == 0:
        logger.warning("Rejected: not graduate")
        st.error("Sorry, you are not eligible for the loan. Applicants must be graduates.")
    elif credit_history == 0:
        logger.warning("Rejected: no credit history")
        st.error("Sorry, you are not eligible for the loan. A valid credit history is required.")
    elif applicant_income < 2500.0 and coapplicant_income < 500.0:
        logger.warning("Rejected: insufficient income")
        st.error("Sorry, you are not eligible for the loan. Get at least 2500 monthly income or a coapplicant with at least 500 income or Applicant over 3000.")
    elif applicant_income < 3000.0 and coapplicant_income == 0.0:
        logger.warning("Rejected: applicant income below 3000 with no coapplicant")
        st.error("Sorry, you are not eligible for the loan. Applicant income must be at least 3000 and coapplicant income cannot be negative.")
    else:
        data = pd.DataFrame([{
            "Married": married_val,
            "Education": edu_val,
            "ApplicantIncome": float(applicant_income),
            "CoapplicantIncome": float(coapplicant_income),
            "LoanAmount": float(loan_amount),
            "Loan_Amount_Term": float(loan_term),
            "Credit_History": int(credit_history)
        }])

        try:
            result = model.predict(data)[0]
            if result == 1:
                logger.info(f"APPROVED - Income: {total_income}, LoanAmount: {loan_amount}")
                st.success("Congratulations! You are eligible for the loan.")
            else:
                logger.info(f"REJECTED - Income: {total_income}, LoanAmount: {loan_amount}")
                st.error("Sorry, you are not eligible for the loan.")
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            st.error(f"Error making prediction: {str(e)}")

#  Show Accuracy Button 
if st.button("Show Model Accuracy"):
    logger.info("Accuracy button clicked")
    try:
        if os.path.exists(ACCURACY_PATH):
            with open(ACCURACY_PATH, 'r') as f:
                acc_text = f.read().strip()
                if acc_text:
                    accuracy = float(acc_text)
                    logger.info(f"Model Accuracy: {accuracy:.1%}")
                    st.info(f"Model Accuracy: {accuracy:.1%}")
                else:
                    logger.warning("Accuracy file is empty")
                    st.warning("Accuracy not available - file is empty")
        else:
            logger.warning("Accuracy file not found")
            st.warning("Accuracy not available - please train the model first")
    except ValueError as e:
        logger.error(f"Error reading accuracy: {str(e)}")
        st.error("Error reading accuracy file")


