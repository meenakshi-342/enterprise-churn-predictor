import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Enterprise Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)

MODEL_PATH = "models/best_churn_model.pkl"


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        st.error("Model file not found! Please run `train.py` first.")
        return None


model = load_model()

st.title("📊 Enterprise Customer Churn & Revenue Risk Predictor")
st.markdown(
    "Predict customer churn probability and analyze revenue at risk."
)

st.sidebar.header("Customer Profile Settings")

# Input features
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)

phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox(
    "Multiple Lines", ["No phone service", "No", "Yes"]
)
internet_service = st.sidebar.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security", ["No internet service", "No", "Yes"]
)
online_backup = st.sidebar.selectbox(
    "Online Backup", ["No internet service", "No", "Yes"]
)
device_protection = st.sidebar.selectbox(
    "Device Protection", ["No internet service", "No", "Yes"]
)
tech_support = st.sidebar.selectbox(
    "Tech Support", ["No internet service", "No", "Yes"]
)
streaming_tv = st.sidebar.selectbox(
    "Streaming TV", ["No internet service", "No", "Yes"]
)
streaming_movies = st.sidebar.selectbox(
    "Streaming Movies", ["No internet service", "No", "Yes"]
)

contract = st.sidebar.selectbox(
    "Contract Type", ["Month-to-month", "One year", "Two year"]
)
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges ($)", min_value=18.0, max_value=150.0, value=65.0
)
total_charges = st.sidebar.number_input(
    "Total Charges ($)", min_value=18.0, max_value=10000.0, value=780.0
)

# Convert categorical inputs to numeric matching trained model encoding
binary_map = {"No": 0, "Yes": 1, "Female": 0, "Male": 1}
multi_map = {
    "No phone service": 0,
    "No": 1,
    "Yes": 2,
    "No internet service": 0,
    "DSL": 1,
    "Fiber optic": 2,
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2,
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 1,
    "Electronic check": 2,
    "Mailed check": 3,
}

input_data = pd.DataFrame(
    [
        {
            "gender": binary_map[gender],
            "SeniorCitizen": binary_map[senior_citizen],
            "Partner": binary_map[partner],
            "Dependents": binary_map[dependents],
            "tenure": tenure,
            "PhoneService": binary_map[phone_service],
            "MultipleLines": multi_map[multiple_lines],
            "InternetService": multi_map[internet_service],
            "OnlineSecurity": multi_map[online_security],
            "OnlineBackup": multi_map[online_backup],
            "DeviceProtection": multi_map[device_protection],
            "TechSupport": multi_map[tech_support],
            "StreamingTV": multi_map[streaming_tv],
            "StreamingMovies": multi_map[streaming_movies],
            "Contract": multi_map[contract],
            "PaperlessBilling": binary_map[paperless_billing],
            "PaymentMethod": multi_map[payment_method],
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
    ]
)

col1, col2 = st.columns(2)

if st.button("🔍 Predict Churn Risk"):
    if model is not None:
        churn_prob = model.predict_proba(input_data)[0][1]
        churn_pred = model.predict(input_data)[0]

        with col1:
            st.subheader("Prediction Result")
            if churn_pred == 1:
                st.error(
                    f"⚠️ **High Churn Risk!** Probability: {churn_prob * 100:.2f}%"
                )
            else:
                st.success(
                    f"✅ **Low Churn Risk.** Probability: {churn_prob * 100:.2f}%"
                )

        with col2:
            st.subheader("Revenue Risk Analysis")
            annual_risk = monthly_charges * 12
            st.metric(
                label="Potential Annual Revenue at Risk",
                value=f"${annual_risk:,.2f}",
            )
            st.info(
                f"Customer Monthly Revenue: **${monthly_charges:.2f}** | Tenure: **{tenure} months**"
            )