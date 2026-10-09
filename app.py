import streamlit as st
import pandas as pd
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Enterprise Churn Predictor",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern Aesthetic Theme Styling
st.markdown("""
    <style>
    /* Main Background & Base Styling */
    .stApp {
        background-color: #faf5f7;
        color: #2d1e2f;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Bar & Container Spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #f3e8ee;
    }
    
    /* Clean Input Labels & Selectboxes */
    section[data-testid="stSidebar"] label {
        color: #5c3d4f !important;
        font-weight: 600 !important;
    }

    /* Custom Header Container */
    .header-card {
        background: linear-gradient(135deg, #ffffff 0%, #fff5f8 100%);
        border: 1px solid #f9a8d4;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(236, 72, 153, 0.06);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #831843;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .header-subtitle {
        color: #9d174d;
        font-size: 1rem;
        margin-top: 0.4rem;
        font-weight: 400;
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #fbcfe8;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(244, 114, 182, 0.08);
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 700;
        color: #9d174d;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        font-size: 2.4rem;
        font-weight: 800;
        color: #831843;
        margin-top: 0.3rem;
    }

    /* Badges */
    .badge-high {
        background-color: #ffe4e6;
        color: #9f1239;
        border: 1px solid #f43f5e;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-medium {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #f59e0b;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-low {
        background-color: #d1fae5;
        color: #065f46;
        border: 1px solid #10b981;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }

    /* Styling Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #ec4899 0%, #be185d 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 700 !important;
        width: 100%;
        box-shadow: 0 4px 12px rgba(236, 72, 153, 0.25);
    }
    </style>
""", unsafe_allow_html=True)

# 3. Header Section
st.markdown("""
    <div class="header-card">
        <div class="header-title">Enterprise Customer Churn Predictor 🌸</div>
        <div class="header-subtitle">Predict customer retention, analyze risk categories, and estimate revenue exposure.</div>
    </div>
""", unsafe_allow_html=True)

# 4. Sidebar Form Inputs
st.sidebar.markdown("### 📋 Customer Profile")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 💳 Account Details")

tenure = st.sidebar.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# 5. Risk Heuristic Logic
if contract == "Month-to-month":
    churn_prob = min(max((120 - tenure * 1.3 + (monthly_charges / 1.8)) / 100, 0.08), 0.92)
else:
    churn_prob = min(max((40 - tenure * 0.5 + (monthly_charges / 3)) / 100, 0.05), 0.45)

revenue_at_risk = round(monthly_charges * 12, 2)

if churn_prob > 0.6:
    risk_label = "HIGH RISK ⚠️"
    badge_class = "badge-high"
elif churn_prob > 0.3:
    risk_label = "MEDIUM RISK ⚡"
    badge_class = "badge-medium"
else:
    risk_label = "LOW RISK ✨"
    badge_class = "badge-low"

# 6. Metrics Grid
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Churn Probability</div>
            <div class="metric-val">{churn_prob*100:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk Category</div>
            <div style="margin-top: 0.8rem;"><span class="{badge_class}">{risk_label}</span></div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Annual Revenue Exposure</div>
            <div class="metric-val">${revenue_at_risk:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 7. Insights Section
st.markdown("### 📊 Retention Insights & Recommended Strategy")

if "HIGH" in risk_label:
    st.error("⚠️ **High Risk Alert**: This customer has a high likelihood of churning due to month-to-month contract terms and elevated monthly charges.")
    st.markdown("- **Recommended Action**: Offer an immediate **15% discount** on an annual contract upgrade.")
    st.markdown("- **Retention Offer**: Complementary premium customer support for 6 months.")
elif "MEDIUM" in risk_label:
    st.warning("⚡ **Moderate Risk Alert**: The customer profile shows moderate retention stability.")
    st.markdown("- **Recommended Action**: Suggest switching to automated payment methods for a minor billing credit.")
else:
    st.success("✨ **Healthy Account**: Low probability of churn. Customer is highly engaged.")
    st.markdown("- **Recommended Action**: Enroll in cross-selling and loyalty benefit programs.")

st.markdown("---")

# 8. Data Summary
st.markdown("### 📈 Customer Benchmark Comparison")
benchmark_df = pd.DataFrame({
    "Metric": ["Tenure (Months)", "Monthly Charges ($)", "Annual Contract Value ($)"],
    "Current Selected Customer": [tenure, monthly_charges, monthly_charges * 12],
    "Portfolio Average": [32, 64.76, 777.12]
})
st.dataframe(benchmark_df, use_container_width=True)