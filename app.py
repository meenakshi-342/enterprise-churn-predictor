import streamlit as st
import pandas as pd
import numpy as np
from data_loader import load_data

# 1. Page Configuration
st.set_page_config(
    page_title="Enterprise Churn Predictor",
    page_icon="💖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Pink & Rose Gold Aesthetic Styling
st.markdown("""
    <style>
    /* Global Styling */
    .stApp {
        background: linear-gradient(135deg, #fff0f5 0%, #ffe4e1 50%, #fbcfe8 100%);
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
        color: #4a154b;
    }
    
    /* Header Section */
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(120deg, #ec4899 0%, #db2777 50%, #831843 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        text-shadow: 0px 4px 12px rgba(236, 72, 153, 0.15);
    }
    .header-subtitle {
        color: #9d174d;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 1.8rem;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.75) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(251, 207, 232, 0.8);
    }
    
    /* Custom Card Containers */
    .card-box {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border: 1px solid #fbcfe8;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(244, 114, 182, 0.15);
        text-align: center;
        transition: transform 0.2s ease;
    }
    .card-box:hover {
        transform: translateY(-3px);
    }
    
    .metric-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #9d174d;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #831843;
        margin-top: 0.3rem;
    }

    /* Risk Badges */
    .badge-high {
        background: linear-gradient(135deg, #fecdd3 0%, #fda4af 100%);
        color: #9f1239;
        border: 1px solid #f43f5e;
        padding: 0.45rem 1.2rem;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 4px 10px rgba(244, 63, 94, 0.2);
    }
    .badge-medium {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
        border: 1px solid #f59e0b;
        padding: 0.45rem 1.2rem;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 4px 10px rgba(245, 158, 11, 0.2);
    }
    .badge-low {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        border: 1px solid #10b981;
        padding: 0.45rem 1.2rem;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.2);
    }

    /* Primary Predict Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #ec4899 0%, #db2777 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-size: 1.05rem;
        font-weight: 700;
        box-shadow: 0 6px 18px rgba(236, 72, 153, 0.35);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #db2777 0%, #be185d 100%);
        box-shadow: 0 8px 22px rgba(219, 39, 119, 0.45);
        transform: translateY(-2px);
        color: white;
    }

    /* Clean Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: rgba(255, 255, 255, 0.6);
        border-radius: 10px;
        color: #9d174d;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f472b6 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. App Header
st.markdown('<div class="header-title">Enterprise Customer Churn Predictor ✨</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Analyze customer behavior, predict churn probability, and quantify revenue at risk</div>', unsafe_allow_html=True)

# 4. Sidebar Inputs
st.sidebar.markdown("### 🌸 Customer Profile")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])

st.sidebar.markdown("---")

tenure = st.sidebar.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

predict_clicked = st.sidebar.button("Predict Churn Risk 💖")

# 5. Risk Calculation Heuristic
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

# 6. Dashboard Display
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
        <div class="card-box">
            <div class="metric-title">Churn Probability</div>
            <div class="metric-value">{churn_prob*100:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="card-box">
            <div class="metric-title">Risk Category</div>
            <div style="margin-top: 0.6rem;"><span class="{badge_class}">{risk_label}</span></div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="card-box">
            <div class="metric-title">Annual Revenue Exposure</div>
            <div class="metric-value">${revenue_at_risk:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 7. Analytics Tabs
tab1, tab2 = st.tabs(["📊 Action Plan & Insights", "📈 Profile Comparison"])

with tab1:
    st.markdown("#### Retention Strategy")
    if "HIGH" in risk_label:
        st.error("⚠️ **High Risk Alert**: Customer is on a short-term plan with high monthly charges.")
        st.markdown("- **Action Needed**: Offer a **15% discount** if upgraded to a 1-Year contract.")
        st.markdown("- **Retention Offer**: Provide free account add-ons for the next 3 months.")
    elif "MEDIUM" in risk_label:
        st.warning("⚡ **Moderate Risk**: Customer engagement is steady, but sensitivity to price is moderate.")
        st.markdown("- **Action Needed**: Recommend switching to automatic credit card billing.")
    else:
        st.success("✨ **Healthy Account**: Low probability of churn.")
        st.markdown("- **Action Needed**: Target for loyalty reward promotions or upselling.")

with tab2:
    st.markdown("#### Benchmark Metrics")
    benchmark_df = pd.DataFrame({
        "Metric": ["Tenure (Months)", "Monthly Charges ($)", "Annual Contract Value ($)"],
        "Selected Customer": [tenure, monthly_charges, monthly_charges * 12],
        "Portfolio Average": [32, 64.76, 777.12]
    })
    st.dataframe(benchmark_df, use_container_width=True)