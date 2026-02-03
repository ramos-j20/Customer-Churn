import streamlit as st
import requests
import os
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

SERVING_URL = os.getenv("SERVING_API_URL", "http://serving:8000")

# ==========================================
# CUSTOM CSS - Sleek Neutral Palette
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Root Variables */
    :root {
        --bg-primary: #0a0a0a;
        --bg-secondary: #171717;
        --bg-tertiary: #262626;
        --border-color: #404040;
        --text-primary: #fafafa;
        --text-secondary: #a3a3a3;
        --text-muted: #737373;
        --accent: #e5e5e5;
        --success: #22c55e;
        --warning: #eab308;
        --danger: #ef4444;
    }
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: var(--bg-primary);
    }
    
    /* Hide Streamlit defaults */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Remove default padding/margins */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0 !important;
        max-width: 1400px !important;
    }
    
    /* Header Section */
    .header-section {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.5rem 0;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 2rem;
    }
    
    .header-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .header-subtitle {
        font-size: 0.875rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }
    
    .status-container {
        display: flex;
        gap: 0.75rem;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        padding: 0.375rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .status-online {
        background: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .status-offline {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Main Container */
    .main-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
        min-height: calc(100vh - 200px);
    }
    
    /* Card Styling */
    .card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 0.75rem;
        padding: 1.5rem;
        height: 100%;
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1.25rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    /* Section Label */
    .section-label {
        font-size: 0.6875rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1.25rem 0 0.75rem 0;
    }
    
    /* Form Inputs */
    .stSelectbox > div > div, .stNumberInput > div > div {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 0.5rem !important;
    }
    
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 0.8125rem !important;
    }
    
    /* Slider Track */
    .stSlider > div > div > div {
        background: var(--bg-tertiary) !important;
    }
    
    /* Button */
    .stButton > button {
        background: var(--text-primary) !important;
        color: var(--bg-primary) !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        border-radius: 0.5rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background: var(--text-secondary) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-tertiary);
        padding: 0.25rem;
        border-radius: 0.5rem;
        border: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 0.375rem;
        color: var(--text-muted);
        font-weight: 500;
        font-size: 0.8125rem;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }
    
    /* Result Card */
    .result-placeholder {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        min-height: 400px;
        color: var(--text-muted);
        text-align: center;
    }
    
    .result-placeholder-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    
    .result-placeholder-text {
        font-size: 0.875rem;
        max-width: 250px;
        line-height: 1.5;
    }
    
    /* Risk Display */
    .risk-result {
        text-align: center;
        padding: 1.5rem;
    }
    
    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1.25rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .risk-high {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .risk-medium {
        background: rgba(234, 179, 8, 0.15);
        color: #eab308;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }
    
    .risk-low {
        background: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    /* Metrics Row */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 1.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--border-color);
    }
    
    .metric-item {
        text-align: center;
        padding: 0.75rem;
        background: var(--bg-tertiary);
        border-radius: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .metric-label {
        font-size: 0.6875rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-tertiary) !important;
        border-radius: 0.5rem !important;
        border: 1px solid var(--border-color) !important;
        font-size: 0.8125rem !important;
    }
    
    /* Remove extra spacing */
    .stMarkdown { margin-bottom: 0 !important; }
    
    div[data-testid="column"] > div { gap: 0.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def create_gauge_chart(probability: float, risk_level: str):
    """Create a minimal gauge chart."""
    colors = {"Low": "#22c55e", "Medium": "#eab308", "High": "#ef4444"}
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': '%', 'font': {'size': 48, 'color': '#fafafa'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#404040', 'tickwidth': 1, 'tickfont': {'color': '#737373'}},
            'bar': {'color': colors.get(risk_level, "#a3a3a3"), 'thickness': 0.75},
            'bgcolor': '#262626',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(34, 197, 94, 0.1)'},
                {'range': [30, 60], 'color': 'rgba(234, 179, 8, 0.1)'},
                {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.1)'}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#fafafa'},
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def check_api_status():
    """Check API health status."""
    try:
        health = requests.get(f"{SERVING_URL}/health", timeout=2)
        if health.status_code == 200:
            data = health.json()
            return True, data.get('model_loaded', False)
        return False, False
    except:
        return False, False

# ==========================================
# HEADER
# ==========================================
api_online, model_loaded = check_api_status()

st.markdown(f"""
<div class="header-section">
    <div>
        <h1 class="header-title">📊 Customer Churn Prediction</h1>
        <p class="header-subtitle">Machine Learning powered customer risk assessment</p>
    </div>
    <div class="status-container">
        <span class="status-badge {'status-online' if api_online else 'status-offline'}">
            {'●' if api_online else '○'} API {'Online' if api_online else 'Offline'}
        </span>
        <span class="status-badge {'status-online' if model_loaded else 'status-offline'}">
            {'●' if model_loaded else '○'} Model {'Ready' if model_loaded else 'Not Loaded'}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# MAIN LAYOUT - Two Column Grid
# ==========================================
col_left, col_right = st.columns(2, gap="large")

# Store form values in session state for display after prediction
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

with col_left:
    st.markdown('<div class="card-header">📋 Customer Profile</div>', unsafe_allow_html=True)
    
    tab_demo, tab_account, tab_services = st.tabs(["Demographics", "Account", "Services"])
    
    with tab_demo:
        st.markdown('<div class="section-label">Personal Information</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            partner = st.selectbox("Partner", ["Yes", "No"])
        with col2:
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            dependents = st.selectbox("Dependents", ["Yes", "No"])
    
    with tab_account:
        st.markdown('<div class="section-label">Subscription</div>', unsafe_allow_html=True)
        tenure = st.slider("Tenure (Months)", 0, 72, 12, help="Time as customer")
        
        col1, col2 = st.columns(2)
        with col1:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        with col2:
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", 
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
        
        st.markdown('<div class="section-label">Billing</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            monthly_charges = st.number_input("Monthly ($)", min_value=0.0, value=70.0)
        with col2:
            total_charges = st.number_input("Total ($)", min_value=0.0, value=840.0)
    
    with tab_services:
        st.markdown('<div class="section-label">Internet Service</div>', unsafe_allow_html=True)
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        if internet != "No":
            st.markdown('<div class="section-label">Add-ons</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                online_security = st.selectbox("Online Security", ["No", "Yes"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes"])
            with col2:
                tech_support = st.selectbox("Tech Support", ["No", "Yes"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
        else:
            online_security = online_backup = device_protection = "No internet service"
            tech_support = streaming_tv = streaming_movies = "No internet service"
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Predict Churn Risk", use_container_width=True)

with col_right:
    st.markdown('<div class="card-header">📈 Prediction Results</div>', unsafe_allow_html=True)
    
    if predict_btn:
        payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies
        }
        
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(f"{SERVING_URL}/predict", json=payload, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    prob = result['churn_probability']
                    risk = result['risk_level']
                    
                    st.session_state.last_prediction = {
                        'prob': prob, 'risk': risk, 
                        'tenure': tenure, 'monthly': monthly_charges, 'contract': contract
                    }
                else:
                    st.error(f"Prediction failed: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Display results
    if st.session_state.last_prediction:
        pred = st.session_state.last_prediction
        prob, risk = pred['prob'], pred['risk']
        
        # Gauge Chart
        st.plotly_chart(create_gauge_chart(prob, risk), use_container_width=True)
        
        # Risk Badge
        risk_class = f"risk-{risk.lower()}"
        risk_icon = "🚨" if risk == "High" else "⚠️" if risk == "Medium" else "✓"
        st.markdown(f"""
        <div style="text-align: center;">
            <span class="risk-badge {risk_class}">{risk_icon} {risk.upper()} RISK</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        st.markdown(f"""
        <div class="metrics-grid">
            <div class="metric-item">
                <div class="metric-value">{pred['tenure']}mo</div>
                <div class="metric-label">Tenure</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">${pred['monthly']:.0f}</div>
                <div class="metric-label">Monthly</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{pred['contract'].split()[0]}</div>
                <div class="metric-label">Contract</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Placeholder
        st.markdown("""
        <div class="result-placeholder">
            <div class="result-placeholder-icon">📊</div>
            <div class="result-placeholder-text">
                Fill out the customer profile and click <strong>Predict Churn Risk</strong> to see results
            </div>
        </div>
        """, unsafe_allow_html=True)

# Bottom spacer for scroll room
st.markdown("<div style='height: 200px;'></div>", unsafe_allow_html=True)
