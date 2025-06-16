import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from auth.auth_manager import create_user_table, signup_user, login_user, get_username_by_email
import utils.helper as helper
from utils.helper import get_questions, retrain_model_with_feedback, append_to_insomnia_data, load_model
from utils.recommender import recommend_song_from_dataset, load_lullaby_dataset
from admin.admin_panel import show_admin_panel
from utils.feedback import save_feedback

# --- Initialize DB ---
create_user_table()

# --- Set Page Config ---
st.set_page_config(
    page_title="Restify | Sleep Wellness Platform", 
    page_icon="🌙", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS Styling with Glassmorphism and Advanced Animations
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --glass-bg: rgba(255, 255, 255, 0.1);
        --glass-border: rgba(255, 255, 255, 0.2);
        --text-primary: #ffffff;
        --text-secondary: #e2e8f0;
        --shadow-lg: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        --shadow-xl: 0 35px 60px -12px rgba(0, 0, 0, 0.35);
    }
    
    /* Reset and Base Styles */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 25%, #16213e 50%, #0f0f23 100%);
        background-attachment: fixed;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
        min-height: 100vh;
    }
    
    /* Animated Background Elements */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 50%, rgba(102, 126, 234, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(118, 75, 162, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 80%, rgba(79, 172, 254, 0.1) 0%, transparent 50%);
        animation: backgroundFloat 20s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes backgroundFloat {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        33% { transform: translateY(-20px) rotate(1deg); }
        66% { transform: translateY(20px) rotate(-1deg); }
    }
    
    /* Container Styles */
    .block-container {
        padding: 2rem 3rem !important;
        max-width: 1200px !important;
        margin: 0 auto !important;
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid var(--glass-border);
        box-shadow: var(--shadow-xl);
        transition: all 0.3s ease;
    }
    
    .block-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 40px 80px -12px rgba(0, 0, 0, 0.4);
    }
    
    /* Typography */
    h1, h2, h3, h4 {
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    h1 {
        font-size: 3rem;
        font-weight: 800;
        text-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        animation: titleGlow 3s ease-in-out infinite alternate;
    }
    
    @keyframes titleGlow {
        from { text-shadow: 0 4px 20px rgba(102, 126, 234, 0.3); }
        to { text-shadow: 0 4px 30px rgba(118, 75, 162, 0.5); }
    }
    
    /* Card Components */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid var(--glass-border);
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: var(--shadow-lg);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.5s;
    }
    
    .glass-card:hover::before {
        left: 100%;
    }
    
    .glass-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: var(--shadow-xl);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    /* Button Styles */
    .stButton > button {
        background: var(--primary-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 1rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3) !important;
        position: relative !important;
        overflow: hidden !important;
        z-index: 1 !important;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: var(--secondary-gradient);
        transition: left 0.3s ease;
        z-index: -1;
    }
    
    .stButton > button:hover::before {
        left: 0;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.02) !important;
    }
    
    /* Input Styles */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: rgba(102, 126, 234, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
        outline: none !important;
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: rgba(15, 15, 35, 0.9) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 2rem 1rem !important;
    }
    
    .css-1d391kg .css-1v3fvcr {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        margin: 0.5rem 0 !important;
        transition: all 0.3s ease !important;
    }
    
    .css-1d391kg .css-1v3fvcr:hover {
        background: rgba(102, 126, 234, 0.2) !important;
        border-color: rgba(102, 126, 234, 0.5) !important;
        transform: translateX(5px) !important;
    }
    
    /* Form Styles */
    .stForm {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(15px) !important;
        border-radius: 20px !important;
        border: 1px solid var(--glass-border) !important;
        padding: 2rem !important;
        box-shadow: var(--shadow-lg) !important;
    }
    
    /* Radio Button Styles */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Selectbox Styles */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Multiselect Styles */
    .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Success/Error Message Styles */
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Chart Styles */
    .stPlotlyChart {
        background: var(--glass-bg) !important;
        border-radius: 20px !important;
        padding: 1rem !important;
        backdrop-filter: blur(15px) !important;
        border: 1px solid var(--glass-border) !important;
    }
    
    /* Metric Styles */
    .metric-card {
        background: var(--glass-bg);
        backdrop-filter: blur(15px);
        border-radius: 16px;
        border: 1px solid var(--glass-border);
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-lg);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-xl);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 1rem;
        color: var(--text-secondary);
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    /* Song Card Styles */
    .song-card {
        background: var(--glass-bg);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid var(--glass-border);
        padding: 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow-lg);
        text-align: center;
        overflow: hidden;
        position: relative;
    }
    
    .song-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        animation: rotate 4s linear infinite;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .song-card:hover::before {
        opacity: 1;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .song-card:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 0 25px 50px rgba(102, 126, 234, 0.3);
    }
    
    .song-card img {
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    .song-card:hover img {
        transform: scale(1.1);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);
    }
    
    /* Spotify Button */
    .spotify-btn {
        background: linear-gradient(135deg, #1db954 0%, #1ed760 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border: none;
        border-radius: 25px;
        font-weight: 600;
        text-decoration: none;
        display: inline-block;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(29, 185, 84, 0.3);
        margin: 0.5rem;
    }
    
    .spotify-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(29, 185, 84, 0.4);
        text-decoration: none;
        color: white;
    }
    
    /* Questionnaire Scale */
    .scale-indicator {
        background: var(--glass-bg);
        backdrop-filter: blur(15px);
        border-radius: 16px;
        border: 1px solid var(--glass-border);
        padding: 1rem;
        margin: 1rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }
    
    .scale-item {
        text-align: center;
        padding: 0.5rem;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.05);
        min-width: 80px;
        transition: all 0.3s ease;
    }
    
    .scale-item:hover {
        background: rgba(102, 126, 234, 0.2);
        transform: translateY(-2px);
    }
    
    .scale-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .scale-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem !important;
            margin: 0.5rem !important;
            border-radius: 16px;
        }
        
        h1 {
            font-size: 2rem;
        }
        
        .glass-card {
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .scale-indicator {
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .scale-item {
            min-width: 60px;
        }
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary-gradient);
    }
    
    /* Loading Animation */
    .loading-spinner {
        border: 3px solid rgba(255, 255, 255, 0.1);
        border-top: 3px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 2rem auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Floating Action Button */
    .fab {
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        background: var(--primary-gradient);
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        box-shadow: var(--shadow-lg);
        cursor: pointer;
        transition: all 0.3s ease;
        z-index: 1000;
    }
    
    .fab:hover {
        transform: scale(1.1);
        box-shadow: var(--shadow-xl);
    }
    
    /* Pulse Animation for Important Elements */
    .pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: .7;
        }
    }
    
    /* Slide-in Animation */
    .slide-in {
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(-100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    /* Fade-in Animation */
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- Session Initialization ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# --- Authentication ---
if not st.session_state.logged_in:
    st.sidebar.markdown("## 🔐 Authentication")
    option = st.sidebar.radio("Choose Option", ("Login", "Signup"), key="auth_radio")

    if option == "Signup":
        with st.sidebar.form("signup_form"):
            st.markdown("### Create Account")
            username = st.text_input("Username", placeholder="Enter your username")
            email = st.text_input("Email", placeholder="Enter your email")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("Create Account")
            if submit:
                signup_result = signup_user(username, email, password)
                if signup_result:
                    st.success("✅ Account created successfully!")
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("❌ Username already exists.")
    else:
        with st.sidebar.form("login_form"):
            st.markdown("### Welcome Back")
            email = st.text_input("Email", placeholder="Enter your email")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("Sign In")
            if submit:
                login_result = login_user(email, password)
                if login_result:
                    st.session_state.logged_in = True
                    st.session_state.username = get_username_by_email(email) or email
                    st.success(f"✅ Welcome back, {st.session_state.username}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials.")

if st.session_state.logged_in:
    st.sidebar.success(f"👋 Welcome, *{st.session_state.username}*")
    if st.sidebar.button("🚪 Logout", key="unique_logout_button"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()


def save_insomnia_entry(user_input, insomnia_level):
    data_file = "data/insomnia_synthetic.csv"
    user_entries_file = "data/user_entries.csv"
    os.makedirs("data", exist_ok=True)

    # Calculate total score
    total_score = sum(user_input[:-1])  # Exclude age from total score calculation

    # Create new data entry for the main model dataset
    new_data = pd.DataFrame([user_input + [total_score, insomnia_level]],
                            columns=FEATURE_COLS + ["Total Score", "Insomnia Level"])

    # Save/update the main model dataset
    if os.path.exists(data_file):
        print(f"DEBUG: Reading {data_file} in app.py (save_insomnia_entry)")
        existing_df = pd.read_csv(data_file, on_bad_lines='skip')
        combined_df = pd.concat([existing_df, new_data], ignore_index=True)
    else:
        combined_df = new_data

    combined_df.to_csv(data_file, index=False)

    # Save user tracking info separately (WITH Username and Timestamp)
    user_entry = pd.DataFrame([{
        "Insomnia Level": insomnia_level,
        "Total Score": total_score
    }])

    if os.path.exists(user_entries_file):
        user_entries_df = pd.read_csv(user_entries_file, on_bad_lines='skip')
        user_entries_df = pd.concat([user_entries_df, user_entry], ignore_index=True)
    else:
        user_entries_df = user_entry

    user_entries_df.to_csv(user_entries_file, index=False)

    # Ensure columns order matches existing dataset
    columns_order = [
        "Level of Insomnia Intensity", "Sleep Efficiency", "Degree of Depression", "Sleep Wellness Practices",
        "Sleep-Related Negative Thinking", "Anxious Thinking Regarding Sleep", "Stress Level",
        "Coping Skills", "Emotion Regulation", "Age"
        # "Level of Insomnia Intensity", "Sleep Efficiency", "Degree of Depression", "Sleep Wellness Practices",
        # "Sleep-Related Negative Thinking", "Anxious Thinking Regarding Sleep", "Stress Level",
        # "Coping Skills", "Emotion Regulation", "Age", "Insomnia Level", "Username", "Timestamp"
    ]
    new_data = new_data[columns_order]

    if os.path.exists(data_file):
        existing_df = pd.read_csv(data_file, on_bad_lines='skip')
        combined_df = pd.concat([existing_df, new_data], ignore_index=True)
    else:
        combined_df = new_data

    combined_df.to_csv(data_file, index=False, sep=',')

# --- Dashboard ---
from utils.helper import FEATURE_COLS
import utils.helper

def show_dashboard(username):
    st.markdown("# 📊 Sleep Analytics Dashboard")
    
    data_path = "data/insomnia_synthetic_clean.csv"
    if not os.path.exists(data_path):
        st.warning("📂 No data available yet. Complete the questionnaire first!")
        return

    df = pd.read_csv(data_path, on_bad_lines='skip' )
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Total Assessments</div>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        avg_age = df['Age'].mean() if 'Age' in df.columns else 0
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.0f}</div>
            <div class="metric-label">Average Age</div>
        </div>
        """.format(avg_age), unsafe_allow_html=True)
    
    with col3:
        unique_users = df['Username'].nunique() if 'Username' in df.columns else 0
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Active Users</div>
        </div>
        """.format(unique_users), unsafe_allow_html=True)
    
    with col4:
        recent_entries = 0
        if 'Timestamp' in df.columns:
            try:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                recent_entries = len(df[df['Timestamp'] >= (datetime.now() - pd.Timedelta(days=7))])
            except Exception:
                recent_entries = 0
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">This Week</div>
        </div>
        """.format(recent_entries), unsafe_allow_html=True)

    st.markdown("---")
    
    # Charts Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Insomnia Level Distribution")
        if "Insomnia Level" in df.columns:
            chart_data = df["Insomnia Level"].value_counts()
            st.bar_chart(chart_data)
        else:
            st.info("No insomnia level data available")

    # Correlation Heatmap
    st.markdown("### 🔥 Feature Correlation Heatmap")
    try:
        from utils.helper import FEATURE_COLS
        feature_cols = [col for col in FEATURE_COLS if col in df.columns]
        if len(feature_cols) > 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            corr = df[feature_cols].corr(numeric_only=True)
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap( 
                 corr,
                 annot=True,
                 cmap=sns.diverging_palette(220, 20, as_cmap=True),
                 center=0,
                 # mask=mask,  # Remove or comment this line
                 annot_kws={
        "size": 10,
        "weight": "bold",
        "color": "white"
    },
    fmt=".2f",
    linewidths=0.5,
    square=True,
    cbar_kws={
        "shrink": 0.8,
        "label": "Correlation Strength"
    }
)
            # Improve readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)

           
               
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Not enough numeric data for correlation analysis")
    except Exception as e:
        st.error(f"Error generating correlation heatmap: {e}")


# --- Feedback Form ---
def collect_feedback(insomnia_level, song_label, user_input):
    st.markdown("## 🗣️ Share Your Feedback")
    with st.form("feedback_form"):
        st.markdown(f"<div style='font-size:20px; font-weight:700; color:#facc15; margin-bottom:10px;'>🎵 Song Recommended: <span style='color:#e0e0e0;'>{song_label}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:20px; font-weight:700; color:#facc15; margin-bottom:20px;'>🛌 Predicted Insomnia Level: <span style='color:#e0e0e0;'>{insomnia_level}</span></div>", unsafe_allow_html=True)

        container_style = "background: #1f2937; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); max-width: 700px; margin-bottom: 20px; color: #e0e0e0;"
        with st.container():
            st.markdown(f"<div style='{container_style}'>", unsafe_allow_html=True)
            relaxed = st.radio("Did the song help you feel relaxed?", ["Yes", "No"], index=0, horizontal=True)
            sleep = st.radio("Did the song help you fall asleep?", ["Yes", "No"], index=0, horizontal=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown(f"<div style='{container_style}'>", unsafe_allow_html=True)
            sleep_quality = st.selectbox("How was your sleep quality?", ["Poor", "Average", "Good", "Excellent"])
            rating = st.radio("⭐ Rate the song experience", [1, 2, 3, 4, 5], format_func=lambda x: "⭐" * x, horizontal=True)
            st.markdown("</div>", unsafe_allow_html=True)

        comments_style = "background: #1f2937; padding: 15px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); max-width: 700px; margin-bottom: 20px; color: #e0e0e0;"
        comments = st.text_area("Additional comments or suggestions (optional)", height=100, placeholder="Your comments here...", key="feedback_comments")
        st.markdown(f"<div style='{comments_style}'></div>", unsafe_allow_html=True)

        submit_style = "background-color: #fbbf24; color: #1f2937; font-weight: 700; border-radius: 8px; padding: 12px 24px; border: none; box-shadow: 0 4px 12px rgba(251, 191, 36, 0.5); cursor: pointer; transition: background-color 0.3s ease;"
        submit_feedback = st.form_submit_button("Submit Feedback ✅", help="Click to submit your feedback")
        if submit_feedback:
            feedback_entry = {
                "Username": st.session_state.username,
                "Timestamp": datetime.now().isoformat(),
                "Insomnia Level": insomnia_level,
                "Recommended Song": song_label,
                "Felt Relaxed": relaxed == "Yes", 
                "Fell Asleep": sleep == "Yes",
                "Sleep Quality": sleep_quality,
                "Rating": rating,
                "Comments": comments,
            }
            for i, q in enumerate(get_questions()):
                feedback_entry[q] = user_input[i]

            feedback_file = "data/feedback.csv"
            os.makedirs("data", exist_ok=True)
            try:
                pd.DataFrame([feedback_entry]).to_csv(feedback_file, mode='a', header=not os.path.exists(feedback_file), index=False)
                st.session_state.feedback_submitted = True
                st.success("✅ Thank you! Your feedback has been saved.")
                with st.expander("📄 See your submitted feedback"):
                    st.json(feedback_entry)
            except Exception as e:
                st.error(f"Error saving feedback: {e}")

# --- Main App ---
def main():
    
    if not st.session_state.logged_in:
        st.markdown(
            """
            <div style="
                display: flex;
                justify-content: center;
                align-items: center;
                height: calc(80vh - 60px);
                width: 100%;
                margin-top: 100px;
                font-size: 1.5rem;
                font-weight: 600;
                color: #f59e0b;
                background: linear-gradient(to right, #1f2937, #111827);
                border-radius: 12px;
                padding: 2rem 3rem;
                text-align: center;
                box-shadow: 0 4px 12px rgba(251, 191, 36, 0.5);
            ">
                ⚠️ Please log in to access the application features.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    user_is_admin = st.session_state.username.lower() == "admin"
    menu_options = ["Home", "Dashboard", "Feedback"]
    if user_is_admin:
        menu_options.append("Admin Panel")

    page = st.sidebar.selectbox("Navigate", menu_options, index=menu_options.index(st.session_state.get("page", "Home")))
    st.session_state.page = page

    questions_with_help = {
        "Level of Insomnia Intensity": "It refers to how severe or strong a person's sleep difficulty is, ranging from mild to severe",
        "Sleep Efficiency": "It is the percentage of time spent asleep while in bed, showing how well a person sleeps during the time allocated for rest",
        "Degree of Depression": "It refers to how severe or intense a person’s depressive symptoms are, typically ranging from mild to severe",
        "Sleep Wellness Practices": "These are daily habits and routines that help improve sleep quality and overall restfulness",
        "Sleep-Related Negative Thinking": "It refers to pessimistic or fearful thoughts about sleep, such as expecting poor sleep or fearing its consequences",
        "Anxious Thinking Regarding Sleep": "It means worrying or overthinking about not being able to sleep well, which can make falling asleep harder",
        "Stress Level": "Your general stress level recently.",
        "Coping Skills": "How effectively you deal with stressors.",
        "Emotion Regulation": "How well you manage difficult emotions.",
        "Age": "What is the age of the user?"
    }
    questions = list(questions_with_help.keys())

    if page == "Dashboard":
        show_dashboard(st.session_state.username)

    elif page == "Admin Panel":
        show_admin_panel()
        # Removed button to regenerate dataset as per user request

    elif page == "Feedback":
        insomnia_level = st.session_state.get("feedback_insomnia_level")
        song_label = st.session_state.get("feedback_song_label")
        user_input = st.session_state.get("feedback_user_input")

        if "feedback_submitted" in st.session_state:
            del st.session_state["feedback_submitted"]

        if insomnia_level and song_label and user_input:
            collect_feedback(insomnia_level, song_label, user_input)
        else:
            st.info("Please get a song recommendation first from the Home page.")

    elif page == "Home":
        st.markdown("""
        <div style='
            background: linear-gradient(to right, #ffecd2 0%, #fcb69f 100%);
            padding: 1.7rem;
            border-radius: 12px;
            text-align: center;
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            font-size: 1.6rem;
            color: #2c3e50;
            font-weight: 600;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 1000px;
            margin-bottom: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        '>
            💤 Questionnaire to predict the Degree of Insomnia
        </div>
""", unsafe_allow_html=True)

        # st.markdown("### Insomnia Symptoms Assessment")
        questions_with_help = {
            "Insomnia Intensity": "It refers to how severe or strong a person's sleep difficulty is, ranging from mild to severe",
            "Sleep Efficiency": "It is the percentage of time spent asleep while in bed, showing how well a person sleeps during the time allocated for rest",
            "Degree of Depression": "It refers to how severe or intense a person’s depressive symptoms are, typically ranging from mild to severe",
            "Sleep Wellness Practices": "These are daily habits and routines that help improve sleep quality and overall restfulness",
            "Sleep-Related Negative Thinking": "It refers to pessimistic or fearful thoughts about sleep, such as expecting poor sleep or fearing its consequences",
            "Anxious Thinking Regarding Sleep": "It means worrying or overthinking about not being able to sleep well, which can make falling asleep harder",
            "Stress Level": "Your general stress level in past few days.",
            "Coping Skills": "How effectively you deal with stressors.",
            "Emotion Regulation": "How well you manage difficult emotions.",
            "Age": "What is the age of the user?"
        }

        st.markdown(
            """
        <div style="
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 8px 12px; border-radius: 10px; color: white; font-weight: 600;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 15px; box-shadow: 1px 1px 8px rgba(0,0,0,0.12);
            max-width: 900px; margin-bottom: 10px; display: flex; align-items: center; justify-content: flex-start;
            width: 100%;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        ">
            <span style="margin: 0 10px 0 0;">Rate symptoms:</span>
            <div style="display: flex; gap: 18px;">
                <div style="text-align:center;">
                    <div style="font-weight:700;">0</div>
                    <div style="font-size:12px;">Negligible</div>
                </div>
                <div style="font-weight:700;">|</div>
                <div style="text-align:center;">
                    <div style="font-weight:700;">1</div>
                    <div style="font-size:12px;">Mild</div>
                </div>
                <div style="font-weight:700;">|</div>
                <div style="text-align:center;">
                    <div style="font-weight:700;">2</div>
                    <div style="font-size:12px;">Moderate</div>
                </div>
                <div style="font-weight:700;">|</div>
                <div style="text-align:center;">
                    <div style="font-weight:700;">3</div>
                    <div style="font-size:12px;">High</div>
                </div>
                <div style="font-weight:700;">|</div>
                <div style="text-align:center;">
                    <div style="font-weight:700;">4</div>
                    <div style="font-size:12px;">Severe</div>
                </div>
            </div>
        </div>
            """,
            unsafe_allow_html=True,
        )


        value_labels = {
            "Insomnia Intensity": "( 0=Negligible , 1=Mild , 2=Moderate , 3=High , 4=Severe )",
            "Sleep Efficiency": "( 0=Very Poor , 1=Poor , 2=Average , 3=Good , 4=Excellent )",
            "Sleep Wellness Practices": "( 0=Very Poor , 1=Poor , 2=Average , 3=Good , 4=Excellent )",
            "Coping Skills": "( 0=Very Poor , 1=Poor , 2=Average , 3=Good , 4=Excellent )",
            "Emotion Regulation": "( 0=Very Poor, 1=Poor, 2=Average, 3=Good, 4=Excellent )",
            "Stress Level": " ( 0=None, 1=Low , 2=Moderate , 3=High , 4=Very High )",
            "Level of Insomnia Intensity": " ( 0=None, 1=Low , 2=Moderate , 3=High , 4=Very High 0 )",
            "Degree of Depression": "( 0=None , 1=Low , 2=Moderate , 3=High , 4=Very High )",
            "Sleep-Related Negative Thinking": "( 0=None , 1=Low , 2=Moderate , 3=High , 4=Very High )",
            "Anxious Thinking Regarding Sleep": "( 0=None, 1=Low , 2=Moderate , 3=High , 4=Very High )"
        }
        user_input = [
            st.number_input(
                f"**{q}**\n\n_{desc}_\n\n{value_labels.get(q, '')}",
                0.0, 4.0, step=0.1, key=f"input_{i}"
            ) if q != "Age" else st.number_input(
                f"**{q}**\n\n_{desc}_", 18, 100, step=1, key=f"input_{i}", value=25
            )
            for i, (q, desc) in enumerate(questions_with_help.items())
        ]

        num_songs = st.number_input("Number of song recommendations:", 1, 10, 1, key="num_songs")

        # Add category multi-select for lullaby songs
        lullaby_df = load_lullaby_dataset()
        categories = lullaby_df['Category'].dropna().unique().tolist()
        selected_categories = st.multiselect("Select song categories:", options=categories)

        # Display the selected number of songs and categories
        st.markdown(f"### 🎶 Number of Songs Selected: **{num_songs}**")
        if selected_categories:
            selected_cats_str = ", ".join(selected_categories)
            st.markdown(f"### 🎼 Selected Categories: **{selected_cats_str}**")
        else:
            st.markdown("### 🎼 No categories selected")

    if st.button("Predict & Recommend"):
        try:
            model, label_encoder, scaler = load_model()
            input_df = pd.DataFrame([user_input], columns=FEATURE_COLS)
            # Add this check:
            if list(input_df.columns) != FEATURE_COLS:
                st.error("Input columns do not match model features.")
                return
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]
            insomnia_level = label_encoder.inverse_transform([prediction])[0]
            # ... (rest of your code here, properly indented)


            st.success(f"🧠 Predicted Insomnia Level: **{insomnia_level}**")

            save_insomnia_entry(user_input, insomnia_level)
            st.success("📁 Your input has been saved to the dataset.")

            labels, links, thumbnails, categories = recommend_song_from_dataset(insomnia_level, num_songs, selected_categories)
            # Set songs_displayed to True only after songs are rendered
            st.session_state.songs_displayed = True
            

            # Handle negligible or no insomnia message and Spotify link button
            if insomnia_level in ["No Insomnia", "Negligible"] and labels and links:
                st.success(labels[0])  # Display the message
                spotify_link = links[0]
                # Use st.markdown with HTML button for custom styling instead of st.button
                st.markdown(
                    f'<a href="{spotify_link}" target="_blank" style="text-decoration:none;">'
                    f'<button style="background-color:#1db954 !important; color:white; padding:10px 20px; border:none; border-radius:5px; font-weight:bold; cursor:pointer;">'
                    f'▶️ Go to Spotify</button></a>',
                    unsafe_allow_html=True
                )
            else:
                # Group songs by category
                songs_by_category = {}
                for label, link, thumb, category in zip(labels, links, thumbnails, categories):
                    if category not in songs_by_category:
                        songs_by_category[category] = []
                    songs_by_category[category].append((label, link, thumb))

                # Distribute the total number of songs requested evenly across selected categories
                total_songs = num_songs
                num_categories = len(songs_by_category)
                base_count = total_songs // num_categories if num_categories > 0 else 0
                remainder = total_songs % num_categories if num_categories > 0 else 0

                categories_to_show = []
                for i, (category, songs) in enumerate(songs_by_category.items()):
                    count = base_count + (1 if i < remainder else 0)
                    categories_to_show.append((category, songs[:count]))

                # Display songs category-wise with exact distribution
                song_index = 1
                for category, songs in categories_to_show:
                    st.markdown(f"## <span style='color:#6a0dad; font-weight: 700;'>🎼 Category:</span> <span style='color:#6a0dad; font-weight: 700;'>{category}</span>", unsafe_allow_html=True)
                    row_size = 3
                    for row_start in range(0, len(songs), row_size):
                        row_songs = songs[row_start:row_start + row_size]
                        cols = st.columns(row_size)
                        for i, (label, link, thumb) in enumerate(row_songs):
                            with cols[i]:
                                st.markdown(f"### 🎵 Song {song_index}")
                                song_index += 1
                                if thumb:
                                    st.image(thumb, width=200, use_container_width=True)
                                # Display only the song name without artist or BPM
                                song_name_only = label.split(" by ")[0]
                                st.markdown(f"**{song_name_only}**")
                                if link:
                                    st.markdown(
                                        f'<a href="{link}" target="_blank">'
                                        f'<button style="background-color:#28a745; color:white; padding:10px; border:none; border-radius:5px;">'
                                        f'▶️ Listen on Spotify</button></a>',
                                        unsafe_allow_html=True
                                    )
                                    st.session_state.songs_displayed = True

            # Save feedback context
            st.session_state.feedback_insomnia_level = insomnia_level
            st.session_state.feedback_song_label = labels[0] if labels else "N/A"
            st.session_state.feedback_user_input = user_input
            st.session_state.show_feedback_button = True

        except Exception as e:
            st.error(f"Error during prediction or recommendation: {e}")

    # Show Go to Feedback button only after all songs are displayed by clicking Predict & Recommend, for all users including admin
    if st.session_state.get("show_feedback_button", False) and st.session_state.get("songs_displayed", True):
        st.markdown("---")


        feedback_button = st.button("🗣️ Give Feedback", key="feedback_button", help="Click to provide feedback on the recommended songs" )

        if feedback_button:
            st.session_state.show_feedback_button = False
            st.session_state.page = "Feedback"
            st.rerun()


# --- Run App ---
main()
