"""
Dark Theme Loader - Apply to all pages
"""
import streamlit as st

def apply_dark_theme():
    """Apply dark theme CSS to any page"""
    st.markdown("""
    <style>
    /* ================================ */
    /* SIMPLE DARK THEME - NO BREAKING */
    /* ================================ */

    /* Main Background */
    .stApp {
        background: linear-gradient(to bottom, #0f172a, #1e293b, #0f172a);
        color: #e2e8f0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #1e293b, #0f172a);
        border-right: 1px solid rgba(102, 126, 234, 0.3);
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    /* All text */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #e2e8f0 !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.5);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }

    /* Input fields */
    input, select, textarea {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 0.375rem !important;
    }

    /* Dataframes */
    .dataframe {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
    }

    .dataframe th {
        background-color: #334155 !important;
        color: #e2e8f0 !important;
    }

    .dataframe td {
        color: #e2e8f0 !important;
        background-color: #1e293b !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #1e293b;
    }

    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 5px;
    }

    /* Links */
    a {
        color: #667eea !important;
    }

    /* Info boxes */
    .stAlert {
        background-color: rgba(30, 41, 59, 0.8) !important;
        color: #e2e8f0 !important;
        border-radius: 0.5rem !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        color: #e2e8f0;
        border-radius: 4px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
    }
    
    /* Select boxes */
    [data-baseweb="select"] {
        background-color: #1e293b !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)