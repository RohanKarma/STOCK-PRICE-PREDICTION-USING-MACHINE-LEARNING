"""
Theme Loader Utility
"""
import streamlit as st
import os


def load_dark_theme():
    """Load dark theme CSS"""
    
    css_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'dark_theme.css')
    
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # Fallback inline CSS
        st.markdown("""
        <style>
        .stApp { 
            background: linear-gradient(to bottom, #0f172a, #1e293b, #0f172a); 
            color: #e2e8f0; 
        }
        [data-testid="stSidebar"] { 
            background: linear-gradient(to bottom, #1e293b, #0f172a); 
        }
        h1, h2, h3, h4, h5, h6, p, span, div { 
            color: #e2e8f0 !important; 
        }
        .stButton > button { 
            background: linear-gradient(135deg, #667eea, #764ba2); 
            color: white !important; 
        }
        </style>
        """, unsafe_allow_html=True)


def apply_page_config(title="StockAI Dashboard", icon="📈"):
    """Apply consistent page config"""
    
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded"
    )