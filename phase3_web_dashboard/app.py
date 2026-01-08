"""
Stock Prediction Dashboard - Simple Dark Theme
"""
import streamlit as st
import pandas as pd
import os
import sys

# Page config
st.set_page_config(
    page_title="StockAI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def load_css():
    css_file = os.path.join(os.path.dirname(__file__), 'assets', 'dark_theme.css')
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # Inline CSS fallback
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
            color: white;
            border: none;
            border-radius: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)

load_css()

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

# Title
st.title("📈 AI Stock Prediction Dashboard")
st.markdown("Powered by Machine Learning & Deep Learning")

st.markdown("---")

# Live Ticker
try:
    from live_ticker import display_live_ticker
    display_live_ticker(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'])
except Exception as e:
    st.info("💡 Live market data loading...")

st.markdown("---")

# Feature Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔮 Predictions")
    st.write("AI-powered stock price predictions for the next 30 days")
    st.write("- Linear Regression")
    st.write("- Random Forest")
    st.write("- LSTM Neural Network")

with col2:
    st.markdown("### 📊 Analytics")
    st.write("Compare model performance and analyze trends")
    st.write("- Model Comparison")
    st.write("- Performance Metrics")
    st.write("- Accuracy Analysis")

with col3:
    st.markdown("### 💼 Portfolio")
    st.write("Track your stock portfolio and predictions")
    st.write("- 5 Major Stocks")
    st.write("- Real-time Updates")
    st.write("- Risk Analysis")

st.markdown("---")

# Quick Overview
st.markdown("## 📊 Quick Overview")

predictions_dir = '../phase2_ml_models/data/predictions'

if os.path.exists(predictions_dir):
    pred_files = [f for f in os.listdir(predictions_dir) if f.endswith('.csv')]
    
    if pred_files:
        cols = st.columns(5)
        
        for idx, file in enumerate(pred_files[:5]):
            ticker = file.replace('_predictions.csv', '')
            df = pd.read_csv(os.path.join(predictions_dir, file))
            
            current_pred = df.iloc[0]['predicted_price']
            future_pred = df.iloc[-1]['predicted_price']
            change_pct = ((future_pred - current_pred) / current_pred) * 100
            
            with cols[idx]:
                st.metric(
                    label=ticker,
                    value=f"${current_pred:.2f}",
                    delta=f"{change_pct:+.2f}%"
                )

st.markdown("---")

# Features
st.markdown("## ✨ Dashboard Features")

feature_col1, feature_col2 = st.columns(2)

with feature_col1:
    st.markdown("### 🎯 What You Can Do")
    st.write("✅ View predictions for 5 major stocks")
    st.write("✅ Compare 3 different ML models")
    st.write("✅ Interactive lives with zoom & pan")
    st.write("✅ Download predictions as CSV")
    st.write("✅ Track prediction accuracy")
    st.write("✅ Analyze risk metrics")
    st.write("✅ Real-time live prices")
    st.write("✅ Live news feed with sentiment")

with feature_col2:
    st.markdown("### 🤖 AI Models Used")
    st.write("📊 **Linear Regression** - Fast baseline predictions")
    st.write("🌲 **Random Forest** - Ensemble learning (81% accuracy)")
    st.write("🧠 **LSTM Neural Network** - Deep learning for time series")
    st.write("")
    st.write("**Total Models Trained:** 15")
    st.write("**Data Points Processed:** 6,000+")
    st.write("**Technical Indicators:** 37")

st.markdown("---")

# Navigation
st.markdown("## 🚀 Get Started")

st.info("👈 Use the **sidebar** to navigate between pages")

st.write("**Available Pages:**")
st.write("1. 📊 **Stock Predictions** - View candlestick live datas and AI forecasts")
st.write("2. 📈 **Model Comparison** - Compare model performance")
st.write("3. 💼 **Portfolio** - Track all your stocks")
st.write("4. 📚 **About** - Learn more about the project")
st.write("5. 🔴 **Live live data** - Real-time intraday data")
st.write("6. 📰 **News Feed** - Live news with sentiment analysis")
st.write("7. ⚖️ **Stock Comparison** - Compare multiple stocks")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p>Built with ❤️ using Streamlit, TensorFlow, and Scikit-learn</p>
    <p>Phase 1: Data Pipeline ✅ | Phase 2: ML Models ✅ | Phase 3: Web Dashboard ✅</p>
</div>
""", unsafe_allow_html=True)