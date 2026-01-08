"""
About Page - Dark Theme Edition with Complete Feature List
"""
import streamlit as st

# Page config
st.set_page_config(page_title="About", page_icon="📚", layout="wide")

# Load dark theme CSS
def load_css():
    import os
    css_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'dark_theme.css')
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Additional custom styling
st.markdown("""
<style>
.feature-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%);
    border-radius: 12px;
    padding: 25px;
    margin: 15px 0;
    border: 1px solid rgba(102, 126, 234, 0.3);
    transition: all 0.3s;
}
.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
    border-color: #667eea;
}
.phase-card {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    border-left: 4px solid #667eea;
    border-radius: 8px;
    padding: 20px;
    margin: 10px 0;
    height: 100%;
}
.tech-badge {
    display: inline-block;
    background: rgba(102, 126, 234, 0.2);
    color: #60a5fa;
    padding: 6px 12px;
    border-radius: 20px;
    margin: 5px;
    font-size: 13px;
    font-weight: 500;
}
.stat-box {
    background: linear-gradient(135deg, rgba(74, 222, 128, 0.1) 0%, rgba(16, 185, 129, 0.1) 100%);
    border-left: 4px solid #4ade80;
    border-radius: 8px;
    padding: 20px;
    margin: 10px 0;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("📚 About This Project")
st.markdown("AI-Powered Stock Prediction Platform - Complete Documentation")
st.markdown("---")

# Hero Section
st.markdown("""
<div class='feature-card' style='text-align: center;'>
    <h2 style='color: #667eea; margin-bottom: 15px;'>🚀 StockAI Pro</h2>
    <p style='font-size: 18px; color: #e2e8f0;'>
        A comprehensive <b>AI-powered stock prediction platform</b> featuring machine learning, 
        real-time data, sentiment analysis, and interactive visualizations.
    </p>
    <p style='color: #94a3b8; margin-top: 15px;'>
        Built with cutting-edge technologies for accurate market predictions and insights.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Project Phases
st.markdown("## 🎯 Project Development Phases")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='phase-card'>
        <h3 style='color: #667eea; margin-bottom: 15px;'>📊 Phase 1</h3>
        <h4 style='color: #e2e8f0;'>Data Pipeline</h4>
        <ul style='color: #94a3b8;'>
            <li>5 years historical data</li>
            <li>Data cleaning & validation</li>
            <li>37 technical indicators</li>
            <li>CSV & SQLite storage</li>
            <li>Modular architecture</li>
        </ul>
        <p style='color: #4ade80; font-weight: bold; margin-top: 15px;'>✅ Complete</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='phase-card'>
        <h3 style='color: #667eea; margin-bottom: 15px;'>🧠 Phase 2</h3>
        <h4 style='color: #e2e8f0;'>Machine Learning</h4>
        <ul style='color: #94a3b8;'>
            <li>Linear Regression</li>
            <li>Random Forest (81.9%)</li>
            <li>LSTM Neural Network</li>
            <li>15 trained models</li>
            <li>Model comparison</li>
        </ul>
        <p style='color: #4ade80; font-weight: bold; margin-top: 15px;'>✅ Complete</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='phase-card'>
        <h3 style='color: #667eea; margin-bottom: 15px;'>🌐 Phase 3</h3>
        <h4 style='color: #e2e8f0;'>Web Dashboard</h4>
        <ul style='color: #94a3b8;'>
            <li>Interactive charts</li>
            <li>Real-time predictions</li>
            <li>Portfolio tracking</li>
            <li>7 dashboard pages</li>
            <li>Dark theme UI</li>
        </ul>
        <p style='color: #4ade80; font-weight: bold; margin-top: 15px;'>✅ Complete</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Bonus Features
st.markdown("## 🎁 Bonus Features")

bonus_col1, bonus_col2 = st.columns(2)

with bonus_col1:
    st.markdown("""
    <div class='phase-card'>
        <h3 style='color: #f093fb; margin-bottom: 15px;'>🔴 Real-Time Data</h3>
        <ul style='color: #94a3b8;'>
            <li>Live stock prices</li>
            <li>Real-time ticker display</li>
            <li>Auto-refresh functionality</li>
            <li>Live vs Predicted comparison</li>
            <li>Intraday candlestick charts</li>
            <li>Market status indicators</li>
            <li>Multiple time intervals (1m, 5m, 15m, 30m, 1h)</li>
        </ul>
        <p style='color: #4ade80; font-weight: bold; margin-top: 15px;'>✅ Complete</p>
    </div>
    """, unsafe_allow_html=True)

with bonus_col2:
    st.markdown("""
    <div class='phase-card'>
        <h3 style='color: #fbbf24; margin-bottom: 15px;'>📰 News Integration</h3>
        <ul style='color: #94a3b8;'>
            <li>Real-time news feed</li>
            <li>AI sentiment analysis (VADER)</li>
            <li>Positive/Negative/Neutral classification</li>
            <li>Trending topics extraction</li>
            <li>News filtering & sorting</li>
            <li>Source attribution</li>
            <li>Auto-refreshing news</li>
        </ul>
        <p style='color: #4ade80; font-weight: bold; margin-top: 15px;'>✅ Complete</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Complete Feature List
st.markdown("## ✨ Complete Feature List")

feature_col1, feature_col2 = st.columns(2)

with feature_col1:
    st.markdown("""
    <div class='feature-card'>
        <h3 style='color: #60a5fa;'>📊 Data & Predictions</h3>
        <ul style='color: #e2e8f0; line-height: 1.8;'>
            <li>✅ Historical data (5 years)</li>
            <li>✅ 37 technical indicators</li>
            <li>✅ 30-day price predictions</li>
            <li>✅ Confidence intervals (95%)</li>
            <li>✅ Multiple ML models</li>
            <li>✅ Model performance comparison</li>
            <li>✅ Directional accuracy tracking</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='feature-card'>
        <h3 style='color: #4ade80;'>📈 Visualizations</h3>
        <ul style='color: #e2e8f0; line-height: 1.8;'>
            <li>✅ Candlestick charts</li>
            <li>✅ Line charts with SMA</li>
            <li>✅ Volume analysis</li>
            <li>✅ Interactive time periods (1D, 5D, 1M, 1Y, 5Y, MAX)</li>
            <li>✅ Projection mode</li>
            <li>✅ Real-time intraday charts</li>
            <li>✅ Portfolio distribution charts</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with feature_col2:
    st.markdown("""
    <div class='feature-card'>
        <h3 style='color: #fbbf24;'>🔴 Live Features</h3>
        <ul style='color: #e2e8f0; line-height: 1.8;'>
            <li>✅ Real-time price updates</li>
            <li>✅ Live market ticker</li>
            <li>✅ Auto-refresh (configurable)</li>
            <li>✅ Market open/closed status</li>
            <li>✅ Live vs Predicted metrics</li>
            <li>✅ Intraday data (5-min intervals)</li>
            <li>✅ Current market sentiment</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='feature-card'>
        <h3 style='color: #f093fb;'>💼 Portfolio & Analysis</h3>
        <ul style='color: #e2e8f0; line-height: 1.8;'>
            <li>✅ Multi-stock tracking</li>
            <li>✅ Portfolio overview</li>
            <li>✅ Risk analysis</li>
            <li>✅ Performance comparison</li>
            <li>✅ Stock correlation matrix</li>
            <li>✅ Allocation suggestions</li>
            <li>✅ Top gainers/losers</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Dashboard Pages
st.markdown("## 📱 Dashboard Pages")

st.markdown("""
<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0;'>
    <div class='feature-card'>
        <h4 style='color: #667eea;'>🏠 Homepage</h4>
        <p style='color: #94a3b8;'>Overview with live ticker and feature showcase</p>
    </div>
    <div class='feature-card'>
        <h4 style='color: #60a5fa;'>📊 Stock Predictions</h4>
        <p style='color: #94a3b8;'>Candlestick charts, AI forecasts, technical indicators</p>
    </div>
    <div class='feature-card'>
        <h4 style='color: #4ade80;'>📈 Model Comparison</h4>
        <p style='color: #94a3b8;'>Compare ML models, accuracy metrics, performance</p>
    </div>
    <div class='feature-card'>
        <h4 style='color: #fbbf24;'>💼 Portfolio</h4>
        <p style='color: #94a3b8;'>Track all stocks, risk analysis, allocation</p>
    </div>
    <div class='feature-card'>
        <h4 style='color: #f87171;'>🔴 Live Chart</h4>
        <p style='color: #94a3b8;'>Real-time intraday data, auto-refresh</p>
    </div>
    <div class='feature-card'>
        <h4 style='color: #a78bfa;'>📰 News Feed</h4>
        <p style='color: #94a3b8;'>Live news, sentiment analysis, trending topics</p>
    </div>
    <div class='feature-card'>
        <h4 style='color: #f093fb;'>⚖️ Stock Comparison</h4>
        <p style='color: #94a3b8;'>Multi-stock analysis, correlation, risk vs return</p>
    </div>
    <div class='feature-card'>
        <h4 style='color: #64748b;'>📚 About</h4>
        <p style='color: #94a3b8;'>Project documentation and information</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Technology Stack
st.markdown("## 🛠️ Technology Stack")

tech_col1, tech_col2, tech_col3 = st.columns(3)

with tech_col1:
    st.markdown("""
    <div class='feature-card'>
        <h3 style='color: #60a5fa;'>🐍 Backend</h3>
        <div style='margin-top: 15px;'>
            <span class='tech-badge'>Python 3.10</span>
            <span class='tech-badge'>Pandas</span>
            <span class='tech-badge'>NumPy</span>
            <span class='tech-badge'>yFinance</span>
            <span class='tech-badge'>SQLite</span>
            <span class='tech-badge'>NewsAPI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with tech_col2:
    st.markdown("""
    <div class='feature-card'>
        <h3 style='color: #4ade80;'>🤖 Machine Learning</h3>
        <div style='margin-top: 15px;'>
            <span class='tech-badge'>Scikit-learn</span>
            <span class='tech-badge'>TensorFlow</span>
            <span class='tech-badge'>Keras</span>
            <span class='tech-badge'>VADER Sentiment</span>
            <span class='tech-badge'>TA-Lib</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with tech_col3:
    st.markdown("""
    <div class='feature-card'>
        <h3 style='color: #fbbf24;'>🌐 Frontend</h3>
        <div style='margin-top: 15px;'>
            <span class='tech-badge'>Streamlit</span>
            <span class='tech-badge'>Plotly</span>
            <span class='tech-badge'>Custom CSS</span>
            <span class='tech-badge'>HTML5</span>
            <span class='tech-badge'>Dark Theme</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Project Statistics
st.markdown("## 📊 Project Statistics")

stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

with stat_col1:
    st.markdown("""
    <div class='stat-box'>
        <h2 style='color: #4ade80; margin: 0;'>60+</h2>
        <p style='color: #94a3b8; margin: 5px 0 0 0;'>Files Created</p>
    </div>
    """, unsafe_allow_html=True)

with stat_col2:
    st.markdown("""
    <div class='stat-box'>
        <h2 style='color: #60a5fa; margin: 0;'>3,500+</h2>
        <p style='color: #94a3b8; margin: 5px 0 0 0;'>Lines of Code</p>
    </div>
    """, unsafe_allow_html=True)

with stat_col3:
    st.markdown("""
    <div class='stat-box'>
        <h2 style='color: #fbbf24; margin: 0;'>15</h2>
        <p style='color: #94a3b8; margin: 5px 0 0 0;'>ML Models Trained</p>
    </div>
    """, unsafe_allow_html=True)

with stat_col4:
    st.markdown("""
    <div class='stat-box'>
        <h2 style='color: #f093fb; margin: 0;'>6,000+</h2>
        <p style='color: #94a3b8; margin: 5px 0 0 0;'>Data Points</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Model Details
st.markdown("## 🤖 Machine Learning Models")

model_tab1, model_tab2, model_tab3 = st.tabs(["📊 Linear Regression", "🌲 Random Forest", "🧠 LSTM Network"])

with model_tab1:
    st.markdown("""
    <div class='feature-card'>
        <h3 style='color: #60a5fa;'>Linear Regression - Baseline Model</h3>
        <ul style='color: #e2e8f0; line-height: 1.8;'>
            <li><strong>Type:</strong> Linear regression model</li>
            <li><strong>Training Time:</strong> ~0.02 seconds</li>
            <li><strong>Features Used:</strong> 27 technical indicators</li>
            <li><strong>Best For:</strong> Quick predictions, understanding linear trends</li>
            <li><strong>Advantages:</strong> Fast, interpretable, low computational cost</li>
            <li><strong>Limitations:</strong> Assumes linear relationships, may underfit complex patterns</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with model_tab2:
    st.markdown("""
    <div class='feature-card'>
        <h3 style='color: #4ade80;'>Random Forest - Production Model</h3>
        <ul style='color: #e2e8f0; line-height: 1.8;'>
            <li><strong>Type:</strong> Ensemble learning (100 decision trees)</li>
            <li><strong>Training Time:</strong> ~0.6 seconds</li>
            <li><strong>Accuracy:</strong> 81.9% directional accuracy</li>
            <li><strong>Features Used:</strong> 27 technical indicators</li>
            <li><strong>Best For:</strong> Production use, balanced accuracy and speed</li>
            <li><strong>Advantages:</strong> Handles non-linear patterns, feature importance analysis</li>
            <li><strong>Parameters:</strong> n_estimators=100, max_depth=10, random_state=42</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with model_tab3:
    st.markdown("""
    <div class='feature-card'>
        <h3 style='color: #a78bfa;'>LSTM Neural Network - Deep Learning</h3>
        <ul style='color: #e2e8f0; line-height: 1.8;'>
            <li><strong>Type:</strong> Recurrent Neural Network (Long Short-Term Memory)</li>
            <li><strong>Training Time:</strong> ~60 seconds</li>
            <li><strong>Architecture:</strong> 2 LSTM layers (100, 50 units) + Dropout (0.2)</li>
            <li><strong>Input Shape:</strong> 60-day sequences of features</li>
            <li><strong>Best For:</strong> Capturing long-term dependencies and complex temporal patterns</li>
            <li><strong>Advantages:</strong> Handles sequential data, learns temporal relationships</li>
            <li><strong>Optimizer:</strong> Adam, Loss: MSE, Epochs: 50, Batch Size: 32</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Technical Indicators
st.markdown("## 📈 Technical Indicators (37 Total)")

ind_col1, ind_col2, ind_col3 = st.columns(3)

with ind_col1:
    st.markdown("""
    <div class='feature-card'>
        <h4 style='color: #60a5fa;'>Trend Indicators</h4>
        <ul style='color: #e2e8f0;'>
            <li>SMA (20, 50, 200)</li>
            <li>EMA (12, 26)</li>
            <li>MACD</li>
            <li>ADX</li>
            <li>Aroon</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with ind_col2:
    st.markdown("""
    <div class='feature-card'>
        <h4 style='color: #4ade80;'>Momentum Indicators</h4>
        <ul style='color: #e2e8f0;'>
            <li>RSI (14)</li>
            <li>Stochastic Oscillator</li>
            <li>Williams %R</li>
            <li>ROC</li>
            <li>CCI</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with ind_col3:
    st.markdown("""
    <div class='feature-card'>
        <h4 style='color: #fbbf24;'>Volatility & Volume</h4>
        <ul style='color: #e2e8f0;'>
            <li>Bollinger Bands</li>
            <li>ATR</li>
            <li>Standard Deviation</li>
            <li>Volume indicators</li>
            <li>OBV</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Disclaimer
st.markdown("## ⚠️ Important Disclaimer")

st.markdown("""
<div style='
    background: linear-gradient(135deg, rgba(248, 113, 113, 0.1) 0%, rgba(239, 68, 68, 0.1) 100%);
    border-left: 4px solid #f87171;
    border-radius: 8px;
    padding: 25px;
    margin: 20px 0;
'>
    <h3 style='color: #f87171; margin-top: 0;'>⚠️ DISCLAIMER - READ CAREFULLY</h3>
    <p style='color: #e2e8f0; line-height: 1.8;'>
        This project is for <strong>EDUCATIONAL PURPOSES ONLY</strong>. The predictions made by this system 
        should <strong>NOT</strong> be used as financial advice or as the sole basis for investment decisions.
    </p>
    <ul style='color: #e2e8f0; line-height: 1.8;'>
        <li>Stock market predictions are inherently uncertain and speculative</li>
        <li>Past performance does not guarantee future results</li>
        <li>Machine learning models can be wrong and make inaccurate predictions</li>
        <li>Market conditions change and models may not adapt in real-time</li>
        <li>Always conduct your own research (DYOR)</li>
        <li>Consult with licensed financial advisors before making investment decisions</li>
        <li>Only invest money you can afford to lose</li>
        <li>The developers assume no liability for financial losses</li>
    </ul>
    <p style='color: #94a3b8; font-style: italic; margin-bottom: 0;'>
        This platform is a demonstration of machine learning and web development skills, 
        not a professional trading system.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Development Info
st.markdown("## 👨‍💻 Development Information")

dev_col1, dev_col2 = st.columns(2)

with dev_col1:
    st.markdown("""
    <div class='feature-card'>
        <h3 style='color: #667eea;'>📅 Project Timeline</h3>
        <ul style='color: #e2e8f0; line-height: 1.8;'>
            <li><strong>Phase 1 (Data):</strong> Week 1</li>
            <li><strong>Phase 2 (ML):</strong> Week 2</li>
            <li><strong>Phase 3 (Web):</strong> Week 3</li>
            <li><strong>Bonus Features:</strong> Week 4</li>
            <li><strong>Total Duration:</strong> 1 Month</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with dev_col2:
    st.markdown("""
    <div class='feature-card'>
        <h3 style='color: #4ade80;'>🎯 Learning Outcomes</h3>
        <ul style='color: #e2e8f0; line-height: 1.8;'>
            <li>Data engineering & ETL</li>
            <li>Machine learning & deep learning</li>
            <li>Web development</li>
            <li>API integration</li>
            <li>Natural language processing</li>
            <li>Data visualization</li>
            <li>Software architecture</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Footer
st.markdown("""
<div style='
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    border-radius: 12px;
    padding: 30px;
    text-align: center;
    margin-top: 40px;
'>
    <h3 style='color: #667eea; margin-top: 0;'>🚀 Built with Passion</h3>
    <p style='color: #e2e8f0; font-size: 16px;'>
        This project demonstrates the power of combining machine learning, 
        real-time data, and modern web technologies to create intelligent applications.
    </p>
    <div style='margin: 20px 0;'>
        <span class='tech-badge'>Python</span>
        <span class='tech-badge'>TensorFlow</span>
        <span class='tech-badge'>Streamlit</span>
        <span class='tech-badge'>Plotly</span>
        <span class='tech-badge'>NewsAPI</span>
    </div>
    <p style='color: #94a3b8; margin-top: 20px;'>
        <strong>Project Status:</strong> <span style='color: #4ade80;'>✅ Complete & Deployed</span>
    </p>
    <p style='color: #94a3b8;'>
        Built with ❤️ using cutting-edge technologies
    </p>
</div>
""", unsafe_allow_html=True)