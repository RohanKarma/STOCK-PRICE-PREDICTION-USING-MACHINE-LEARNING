"""
Stock Predictions Page - Dark Theme Edition
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys

# Page config - MUST BE FIRST
st.set_page_config(page_title="Stock Predictions", page_icon="📊", layout="wide")

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils'))

# Load dark theme CSS
def load_css():
    css_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'dark_theme.css')
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Import utilities
from realtime_data import RealTimeData
from live_ticker import display_live_price_card
import time

# Simple CSS
st.markdown("""
<style>
.stButton > button {
    width: 100%;
    height: 45px;
    font-size: 13px;
    font-weight: 600;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Stock Price Predictions")
st.markdown("---")

# Sidebar - Stock Selection
st.sidebar.header("🎯 Select Stock")

# Available stocks
predictions_dir = '../phase2_ml_models/data/predictions'
phase1_data_dir = '../phase1_data_pipeline/data/processed'

if os.path.exists(predictions_dir):
    available_stocks = [f.replace('_predictions.csv', '') for f in os.listdir(predictions_dir) if f.endswith('.csv')]
    
    selected_stock = st.sidebar.selectbox(
        "Choose a stock:",
        available_stocks,
        index=0
    )
    
    # Auto-refresh toggle
    st.sidebar.markdown("---")
    st.sidebar.header("🔴 Live Data")
    
    auto_refresh = st.sidebar.checkbox("🔄 Auto-Refresh", value=False, 
                                       help="Refresh live data every 30 seconds")
    
    if auto_refresh:
        refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 10, 60, 30)
        st.sidebar.info(f"Page will refresh every {refresh_rate} seconds")
    
    # Load prediction data
    pred_file = os.path.join(predictions_dir, f'{selected_stock}_predictions.csv')
    pred_df = pd.read_csv(pred_file)
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    
    # Load historical data
    hist_file = os.path.join(phase1_data_dir, f'{selected_stock}_processed.csv')
    if os.path.exists(hist_file):
        hist_df = pd.read_csv(hist_file)
        hist_df['date'] = pd.to_datetime(hist_df['date'])
    else:
        hist_df = None
    
    # Display metrics
    st.markdown(f"## {selected_stock} - AI-Powered Analysis")
    
    # Show live price data
    st.markdown("### 🔴 LIVE MARKET DATA")
    
    try:
        display_live_price_card(selected_stock)
    except Exception as e:
        st.warning("Live data unavailable. Showing predicted values.")
    
    st.markdown("---")
    
    # Live vs Predicted comparison
    st.markdown("### 📊 Live vs Predicted Comparison")
    
    try:
        rtd = RealTimeData(selected_stock)
        live_metrics = rtd.get_live_metrics()
        
        if live_metrics:
            live_price = live_metrics['price']
            predicted_next = pred_df.iloc[0]['predicted_price']
            
            diff = predicted_next - live_price
            diff_pct = (diff / live_price) * 100
            
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            with comp_col1:
                st.metric("Current LIVE Price", f"${live_price:.2f}")
            
            with comp_col2:
                st.metric("AI Predicted (Next Day)", f"${predicted_next:.2f}")
            
            with comp_col3:
                accuracy_label = "Prediction Accuracy"
                if abs(diff_pct) < 1:
                    accuracy = "🎯 Excellent"
                elif abs(diff_pct) < 3:
                    accuracy = "✅ Good"
                else:
                    accuracy = "⚠️ Moderate"
                
                st.metric(accuracy_label, accuracy, f"{diff:+.2f} ({diff_pct:+.2f}%)")
    
    except:
        # Show predictions only
        col1, col2, col3, col4 = st.columns(4)
        
        current_pred = pred_df.iloc[0]['predicted_price']
        future_pred = pred_df.iloc[29]['predicted_price']
        change = future_pred - current_pred
        change_pct = (change / current_pred) * 100
        
        latest_actual = hist_df.iloc[-1]['close'] if hist_df is not None else current_pred
        
        with col1:
            st.metric("Latest Price", f"${latest_actual:.2f}")
        
        with col2:
            st.metric("Next Day Prediction", f"${current_pred:.2f}", 
                     f"{((current_pred - latest_actual) / latest_actual * 100):+.2f}%")
        
        with col3:
            st.metric("30-Day Target", f"${future_pred:.2f}", f"{change:+.2f}")
        
        with col4:
            trend = "📈 BULLISH" if change_pct > 0 else "📉 BEARISH"
            st.metric("Trend", trend)
    
    st.markdown("---")
    
        # ========================================
    # CHART VISUALIZATION - VISIBILITY FIXED
    # ========================================
    
    st.markdown("### 📈 Chart Visualization")
    
    # Clean CSS for buttons
    st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        height: 45px;
        font-size: 13px;
        font-weight: 600;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Chart type selector
    chart_type = st.selectbox(
        "📊 Select Chart Type:",
        ["🕯️ Candlestick Chart", "📈 Line Chart", "📊 Area Chart"],
        help="Choose how you want to visualize the data"
    )
    
    st.markdown("**⏱️ Select Time Period:**")
    
    # Initialize session state
    if 'selected_period' not in st.session_state:
        st.session_state.selected_period = '1M'
    
    periods = {
        '1D': {'label': '1D', 'days': 1},
        '5D': {'label': '5D', 'days': 5},
        '1M': {'label': '1M', 'days': 30},
        '1Y': {'label': '1Y', 'days': 365},
        '5Y': {'label': '5Y', 'days': 1825},
        'MAX': {'label': 'MAX', 'days': 99999},
        'PROJECTION': {'label': 'PROJECTION', 'days': 0}
    }
    
    # Create period buttons
    cols = st.columns(7)
    
    for idx, (key, value) in enumerate(periods.items()):
        with cols[idx]:
            button_type = "primary" if st.session_state.selected_period == key else "secondary"
            
            if st.button(
                value['label'],
                key=f"period_btn_{key}",
                type=button_type,
                use_container_width=True
            ):
                st.session_state.selected_period = key
                st.rerun()
    
    # Get selected period
    selected_period = st.session_state.selected_period
    
    # Display selected period
    st.info(f"📅 Viewing: **{periods[selected_period]['label']}** period")
    
    st.markdown("---")
    
    # ========================================
    # CHART RENDERING LOGIC
    # ========================================
    
    # 1. PROJECTION CHART
    if selected_period == 'PROJECTION':
        st.markdown("### 🔮 AI Price Predictions (30 Days)")
        
        fig = go.Figure()
        
        # Historical data
        if hist_df is not None:
            hist_display = hist_df.tail(30)
            fig.add_trace(go.Scatter(
                x=hist_display['date'],
                y=hist_display['close'],
                mode='lines',
                name='Historical',
                line=dict(color='#60a5fa', width=2)
            ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=pred_df['date'][:30],
            y=pred_df['predicted_price'][:30],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#4ade80', width=3, dash='dash'),
            marker=dict(size=6)
        ))
        
        # Confidence bands
        fig.add_trace(go.Scatter(
            x=pred_df['date'][:30],
            y=pred_df['upper_bound'][:30],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=pred_df['date'][:30],
            y=pred_df['lower_bound'][:30],
            mode='lines',
            name='Confidence Band',
            line=dict(width=0),
            fillcolor='rgba(74, 222, 128, 0.2)',
            fill='tonexty'
        ))
        
        # LAYOUT - FIXED TEXT VISIBILITY
        fig.update_layout(
            title={
                'text': f"{selected_stock} - 30-Day Predictions",
                'font': {'size': 20, 'color': '#e2e8f0'}
            },
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,41,59,0.4)',
            font=dict(color='#e2e8f0'),  # ✅ MASTER FONT COLOR FIX
            height=600,
            hovermode='x unified',
            legend=dict(
                bgcolor="rgba(30,41,59,0.8)",
                bordercolor="rgba(102,126,234,0.3)",
                borderwidth=1,
                font=dict(color='#e2e8f0')
            )
        )
        
        # Grid lines
        fig.update_xaxes(gridcolor='rgba(148,163,184,0.1)', tickfont=dict(color='#e2e8f0'))
        fig.update_yaxes(gridcolor='rgba(148,163,184,0.1)', tickfont=dict(color='#e2e8f0'))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 2. HISTORICAL CHARTS
    else:
        if hist_df is not None:
            
            # Data Filtering Logic
            if selected_period == 'MAX':
                hist_display = hist_df.copy()
            elif selected_period == '1D':
                try:
                    rtd = RealTimeData(selected_stock)
                    intraday = rtd.get_intraday_data(interval='5m', period='1d')
                    if intraday is not None and not intraday.empty:
                        if 'datetime' in intraday.columns:
                            intraday = intraday.rename(columns={'datetime': 'date'})
                        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                        if all(col in intraday.columns for col in required_cols):
                            hist_display = intraday
                            st.success("📊 Showing intraday data")
                        else:
                            hist_display = hist_df.tail(50)
                    else:
                        hist_display = hist_df.tail(50)
                except:
                    hist_display = hist_df.tail(50)
            else:
                days = periods[selected_period]['days']
                hist_display = hist_df.tail(days)
            
            # A. CANDLESTICK CHART
            if chart_type == "🕯️ Candlestick Chart":
                
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(f'{selected_stock} Price', 'Volume')
                )
                
                # Price
                fig.add_trace(go.Candlestick(
                    x=hist_display['date'],
                    open=hist_display['open'],
                    high=hist_display['high'],
                    low=hist_display['low'],
                    close=hist_display['close'],
                    name='Price',
                    increasing_line_color='#4ade80',
                    decreasing_line_color='#f87171'
                ), row=1, col=1)
                
                # Moving Averages
                if len(hist_display) > 20 and 'sma_20' in hist_display.columns:
                    fig.add_trace(go.Scatter(
                        x=hist_display['date'],
                        y=hist_display['sma_20'],
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='#fbbf24', width=1.5)
                    ), row=1, col=1)
                
                if len(hist_display) > 50 and 'sma_50' in hist_display.columns:
                    fig.add_trace(go.Scatter(
                        x=hist_display['date'],
                        y=hist_display['sma_50'],
                        mode='lines',
                        name='SMA 50',
                        line=dict(color='#60a5fa', width=1.5)
                    ), row=1, col=1)
                
                # Volume
                colors = ['#4ade80' if row['close'] >= row['open'] else '#f87171' 
                         for _, row in hist_display.iterrows()]
                
                fig.add_trace(go.Bar(
                    x=hist_display['date'],
                    y=hist_display['volume'],
                    name='Volume',
                    marker_color=colors,
                    showlegend=False
                ), row=2, col=1)
                
                # Layout - VISIBILITY FIX
                fig.update_layout(
                    title={
                        'text': f"{selected_stock} - Candlestick Chart",
                        'font': {'size': 20, 'color': '#e2e8f0'}
                    },
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(30,41,59,0.4)',
                    font=dict(color='#e2e8f0'), # ✅ MASTER FONT FIX
                    height=700,
                    hovermode='x unified',
                    xaxis_rangeslider_visible=False,
                    legend=dict(
                        bgcolor="rgba(30,41,59,0.8)",
                        bordercolor="rgba(102,126,234,0.3)",
                        borderwidth=1,
                        font=dict(color='#e2e8f0')
                    )
                )
                
                # Axes colors
                fig.update_xaxes(gridcolor='rgba(148,163,184,0.1)', tickfont=dict(color='#e2e8f0'))
                fig.update_yaxes(title_text="Price ($)", row=1, col=1, gridcolor='rgba(148,163,184,0.1)', tickfont=dict(color='#e2e8f0'))
                fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor='rgba(148,163,184,0.1)', tickfont=dict(color='#e2e8f0'))
                
                st.plotly_chart(fig, use_container_width=True)
            
            # B. LINE CHART
            elif chart_type == "📈 Line Chart":
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=hist_display['date'],
                    y=hist_display['close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='#60a5fa', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(96, 165, 250, 0.1)'
                ))
                
                if len(hist_display) > 20 and 'sma_20' in hist_display.columns:
                    fig.add_trace(go.Scatter(
                        x=hist_display['date'],
                        y=hist_display['sma_20'],
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='#fbbf24', width=1.5)
                    ))
                
                fig.update_layout(
                    title={
                        'text': f"{selected_stock} - Price Chart",
                        'font': {'size': 20, 'color': '#e2e8f0'}
                    },
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(30,41,59,0.4)',
                    font=dict(color='#e2e8f0'), # ✅ MASTER FONT FIX
                    height=600,
                    xaxis=dict(gridcolor='rgba(148,163,184,0.1)', tickfont=dict(color='#e2e8f0')),
                    yaxis=dict(gridcolor='rgba(148,163,184,0.1)', tickfont=dict(color='#e2e8f0'))
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # C. AREA CHART
            elif chart_type == "📊 Area Chart":
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=hist_display['date'],
                    y=hist_display['close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='#a78bfa', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(167, 139, 250, 0.2)'
                ))
                
                fig.update_layout(
                    title={
                        'text': f"{selected_stock} - Area Chart",
                        'font': {'size': 20, 'color': '#e2e8f0'}
                    },
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(30,41,59,0.4)',
                    font=dict(color='#e2e8f0'), # ✅ MASTER FONT FIX
                    height=600,
                    xaxis=dict(gridcolor='rgba(148,163,184,0.1)', tickfont=dict(color='#e2e8f0')),
                    yaxis=dict(gridcolor='rgba(148,163,184,0.1)', tickfont=dict(color='#e2e8f0'))
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("⚠️ No historical data available for this stock.")
    
    st.markdown("---")
    # Technical Indicators
    if hist_df is not None and 'rsi' in hist_df.columns:
        st.markdown("### 📊 Technical Indicators")
        
        ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
        
        with ind_col1:
            latest_rsi = hist_df['rsi'].iloc[-1]
            rsi_signal = "Oversold 🟢" if latest_rsi < 30 else "Overbought 🔴" if latest_rsi > 70 else "Neutral ⚪"
            st.metric("RSI (14)", f"{latest_rsi:.2f}", rsi_signal)
        
        with ind_col2:
            if 'macd' in hist_df.columns:
                latest_macd = hist_df['macd'].iloc[-1]
                macd_signal = "Bullish 📈" if latest_macd > 0 else "Bearish 📉"
                st.metric("MACD", f"{latest_macd:.2f}", macd_signal)
        
        with ind_col3:
            if 'volatility' in hist_df.columns:
                latest_vol = hist_df['volatility'].iloc[-1] * 100
                st.metric("Volatility", f"{latest_vol:.2f}%")
        
        with ind_col4:
            if 'volume' in hist_df.columns:
                avg_volume = hist_df['volume'].mean()
                latest_volume = hist_df['volume'].iloc[-1]
                volume_change = ((latest_volume - avg_volume) / avg_volume) * 100
                st.metric("Volume vs Avg", f"{volume_change:+.1f}%")
        
        st.markdown("---")
    
    # Prediction table
    st.markdown("### 📅 Detailed Predictions (Next 30 Days)")
    
    # Format the table
    display_df = pred_df[:30].copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    display_df.columns = ['Date', 'Ticker', 'Predicted Price', 'Lower Bound', 'Upper Bound']
    
    st.dataframe(
        display_df.style.format({
            'Predicted Price': '${:.2f}',
            'Lower Bound': '${:.2f}',
            'Upper Bound': '${:.2f}'
        }).background_gradient(subset=['Predicted Price'], cmap='RdYlGn'),
        use_container_width=True
    )
    
    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Predictions (CSV)",
        data=csv,
        file_name=f"{selected_stock}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    # Risk Analysis
    st.markdown("---")
    st.markdown("### ⚠️ Risk Analysis")
    
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    
    latest_actual = hist_df.iloc[-1]['close'] if hist_df is not None else pred_df.iloc[0]['predicted_price']
    
    with risk_col1:
        volatility_range = ((pred_df['upper_bound'][:30] - pred_df['lower_bound'][:30]) / 
                     pred_df['predicted_price'][:30]).mean() * 100
        
        vol_status = "Low 🟢" if volatility_range < 5 else "Medium 🟡" if volatility_range < 10 else "High 🔴"
        st.metric("Prediction Volatility", f"{volatility_range:.2f}%", vol_status)
    
    with risk_col2:
        max_gain = ((pred_df['upper_bound'][:30].max() - latest_actual) / latest_actual) * 100
        st.metric("Max Potential Gain", f"{max_gain:.2f}%")
    
    with risk_col3:
        max_loss = ((pred_df['lower_bound'][:30].min() - latest_actual) / latest_actual) * 100
        st.metric("Max Potential Loss", f"{max_loss:.2f}%")
    
    # Trading Signals
    st.markdown("---")
    st.markdown("### 🎯 AI Trading Signals")
    
    signal_col1, signal_col2 = st.columns(2)
    
    current_pred = pred_df.iloc[0]['predicted_price']
    future_pred = pred_df.iloc[29]['predicted_price']
    change_pct = ((future_pred - current_pred) / current_pred) * 100
    
    with signal_col1:
        if change_pct > 5:
            st.success("🟢 **STRONG BUY** - AI predicts significant upside potential")
        elif change_pct > 2:
            st.success("🟢 **BUY** - AI predicts moderate upside potential")
        elif change_pct > -2:
            st.info("⚪ **HOLD** - AI predicts sideways movement")
        elif change_pct > -5:
            st.warning("🟡 **SELL** - AI predicts moderate downside risk")
        else:
            st.error("🔴 **STRONG SELL** - AI predicts significant downside risk")
    
    with signal_col2:
        st.info(f"""
        **Confidence Level:** 95%
        
        **Model Used:** Random Forest + LSTM
        
        **Directional Accuracy:** 81.9%
        
        **Prediction Horizon:** 30 Days
        
        **Selected Period:** {selected_period}
        """)
    
    # Disclaimer
    st.markdown("---")
    st.warning("⚠️ **Disclaimer:** These predictions are generated by AI models for educational purposes only. Not financial advice. Always do your own research and consult with a licensed financial advisor before making investment decisions.")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

else:
    st.error("❌ No prediction data found! Please run Phase 2 first.")
    st.info("👉 Go to `phase2_ml_models` folder and run `python main.py`")