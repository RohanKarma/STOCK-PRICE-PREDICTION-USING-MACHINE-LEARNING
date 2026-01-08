"""
Live Stock Chart Page - Dark Theme Edition
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import time

# Page config
st.set_page_config(page_title="Live Chart", page_icon="🔴", layout="wide")

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

st.title("🔴 Live Stock Chart")
st.markdown("Real-time intraday price movements with auto-refresh")
st.markdown("---")

# Sidebar
st.sidebar.header("🎯 Chart Settings")

ticker = st.sidebar.selectbox(
    "Select Stock:",
    ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
)

interval = st.sidebar.selectbox(
    "Time Interval:",
    ['1m', '5m', '15m', '30m', '1h'],
    index=1,
    help="Smaller intervals = more frequent updates"
)

period = st.sidebar.selectbox(
    "Period:",
    ['1d', '5d'],
    index=0,
    help="Total time range to display"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 Auto-Refresh Settings")

auto_refresh = st.sidebar.checkbox("🔄 Enable Auto-Refresh", value=True)

if auto_refresh:
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 10, 60, 30)
    st.sidebar.info(f"Chart will update every {refresh_rate} seconds")
else:
    refresh_rate = 30
    st.sidebar.warning("Auto-refresh is disabled. Click 'R' to refresh manually.")

# Fetch live data
rtd = RealTimeData(ticker)

# Live metrics header
st.markdown(f"## {ticker} - Live Intraday Chart")
st.markdown(f"**Interval:** {interval} | **Period:** {period}")

st.markdown("---")

# Live metrics cards
st.markdown("### 🔴 Live Market Data")

metrics = rtd.get_live_metrics()

if metrics:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🔴 LIVE PRICE",
            f"${metrics['price']:.2f}",
            f"{metrics['change']:+.2f} ({metrics['change_pct']:+.2f}%)"
        )
    
    with col2:
        st.metric("Day High", f"${metrics['day_high']:.2f}")
    
    with col3:
        st.metric("Day Low", f"${metrics['day_low']:.2f}")
    
    with col4:
        st.metric("Previous Close", f"${metrics['previous_close']:.2f}")
    
    with col5:
        status = "🟢 OPEN" if metrics['is_market_open'] else "🔴 CLOSED"
        st.metric("Market Status", status)

st.markdown("---")

# Fetch intraday data
with st.spinner(f'Loading {interval} intraday data...'):
    df = rtd.get_intraday_data(interval=interval, period=period)

if df is not None and not df.empty:
    
    st.success(f"✅ Loaded {len(df)} data points for {ticker}")
    
    # Create candlestick chart with DARK THEME
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker} Intraday Price ({interval})', 'Volume')
    )
    
    # Candlestick - DARK THEME COLORS
    fig.add_trace(go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='#4ade80',  # Bright green
        decreasing_line_color='#f87171',  # Bright red
        increasing_fillcolor='rgba(74, 222, 128, 0.3)',
        decreasing_fillcolor='rgba(248, 113, 113, 0.3)'
    ), row=1, col=1)
    
    # Volume bars - DARK THEME COLORS
    colors = ['rgba(74, 222, 128, 0.7)' if row['close'] >= row['open'] 
              else 'rgba(248, 113, 113, 0.7)' 
              for _, row in df.iterrows()]
    
    fig.add_trace(go.Bar(
        x=df['datetime'],
        y=df['volume'],
        name='Volume',
        marker_color=colors,
        showlegend=False,
        hovertemplate='Volume: %{y:,.0f}<extra></extra>'
    ), row=2, col=1)
    
    # Update layout - DARK THEME
    title_text = f"{ticker} - {interval} Live Chart"
    if auto_refresh:
        title_text += f" (Auto-refresh: {refresh_rate}s)"
    
    fig.update_layout(
        title={
            'text': title_text,
            'font': {'size': 20, 'color': '#e2e8f0'}
        },
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,41,59,0.4)',
        font=dict(color='#e2e8f0', size=12),
        height=750,
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(30,41,59,0.8)",
            bordercolor="rgba(102,126,234,0.3)",
            borderwidth=1,
            font=dict(color='#e2e8f0', size=11)
        )
    )
    
    # Update axes - PROPER SYNTAX
    fig.update_xaxes(
        title_text="Time",
        title_font=dict(color='#e2e8f0', size=13),
        tickfont=dict(color='#e2e8f0', size=10),
        gridcolor='rgba(148,163,184,0.1)',
        row=2, col=1
    )
    
    fig.update_yaxes(
        title_text="Price (USD)",
        title_font=dict(color='#e2e8f0', size=13),
        tickfont=dict(color='#e2e8f0', size=10),
        gridcolor='rgba(148,163,184,0.1)',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Volume",
        title_font=dict(color='#e2e8f0', size=13),
        tickfont=dict(color='#e2e8f0', size=10),
        gridcolor='rgba(148,163,184,0.1)',
        row=2, col=1
    )
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Session Statistics
    st.markdown("---")
    st.markdown("### 📊 Session Statistics")
    
    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
    
    session_high = df['high'].max()
    session_low = df['low'].min()
    session_open = df['open'].iloc[0]
    session_close = df['close'].iloc[-1]
    session_change = ((session_close - session_open) / session_open) * 100
    avg_volume = df['volume'].mean()
    total_volume = df['volume'].sum()
    total_trades = len(df)
    
    with stat_col1:
        st.metric("Session High", f"${session_high:.2f}")
    
    with stat_col2:
        st.metric("Session Low", f"${session_low:.2f}")
    
    with stat_col3:
        st.metric(
            "Session Change",
            f"{session_change:+.2f}%",
            f"${session_close - session_open:+.2f}"
        )
    
    with stat_col4:
        st.metric("Avg Volume", f"{avg_volume:,.0f}")
    
    with stat_col5:
        st.metric("Data Points", f"{total_trades}")
    
    # Additional stats in expandable section
    with st.expander("📈 Detailed Statistics"):
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        
        with detail_col1:
            st.markdown("#### Price Action")
            price_range = session_high - session_low
            st.write(f"**Range:** ${price_range:.2f}")
            st.write(f"**Opening:** ${session_open:.2f}")
            st.write(f"**Current:** ${session_close:.2f}")
            
            # Price movement
            if session_close > session_open:
                st.success(f"📈 Up ${session_close - session_open:.2f} from open")
            else:
                st.error(f"📉 Down ${session_open - session_close:.2f} from open")
        
        with detail_col2:
            st.markdown("#### Volume Analysis")
            st.write(f"**Total Volume:** {total_volume:,.0f}")
            st.write(f"**Average Volume:** {avg_volume:,.0f}")
            max_volume = df['volume'].max()
            st.write(f"**Peak Volume:** {max_volume:,.0f}")
            
            # Volume trend
            recent_vol = df['volume'].tail(10).mean()
            if recent_vol > avg_volume:
                st.info("📊 Recent volume above average")
            else:
                st.info("📊 Recent volume below average")
        
        with detail_col3:
            st.markdown("#### Market Info")
            st.write(f"**Interval:** {interval}")
            st.write(f"**Period:** {period}")
            st.write(f"**Data Points:** {total_trades}")
            
            if metrics:
                st.write(f"**Market:** {'🟢 Open' if metrics['is_market_open'] else '🔴 Closed'}")
    
    # Latest candles table
    st.markdown("---")
    st.markdown("### 📋 Latest Candles")
    
    # Show last 10 candles
    latest_df = df.tail(10).copy()
    latest_df['datetime'] = latest_df['datetime'].dt.strftime('%H:%M:%S')
    latest_df['Change'] = ((latest_df['close'] - latest_df['open']) / latest_df['open'] * 100).round(2)
    
    display_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'Change']
    latest_df = latest_df[display_cols]
    latest_df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change (%)']
    
    st.dataframe(
        latest_df.style.format({
            'Open': '${:.2f}',
            'High': '${:.2f}',
            'Low': '${:.2f}',
            'Close': '${:.2f}',
            'Volume': '{:,.0f}',
            'Change (%)': '{:+.2f}%'
        }),
        use_container_width=True,
        height=400
    )
    
    # Last update time
    st.markdown("---")
    if metrics:
        update_col1, update_col2, update_col3 = st.columns(3)
        
        with update_col1:
            st.info(f"🕐 Last Updated: {metrics['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        with update_col2:
            if auto_refresh:
                st.success(f"🔄 Auto-refresh: ON ({refresh_rate}s)")
            else:
                st.warning("🔄 Auto-refresh: OFF")
        
        with update_col3:
            st.info(f"📊 Showing: {len(df)} candles")
    
else:
    st.error("❌ No intraday data available")
    
    if metrics and not metrics['is_market_open']:
        st.warning("⚠️ Market is currently closed. Intraday data may not be available.")
        st.info("💡 Try viewing historical daily data in the Stock Predictions page.")
    else:
        st.warning("⚠️ Unable to fetch intraday data. Please try again later.")
    
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        **Possible reasons for no data:**
        
        1. **Market is closed** - Intraday data is only available during market hours
        2. **Weekend/Holiday** - Markets are closed
        3. **Data provider issue** - Temporary API issue
        4. **Invalid ticker** - Try a different stock symbol
        5. **Internet connection** - Check your connection
        
        **Solutions:**
        
        - Wait for market to open (9:30 AM - 4:00 PM EST)
        - Try a different time interval
        - Refresh the page
        - View historical data in Stock Predictions page
        """)

# Auto-refresh logic
if auto_refresh and df is not None:
    time.sleep(refresh_rate)
    st.rerun()