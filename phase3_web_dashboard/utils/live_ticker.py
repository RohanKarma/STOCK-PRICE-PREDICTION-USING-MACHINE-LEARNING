"""
Live price ticker component - Pure Streamlit Version
"""
import streamlit as st
from utils.realtime_data import RealTimeData


def display_live_ticker(tickers):
    """
    Display live price ticker using pure Streamlit components
    
    Args:
        tickers: List of ticker symbols
    """
    
    # Create a container with columns
    st.markdown("### 🔴 Live Market Prices")
    
    # Create columns for each ticker + market status
    cols = st.columns(len(tickers) + 1)
    
    # Market status in first column
    with cols[0]:
        try:
            rtd_first = RealTimeData(tickers[0])
            is_open = rtd_first.is_market_open()
            
            if is_open:
                st.success("🟢 Market Open")
            else:
                st.error("🔴 Market Closed")
        except:
            st.info("⚪ Loading...")
    
    # Display each ticker
    for idx, ticker in enumerate(tickers):
        with cols[idx + 1]:
            try:
                rtd = RealTimeData(ticker)
                metrics = rtd.get_live_metrics()
                
                if metrics:
                    price = metrics['price']
                    change = metrics['change']
                    change_pct = metrics['change_pct']
                    
                    # Display metric with delta
                    st.metric(
                        label=f"**{ticker}**",
                        value=f"${price:.2f}",
                        delta=f"{change_pct:+.2f}%",
                        delta_color="normal"
                    )
                else:
                    st.metric(
                        label=f"**{ticker}**",
                        value="--",
                        delta="Loading..."
                    )
            except Exception as e:
                st.metric(
                    label=f"**{ticker}**",
                    value="--",
                    delta="Error"
                )
    
    st.markdown("---")


def display_live_price_card(ticker):
    """
    Display detailed live price card for a single stock
    
    Args:
        ticker: Stock symbol
    """
    rtd = RealTimeData(ticker)
    metrics = rtd.get_live_metrics()
    
    if not metrics:
        st.error(f"Unable to fetch live data for {ticker}")
        return
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🔴 LIVE PRICE",
            value=f"${metrics['price']:.2f}",
            delta=f"{metrics['change']:+.2f} ({metrics['change_pct']:+.2f}%)"
        )
    
    with col2:
        st.metric(
            label="Previous Close",
            value=f"${metrics['previous_close']:.2f}"
        )
    
    with col3:
        st.metric(
            label="Day Range",
            value=f"${metrics['day_low']:.2f} - ${metrics['day_high']:.2f}"
        )
    
    with col4:
        market_status = "🟢 Open" if metrics['is_market_open'] else "🔴 Closed"
        st.metric(
            label="Market Status",
            value=market_status
        )
    
    # Last update time
    st.caption(f"Last updated: {metrics['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")