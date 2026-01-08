"""
Multi-Stock Comparison Page - Dark Theme Edition
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Page config
st.set_page_config(page_title="Stock Comparison", page_icon="⚖️", layout="wide")

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
try:
    from realtime_data import RealTimeData
except:
    pass

st.title("⚖️ Multi-Stock Comparison")
st.markdown("Compare multiple stocks side-by-side with AI predictions and analytics")
st.markdown("---")

# Load available stocks
predictions_dir = '../phase2_ml_models/data/predictions'
phase1_data_dir = '../phase1_data_pipeline/data/processed'

if os.path.exists(predictions_dir):
    available_stocks = [f.replace('_predictions.csv', '') for f in os.listdir(predictions_dir) if f.endswith('.csv')]
    
    # Sidebar - Stock Selection
    st.sidebar.header("🎯 Select Stocks to Compare")
    
    selected_stocks = st.sidebar.multiselect(
        "Choose stocks (2-5 recommended):",
        available_stocks,
        default=available_stocks[:3] if len(available_stocks) >= 3 else available_stocks
    )
    
    if len(selected_stocks) < 2:
        st.warning("⚠️ Please select at least 2 stocks to compare")
        st.stop()
    
    # Comparison settings
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Comparison Settings")
    
    time_period = st.sidebar.selectbox(
        "Time Period:",
        ["1M", "3M", "6M", "1Y", "5Y", "MAX"],
        index=3
    )
    
    show_predictions = st.sidebar.checkbox("Show AI Predictions", value=True)
    show_correlation = st.sidebar.checkbox("Show Correlation Analysis", value=True)
    normalize_prices = st.sidebar.checkbox("Normalize Prices (% Change)", value=False)
    
    # Map time periods to days
    period_days = {
        "1M": 30,
        "3M": 90,
        "6M": 180,
        "1Y": 365,
        "5Y": 1825,
        "MAX": 99999
    }
    
    days = period_days[time_period]
    
    # Load data
    stock_data = {}
    prediction_data = {}
    
    with st.spinner('Loading data for selected stocks...'):
        for ticker in selected_stocks:
            # Load historical data
            hist_file = os.path.join(phase1_data_dir, f'{ticker}_processed.csv')
            if os.path.exists(hist_file):
                df = pd.read_csv(hist_file)
                df['date'] = pd.to_datetime(df['date'])
                df = df.tail(days)
                stock_data[ticker] = df
            
            # Load predictions
            pred_file = os.path.join(predictions_dir, f'{ticker}_predictions.csv')
            if os.path.exists(pred_file):
                pred_df = pd.read_csv(pred_file)
                pred_df['date'] = pd.to_datetime(pred_df['date'])
                prediction_data[ticker] = pred_df
    
    # Overview Metrics
    st.markdown("## 📊 Quick Comparison")
    
    metrics_cols = st.columns(len(selected_stocks))
    
    for idx, ticker in enumerate(selected_stocks):
        with metrics_cols[idx]:
            if ticker in stock_data:
                df = stock_data[ticker]
                current_price = df['close'].iloc[-1]
                start_price = df['close'].iloc[0]
                change = current_price - start_price
                change_pct = (change / start_price) * 100
                
                st.metric(
                    label=f"**{ticker}**",
                    value=f"${current_price:.2f}",
                    delta=f"{change_pct:+.2f}%"
                )
                
                # Try live price
                try:
                    rtd = RealTimeData(ticker)
                    live = rtd.get_live_metrics()
                    if live:
                        st.caption(f"🔴 Live: ${live['price']:.2f}")
                except:
                    pass
    
    st.markdown("---")
    
    # Price Comparison Chart - DARK THEME
    st.markdown("## 📈 Price Comparison")
    
    fig = go.Figure()
    
    colors = ['#60a5fa', '#4ade80', '#fbbf24', '#f87171', '#a78bfa']
    
    for idx, ticker in enumerate(selected_stocks):
        if ticker in stock_data:
            df = stock_data[ticker]
            
            if normalize_prices:
                base_price = df['close'].iloc[0]
                y_values = ((df['close'] - base_price) / base_price) * 100
                y_label = "% Change"
            else:
                y_values = df['close']
                y_label = "Price (USD)"
            
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=y_values,
                mode='lines',
                name=ticker,
                line=dict(color=colors[idx % len(colors)], width=3),
                hovertemplate=f'<b>{ticker}</b><br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
            ))
    
    fig.update_layout(
        title={
            'text': f"Stock Price Comparison ({time_period})",
            'font': {'size': 20, 'color': '#e2e8f0'}
        },
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,41,59,0.4)',
        font=dict(color='#e2e8f0', size=12),
        height=550,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            bgcolor="rgba(30,41,59,0.8)",
            bordercolor="rgba(102,126,234,0.3)",
            borderwidth=1,
            font=dict(color='#e2e8f0', size=11)
        )
    )
    
    fig.update_xaxes(
        title_text="Date",
        title_font=dict(color='#e2e8f0', size=13),
        tickfont=dict(color='#e2e8f0', size=10),
        gridcolor='rgba(148,163,184,0.1)'
    )
    
    fig.update_yaxes(
        title_text=y_label,
        title_font=dict(color='#e2e8f0', size=13),
        tickfont=dict(color='#e2e8f0', size=10),
        gridcolor='rgba(148,163,184,0.1)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Performance Metrics Table
    st.markdown("## 📊 Performance Metrics")
    
    performance_data = []
    
    for ticker in selected_stocks:
        if ticker in stock_data:
            df = stock_data[ticker]
            
            current_price = df['close'].iloc[-1]
            start_price = df['close'].iloc[0]
            high = df['high'].max()
            low = df['low'].min()
            avg_volume = df['volume'].mean()
            
            total_return = ((current_price - start_price) / start_price) * 100
            volatility = df['close'].pct_change().std() * np.sqrt(252) * 100
            
            performance_data.append({
                'Stock': ticker,
                'Price': current_price,
                'Return': total_return,
                'High': high,
                'Low': low,
                'Volatility': volatility,
                'Volume': avg_volume
            })
    
    perf_df = pd.DataFrame(performance_data)
    
    styled_perf = perf_df.style.format({
        'Price': '${:.2f}',
        'Return': '{:+.2f}%',
        'High': '${:.2f}',
        'Low': '${:.2f}',
        'Volatility': '{:.2f}%',
        'Volume': '{:,.0f}'
    }).background_gradient(
        subset=['Return'],
        cmap='RdYlGn',
        vmin=-20,
        vmax=20
    ).background_gradient(
        subset=['Volatility'],
        cmap='YlOrRd_r',
        vmin=0,
        vmax=50
    )
    
    st.dataframe(styled_perf, use_container_width=True)
    
    st.markdown("---")
    
    # Predictions Comparison - DARK THEME
    if show_predictions and prediction_data:
        st.markdown("## 🔮 AI Predictions Comparison")
        
        fig_pred = go.Figure()
        
        for idx, ticker in enumerate(selected_stocks):
            if ticker in prediction_data:
                pred_df = prediction_data[ticker]
                pred_display = pred_df.head(30)
                
                fig_pred.add_trace(go.Scatter(
                    x=pred_display['date'],
                    y=pred_display['predicted_price'],
                    mode='lines+markers',
                    name=ticker,
                    line=dict(color=colors[idx % len(colors)], width=3),
                    marker=dict(size=5)
                ))
        
        fig_pred.update_layout(
            title={
                'text': "30-Day Price Predictions",
                'font': {'size': 20, 'color': '#e2e8f0'}
            },
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,41,59,0.4)',
            font=dict(color='#e2e8f0', size=12),
            height=500,
            hovermode='x unified',
            legend=dict(
                bgcolor="rgba(30,41,59,0.8)",
                bordercolor="rgba(102,126,234,0.3)",
                borderwidth=1,
                font=dict(color='#e2e8f0', size=11)
            )
        )
        
        fig_pred.update_xaxes(
            title_text="Date",
            title_font=dict(color='#e2e8f0', size=13),
            tickfont=dict(color='#e2e8f0', size=10),
            gridcolor='rgba(148,163,184,0.1)'
        )
        
        fig_pred.update_yaxes(
            title_text="Predicted Price (USD)",
            title_font=dict(color='#e2e8f0', size=13),
            tickfont=dict(color='#e2e8f0', size=10),
            gridcolor='rgba(148,163,184,0.1)'
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Prediction Summary
        st.markdown("### 📊 Prediction Summary")
        
        pred_summary = []
        
        for ticker in selected_stocks:
            if ticker in prediction_data:
                pred_df = prediction_data[ticker]
                
                current_pred = pred_df.iloc[0]['predicted_price']
                future_pred = pred_df.iloc[29]['predicted_price']
                expected_change = ((future_pred - current_pred) / current_pred) * 100
                
                if ticker in stock_data:
                    actual_price = stock_data[ticker]['close'].iloc[-1]
                else:
                    actual_price = current_pred
                
                if expected_change > 2:
                    signal = '🟢 BUY'
                elif expected_change < -2:
                    signal = '🔴 SELL'
                else:
                    signal = '⚪ HOLD'
                
                pred_summary.append({
                    'Stock': ticker,
                    'Current': actual_price,
                    'Next Day': current_pred,
                    '30-Day': future_pred,
                    'Change %': expected_change,
                    'Signal': signal
                })
        
        pred_summary_df = pd.DataFrame(pred_summary)
        
        styled_pred = pred_summary_df.style.format({
            'Current': '${:.2f}',
            'Next Day': '${:.2f}',
            '30-Day': '${:.2f}',
            'Change %': '{:+.2f}%'
        }).background_gradient(
            subset=['Change %'],
            cmap='RdYlGn',
            vmin=-10,
            vmax=10
        )
        
        st.dataframe(styled_pred, use_container_width=True)
        
        st.markdown("---")
    
    # Correlation Analysis - FIXED VERSION
    if show_correlation and len(selected_stocks) > 1:
        st.markdown("## 🎯 Correlation Analysis")
        
        # Create correlation matrix
        price_data = {}
        
        for ticker in selected_stocks:
            if ticker in stock_data:
                df = stock_data[ticker]
                price_data[ticker] = df.set_index('date')['close']
        
        # Combine into single dataframe
        combined_df = pd.DataFrame(price_data)
        
        # Calculate correlation
        correlation = combined_df.corr()
        
        # Create heatmap - FIXED COLORBAR SYNTAX
        fig_corr = go.Figure(data=go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.columns,
            colorscale='RdYlGn',
            zmid=0,
            text=correlation.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 13, "color": "#000000"},
            colorbar=dict(
                title="Correlation",
                title_font=dict(color='#e2e8f0'),  # FIXED: title_font
                tickfont=dict(color='#e2e8f0')
            )
        ))
        
        fig_corr.update_layout(
            title={
                'text': "Stock Price Correlation Matrix",
                'font': {'size': 20, 'color': '#e2e8f0'}
            },
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,41,59,0.4)',
            font=dict(color='#e2e8f0', size=12),
            height=450,
            xaxis={'side': 'bottom'}
        )
        
        fig_corr.update_xaxes(
            tickfont=dict(color='#e2e8f0', size=11)
        )
        
        fig_corr.update_yaxes(
            tickfont=dict(color='#e2e8f0', size=11)
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Correlation insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔗 Highest Correlation")
            
            # Find highest correlation (excluding diagonal)
            corr_values = []
            for i in range(len(correlation)):
                for j in range(i+1, len(correlation)):
                    corr_values.append({
                        'Pair': f"{correlation.index[i]} - {correlation.columns[j]}",
                        'Correlation': correlation.iloc[i, j]
                    })
            
            if corr_values:
                corr_df = pd.DataFrame(corr_values).sort_values('Correlation', ascending=False)
                st.dataframe(corr_df.head(5), use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Correlation Strength")
            
            avg_corr = correlation.values[np.triu_indices_from(correlation.values, k=1)].mean()
            
            st.metric("Average Correlation", f"{avg_corr:.2f}")
            
            if avg_corr > 0.7:
                st.info("🔗 **High correlation** - Stocks move together")
            elif avg_corr > 0.3:
                st.info("↔️ **Moderate correlation** - Some relationship")
            else:
                st.info("🔀 **Low correlation** - Independent movements")
        
        st.markdown("---")
        
    
    # Volume Comparison - DARK THEME
    st.markdown("## 📦 Volume Comparison")
    
    volume_data = []
    
    for ticker in selected_stocks:
        if ticker in stock_data:
            df = stock_data[ticker]
            avg_volume = df['volume'].mean()
            latest_volume = df['volume'].iloc[-1]
            
            volume_data.append({
                'Stock': ticker,
                'Average': avg_volume,
                'Latest': latest_volume
            })
    
    vol_df = pd.DataFrame(volume_data)
    
    fig_vol = go.Figure()
    
    fig_vol.add_trace(go.Bar(
        x=vol_df['Stock'],
        y=vol_df['Average'],
        name='Average Volume',
        marker_color='#60a5fa',
        text=vol_df['Average'],
        texttemplate='%{text:,.0f}',
        textposition='outside',
        textfont=dict(color='#e2e8f0', size=11)
    ))
    
    fig_vol.add_trace(go.Bar(
        x=vol_df['Stock'],
        y=vol_df['Latest'],
        name='Latest Volume',
        marker_color='#4ade80',
        text=vol_df['Latest'],
        texttemplate='%{text:,.0f}',
        textposition='outside',
        textfont=dict(color='#e2e8f0', size=11)
    ))
    
    fig_vol.update_layout(
        title={
            'text': "Trading Volume Comparison",
            'font': {'size': 20, 'color': '#e2e8f0'}
        },
        barmode='group',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,41,59,0.4)',
        font=dict(color='#e2e8f0', size=12),
        height=450,
        legend=dict(
            bgcolor="rgba(30,41,59,0.8)",
            bordercolor="rgba(102,126,234,0.3)",
            borderwidth=1,
            font=dict(color='#e2e8f0', size=11)
        )
    )
    
    fig_vol.update_xaxes(
        title_text="Stock",
        title_font=dict(color='#e2e8f0', size=13),
        tickfont=dict(color='#e2e8f0', size=11),
        gridcolor='rgba(148,163,184,0.1)'
    )
    
    fig_vol.update_yaxes(
        title_text="Volume",
        title_font=dict(color='#e2e8f0', size=13),
        tickfont=dict(color='#e2e8f0', size=11),
        gridcolor='rgba(148,163,184,0.1)'
    )
    
    st.plotly_chart(fig_vol, use_container_width=True)
    
    st.markdown("---")
    
    # Risk vs Return - DARK THEME
    st.markdown("## 📈 Risk vs Return Analysis")
    
    risk_return_data = []
    
    for ticker in selected_stocks:
        if ticker in stock_data:
            df = stock_data[ticker]
            returns = df['close'].pct_change().dropna()
            total_return = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
            volatility = returns.std() * np.sqrt(252) * 100
            
            risk_return_data.append({
                'Stock': ticker,
                'Return': total_return,
                'Risk': volatility
            })
    
    rr_df = pd.DataFrame(risk_return_data)
    
    fig_rr = go.Figure()
    
    fig_rr.add_trace(go.Scatter(
        x=rr_df['Risk'],
        y=rr_df['Return'],
        mode='markers+text',
        marker=dict(size=25, color=colors[:len(rr_df)], opacity=0.8),
        text=rr_df['Stock'],
        textposition='top center',
        textfont=dict(size=14, color='#e2e8f0', family='Arial Black'),
        hovertemplate='<b>%{text}</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
    ))
    
    avg_risk = rr_df['Risk'].mean()
    avg_return = rr_df['Return'].mean()
    
    fig_rr.add_hline(y=avg_return, line_dash="dash", line_color="#94a3b8", opacity=0.5)
    fig_rr.add_vline(x=avg_risk, line_dash="dash", line_color="#94a3b8", opacity=0.5)
    
    fig_rr.update_layout(
        title={
            'text': "Risk vs Return Profile",
            'font': {'size': 20, 'color': '#e2e8f0'}
        },
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,41,59,0.4)',
        font=dict(color='#e2e8f0', size=12),
        height=550,
        showlegend=False
    )
    
    fig_rr.update_xaxes(
        title_text="Risk (Volatility %)",
        title_font=dict(color='#e2e8f0', size=13),
        tickfont=dict(color='#e2e8f0', size=11),
        gridcolor='rgba(148,163,184,0.1)'
    )
    
    fig_rr.update_yaxes(
        title_text="Return (%)",
        title_font=dict(color='#e2e8f0', size=13),
        tickfont=dict(color='#e2e8f0', size=11),
        gridcolor='rgba(148,163,184,0.1)'
    )
    
    st.plotly_chart(fig_rr, use_container_width=True)
    
    # Quadrant analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Best Risk-Adjusted Performance")
        
        rr_df['Sharpe'] = rr_df['Return'] / rr_df['Risk']
        best_sharpe = rr_df.sort_values('Sharpe', ascending=False).iloc[0]
        
        st.success(f"**{best_sharpe['Stock']}** - Best risk-adjusted returns")
        st.metric("Sharpe Ratio", f"{best_sharpe['Sharpe']:.2f}")
    
    with col2:
        st.markdown("### 📊 Risk Categories")
        
        for _, row in rr_df.iterrows():
            if row['Risk'] < avg_risk and row['Return'] > avg_return:
                category = "🟢 Low Risk, High Return"
            elif row['Risk'] > avg_risk and row['Return'] > avg_return:
                category = "🟡 High Risk, High Return"
            elif row['Risk'] < avg_risk and row['Return'] < avg_return:
                category = "🔵 Low Risk, Low Return"
            else:
                category = "🔴 High Risk, Low Return"
            
            st.caption(f"**{row['Stock']}**: {category}")
    
    st.markdown("---")
    
    # Portfolio Allocation - DARK THEME
    st.markdown("## 💼 Suggested Portfolio Allocation")
    
    total_sharpe = rr_df['Sharpe'].sum()
    rr_df['Allocation'] = (rr_df['Sharpe'] / total_sharpe) * 100
    
    fig_allocation = go.Figure(data=[go.Pie(
        labels=rr_df['Stock'],
        values=rr_df['Allocation'],
        hole=0.4,
        marker_colors=colors[:len(rr_df)],
        textfont=dict(color='#ffffff', size=13),
        hovertemplate='<b>%{label}</b><br>Allocation: %{value:.1f}%<extra></extra>'
    )])
    
    fig_allocation.update_layout(
        title={
            'text': 'Recommended Portfolio Distribution (Risk-Adjusted)',
            'font': {'size': 20, 'color': '#e2e8f0'}
        },
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,41,59,0.4)',
        font=dict(color='#e2e8f0', size=12),
        height=450,
        legend=dict(
            bgcolor="rgba(30,41,59,0.8)",
            bordercolor="rgba(102,126,234,0.3)",
            borderwidth=1,
            font=dict(color='#e2e8f0', size=11)
        )
    )
    
    st.plotly_chart(fig_allocation, use_container_width=True)
    
    # Allocation table
    alloc_display = rr_df[['Stock', 'Allocation', 'Return', 'Risk']].copy()
    alloc_display['Allocation'] = alloc_display['Allocation'].apply(lambda x: f"{x:.1f}%")
    alloc_display['Return'] = alloc_display['Return'].apply(lambda x: f"{x:.2f}%")
    alloc_display['Risk'] = alloc_display['Risk'].apply(lambda x: f"{x:.2f}%")
    
    st.dataframe(alloc_display, use_container_width=True)
    
    st.info("💡 **Note:** This is a simplified allocation based on historical performance. Not financial advice!")
    
    # Export
    st.markdown("---")
    st.markdown("### 📥 Export Comparison Data")
    
    export_data = perf_df.merge(
        pred_summary_df if show_predictions and 'pred_summary_df' in locals() else pd.DataFrame(),
        on='Stock',
        how='left'
    )
    
    csv = export_data.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Comparison Report (CSV)",
        data=csv,
        file_name=f"stock_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

else:
    st.error("❌ No stock data found! Please run Phase 2 first.")
    st.info("👉 Go to `phase2_ml_models` folder and run `python main.py`")