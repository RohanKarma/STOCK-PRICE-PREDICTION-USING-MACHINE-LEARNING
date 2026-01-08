"""
Portfolio Tracker Page - Dark Theme Edition (Complete Fixed)
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

# Page config
st.set_page_config(page_title="Portfolio", page_icon="💼", layout="wide")

# Load dark theme CSS
def load_css():
    css_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'dark_theme.css')
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Additional CSS for this page
st.markdown("""
<style>
/* Download button styling */
.stDownloadButton > button {
    background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%) !important;
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 12px rgba(74, 222, 128, 0.3) !important;
}

.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(74, 222, 128, 0.5) !important;
}

.stDownloadButton > button > div,
.stDownloadButton > button > div > p,
.stDownloadButton > button p {
    color: #ffffff !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("💼 Stock Portfolio Tracker")
st.markdown("Track all your stocks in one place")
st.markdown("---")

# Load all predictions
predictions_dir = '../phase2_ml_models/data/predictions'

if os.path.exists(predictions_dir):
    pred_files = [f for f in os.listdir(predictions_dir) if f.endswith('.csv')]
    
    if pred_files:
        # Create portfolio overview
        st.markdown("## 📊 Portfolio Overview")
        
        portfolio_data = []
        
        for file in pred_files:
            ticker = file.replace('_predictions.csv', '')
            df = pd.read_csv(os.path.join(predictions_dir, file))
            
            current = df.iloc[0]['predicted_price']
            future_30d = df.iloc[-1]['predicted_price']
            change_pct = ((future_30d - current) / current) * 100
            
            portfolio_data.append({
                'Ticker': ticker,
                'Current_Price': current,
                '30D_Prediction': future_30d,
                'Expected_Change_%': change_pct,
                'Trend': '📈 Bullish' if change_pct > 0 else '📉 Bearish'
            })
        
        portfolio_df = pd.DataFrame(portfolio_data)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Stocks", len(portfolio_df))
        
        with col2:
            bullish_count = (portfolio_df['Expected_Change_%'] > 0).sum()
            st.metric("Bullish Stocks", bullish_count, f"{(bullish_count/len(portfolio_df)*100):.0f}%")
        
        with col3:
            avg_change = portfolio_df['Expected_Change_%'].mean()
            st.metric("Avg Expected Change", f"{avg_change:+.2f}%")
        
        with col4:
            best_stock = portfolio_df.loc[portfolio_df['Expected_Change_%'].idxmax(), 'Ticker']
            st.metric("Best Performer", best_stock)
        
        st.markdown("---")
        
          # Portfolio table - MODERN GRID LAYOUT
        st.markdown("### 📋 Stock List")
        
        st.markdown("""
        <style>
        .stock-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stock-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(102, 126, 234, 0.2);
            transition: all 0.3s;
        }
        .stock-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
            border-color: #667eea;
        }
        .stock-ticker {
            font-size: 24px;
            font-weight: bold;
            color: #60a5fa;
            margin-bottom: 10px;
        }
        .stock-trend {
            font-size: 14px;
            color: #94a3b8;
            margin-bottom: 15px;
        }
        .stock-prices {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
        }
        .price-block {
            text-align: center;
        }
        .price-label {
            font-size: 11px;
            color: #94a3b8;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .price-value {
            font-size: 16px;
            color: #e2e8f0;
            font-weight: bold;
        }
        .change-block {
            background: rgba(102, 126, 234, 0.1);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        .change-positive {
            color: #4ade80;
            font-size: 20px;
            font-weight: bold;
        }
        .change-negative {
            color: #f87171;
            font-size: 20px;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create grid
        st.markdown('<div class="stock-grid">', unsafe_allow_html=True)
        
        for _, row in portfolio_df.iterrows():
            change_class = 'change-positive' if row['Expected_Change_%'] > 0 else 'change-negative'
            
            st.markdown(f"""
            <div class="stock-card">
                <div class="stock-ticker">{row['Ticker']}</div>
                <div class="stock-trend">{row['Trend']}</div>
                <div class="stock-prices">
                    <div class="price-block">
                        <div class="price-label">Current</div>
                        <div class="price-value">${row['Current_Price']:.2f}</div>
                    </div>
                    <div class="price-block">
                        <div class="price-label">30D Target</div>
                        <div class="price-value">${row['30D_Prediction']:.2f}</div>
                    </div>
                </div>
                <div class="change-block">
                    <div class="price-label">Expected Change</div>
                    <div class="{change_class}">{row['Expected_Change_%']:+.2f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        # Visualization
        st.markdown("---")
        st.markdown("### 📊 Performance Comparison")
        
        # Expected change chart - DARK THEME
        fig1 = go.Figure()
        
        # Create color array based on values
        colors = ['#4ade80' if x > 0 else '#f87171' for x in portfolio_df['Expected_Change_%']]
        
        fig1.add_trace(go.Bar(
            x=portfolio_df['Ticker'],
            y=portfolio_df['Expected_Change_%'],
            marker_color=colors,
            text=portfolio_df['Expected_Change_%'].round(2),
            textposition='outside',
            textfont=dict(color='#e2e8f0', size=12),
            hovertemplate='<b>%{x}</b><br>Expected Change: %{y:+.2f}%<extra></extra>'
        ))
        
        fig1.update_layout(
            title={
                'text': "30-Day Expected Returns by Stock",
                'font': {'size': 18, 'color': '#e2e8f0'}
            },
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,41,59,0.4)',
            font=dict(color='#e2e8f0', size=12),
            height=450,
            showlegend=False
        )
        
        fig1.update_xaxes(
            title_text="Stock",
            title_font=dict(color='#e2e8f0', size=13),
            tickfont=dict(color='#e2e8f0', size=11),
            gridcolor='rgba(148,163,184,0.1)'
        )
        
        fig1.update_yaxes(
            title_text="Expected Change (%)",
            title_font=dict(color='#e2e8f0', size=13),
            tickfont=dict(color='#e2e8f0', size=11),
            gridcolor='rgba(148,163,184,0.1)',
            zeroline=True,
            zerolinecolor='rgba(148,163,184,0.3)',
            zerolinewidth=2
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Price comparison - DARK THEME
        st.markdown("---")
        st.markdown("### 💰 Price Levels")
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Bar(
            name='Current Price',
            x=portfolio_df['Ticker'],
            y=portfolio_df['Current_Price'],
            marker_color='#60a5fa',
            text=portfolio_df['Current_Price'].round(2),
            textposition='outside',
            textfont=dict(color='#e2e8f0', size=11),
            hovertemplate='<b>%{x}</b><br>Current: $%{y:.2f}<extra></extra>'
        ))
        
        fig2.add_trace(go.Bar(
            name='30-Day Prediction',
            x=portfolio_df['Ticker'],
            y=portfolio_df['30D_Prediction'],
            marker_color='#4ade80',
            text=portfolio_df['30D_Prediction'].round(2),
            textposition='outside',
            textfont=dict(color='#e2e8f0', size=11),
            hovertemplate='<b>%{x}</b><br>Predicted: $%{y:.2f}<extra></extra>'
        ))
        
        fig2.update_layout(
            title={
                'text': "Current vs Predicted Prices",
                'font': {'size': 18, 'color': '#e2e8f0'}
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
        
        fig2.update_xaxes(
            title_text="Stock",
            title_font=dict(color='#e2e8f0', size=13),
            tickfont=dict(color='#e2e8f0', size=11),
            gridcolor='rgba(148,163,184,0.1)'
        )
        
        fig2.update_yaxes(
            title_text="Price ($)",
            title_font=dict(color='#e2e8f0', size=13),
            tickfont=dict(color='#e2e8f0', size=11),
            gridcolor='rgba(148,163,184,0.1)'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Risk Analysis
        st.markdown("---")
        st.markdown("### ⚠️ Risk Distribution")
        
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        
        high_risk = portfolio_df[portfolio_df['Expected_Change_%'].abs() > 5]
        medium_risk = portfolio_df[(portfolio_df['Expected_Change_%'].abs() > 2) & (portfolio_df['Expected_Change_%'].abs() <= 5)]
        low_risk = portfolio_df[portfolio_df['Expected_Change_%'].abs() <= 2]
        
        with risk_col1:
            st.metric("High Volatility", len(high_risk), "±>5%")
        
        with risk_col2:
            st.metric("Medium Volatility", len(medium_risk), "±2-5%")
        
        with risk_col3:
            st.metric("Low Volatility", len(low_risk), "±<2%")
        
        # Allocation suggestion
        st.markdown("---")
        st.markdown("### 💡 Portfolio Allocation Suggestion")
        
        alloc_col1, alloc_col2 = st.columns([2, 1])
        
        with alloc_col1:
            st.info("""
            **Based on predicted returns, consider:**
            
            - **High allocation (40-50%)**: Stocks with positive expected returns (>3%)
            - **Medium allocation (30-40%)**: Stocks with slight positive returns (0-3%)
            - **Low allocation (10-20%)**: Stocks with negative expected returns
            
            ⚠️ **Disclaimer:** These are AI predictions, not financial advice!
            """)
        
        with alloc_col2:
            # Summary box
            st.markdown("""
            <div style='
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                border-left: 4px solid #667eea;
                border-radius: 8px;
                padding: 20px;
            '>
                <h4 style='color: #667eea; margin: 0 0 15px 0;'>📊 Summary</h4>
                <p style='color: #e2e8f0; margin: 5px 0;'><strong>Total Stocks:</strong> {}</p>
                <p style='color: #4ade80; margin: 5px 0;'><strong>Bullish:</strong> {}</p>
                <p style='color: #f87171; margin: 5px 0;'><strong>Bearish:</strong> {}</p>
                <p style='color: #e2e8f0; margin: 5px 0;'><strong>Avg Change:</strong> {:+.2f}%</p>
            </div>
            """.format(
                len(portfolio_df),
                len(portfolio_df[portfolio_df['Expected_Change_%'] > 0]),
                len(portfolio_df[portfolio_df['Expected_Change_%'] < 0]),
                portfolio_df['Expected_Change_%'].mean()
            ), unsafe_allow_html=True)
        
        # Pie chart - DARK THEME
        st.markdown("---")
        st.markdown("### 🥧 Portfolio Distribution")
        
        fig3 = go.Figure(data=[go.Pie(
            labels=portfolio_df['Ticker'],
            values=[1]*len(portfolio_df),
            hole=0.3,
            marker=dict(
                colors=['#667eea', '#60a5fa', '#4ade80', '#fbbf24', '#f87171', '#a78bfa', '#f093fb'][:len(portfolio_df)]
            ),
            textfont=dict(color='#ffffff', size=13),
            hovertemplate='<b>%{label}</b><br>Allocation: %{percent}<extra></extra>'
        )])
        
        fig3.update_layout(
            title={
                'text': 'Current Portfolio Distribution (Equal Weight)',
                'font': {'size': 18, 'color': '#e2e8f0'}
            },
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,41,59,0.4)',
            font=dict(color='#e2e8f0', size=12),
            height=450,
            showlegend=True,
            legend=dict(
                bgcolor="rgba(30,41,59,0.8)",
                bordercolor="rgba(102,126,234,0.3)",
                borderwidth=1,
                font=dict(color='#e2e8f0', size=11)
            )
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Performance breakdown
        st.markdown("---")
        st.markdown("### 📈 Performance Breakdown")
        
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            st.markdown("#### 🟢 Top Gainers")
            top_gainers = portfolio_df.nlargest(3, 'Expected_Change_%')[['Ticker', 'Expected_Change_%']]
            for _, row in top_gainers.iterrows():
                st.success(f"**{row['Ticker']}**: {row['Expected_Change_%']:+.2f}%")
        
        with perf_col2:
            st.markdown("#### 🔴 Top Losers")
            top_losers = portfolio_df.nsmallest(3, 'Expected_Change_%')[['Ticker', 'Expected_Change_%']]
            for _, row in top_losers.iterrows():
                st.error(f"**{row['Ticker']}**: {row['Expected_Change_%']:+.2f}%")
        
        # Download portfolio data - FIXED BUTTON
        st.markdown("---")
        
        csv = portfolio_df.to_csv(index=False)
        
        # Center the download button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="📥 Download Portfolio Data (CSV)",
                data=csv,
                file_name=f"portfolio_summary_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
    else:
        st.error("❌ No prediction files found!")
        st.info("👉 Please run Phase 2 model training first")
        
        with st.expander("📝 How to generate predictions"):
            st.code("""
# Navigate to phase2 directory
cd ../phase2_ml_models

# Run training
python main.py

# This will create predictions for all stocks
            """)
        
else:
    st.error("❌ Predictions directory not found!")
    st.info("👉 Please run Phase 2 to generate predictions")
    
    with st.expander("📝 Setup Instructions"):
        st.markdown("""
        **Steps to set up predictions:**
        
        1. Navigate to Phase 2 directory:
        ```bash
        cd ../phase2_ml_models
        ```
        
        2. Run the main script:
        ```bash
        python main.py
        ```
        
        3. Wait for models to train and predictions to generate
        
        4. Return to dashboard and refresh this page
        """)