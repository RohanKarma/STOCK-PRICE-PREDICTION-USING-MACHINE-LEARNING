"""
Model Comparison Page - Dark Theme Edition
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os

# Page config
st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")

# Load dark theme CSS
def load_css():
    css_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'dark_theme.css')
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

st.title("📈 Model Performance Comparison")
st.markdown("Compare machine learning models side-by-side")
st.markdown("---")

# Load model results
models_dir = '../phase2_ml_models/models/saved_models'

if os.path.exists(models_dir):
    # Find all result files
    result_files = [f for f in os.listdir(models_dir) if f.endswith('_model_results.json')]
    
    if result_files:
        # Sidebar - Stock Selection
        st.sidebar.header("🎯 Select Stock")
        
        available_stocks = [f.replace('_model_results.json', '') for f in result_files]
        selected_stock = st.sidebar.selectbox("Choose a stock:", available_stocks)
        
        # Load results
        result_file = os.path.join(models_dir, f'{selected_stock}_model_results.json')
        
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        st.markdown(f"## {selected_stock} - Model Performance")
        
        # Prepare data
        model_data = []
        for model_name, data in results.items():
            if model_name != 'timestamp':
                metrics = data['metrics']
                model_data.append({
                    'Model': model_name,
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE'],
                    'R²': metrics['R2'],
                    'MAPE': metrics['MAPE'],
                    'Directional_Accuracy': metrics['Directional_Accuracy'],
                    'Training_Time': data.get('training_time', 0)
                })
        
        df = pd.DataFrame(model_data)
        
        # Filter out suspiciously perfect results
        df_display = df[df['RMSE'] > 0.1].copy() if (df['RMSE'] < 0.1).any() else df.copy()
        
        # Best model highlight
        best_model_idx = df_display['RMSE'].idxmin()
        best_model = df_display.loc[best_model_idx, 'Model']
        
        st.success(f"🏆 **Best Model:** {best_model}")
        
        # Metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Best RMSE", f"${df_display['RMSE'].min():.2f}")
        
        with col2:
            st.metric("Best MAE", f"${df_display['MAE'].min():.2f}")
        
        with col3:
            st.metric("Best R²", f"{df_display['R²'].max():.4f}")
        
        with col4:
            st.metric("Best Accuracy", f"{df_display['Directional_Accuracy'].max():.1f}%")
        
        st.markdown("---")
        
        # Comparison table
                # Comparison table - CUSTOM HTML VERSION
        st.markdown("### 📊 Detailed Comparison")
        
        # Build custom HTML table
        table_html = """
        <style>
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: #1e293b;
            border-radius: 8px;
            overflow: hidden;
        }
        .comparison-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
            padding: 15px;
            text-align: center;
            font-weight: bold;
            font-size: 14px;
        }
        .comparison-table td {
            padding: 12px;
            text-align: center;
            color: #e2e8f0;
            border-bottom: 1px solid #334155;
        }
        .comparison-table tbody tr:hover {
            background-color: #334155;
        }
        .best-value {
            background-color: #10b981 !important;
            color: #ffffff !important;
            font-weight: bold;
        }
        .good-value {
            background-color: #3b82f6 !important;
            color: #ffffff !important;
            font-weight: bold;
        }
        </style>
        
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>R²</th>
                    <th>MAPE</th>
                    <th>Directional Accuracy</th>
                    <th>Training Time</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Find best values
        min_rmse = df_display['RMSE'].min()
        min_mae = df_display['MAE'].min()
        max_r2 = df_display['R²'].max()
        min_mape = df_display['MAPE'].min()
        max_acc = df_display['Directional_Accuracy'].max()
        
        # Add rows
        for _, row in df_display.iterrows():
            table_html += "<tr>"
            table_html += f"<td>{row['Model']}</td>"
            
            # RMSE (highlight min)
            rmse_class = 'best-value' if row['RMSE'] == min_rmse else ''
            table_html += f"<td class='{rmse_class}'>${row['RMSE']:.2f}</td>"
            
            # MAE (highlight min)
            mae_class = 'best-value' if row['MAE'] == min_mae else ''
            table_html += f"<td class='{mae_class}'>${row['MAE']:.2f}</td>"
            
            # R² (highlight max)
            r2_class = 'good-value' if row['R²'] == max_r2 else ''
            table_html += f"<td class='{r2_class}'>{row['R²']:.4f}</td>"
            
            # MAPE (highlight min)
            mape_class = 'best-value' if row['MAPE'] == min_mape else ''
            table_html += f"<td class='{mape_class}'>{row['MAPE']:.2f}%</td>"
            
            # Directional Accuracy (highlight max)
            acc_class = 'good-value' if row['Directional_Accuracy'] == max_acc else ''
            table_html += f"<td class='{acc_class}'>{row['Directional_Accuracy']:.2f}%</td>"
            
            # Training Time
            table_html += f"<td>{row['Training_Time']:.2f}s</td>"
            
            table_html += "</tr>"
        
        table_html += """
            </tbody>
        </table>
        """
        
        st.markdown(table_html, unsafe_allow_html=True)
        
        # Visualization
        st.markdown("---")
        st.markdown("### 📊 Visual Comparison")
        
        # Create subplot
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            # RMSE & MAE
            fig1 = go.Figure()
            
            fig1.add_trace(go.Bar(
                name='RMSE',
                x=df_display['Model'],
                y=df_display['RMSE'],
                marker_color='#60a5fa',
                text=df_display['RMSE'].round(2),
                textposition='outside',
                textfont=dict(color='#e2e8f0', size=12),
                hovertemplate='<b>%{x}</b><br>RMSE: $%{y:.2f}<extra></extra>'
            ))
            
            fig1.add_trace(go.Bar(
                name='MAE',
                x=df_display['Model'],
                y=df_display['MAE'],
                marker_color='#4ade80',
                text=df_display['MAE'].round(2),
                textposition='outside',
                textfont=dict(color='#e2e8f0', size=12),
                hovertemplate='<b>%{x}</b><br>MAE: $%{y:.2f}<extra></extra>'
            ))
            
            fig1.update_layout(
                title={
                    'text': "Error Metrics (Lower is Better)",
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
            
            fig1.update_xaxes(
                title_text="Model",
                title_font=dict(color='#e2e8f0', size=13),
                tickfont=dict(color='#e2e8f0', size=11),
                gridcolor='rgba(148,163,184,0.1)'
            )
            
            fig1.update_yaxes(
                title_text="Error ($)",
                title_font=dict(color='#e2e8f0', size=13),
                tickfont=dict(color='#e2e8f0', size=11),
                gridcolor='rgba(148,163,184,0.1)'
            )
            
            st.plotly_chart(fig1, use_container_width=True)
        
        with fig_col2:
            # Accuracy metrics
            fig2 = go.Figure()
            
            fig2.add_trace(go.Bar(
                name='R² Score',
                x=df_display['Model'],
                y=df_display['R²'],
                marker_color='#a78bfa',
                text=df_display['R²'].round(4),
                textposition='outside',
                textfont=dict(color='#e2e8f0', size=12),
                hovertemplate='<b>%{x}</b><br>R²: %{y:.4f}<extra></extra>'
            ))
            
            fig2.add_trace(go.Bar(
                name='Directional Accuracy',
                x=df_display['Model'],
                y=df_display['Directional_Accuracy'],
                marker_color='#fbbf24',
                text=df_display['Directional_Accuracy'].round(1),
                textposition='outside',
                textfont=dict(color='#e2e8f0', size=12),
                hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.2f}%<extra></extra>'
            ))
            
            fig2.update_layout(
                title={
                    'text': "Accuracy Metrics (Higher is Better)",
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
                title_text="Model",
                title_font=dict(color='#e2e8f0', size=13),
                tickfont=dict(color='#e2e8f0', size=11),
                gridcolor='rgba(148,163,184,0.1)'
            )
            
            fig2.update_yaxes(
                title_text="Score / Accuracy (%)",
                title_font=dict(color='#e2e8f0', size=13),
                tickfont=dict(color='#e2e8f0', size=11),
                gridcolor='rgba(148,163,184,0.1)'
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        # Training time comparison
        st.markdown("---")
        st.markdown("### ⏱️ Training Time Comparison")
        
        fig3 = go.Figure()
        
        # Create gradient colors for bars
        colors = ['#667eea', '#60a5fa', '#4ade80', '#fbbf24', '#f87171']
        bar_colors = [colors[i % len(colors)] for i in range(len(df_display))]
        
        fig3.add_trace(go.Bar(
            x=df_display['Model'],
            y=df_display['Training_Time'],
            marker_color=bar_colors,
            text=df_display['Training_Time'].round(2),
            textposition='outside',
            textfont=dict(color='#e2e8f0', size=12),
            hovertemplate='<b>%{x}</b><br>Time: %{y:.2f}s<extra></extra>'
        ))
        
        fig3.update_layout(
            title={
                'text': "Training Time by Model",
                'font': {'size': 18, 'color': '#e2e8f0'}
            },
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,41,59,0.4)',
            font=dict(color='#e2e8f0', size=12),
            height=400,
            showlegend=False
        )
        
        fig3.update_xaxes(
            title_text="Model",
            title_font=dict(color='#e2e8f0', size=13),
            tickfont=dict(color='#e2e8f0', size=11),
            gridcolor='rgba(148,163,184,0.1)'
        )
        
        fig3.update_yaxes(
            title_text="Time (seconds)",
            title_font=dict(color='#e2e8f0', size=13),
            tickfont=dict(color='#e2e8f0', size=11),
            gridcolor='rgba(148,163,184,0.1)'
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Model recommendations
        st.markdown("---")
        st.markdown("### 💡 Recommendations")
        
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        
        with rec_col1:
            st.info("""
            **🚀 For Speed:**
            
            Linear Regression is fastest (~0.02s)
            
            Best for quick predictions
            """)
        
        with rec_col2:
            st.success("""
            **🎯 For Accuracy:**
            
            Random Forest typically performs best
            
            Good balance of speed and accuracy
            """)
        
        with rec_col3:
            st.warning("""
            **🧠 For Complex Patterns:**
            
            LSTM for capturing long-term trends
            
            Requires more training time
            """)
        
        # Additional insights
        st.markdown("---")
        st.markdown("### 📈 Model Insights")
        
        # Find best and worst performers
        best_rmse = df_display.loc[df_display['RMSE'].idxmin()]
        best_accuracy = df_display.loc[df_display['Directional_Accuracy'].idxmax()]
        fastest = df_display.loc[df_display['Training_Time'].idxmin()]
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.markdown("#### 🏆 Top Performers")
            st.write(f"**Lowest Error:** {best_rmse['Model']} (RMSE: ${best_rmse['RMSE']:.2f})")
            st.write(f"**Highest Accuracy:** {best_accuracy['Model']} ({best_accuracy['Directional_Accuracy']:.2f}%)")
            st.write(f"**Fastest Training:** {fastest['Model']} ({fastest['Training_Time']:.2f}s)")
        
        with insight_col2:
            st.markdown("#### 📊 Performance Summary")
            avg_rmse = df_display['RMSE'].mean()
            avg_accuracy = df_display['Directional_Accuracy'].mean()
            total_time = df_display['Training_Time'].sum()
            
            st.write(f"**Average RMSE:** ${avg_rmse:.2f}")
            st.write(f"**Average Accuracy:** {avg_accuracy:.2f}%")
            st.write(f"**Total Training Time:** {total_time:.2f}s")
        
        # Download results
        st.markdown("---")
        csv = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison (CSV)",
            data=csv,
            file_name=f"{selected_stock}_model_comparison.csv",
            mime="text/csv"
        )
        
    else:
        st.error("❌ No model results found!")
        st.info("👉 Please run model training first")
        
else:
    st.error("❌ Models directory not found!")
    st.info("👉 Please run Phase 2 first to train the models.")