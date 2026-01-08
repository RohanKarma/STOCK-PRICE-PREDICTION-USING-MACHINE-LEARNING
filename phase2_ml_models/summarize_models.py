"""
Summarize all model performances
"""
import json
import os
import pandas as pd

saved_models_dir = 'models/saved_models'

print("\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY - ALL STOCKS")
print("="*80 + "\n")

all_results = []

for file in sorted(os.listdir(saved_models_dir)):
    if file.endswith('_model_results.json'):
        ticker = file.replace('_model_results.json', '')
        
        with open(os.path.join(saved_models_dir, file), 'r') as f:
            results = json.load(f)
        
        for model_name, data in results.items():
            if model_name != 'timestamp':
                metrics = data['metrics']
                all_results.append({
                    'Ticker': ticker,
                    'Model': model_name,
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE'],
                    'R²': metrics['R2'],
                    'Directional_Accuracy': metrics['Directional_Accuracy'],
                    'Training_Time': data.get('training_time', 0)
                })

df = pd.DataFrame(all_results)

# Show best model for each stock
print("\n🏆 BEST MODEL PER STOCK (by RMSE):")
print("="*80)

for ticker in df['Ticker'].unique():
    ticker_df = df[df['Ticker'] == ticker]
    # Exclude Linear Regression if it has suspiciously low RMSE
    ticker_df_filtered = ticker_df[ticker_df['RMSE'] > 0.1]
    
    if not ticker_df_filtered.empty:
        best = ticker_df_filtered.loc[ticker_df_filtered['RMSE'].idxmin()]
    else:
        best = ticker_df.loc[ticker_df['RMSE'].idxmin()]
    
    print(f"\n{ticker}:")
    print(f"  Best Model: {best['Model']}")
    print(f"  RMSE: ${best['RMSE']:.2f}")
    print(f"  MAE: ${best['MAE']:.2f}")
    print(f"  R²: {best['R²']:.4f}")
    print(f"  Directional Accuracy: {best['Directional_Accuracy']:.2f}%")
    print(f"  Training Time: {best['Training_Time']:.2f}s")

# Average performance by model type
print("\n\n📊 AVERAGE PERFORMANCE BY MODEL TYPE:")
print("="*80)

for model in df['Model'].unique():
    model_df = df[df['Model'] == model]
    
    print(f"\n{model}:")
    print(f"  Avg RMSE: ${model_df['RMSE'].mean():.2f}")
    print(f"  Avg MAE: ${model_df['MAE'].mean():.2f}")
    print(f"  Avg R²: {model_df['R²'].mean():.4f}")
    print(f"  Avg Directional Accuracy: {model_df['Directional_Accuracy'].mean():.2f}%")
    print(f"  Avg Training Time: {model_df['Training_Time'].mean():.2f}s")

print("\n" + "="*80)