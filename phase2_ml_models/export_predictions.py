"""
Export all predictions to a single Excel file
"""
import pandas as pd
import os

predictions_dir = 'data/predictions'

# Create Excel writer
with pd.ExcelWriter('All_Stock_Predictions.xlsx', engine='openpyxl') as writer:
    
    for file in sorted(os.listdir(predictions_dir)):
        if file.endswith('_predictions.csv'):
            ticker = file.replace('_predictions.csv', '')
            df = pd.read_csv(os.path.join(predictions_dir, file))
            
            # Write each stock to a separate sheet
            df.to_excel(writer, sheet_name=ticker, index=False)
    
    print("✅ Exported all predictions to 'All_Stock_Predictions.xlsx'")

# Also create a summary sheet
summary_data = []

for file in sorted(os.listdir(predictions_dir)):
    if file.endswith('_predictions.csv'):
        ticker = file.replace('_predictions.csv', '')
        df = pd.read_csv(os.path.join(predictions_dir, file))
        
        summary_data.append({
            'Ticker': ticker,
            'Current_Predicted': df.iloc[0]['predicted_price'],
            '30_Day_Predicted': df.iloc[-1]['predicted_price'],
            'Expected_Change_%': ((df.iloc[-1]['predicted_price'] - df.iloc[0]['predicted_price']) / df.iloc[0]['predicted_price']) * 100
        })

summary_df = pd.DataFrame(summary_data)

with pd.ExcelWriter('All_Stock_Predictions.xlsx', engine='openpyxl', mode='a') as writer:
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

print("✅ Added summary sheet")
print(f"\n📊 Prediction Summary:")
print(summary_df.to_string(index=False))