"""
View all stock predictions
"""
import pandas as pd
import os

predictions_dir = 'data/predictions'

print("\n" + "="*80)
print("STOCK PRICE PREDICTIONS - NEXT 30 DAYS")
print("="*80 + "\n")

for file in sorted(os.listdir(predictions_dir)):
    if file.endswith('_predictions.csv'):
        ticker = file.replace('_predictions.csv', '')
        df = pd.read_csv(os.path.join(predictions_dir, file))
        
        print(f"\n{'='*80}")
        print(f"📊 {ticker} - Next 10 Days Forecast")
        print(f"{'='*80}")
        
        # Show first 10 predictions
        print(f"{'Date':<12} {'Predicted Price':<15} {'Lower Bound':<15} {'Upper Bound':<15}")
        print("-" * 80)
        
        for i in range(min(10, len(df))):
            row = df.iloc[i]
            date = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
            print(f"{date:<12} ${row['predicted_price']:>10.2f}     "
                  f"${row['lower_bound']:>10.2f}     ${row['upper_bound']:>10.2f}")
        
        # Summary
        start_price = df.iloc[0]['predicted_price']
        end_price = df.iloc[-1]['predicted_price']
        change_pct = ((end_price - start_price) / start_price) * 100
        
        print("-" * 80)
        print(f"30-Day Outlook: ${start_price:.2f} → ${end_price:.2f} ({change_pct:+.2f}%)")
        
        if change_pct > 5:
            print(f"📈 Trend: BULLISH (Strong Upward)")
        elif change_pct > 0:
            print(f"📈 Trend: BULLISH (Slight Upward)")
        elif change_pct > -5:
            print(f"📉 Trend: BEARISH (Slight Downward)")
        else:
            print(f"📉 Trend: BEARISH (Strong Downward)")

print("\n" + "="*80)
print("✅ All predictions loaded successfully!")
print("="*80 + "\n")