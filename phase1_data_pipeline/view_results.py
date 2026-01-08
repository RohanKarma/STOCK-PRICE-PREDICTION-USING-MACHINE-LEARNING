"""
View Phase 1 Results - Beautiful Summary
"""
import pandas as pd
import os

def print_header(text, char='='):
    print(f"\n{char * 70}")
    print(f"{text.center(70)}")
    print(f"{char * 70}\n")

# Main summary
print_header("🎉 PHASE 1 - DATA PIPELINE RESULTS 🎉", '=')

processed_dir = 'data/processed'
files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]

print(f"📁 Total Files Created: {len(files)}")
print(f"📊 Total Stocks Processed: {len(files)}")

print("\n" + "─" * 70)
print(f"{'TICKER':<10} {'ROWS':<10} {'COLUMNS':<10} {'FILE SIZE':<15} {'STATUS'}")
print("─" * 70)

total_rows = 0
for file in sorted(files):
    ticker = file.replace('_processed.csv', '')
    filepath = os.path.join(processed_dir, file)
    df = pd.read_csv(filepath)
    file_size = os.path.getsize(filepath) / 1024  # KB
    
    total_rows += len(df)
    
    print(f"{ticker:<10} {len(df):<10} {len(df.columns):<10} {file_size:>10.1f} KB   ✓")

print("─" * 70)
print(f"{'TOTAL':<10} {total_rows:<10} {'37':<10} {sum(os.path.getsize(os.path.join(processed_dir, f))/1024 for f in files):>10.1f} KB")
print("─" * 70)

# Sample data from AAPL
print_header("📈 SAMPLE DATA - AAPL (Last 5 Days)", '=')

df = pd.read_csv('data/processed/AAPL_processed.csv')
sample_cols = ['date', 'close', 'volume', 'sma_20', 'sma_50', 'rsi', 'macd']
print(df[sample_cols].tail().to_string(index=False))

# Technical Indicators Summary
print_header("🔧 TECHNICAL INDICATORS CALCULATED", '=')

indicator_cols = [col for col in df.columns if col not in 
                  ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]

print(f"Total Indicators: {len(indicator_cols)}\n")

categories = {
    '📊 Trend Indicators': ['sma_', 'ema_', 'macd'],
    '⚡ Momentum Indicators': ['rsi', 'roc', 'stoch'],
    '📉 Volatility Indicators': ['bb_', 'atr', 'volatility'],
    '📦 Volume Indicators': ['volume_', 'obv'],
    '💰 Price Features': ['price_change', 'daily_range', 'gap']
}

for category, keywords in categories.items():
    indicators = [col for col in indicator_cols if any(kw in col for kw in keywords)]
    print(f"\n{category} ({len(indicators)}):")
    for ind in indicators:
        print(f"  • {ind}")

# Current Market Snapshot
print_header("💹 CURRENT MARKET SNAPSHOT", '=')

for file in sorted(files):
    ticker = file.replace('_processed.csv', '')
    df = pd.read_csv(os.path.join(processed_dir, file))
    
    current_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2]
    change = ((current_price / prev_price) - 1) * 100
    rsi = df['rsi'].iloc[-1]
    
    arrow = "🟢" if change > 0 else "🔴"
    
    print(f"{ticker:<8} ${current_price:>8.2f}  {arrow} {change:>6.2f}%  |  RSI: {rsi:>5.1f}")

# Performance Statistics
print_header("📊 PERFORMANCE OVER 5 YEARS", '=')

for file in sorted(files):
    ticker = file.replace('_processed.csv', '')
    df = pd.read_csv(os.path.join(processed_dir, file))
    
    start_price = df['close'].iloc[0]
    end_price = df['close'].iloc[-1]
    total_return = ((end_price / start_price) - 1) * 100
    
    max_price = df['close'].max()
    min_price = df['close'].min()
    avg_volume = df['volume'].mean()
    
    print(f"\n{ticker}:")
    print(f"  Start Price:  ${start_price:>8.2f}")
    print(f"  Current:      ${end_price:>8.2f}")
    print(f"  Total Return: {total_return:>7.2f}%")
    print(f"  High:         ${max_price:>8.2f}")
    print(f"  Low:          ${min_price:>8.2f}")
    print(f"  Avg Volume:   {avg_volume:>12,.0f}")

print_header("✅ PHASE 1 COMPLETE - READY FOR MACHINE LEARNING!", '=')
print("\n🎯 Next Step: Phase 2 - Build ML Models")
print("   • LSTM Neural Networks")
print("   • Random Forest")
print("   • XGBoost")
print("   • Model Comparison & Backtesting\n")